"""Transparent XPolicyLab deploy instrumentation for direct OpenWAM evaluation.

The existing OpenWAM deploy loop and model remain authoritative. This wrapper
only projects permitted observations and records what crosses the policy and
native-action boundaries. It never interprets success or changes actions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from .record import OBSERVATION_FORMAT, OBSERVATION_PACK, OBSERVATION_STORED, encoded, lossless_png, npy_bytes, sha_bytes
from .runtime_config import report_env

ALLOWED_OBSERVATION = {"vision", "state", "instruction", "task_instruction", "env_idx"}
FORBIDDEN_OBSERVATION = {"reward", "success", "score", "layout_id", "oracle", "task_env", "remaining_steps"}
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_FIELDS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")
# What the launcher passes (robodojo.launch, from the frozen campaign config's observation_evidence block) to say
# which observation bytes BridgeLog keeps; absent, everything is kept. JSON: {"retention": ..., "head_every": k}.
RETENTION_ENV = "OEH_EVAL_OBSERVATION_RETENTION"
RETENTION_ALL = "all"
RETENTION_READ = "read-endpoints-head"
RETENTIONS = (RETENTION_ALL, RETENTION_READ)
HEAD_CAMERA_LABEL = "obs/vision/cam_head/color"


def _simple(value: Any) -> Any:
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _simple(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_simple(v) for v in value]
    return value


def _encoded_buffer(value: Any) -> bool:
    """XPolicyLab clients may send a camera's encoded image bits as a 1-D uint8 array instead of bytes (its debug
    client alternates the two; `decode_image_bit` accepts both). Such a frame is recorded as the encoded bytes it is."""
    import numpy as np
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.uint8 and value.size > 0


def parse_retention(value: Any) -> dict[str, Any]:
    """The retention policy as recorded: {"retention": all | read-endpoints-head, "head_every": k >= 1}.

    ``all`` stores every frame and array. ``read-endpoints-head`` stores every array, every frame of an observation
    the model read (a get_action followed it before the next observation), every frame of the first and the last
    observation, and the head camera of every k-th observation; the other frames are dropped after the next
    observation replaces them, their hashes kept in the observation event."""
    if value is None:
        return {"retention": RETENTION_ALL, "head_every": 1}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping) or set(value) - {"retention", "head_every"} or \
            value.get("retention") not in RETENTIONS:
        raise ValueError(f"observation retention must be {{retention: one of {RETENTIONS}, head_every}}")
    head_every = value.get("head_every", 1 if value["retention"] == RETENTION_ALL else 5)
    if type(head_every) is not int or head_every < 1:
        raise ValueError("observation retention head_every must be a positive integer")
    return {"retention": value["retention"], "head_every": head_every}


class BridgeLog:
    """The attempt's policy event sidecar and its observation evidence (record.OBSERVATION_FORMAT).

    Every observation's event carries the identity of each frame and array (sha256 of its .npy bytes, the PNG's own
    hash and size) the moment it crosses the boundary. Which bytes are then kept follows the retention policy: the
    kept slices go to one append-only pack per attempt and are named by an observation_stored event, so the event
    chain stays append-only and a validator checks exactly the slices that exist.
    """

    def __init__(self, path: Path, permitted: set[str] | None = None, *, stmem_calibration: bool = False,
                 retention: Mapping[str, Any] | str | None = None):
        self.stmem_calibration = stmem_calibration
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # RoboDojo re-enters eval_one_episode after an exception; one attempt records exactly one episode.
            raise FileExistsError(f"policy event sidecar already exists: {self.path} (the runner re-entered the episode "
                                  "after a failure; the campaign records one episode per attempt)")
        self.path.touch(exist_ok=False)
        self.directory = self.path.parent / "policy-observations"
        self.directory.mkdir(exist_ok=False)
        self.pack = self.path.parent / OBSERVATION_PACK
        self.pack.touch(exist_ok=False)
        self.retention = parse_retention(os.environ.get(RETENTION_ENV) if retention is None else retention)
        self.observation_index = 0
        self.action_index = 0
        self.closed = False
        # Observations the model holds but has not read: [(index, files, blobs, slots not yet stored)]. They are
        # stored when a get_action follows (read) or at close (last), and dropped when the next observation replaces
        # them.
        self.pending: list[tuple[int, list[dict], list[bytes], list[int]]] = []
        self.counts = {"observations": 0, "stored_read": 0, "stored_last": 0, "dropped": 0}
        if permitted is not None and permitted - ALLOWED_OBSERVATION:
            raise ValueError(f"unsupported policy observation fields: {sorted(permitted - ALLOWED_OBSERVATION)}")
        self.permitted = ALLOWED_OBSERVATION if permitted is None else permitted
        # Camera frames are encoded (lossless PNG, record.lossless_png) on worker threads.
        self.writers = ThreadPoolExecutor(max_workers=len(CAMERAS), thread_name_prefix="oeh-evidence")
        self.append("evidence_policy", {"format": OBSERVATION_FORMAT, "pack": OBSERVATION_PACK, **self.retention})

    def _store(self, index: int, files: list[dict], blobs: list[bytes], slots: list[int], reason: str) -> None:
        """Append the given slots of one observation to the pack (fsync once) and name them in a storage event."""
        if not slots:
            return
        named = []
        with self.pack.open("ab") as stream:
            for slot in slots:
                offset = stream.tell()
                stream.write(blobs[slot])
                named.append({"slot": slot, "path": OBSERVATION_PACK, "offset": offset, "length": len(blobs[slot])})
            stream.flush()
            os.fsync(stream.fileno())
        self.append(OBSERVATION_STORED, {"index": index, "reason": reason, "files": named})

    def _keep_now(self, index: int, files: list[dict]) -> list[int]:
        """Slots stored at observation time, whatever the model does next."""
        if self.retention["retention"] == RETENTION_ALL or index == 0:
            return list(range(len(files)))
        return [slot for slot, item in enumerate(files)
                if item.get("label") not in {f"obs/vision/{camera}/color" for camera in CAMERAS}
                or (item.get("label") == HEAD_CAMERA_LABEL and index % self.retention["head_every"] == 0)]

    def observations_replaced(self) -> None:
        """The model is handed new observations: the unread ones it held are dropped (their hashes stay)."""
        for _, _, _, remaining in self.pending:
            self.counts["dropped"] += len(remaining)
        self.pending = []

    def observations_read(self) -> None:
        """The model asked for an action: every observation it holds was read, so its frames are stored."""
        for index, files, blobs, remaining in self.pending:
            self._store(index, files, blobs, remaining, "read")
            self.counts["stored_read"] += len(remaining)
        self.pending = []

    def close(self) -> None:
        """The episode ended: the observation the model still holds is the last one, kept whole."""
        if self.closed:
            return
        self.closed = True
        for index, files, blobs, remaining in self.pending:
            self._store(index, files, blobs, remaining, "last")
            self.counts["stored_last"] += len(remaining)
        self.pending = []
        self.writers.shutdown(wait=True)
        self.append("evidence_retention", dict(self.counts))

    def append(self, kind: str, payload: Mapping[str, Any]) -> None:
        row = {"kind": kind, "payload": _simple(payload), "mono_ns": time.monotonic_ns()}
        with self.path.open("ab") as stream:
            stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())

    def observation(self, obs: Mapping[str, Any]) -> Mapping[str, Any]:
        import numpy as np
        if self.closed:
            raise RuntimeError("the evidence log is closed")
        if not isinstance(obs, Mapping) or FORBIDDEN_OBSERVATION.intersection(obs):
            raise ValueError("evaluator-only state cannot enter a policy observation")
        projected = {key: obs[key] for key in obs if key in self.permitted}
        if "vision" not in projected or "state" not in projected:
            raise ValueError("RoboDojo policy observation needs vision and state")
        if not isinstance(projected["vision"], Mapping) or not isinstance(projected["state"], Mapping):
            raise ValueError("vision and state must be mappings")
        if any(key not in projected["vision"] for key in CAMERAS) or any(
            key not in projected["state"] for key in STATE_FIELDS
        ):
            raise ValueError("OpenWAM's required cameras or proprioception are missing")
        vision = {}
        for key in CAMERAS:
            camera = projected["vision"][key]
            if not isinstance(camera, Mapping):
                raise ValueError(f"{key} color must be an image array or encoded bytes")
            color = camera.get("color")
            if _encoded_buffer(color):
                pass  # XPolicyLab's encoded image bits in a 1-D uint8 buffer (decode_image_bit reads both forms)
            elif isinstance(color, np.ndarray):
                if color.dtype.kind not in "biuf" or color.ndim != 3:
                    raise ValueError(f"{key} color must be a numeric image array")
            elif not isinstance(color, (bytes, bytearray, memoryview)) or not color:
                raise ValueError(f"{key} color must be an image array or nonempty encoded bytes")
            vision[key] = {"color": color}
            if self.stmem_calibration and (key=='cam_head' or camera.get('stmem_calibration') is not None):
                from oeh_policy.stmem_calibration import project_calibration
                vision[key]["stmem_calibration"] = project_calibration(camera)
        # RoboDojo emits poses as arrays and the gripper openings as one-element lists; the OEH gateway hands its
        # projected arrays. The policy receives the object as given; the record hashes one canonical form
        # (float64, fixed shape) so both conditions fingerprint the same observation identically.
        state, canonical_state = {}, {}
        for key in STATE_FIELDS:
            value = projected["state"][key]
            raw = value.detach().cpu().numpy() if hasattr(value, "detach") else value
            try:
                array = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be numeric state data") from exc
            expected = (7,) if key.endswith("_pose") else (1,)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"{key} must be a finite numeric state of shape {expected}")
            state[key] = value
            canonical_state[key] = array
        for key in ("instruction", "task_instruction"):
            if key in projected and not isinstance(projected[key], str):
                raise ValueError(f"{key} must be policy text")
        if "env_idx" in projected and not isinstance(projected["env_idx"], (int, np.integer)):
            raise ValueError("env_idx must be an integer")
        projected["vision"] = vision
        projected["state"] = state
        index = self.observation_index
        files: list[dict] = []
        jobs: list = []  # per file entry: its bytes, or (PNG encoding job, .npy identity) for an array

        def materialize(value: Any, label: str) -> Any:
            if isinstance(value, (bytes, bytearray, memoryview)) or (label.endswith("/color") and _encoded_buffer(value)):
                data = bytes(value)
                item = {"sha256": sha_bytes(data), "kind": "encoded_bytes", "size": len(data), "label": label}
                files.append(item)
                jobs.append(data)
                return {"bytes_sha256": item["sha256"], "size": item["size"]}
            if isinstance(value, np.ndarray):
                # The identity is the .npy serialization's sha256 whatever the stored format (record.lossless_png).
                identity = npy_bytes(value)
                item = {"sha256": sha_bytes(identity), "dtype": str(value.dtype), "shape": list(value.shape),
                        "label": label}
                files.append(item)
                jobs.append((self.writers.submit(lossless_png, value, identity), identity))
                return {"array_sha256": item["sha256"], "dtype": item["dtype"], "shape": item["shape"]}
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Mapping):
                return {str(k): materialize(v, label + "/" + str(k)) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [materialize(v, label + "/" + str(i)) for i, v in enumerate(value)]
            return value

        descriptor = materialize({**projected, "state": canonical_state}, "obs")
        blobs: list[bytes] = []
        for item, job in zip(files, jobs):  # every frame encoded (or the exception that stopped it) before the event
            if isinstance(job, tuple):
                png = job[0].result()
                if png is not None:
                    item.update(format="png", file_sha256=sha_bytes(png), bytes=len(png))
                else:
                    item["bytes"] = len(job[1])
                blobs.append(job[1] if png is None else png)
            else:
                blobs.append(job)
        digest = sha_bytes(encoded(descriptor))
        self.append("observation", {"index": index, "sha256": digest, "evidence_format": OBSERVATION_FORMAT,
                                    "fields": sorted(projected), "descriptor": descriptor, "files": files})
        self.observation_index += 1
        self.counts["observations"] += 1
        now = self._keep_now(index, files)
        self._store(index, files, blobs, now, "retained")
        remaining = [slot for slot in range(len(files)) if slot not in now]
        if remaining:
            self.pending.append((index, files, blobs, remaining))
        return projected

    def action(self, action: Any) -> None:
        value = _simple(action)
        self.append("action_proposed", {"index": self.action_index, "sha256": sha_bytes(encoded(value)),
                                        "action": value})
        self.action_index += 1


class TaskEnvProxy:
    def __init__(self, source: Any, log: BridgeLog, *, record_proposals: bool = True):
        self.source, self.log = source, log
        self.record_proposals = record_proposals

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def get_obs(self):
        return self.source.get_obs()

    def get_obs_batch(self, env_idx_list=None, last_frame=False):
        return self.source.get_obs_batch(env_idx_list, last_frame=last_frame)

    def take_action(self, action):
        if self.record_proposals:
            self.log.action(action)
        else:
            value = _simple(action)
            self.log.append("native_action_submitted", {"sha256": sha_bytes(encoded(value)), "action": value})
        start = time.monotonic_ns()
        result = self.source.take_action(action)
        payload = {"duration_ns": time.monotonic_ns() - start}
        if self.record_proposals:
            payload["action_index"] = self.log.action_index - 1
        self.log.append("native_step_completed", payload)
        return result

    def take_action_batch(self, actions_list, env_idx_list=None):
        for action in actions_list:
            if self.record_proposals:
                self.log.action(action)
            else:
                value = _simple(action)
                self.log.append("native_action_submitted", {"sha256": sha_bytes(encoded(value)), "action": value})
        start = time.monotonic_ns()
        result = self.source.take_action_batch(actions_list, env_idx_list=env_idx_list)
        payload = {"batch_size": len(actions_list), "duration_ns": time.monotonic_ns() - start}
        if self.record_proposals:
            payload["action_index"] = self.log.action_index - 1
        self.log.append("native_step_completed", payload)
        return result


class ModelClientProxy:
    def __init__(self, source: Any, log: BridgeLog):
        self.source, self.log = source, log

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        method = kwargs.get("func_name") or (args[0] if args else None)
        if method not in {"reset", "update_obs", "update_obs_batch", "get_action", "get_action_batch"}:
            raise ValueError(f"unexpected XPolicyLab model method: {method}")
        if method == "update_obs":
            if "obs" not in kwargs:
                raise ValueError("update_obs requires an observation")
            self.log.observations_replaced()  # the adapter keeps only the latest batch (XPolicyLab model.py)
            kwargs = {**kwargs, "obs": self.log.observation(kwargs["obs"])}
        elif method == "update_obs_batch":
            if not isinstance(kwargs.get("obs"), list):
                raise ValueError("update_obs_batch requires an observation list")
            self.log.observations_replaced()
            kwargs = {**kwargs, "obs": [self.log.observation(item) for item in kwargs["obs"]]}
        elif method in {"get_action", "get_action_batch"}:
            self.log.observations_read()  # the held observations are what this action is computed from
        start = time.monotonic_ns()
        result = self.source.call(*args, **kwargs)
        self.log.append("model_call", {"method": method, "duration_ms": (time.monotonic_ns() - start) / 1e6,
                                       "response_sha256": sha_bytes(encoded(_simple(result)))
                                       if method in {"get_action", "get_action_batch"} else None})
        return result


def _evaluate_with_boundary(task_env: Any, model_client: Any, *, direct: bool) -> Any:
    from XPolicyLab.policy.OpenWAM.deploy import eval_one_episode
    event_path = os.environ.get("OEH_EVAL_EVENTS")
    if not event_path:
        raise RuntimeError("OEH_EVAL_EVENTS must name a fresh sidecar path")
    permitted = os.environ.get("OEH_EVAL_PERMITTED_OBSERVATIONS")
    allowed = set(permitted.split(",")) if permitted else None
    log = BridgeLog(Path(event_path), allowed)
    report_env(task_env)
    try:
        return eval_one_episode(TaskEnvProxy(task_env, log, record_proposals=direct),
                                ModelClientProxy(model_client, log))
    finally:
        log.close()  # the observation the model still holds is the episode's last: stored whole


def evaluate_direct(task_env: Any, model_client: Any) -> Any:
    """Run the official loop through the direct OpenWAM model-client boundary."""
    return _evaluate_with_boundary(task_env, model_client, direct=True)


def evaluate_oeh(task_env: Any, model_client: Any) -> Any:
    """Run the official loop through the external OEH model-client boundary.

    It holds no OEH execution leases; the campaign's OEH adapters run `oeh_policy.deploy` instead
    (`oeh_eval.install.OEH_DEPLOY_SOURCE`), which records through this module's boundary and gates each step.
    """
    return _evaluate_with_boundary(task_env, model_client, direct=False)
