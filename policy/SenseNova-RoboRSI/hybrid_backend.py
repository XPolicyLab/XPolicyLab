"""Run the pinned hybrid rules against public observations and deferred joints.

This boundary never creates, resets, advances or scores a simulator. Its only
motion output is a joint request that Model returns to the official evaluator.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import math
import queue
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_SHA = re.compile(r"[a-f0-9]{64}\Z")
_IMPORT_LOCK = threading.Lock()
_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_JOINT_CHANNELS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


class BackendClosed(RuntimeError):
    """The owning episode was cancelled; no motion may be retried."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise ValueError("An externally pinned SHA256 is required")
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"Missing or changed pinned resource: {path.name}")


@dataclass(frozen=True)
class ReferenceSource:
    """Verified original modules, shared with the external tool host."""

    source_dir: Path
    manifest_sha256: str
    HybridAdapter: type
    DualKinematics: type
    InvalidAction: type
    runner: Any
    hybrid_host: Any


def load_reference_source(
    source_dir: Path, manifest_path: Path, expected_manifest_sha256: str
) -> ReferenceSource:
    """Verify original provenance before importing any loose reference module.

    The original adaptive provenance uses entries ``source/<relative path>``.
    Existing imports from another source tree are rejected rather than reused.
    Importing these modules does not call their simulator-owning constructors.
    """
    import json

    source_dir, manifest_path = Path(source_dir).absolute(), Path(manifest_path)
    _verify(manifest_path, expected_manifest_sha256)
    rows = json.loads(manifest_path.read_text())["files"]
    verified = set()
    for row in rows:
        name = row["path"]
        if not name.startswith("source/"):
            continue
        relative = Path(name.removeprefix("source/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe reference provenance path")
        path = source_dir / relative
        if not path.resolve().is_relative_to(source_dir.resolve()):
            raise ValueError("Reference source escapes its verified directory")
        _verify(path, row["sha256"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Reference source size differs")
        if relative.as_posix() in verified:
            raise ValueError("Duplicate reference provenance path")
        verified.add(relative.as_posix())
    adapters = "code/configurations/robodojo/20261005_standard_1x_10x/adapters"
    modules = {
        "policylab": "code/policylab/__init__.py",
        **{
            f"policylab.{name}": f"code/policylab/{name}.py"
            for name in ("api", "common", "codex", "backend", "runner")
        },
        "adapter": f"{adapters}/adapter.py",
        "trial_adapter": f"{adapters}/timeout_chunk_trials/trial_adapter.py",
        "eef_kinematics": f"{adapters}/eef_kinematics.py",
        "waypoints": "waypoints.py",
        "hybrid_adapter": "hybrid_adapter.py",
        "hybrid_host": "hybrid_host.py",
    }
    if not set(modules.values()).issubset(verified):
        raise ValueError("Reference provenance omits an imported module")
    with _IMPORT_LOCK:
        for name, relative in modules.items():
            if (
                name in sys.modules
                and Path(getattr(sys.modules[name], "__file__", "")).resolve()
                != (source_dir / relative).resolve()
            ):
                raise ValueError(
                    f"Reference module already loaded from another tree: {name}"
                )
        paths = [
            source_dir,
            source_dir / "code",
            source_dir / adapters,
            source_dir / adapters / "timeout_chunk_trials",
        ]
        sys.path[:0] = [str(path) for path in paths if str(path) not in sys.path]
        imported = {name: importlib.import_module(name) for name in modules}
    return ReferenceSource(
        source_dir,
        expected_manifest_sha256,
        imported["hybrid_adapter"].Adapter,
        imported["eef_kinematics"].DualKinematics,
        imported["policylab.api"].InvalidAction,
        imported["policylab.runner"],
        imported["hybrid_host"],
    )


class _CpuArray:
    """Read-only numpy facade for the original robot metadata access pattern."""

    def __init__(self, data):
        self.data = np.asarray(data)

    def __getitem__(self, key):
        return _CpuArray(self.data[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.data.copy()


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} requires {size} finite values")
    array = array.copy()
    array.flags.writeable = False
    return array


def _observation(obs: dict) -> dict:
    if (
        not isinstance(obs, dict)
        or not isinstance(obs.get("state"), dict)
        or not isinstance(obs.get("vision"), dict)
    ):
        raise ValueError("Hybrid requires official state and RGB vision mappings")
    instruction = obs.get("instruction") or obs.get("instructions")
    if (
        isinstance(instruction, (list, tuple))
        and instruction
        and all(isinstance(item, str) and item.strip() for item in instruction)
    ):
        instruction = "\n".join(instruction)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Hybrid requires an instruction")
    state = {
        name: _finite_vector(obs["state"][name], size, name)
        for name, size in (*_JOINT_CHANNELS, ("left_ee_pose", 7), ("right_ee_pose", 7))
    }
    vision = {}
    for camera in _CAMERAS:
        color = obs["vision"][camera]["color"]
        if (
            not isinstance(color, np.ndarray)
            or color.dtype != np.uint8
            or color.ndim != 3
            or color.shape[2] != 3
            or not all(color.shape[:2])
        ):
            raise ValueError("Hybrid requires three decoded uint8 H,W,3 RGB cameras")
        color = color.copy()
        color.flags.writeable = False
        vision[camera] = {"color": color}
    frequency = obs.get("additional_info", {}).get("frequency", 25)
    if (
        isinstance(frequency, bool)
        or not isinstance(frequency, (int, float))
        or not math.isfinite(frequency)
        or frequency <= 0
    ):
        raise ValueError("Invalid official observation frequency")
    return {
        "instruction": instruction,
        "state": state,
        "vision": vision,
        "additional_info": {"frequency": frequency},
    }


@dataclass
class PendingJointAction:
    """One action awaiting its official executed observation acknowledgement."""

    request_id: int
    step_index: int
    action: dict[str, list[float]]
    kind: str = "joint"
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _delivered: bool = field(default=False, repr=False)
    _error: BaseException | None = field(default=None, repr=False)


class HybridBackend:
    """Original hybrid adapter rules with an observation-only environment facade."""

    def __init__(
        self,
        reference: ReferenceSource,
        *,
        task_name: str,
        episode_dir: Path,
        initial_observation: dict,
        step_limit: int,
        metadata_path: Path,
        expected_metadata_sha256: str,
        resource_root: Path,
        vla_port: int,
        replay: dict | None = None,
        vla_session_id: str | None = None,
    ):
        import json

        if not isinstance(task_name, str) or not task_name.isidentifier():
            raise ValueError("An exact RoboDojo task name is required")
        if type(step_limit) is not int or step_limit <= 0:
            raise ValueError("The exact public native step limit is required")
        if type(vla_port) is not int or not 1 <= vla_port <= 65535:
            raise ValueError("An explicit local VLA port is required")
        self.reference, self.task_name = reference, task_name
        self.directory = Path(episode_dir).resolve()
        if not self.directory.is_dir():
            raise ValueError("Create the owned episode directory first")
        self.step_limit = step_limit
        self._latest = _observation(initial_observation)
        self._step, self._serial = 0, 0
        self._terminated, self._closed, self._reset = False, False, False
        self._metrics = {}
        self._lock = threading.RLock()
        self._actions = queue.Queue()
        self._pending: PendingJointAction | None = None
        self._connection = None
        _verify(Path(metadata_path), expected_metadata_sha256)
        metadata = json.loads(Path(metadata_path).read_text())
        if metadata.get("schema_version") != 1 or metadata.get("robot") != "arx_x5":
            raise ValueError("Expected pinned arx_x5 public robot metadata")
        sources = metadata.get("sources")
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                not isinstance(v, str) or not _SHA.fullmatch(v)
                for v in sources.values()
            )
        ):
            raise ValueError(
                "Public robot metadata requires exact source SHA256 identities"
            )
        if set(metadata["arms"]) != {"left", "right"}:
            raise ValueError("Public robot metadata requires left/right arms")
        resource_root = Path(resource_root).resolve()
        robots, keys, root_poses = [], [], {}
        for arm in ("left", "right"):
            spec = metadata["arms"][arm]
            relative = Path(spec["urdf_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Robot URDF must be inside the pinned resource root")
            urdf = resource_root / relative
            if not urdf.resolve().is_relative_to(resource_root):
                raise ValueError("Robot URDF escapes resource root")
            _verify(urdf, spec["urdf_sha256"])
            names = spec["joint_names"]
            if (
                not isinstance(names, list)
                or len(names) != 6
                or len(set(names)) != 6
                or any(not isinstance(n, str) or not n for n in names)
            ):
                raise ValueError("Exact six-joint native ordering is required")
            limits = np.asarray(spec["soft_joint_pos_limits"], dtype=float)
            if (
                limits.shape != (6, 2)
                or not np.isfinite(limits).all()
                or np.any(limits[:, 0] >= limits[:, 1])
            ):
                raise ValueError("Exact six-pair native soft joint limits are required")
            root_pose = _finite_vector(spec["root_pose"], 7, "root_pose")
            if abs(np.linalg.norm(root_pose[3:]) - 1) > 1e-4:
                raise ValueError("Robot root quaternion must be unit wxyz")
            if not all(
                isinstance(spec[name], str) and spec[name]
                for name in ("base_link", "ee_link")
            ):
                raise ValueError("Exact native base and EEF links are required")
            robot = SimpleNamespace(
                type="target",
                arm_name=arm + "_arm",
                urdf_path=str(urdf),
                arm_joints_name=names,
                base_link=spec["base_link"],
                ee_link_name=spec["ee_link"],
                arm_joint_indices=np.arange(6),
            )
            robots.append(robot)
            keys.append(
                SimpleNamespace(
                    data=SimpleNamespace(soft_joint_pos_limits=_CpuArray(limits[None]))
                )
            )
            root_poses[arm] = root_pose

        owner = self

        class Manager:
            robot_list, robot_key = robots, keys

            @staticmethod
            def get_joint(robot):
                return {
                    0: owner._latest["state"][
                        robot.arm_name.split("_")[0] + "_arm_joint_state"
                    ]
                }

            @staticmethod
            def get_real_endpose(robot):
                return {
                    0: owner._latest["state"][robot.arm_name.split("_")[0] + "_ee_pose"]
                }

            @staticmethod
            def get_link_pose(robot, link_name, *, is_relative):
                if link_name != robot.base_link or is_relative is not True:
                    raise ValueError(
                        "Only public environment-origin robot root is available"
                    )
                return {0: root_poses[robot.arm_name.split("_")[0]]}

        class Environment:
            step_lim = step_limit
            robot_manager = Manager()

            @property
            def end_flag(self):
                return [owner._terminated]

            @property
            def take_action_cnt(self):
                return [owner._step]

            @staticmethod
            def get_obs():
                return owner._latest

        class PublicAdapter(reference.HybridAdapter):
            def __init__(self):
                # Do not call NativeAdapter.__init__: it owns a simulator.
                self.env = Environment()
                self.task, self.action_mode = task_name, "eef"
                self.vla_horizon, self.vla_port = 50, vla_port
                self.record_dir = self.observation_dir = None
                self.frame = 0
                self.replay = copy.deepcopy(replay)
                self.replay_consumed = False
                self.kinematics = reference.DualKinematics(self.env)

            @property
            def connection(self):
                return owner._connection

            @connection.setter
            def connection(self, value):
                if value is not None and vla_session_id is not None:
                    from .simate_bridge import SimateConnection

                    value = SimateConnection(value, vla_session_id, lambda: owner._step)
                with owner._lock:
                    closed = owner._closed
                    if not closed:
                        owner._connection = value
                if closed and value is not None:
                    value.close()
                    raise BackendClosed("Hybrid episode closed during VLA connection")

            def reset(self, episode, record_dir, observation_dir):
                if self.replay is not None:
                    from .hybrid_resume import original_replay_reset

                    return original_replay_reset(
                        reference,
                        self,
                        self._begin_reset,
                        episode=episode,
                        record_dir=record_dir,
                        observation_dir=observation_dir,
                    )
                return self._begin_reset(episode, record_dir, observation_dir)

            def _begin_reset(self, episode, record_dir, observation_dir):
                with owner._lock:
                    if owner._reset:
                        raise reference.InvalidAction(
                            "The official evaluator owns resets"
                        )
                    if (
                        episode.get("task") != task_name
                        or episode.get("robot") != "arx_x5"
                    ):
                        raise ValueError("Hybrid reset differs from the public task")
                    self.record_dir = owner._owned_directory(record_dir)
                    self.observation_dir = owner._owned_directory(observation_dir)
                    owner._reset = True
                self.kinematics.check()
                return self._transition()

            def _transition(self, applied_action=None):
                from PIL import Image

                raw = owner._latest
                pictures = {}
                for camera in _CAMERAS:
                    name = f"{self.frame:05d}_{camera}.png"
                    path = self.observation_dir / name
                    if path.exists() or path.is_symlink():
                        raise ValueError("Refusing to overwrite an episode observation")
                    # Match the original controller's lossless RGB tool images.
                    Image.fromarray(raw["vision"][camera]["color"]).save(
                        path, format="PNG"
                    )
                    pictures[camera] = "obs://" + name
                observation = {
                    "instruction": raw["instruction"],
                    "state": {k: v.tolist() for k, v in raw["state"].items()},
                    "cameras": pictures,
                    "control_step": owner._step,
                    "control_frequency_hz": raw["additional_info"]["frequency"],
                    "step_limit": step_limit,
                    "steps_remaining": max(0, step_limit - owner._step),
                }
                self.frame += 1
                result = {
                    "observation": observation,
                    "done": owner._terminated,
                    "metrics": copy.deepcopy(owner._metrics)
                    if owner._terminated
                    else {},
                }
                if applied_action is not None:
                    result["applied_action"] = applied_action
                return result

            def _step_joint(self, action):
                owner._submit_joint(action)
                return self._transition()

            def finish(self):
                if not owner._terminated:
                    owner._submit_finish()
                return self._transition()

            def close(self):
                owner.cancel()

        self._adapter = PublicAdapter()
        self.metadata_identity = {
            "sha256": expected_metadata_sha256,
            "sources": sources,
            "reference_manifest_sha256": reference.manifest_sha256,
        }

    def _owned_directory(self, path) -> Path:
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory):
            raise ValueError(
                "Hybrid artifacts must stay in the owned episode directory"
            )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def control_step(self) -> int:
        return self._step

    def request(self, **request):
        if self._closed:
            raise BackendClosed("Hybrid episode has been closed")
        op = request.pop("op", None)
        if op == "reset":
            return self._adapter.reset(**request)
        if not self._reset:
            raise ValueError("Reset the public hybrid bridge before requesting tools")
        if op == "vla_propose":
            return self._adapter.vla_propose(**request)
        if op == "step":
            return self._adapter.step(**request)
        if op == "finish":
            return self._adapter.finish(**request)
        raise ValueError("Unsupported hybrid backend operation")

    def _submit_joint(self, action: dict) -> None:
        if not isinstance(action, dict) or set(action) != {
            name for name, _ in _JOINT_CHANNELS
        }:
            raise self.reference.InvalidAction(
                "Explicit dual-arm joint and gripper targets required"
            )
        for name, size in _JOINT_CHANNELS:
            values = action[name]
            if (
                not isinstance(values, list)
                or len(values) != size
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in values
                )
            ):
                raise self.reference.InvalidAction(
                    "Finite native joint targets required"
                )
            if size == 1 and not 0 <= values[0] <= 1:
                raise self.reference.InvalidAction(
                    "Native gripper opening must be in [0,1]"
                )
        with self._lock:
            if self._closed:
                raise BackendClosed("Hybrid episode closed before motion")
            if self._terminated or self._step >= self.step_limit:
                raise self.reference.InvalidAction("Native episode is over")
            if self._pending is not None:
                raise RuntimeError("A native action already awaits acknowledgement")
            self._serial += 1
            pending = PendingJointAction(
                self._serial, self._step, copy.deepcopy(action)
            )
            self._pending = pending
            self._actions.put(pending)
        pending._event.wait()
        if pending._error is not None:
            raise pending._error

    def _submit_finish(self) -> None:
        """Ask the native deploy owner to finalize; spend no control step."""
        with self._lock:
            if self._closed:
                raise BackendClosed("Hybrid episode closed before finalization")
            if self._pending is not None:
                raise RuntimeError("A native action already awaits acknowledgement")
            self._serial += 1
            pending = PendingJointAction(self._serial, self._step, {}, kind="finish")
            self._pending = pending
            self._actions.put(pending)
        pending._event.wait()
        if pending._error is not None:
            raise pending._error

    def next_action(self, timeout: float | None = None) -> PendingJointAction:
        if self._closed:
            raise BackendClosed("Hybrid episode is closed")
        pending = self._actions.get(timeout=timeout)
        if isinstance(pending, BackendClosed):
            raise pending
        with self._lock:
            if self._closed or pending is not self._pending:
                raise BackendClosed("Queued hybrid action is no longer current")
            pending._delivered = True
        return pending

    def acknowledge(
        self,
        request_id: int,
        observation: dict,
        *,
        terminated: bool = False,
        native_metrics: dict | None = None,
    ) -> None:
        """ACK one real executed action; unknown scoring remains explicitly None."""
        latest = _observation(observation)
        if type(terminated) is not bool:
            raise ValueError("Native termination must be explicit boolean")
        if native_metrics is None:
            metrics = (
                {"success": None, "source": "not_exposed_to_policy"}
                if terminated
                else {}
            )
        else:
            metrics = copy.deepcopy(native_metrics)
            if (
                metrics.get("success") is not None
                and type(metrics["success"]) is not bool
            ):
                raise ValueError("Native success must be boolean or explicitly unknown")
            if "success" not in metrics:
                raise ValueError(
                    "Native metrics must explicitly identify success or unknown"
                )
        with self._lock:
            pending = self._pending
            if self._closed:
                raise BackendClosed("Hybrid episode closed before acknowledgement")
            if (
                pending is None
                or pending.request_id != request_id
                or not pending._delivered
            ):
                raise ValueError("Acknowledge only the current delivered native action")
            if pending.kind == "finish" and not terminated:
                raise ValueError("Native finalization requires terminal acknowledgement")
            self._latest, self._terminated, self._metrics = latest, terminated, metrics
            self._step += int(pending.kind == "joint")
            self._pending = None
            pending._event.set()

    def cancel(self) -> None:
        """Release waits without acquiring Experiment's lock held by a review."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            error = BackendClosed("Hybrid episode cancelled")
            if self._pending is not None:
                self._pending._error = error
                self._pending._event.set()
                self._pending = None
            self._actions.put(error)
            connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    close = cancel
