"""Client-owned episode context for atomic, multi-client policy requests."""

import math
import os
from uuid import uuid4

import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit

from ..config import ObservationConfig
from ..model.memory import EpisodeMemory, resize_head


class ContextClient:
    """One outstanding request per client; no episode state lives on the server."""

    def __init__(self, metadata: dict, *, sparse_capture: bool = False):
        if (
            not isinstance(metadata, dict)
            or set(metadata)
            != {"protocol", "config_hash", "observation", "cameras", "replan_steps"}
            or metadata["protocol"] != 2
        ):
            raise ValueError("Server must supply the HWM v2 observation contract")
        self.config = ObservationConfig(metadata["observation"])
        self.config_hash = metadata["config_hash"]
        if (
            not isinstance(self.config_hash, str)
            or len(self.config_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.config_hash)
        ):
            raise ValueError("Invalid server configuration fingerprint")
        self.cameras = metadata["cameras"]
        if (
            not isinstance(self.cameras, list)
            or len(self.cameras) != 3
            or any(not isinstance(x, str) or not x for x in self.cameras)
            or len(set(self.cameras)) != 3
        ):
            raise ValueError("Server must supply three distinct camera fields")
        self.replan_steps = metadata["replan_steps"]
        if (
            type(self.replan_steps) is not int
            or not 1 <= self.replan_steps <= self.config.horizon
        ):
            raise ValueError("Invalid server replan_steps")
        self.episodes = {}
        self.pending = None
        self.sparse_capture = sparse_capture

    def capture_due(self, indices):
        """Select only frames needed by future configured history samples."""
        return [
            index
            for index in indices
            if self.episodes[index]["control_step"]
            % self.episodes[index]["memory"].capture_stride
            == 0
        ]

    def advance(self, indices):
        """Advance actual control time independently of observation capture."""
        if self.pending is not None:
            raise RuntimeError("Commit the pending response before advancing")
        for index in indices:
            self.episodes[index]["control_step"] += 1

    def observe(self, observations):
        if self.pending is not None:
            raise RuntimeError("Commit the pending response before observing again")
        seen = set()
        for obs in observations:
            index = int(obs.get("env_idx", 0))
            if index in seen:
                raise ValueError("Duplicate environment observation")
            seen.add(index)
            if index not in self.episodes:
                self.episodes[index] = {
                    "id": uuid4().hex,
                    "version": 0,
                    "memory": EpisodeMemory(self.config),
                    "control_step": 0,
                    "replan_steps": None,
                }
            slot = self.episodes[index]
            head = resize_head(
                decode_image_bit(obs["vision"][self.cameras[0]]["color"]), self.config
            )
            slot["memory"].observe(
                obs.get("instruction", obs.get("instructions")),
                None,
                head,
                None,
                step=slot["control_step"] if self.sparse_capture else None,
            )
            slot["observation"] = obs

    def request(self, indices):
        if self.pending is not None:
            raise RuntimeError("A context request is already pending")
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Expected nonempty unique environment indices")
        request_id = uuid4().hex
        rows = []
        expected = []
        for index in indices:
            slot = self.episodes[index]
            memory, obs = slot["memory"], slot["observation"]
            if self.sparse_capture and memory.step != slot["control_step"]:
                raise RuntimeError("Capture the current boundary before inference")
            frames, offsets = memory.history()
            context = {
                "protocol": 2,
                "config_hash": self.config_hash,
                "request_id": request_id,
                "episode_id": slot["id"],
                "step": memory.step,
                "state_version": slot["version"],
                "semantic": memory.semantic,
                "subtask": memory.subtask,
                "cache_valid": memory.cache_valid,
                "history_images": (
                    np.stack(frames)
                    if frames
                    else np.empty(
                        (0, self.config.height, self.config.width, 3), dtype=np.uint8
                    )
                ),
                "history_offsets": offsets,
            }
            # Top-level observation rows are required by the official RGB decoder;
            # wrapping these rows in another dict would bypass its vision handling.
            rows.append(
                {
                    "env_idx": index,
                    "instruction": memory.instruction,
                    "state": obs["state"],
                    "vision": {
                        key: {"color": obs["vision"][key]["color"]}
                        for key in self.cameras
                    },
                    "hwm_context": context,
                }
            )
            expected.append((index, slot["id"], memory.step, slot["version"]))
        self.pending = (request_id, expected)
        return rows

    def commit(self, response):
        if self.pending is None:
            raise ValueError("No pending request; duplicate or unsolicited response")
        request_id, expected = self.pending
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise ValueError("Response request ID does not match")
        items = response.get("items")
        if not isinstance(items, list) or len(items) != len(expected):
            raise ValueError("Response environment count does not match")
        # Validate the whole response before committing any environment.
        for item, (index, episode_id, step, version) in zip(
            items, expected, strict=True
        ):
            if (
                item.get("env_idx"),
                item.get("episode_id"),
                item.get("step"),
                item.get("state_version"),
            ) != (index, episode_id, step, version + 1):
                raise ValueError("Stale, reordered or mismatched environment response")
            if self.episodes[index]["version"] != version:
                raise ValueError("Client state changed during inference")
            if (
                not all(
                    isinstance(item.get(key), str) and item[key].strip()
                    for key in ("semantic", "subtask")
                )
                or item.get("cache_valid") is not True
            ):
                raise ValueError("Invalid returned semantic context")
            chunk = item.get("actions")
            if not isinstance(chunk, list) or not chunk:
                raise ValueError("Empty or malformed action chunk")
            if len(chunk) != self.replan_steps:
                raise ValueError(
                    "Action chunk length disagrees with negotiated replan_steps"
                )
            if self.sparse_capture:
                previous = self.episodes[index]["replan_steps"]
                if previous is not None and len(chunk) != previous:
                    raise ValueError("replan_steps changed within an episode")
                if previous is None and step != 0:
                    raise ValueError("Sparse capture must start at control step zero")
            for action in chunk:
                if (
                    not isinstance(action, dict)
                    or not action
                    or any(
                        np.asarray(value).ndim != 1 or not np.isfinite(value).all()
                        for value in action.values()
                    )
                ):
                    raise ValueError("Malformed or nonfinite action")
        if len({len(item["actions"]) for item in items}) != 1:
            raise ValueError("Different action chunk lengths in one batch")
        for item, (index, _, _, version) in zip(items, expected, strict=True):
            slot = self.episodes[index]
            memory = slot["memory"]
            if self.sparse_capture and slot["replan_steps"] is None:
                slot["replan_steps"] = len(item["actions"])
                memory.configure_capture(slot["replan_steps"])
                print(
                    f"[HWM] sparse head capture: replan_steps={slot['replan_steps']}, "
                    f"stride={memory.capture_stride}, max_frames={memory.frames.maxlen}",
                    flush=True,
                )
            memory.semantic, memory.subtask = item["semantic"], item["subtask"]
            memory.cache_valid = item["cache_valid"]
            slot["version"] = version + 1
        self.pending = None
        return [item["actions"] for item in items]

    def infer(self, model_client, indices):
        payload = self.request(indices)
        response = model_client.call(func_name="infer_context", obs=payload)
        return self.commit(response)


def eval_context_episode(task_env, model_client, *, batched):
    """Upload once per chunk, with opt-in intermediate capture reduction."""
    head_only = os.environ.get("HWM_INTERMEDIATE_HEAD_ONLY", "0")
    if head_only not in {"0", "1"}:
        raise ValueError("HWM_INTERMEDIATE_HEAD_ONLY must be 0 or 1")
    sparse_capture = os.environ.get("HWM_SPARSE_HEAD_CAPTURE", "0")
    if sparse_capture not in {"0", "1"}:
        raise ValueError("HWM_SPARSE_HEAD_CAPTURE must be 0 or 1")
    timeout = os.environ.get("HWM_REQUEST_TIMEOUT_S")
    if timeout is not None:
        seconds = float(timeout)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("HWM_REQUEST_TIMEOUT_S must be finite and positive")

        # RoboDojo does not forward this option when constructing its client.
        # Set the protocol config on its owning loop before adapter requests.
        async def configure_timeout():
            model_client._client.config.request_timeout_s = seconds

        model_client._run(configure_timeout())
    if head_only == "1" or sparse_capture == "1":
        from .robodojo import check_intermediate_safety, history_head_observations

    metadata = model_client.call(func_name="get_policy_metadata")
    context = ContextClient(metadata, sparse_capture=sparse_capture == "1")
    while not task_env.is_episode_end():
        indices = task_env.get_running_env_idx_list() if batched else [0]
        if not indices:
            break
        observations = (
            task_env.get_obs_batch(indices) if batched else [task_env.get_obs()]
        )
        if not batched:
            indices = [int(observations[0].get("env_idx", 0))]
        context.observe(observations)
        chunks = context.infer(model_client, indices)
        for step in range(len(chunks[0])):
            if batched:
                task_env.take_action_batch([chunk[step] for chunk in chunks], indices)
            else:
                task_env.take_action(chunks[0][step])
            context.advance(indices)
            if task_env.is_episode_end() or step + 1 == len(chunks[0]):
                break
            if batched:
                running = set(task_env.get_running_env_idx_list())
                active = [
                    (index, chunk)
                    for index, chunk in zip(indices, chunks, strict=True)
                    if index in running
                ]
                if not active:
                    break
                indices, chunks = map(list, zip(*active, strict=True))
            if sparse_capture == "1":
                check_intermediate_safety(task_env, indices)
                due = context.capture_due(indices)
                if not due:
                    # Preserve observation-side temporal rendering even when
                    # no history image needs to be read or retained.
                    task_env.obs_manager.render_for_capture()
                    continue
                observations = history_head_observations(
                    task_env, due, head_camera=context.cameras[0], check_safety=False
                )
            elif head_only == "1":
                observations = history_head_observations(
                    task_env, indices, head_camera=context.cameras[0]
                )
            else:
                observations = (
                    task_env.get_obs_batch(indices) if batched else [task_env.get_obs()]
                )
            context.observe(observations)
