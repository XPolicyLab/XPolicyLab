"""Select a real native scene and publish simulator-owned episode identity.

The scorer, reward calculation and saved native progress remain upstream code.
An explicit layout narrows the pending queue; it never invents completed seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MethodType


class NativeLayoutError(ValueError):
    """Native episode identity does not satisfy the requested selection."""


def _integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise NativeLayoutError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise NativeLayoutError(f"{label} must be a nonnegative integer") from exc
    if result < 0 or str(result) != str(value):
        raise NativeLayoutError(f"{label} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class NativeSelection:
    task: str
    eval_seed: int
    layout: int | None = None
    run_id: str | None = None
    scene_sha256: str | None = None

    @classmethod
    def from_launch(cls, task: str, eval_seed, layout=None, *, environ=None):
        environ = os.environ if environ is None else environ
        seed = _integer(eval_seed, "eval_seed")
        declared = layout if layout is not None else environ.get("ROBORSI_NATIVE_LAYOUT_ID")
        selected = None if declared in (None, "") else _integer(declared, "layout")
        expected_sha = environ.get("ROBORSI_NATIVE_SCENE_SHA256") or None
        run_id = environ.get("ROBODOJO_RUN_ID") or None
        request_path = environ.get("ROBORSI_RESUME_REQUEST")
        if request_path:
            payload = Path(request_path).read_bytes()
            expected_request_sha = environ.get("ROBORSI_RESUME_REQUEST_SHA256")
            if expected_request_sha and hashlib.sha256(payload).hexdigest() != expected_request_sha:
                raise NativeLayoutError("Resume request changed before native launch")
            request = json.loads(payload)
            if request.get("format") != "xpolicylab_own_episode_resume_v1":
                raise NativeLayoutError("Unsupported native resume request")
            identity = request["identity"]
            actual_layout = _integer(identity.get("layout"), "resume layout")
            if identity.get("task") != task or identity.get("eval_seed") != seed:
                raise NativeLayoutError("Resume task/eval_seed differs from native launch")
            if selected is not None and selected != actual_layout:
                raise NativeLayoutError("Explicit layout differs from resume layout")
            if run_id and run_id != identity.get("run_id"):
                raise NativeLayoutError("Resume run_id differs from native launch")
            selected = actual_layout
            run_id = identity.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise NativeLayoutError("Resume request lacks native run_id")
            native = request.get("native_identity") or identity
            source_sha = native.get("scene_sha256") or native.get("layout_sha256")
            if expected_sha and source_sha and expected_sha != source_sha:
                raise NativeLayoutError("Resume scene SHA differs from requested scene")
            expected_sha = source_sha or expected_sha
            if not expected_sha:
                raise NativeLayoutError("Resume request lacks verified native scene SHA")
        if expected_sha and (len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha)):
            raise NativeLayoutError("Native scene SHA must be lowercase SHA-256")
        return cls(task, seed, selected, run_id, expected_sha)


def _check_progress(state: dict | None, selection: NativeSelection) -> None:
    if state is None:
        return
    expected = {"task_name": selection.task, "eval_seed": selection.eval_seed}
    if selection.run_id is not None:
        expected["run_id"] = selection.run_id
    for key, value in expected.items():
        if state.get(key) != value:
            raise NativeLayoutError(f"Native progress {key} differs from launch")
    completed = set(int(v) for v in state.get("completed_layout_ids", []))
    for detail in (state.get("details") or {}).values():
        if isinstance(detail, dict) and "layout_id" in detail:
            completed.add(int(detail["layout_id"]))
    abandoned = set(int(v) for v in state.get("abandoned_layout_ids", []))
    if selection.layout is not None and selection.layout in completed | abandoned:
        raise NativeLayoutError("Requested layout is already completed or abandoned")


def _scene(env, layout: int) -> tuple[str, str]:
    info = env.seed_manager.seed_info.get(layout)
    if not isinstance(info, dict) or not info.get("scene_layout"):
        raise NativeLayoutError("Requested native scene is unavailable")
    path = Path(info["scene_layout"]).resolve(strict=True)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def bind_native_environment(env, selection: NativeSelection, *, resume_state=None):
    """Wrap one official EvalEnv instance, before its first native reset."""
    _check_progress(resume_state, selection)
    if env.task_name != selection.task or env.eval_seed != selection.eval_seed:
        raise NativeLayoutError("Constructed native task/eval_seed differs from launch")
    if selection.run_id and env.run_id != selection.run_id:
        raise NativeLayoutError("Constructed native run_id differs from launch")
    manager = env.seed_manager
    if selection.layout is not None:
        if env.num_envs != 1:
            raise NativeLayoutError("Explicit layout requires one native environment")
        if selection.layout not in manager.seed_list:
            raise NativeLayoutError("Requested layout is unavailable in native pending queue")
        _, sha = _scene(env, selection.layout)
        if selection.scene_sha256 and sha != selection.scene_sha256:
            raise NativeLayoutError("Requested native scene SHA differs from checkpoint")
        manager.seed_list = [selection.layout]
        manager.st_idx = manager.idx = 0
        manager.ed_idx = 1
        manager._current_batch_seeds = None
    original_reset = env.reset
    original_get_obs_batch = env.get_obs_batch
    current = {}
    consumed = False

    def reset(self, seed=None, options=None):
        nonlocal consumed
        seeds = list(seed) if seed is not None else []
        actual = [int(value) for value in seeds if value is not None]
        if selection.layout is not None and (consumed or actual != [selection.layout]):
            # Upstream main catches Exception and skips a layout. Identity
            # violations must instead leave the process without a score.
            raise SystemExit("Native layout selection refused an unrelated reset")
        receipts = {}
        for index, layout_id in enumerate(seeds):
            if layout_id is None:
                continue
            path, sha = _scene(self, int(layout_id))
            if selection.scene_sha256 and sha != selection.scene_sha256:
                raise SystemExit("Native scene SHA changed before reset")
            receipts[index] = {
                "schema": 1, "task": self.task_name, "task_name": self.task_name,
                "eval_seed": int(self.eval_seed), "layout": int(layout_id),
                "layout_id": int(layout_id), "run_id": self.run_id,
                "scene_sha256": sha, "scene_path": path,
            }
        result = original_reset(seed=seeds, options=options)
        mapping = self.current_env_seed_map
        for index, receipt in receipts.items():
            if mapping.get(index) != receipt["layout"] or self.env_seeds[index] != receipt["layout"]:
                raise SystemExit("Native reset did not select the requested scene")
            if _scene(self, receipt["layout"])[1] != receipt["scene_sha256"]:
                raise SystemExit("Native scene changed during reset")
        current.clear()
        current.update(receipts)
        consumed = selection.layout is not None
        return result

    def get_obs_batch(self, env_idx_list=None, last_frame=False):
        values = original_get_obs_batch(env_idx_list=env_idx_list, last_frame=last_frame)
        indices = list(range(self.num_envs)) if env_idx_list is None else list(env_idx_list)
        if len(indices) != len(values):
            raise SystemExit("Native observation count differs from environment selection")
        for index, obs in zip(indices, values):
            if index not in current:
                raise SystemExit("Native observation precedes verified scene reset")
            receipt = dict(current[index])
            receipt["control_step"] = int(self.take_action_cnt[index])
            receipt["step_limit"] = int(self.step_lim)
            obs["roborsi_native_identity"] = receipt
        return values

    env.reset = MethodType(reset, env)
    env.get_obs_batch = MethodType(get_obs_batch, env)
    return env
