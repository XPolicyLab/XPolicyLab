"""Wall-X evaluation loop: chunked requests with client-held keyframe memory.

The checkpoint predicts a 32-step action chunk and the simulator replays 16 of
those steps before the next request (``replan_steps``). Two things therefore have
to happen on the client:

* every simulator step is recorded, because the memory selection looks at the
  raw frames of the window -- requesting every 16 steps but recording only those
  would starve the selection pass;
* the memory state returned by the server is carried into the next request, since
  it holds the frame ids the checkpoint chose to remember.

Both are what ``memory_client.KFMemClientState`` does. The observation layout is
RoboDojo's: three cameras under ``vision``, end-effector pose plus gripper under
``state``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from .memory_client import KFMemClientState

logger = logging.getLogger(__name__)

#: Camera names RoboDojo environments use for the head camera, in priority order.
FACE_CAMERAS = ("cam_head", "cam_high", "head_camera", "camera_front", "face_view")

#: The checkpoint emits this many steps; only the first ``REPLAN_STEPS`` are used.
ACTION_CHUNK = 32
REPLAN_STEPS = 16


class _FrameCounter:
    """Fallback frame index for environments that do not report ``sim_frame``."""

    def __init__(self):
        self.value = -1

    def next(self, obs):
        frame = obs.get("sim_frame")
        if isinstance(frame, int) and frame >= 0:
            self.value = frame
        else:
            self.value += 1
        return self.value


def _face_image(obs):
    """Head-camera pixels for the client-held frame ring.

    The policy server decodes ``vision`` before the model sees it, but the ring
    travels inside ``kfmem_client_state_v1``, which the server never decodes, so
    an encoded color (JPEG / raw bytes) is decoded here with XPolicyLab's own
    decoder. Plain arrays pass through untouched.
    """
    vision = obs.get("vision") or {}
    for name in FACE_CAMERAS:
        entry = vision.get(name)
        if isinstance(entry, dict):
            image = entry.get("color")
            if image is None:
                continue
            if isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3:
                return image
            from XPolicyLab.utils.process_data import decode_image_bit

            return decode_image_bit(image)
    raise ValueError(
        "observation has no head camera; expected one of " + ", ".join(FACE_CAMERAS)
    )


def _task_id(obs, task_env, context_key):
    for candidate in (obs.get("task_id"), getattr(task_env, "task_name", None)):
        if isinstance(candidate, str) and candidate:
            return candidate
    return context_key


def _validate_chunk(result, horizon):
    if not isinstance(result, dict):
        raise ValueError("infer must return a mapping")
    actions = result.get("actions")
    if not isinstance(actions, (list, tuple)) or len(actions) != horizon:
        raise ValueError(
            f"infer must return the full {horizon}-step action chunk, "
            f"got {None if actions is None else len(actions)}"
        )
    state = result.get("kfmem_state")
    if not isinstance(state, dict):
        raise ValueError("infer must return the updated kfmem_state")
    return list(actions), state


def eval_one_episode(TASK_ENV, model_client):
    """Run one episode against the Wall-X server."""
    context_key = f"xpl-{id(TASK_ENV) & 0xFFFFFFFF:08x}"
    state = KFMemClientState(context_key)
    state.reset(model_client.call(func_name="reset"))

    frames = _FrameCounter()
    chunk = _setting(model_client, "action_chunk_length", ACTION_CHUNK)
    replan = _setting(model_client, "replan_steps", REPLAN_STEPS)
    if replan > chunk:
        raise ValueError(f"replan_steps {replan} exceeds action_chunk_length {chunk}")

    actions = []
    since_request = replan  # force a request on the first step
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        frame = frames.next(obs)
        state.record_frame(frame, _face_image(obs))

        if since_request >= replan:
            observation = {
                **obs,
                "task_id": _task_id(obs, TASK_ENV, context_key),
                "sim_frame": frame,
                "kfmem_client_state_v1": state.request_state(frame),
                "kfmem_history": state.history(frame),
            }
            result = model_client.call(func_name="infer", obs=observation)
            actions, exported = _validate_chunk(result, chunk)
            state.accept_result(exported)
            since_request = 0

        if not actions:  # pragma: no cover - guarded by _validate_chunk
            raise RuntimeError("action queue drained before the next request")
        TASK_ENV.take_action(actions.pop(0))
        since_request += 1

    model_client.call(func_name="reset")


def _setting(model_client, name, default):
    """Chunk/replan length from this policy's deploy.yml (the client cannot ask
    the server); ``model_client`` is accepted for an explicit attribute override."""
    value = getattr(model_client, name, None)
    if not value:
        value = _deploy_config().get(name)
    return int(value) if value else default


def _deploy_config():
    path = Path(__file__).resolve().with_name("deploy.yml")
    try:
        with path.open() as handle:
            return yaml.safe_load(handle) or {}
    except OSError:
        return {}


def eval_one_episode_batch(TASK_ENV, model_client):
    """Run one episode in each of the simulator's environments.

    Every environment keeps its own ``KFMemClientState`` and its own action queue,
    so each one sends the same request sequence as :func:`eval_one_episode`. The
    environments that are due for a new chunk at a step go to the server in one
    ``infer_batch`` call. Environments that finish early drop out.
    """
    base_key = f"xpl-{id(TASK_ENV) & 0xFFFFFFFF:08x}"
    chunk = _setting(model_client, "action_chunk_length", ACTION_CHUNK)
    replan = _setting(model_client, "replan_steps", REPLAN_STEPS)
    if replan > chunk:
        raise ValueError(f"replan_steps {replan} exceeds action_chunk_length {chunk}")

    envs = {}
    for env_idx in TASK_ENV.get_running_env_idx_list():
        context_key = f"{base_key}-e{env_idx}"
        state = KFMemClientState(context_key)
        state.reset(model_client.call(func_name="reset"))
        envs[env_idx] = {
            "key": context_key,
            "state": state,
            "frames": _FrameCounter(),
            "actions": [],
            "since_request": replan,  # force a request on the first step
        }

    while not TASK_ENV.is_episode_end():
        running = [e for e in TASK_ENV.get_running_env_idx_list() if e in envs]
        if not running:
            break
        due, requests = [], []
        for env_idx, obs in zip(running, TASK_ENV.get_obs_batch(running)):
            env = envs[env_idx]
            frame = env["frames"].next(obs)
            env["state"].record_frame(frame, _face_image(obs))
            if env["since_request"] < replan:
                continue
            due.append(env_idx)
            requests.append({
                **obs,
                "task_id": _task_id(obs, TASK_ENV, env["key"]),
                "sim_frame": frame,
                "kfmem_client_state_v1": env["state"].request_state(frame),
                "kfmem_history": env["state"].history(frame),
            })

        if requests:
            results = model_client.call(func_name="infer_batch", obs=requests)
            if not isinstance(results, list) or len(results) != len(requests):
                raise ValueError("infer_batch must return one result per observation")
            for env_idx, result in zip(due, results):
                actions, exported = _validate_chunk(result, chunk)
                envs[env_idx]["state"].accept_result(exported)
                envs[env_idx]["actions"] = actions
                envs[env_idx]["since_request"] = 0

        step_actions = []
        for env_idx in running:
            if not envs[env_idx]["actions"]:  # pragma: no cover - guarded by _validate_chunk
                raise RuntimeError(f"env {env_idx}: action queue drained")
            step_actions.append(envs[env_idx]["actions"].pop(0))
        TASK_ENV.take_action_batch(step_actions, running)
        for env_idx in running:
            envs[env_idx]["since_request"] += 1

    model_client.call(func_name="reset")
