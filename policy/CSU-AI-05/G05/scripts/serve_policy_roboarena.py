"""RoboArena/OpenPI-compatible WebSocket policy server for G05.

This file is intentionally a thin adapter around ``serve_policy.py``:

- ``serve_policy.py`` owns model loading, processor construction, and the
  training-consistent raw_obs -> obs_dict -> inferencer path.
- this file owns only the RoboArena protocol, namely flat DROID observations in
  and ``{"actions": ndarray[N, 8]}`` out.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import rootutils
import torch
import websockets

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.models.g05.inferencer import PolicyInferencer, resolve_processor
from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.eval.eval_utils import filter_embodiment

register_default_resolvers()

# ``scripts`` is not a package. Add it explicitly before importing serve_policy.
_scripts_dir = Path(__file__).parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from serve_policy import build_obs_dict, setup, unpackb  # noqa: E402

logger = logging.getLogger(__name__)


_DROID_MAX_JOINT_DELTA = 0.2  # RobotIKSolver.max_joint_delta (rad/step @ 15Hz)
_DEFAULT_FREQUENCY = 15.0
_UNSUPPORTED_NUMPY_KINDS = ("V", "O", "c")


def _pack_roboarena(obj: Any) -> Any:
    """Pack numpy arrays with RoboArena/OpenPI's bytes-key ndarray convention."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in _UNSUPPORTED_NUMPY_KINDS:
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def packb_roboarena(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_roboarena)


def _shape_dim(meta: dict[str, Any]) -> int | tuple[int, ...]:
    return meta.get("raw_shape", meta.get("shape"))


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _strip_batch_dim(value: Any) -> np.ndarray:
    arr = _as_numpy(value)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32, copy=False)


def _to_chw_uint8(image: Any) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError(f"image must be 3D, got shape={img.shape}")
    if img.shape[-1] in (1, 3, 4):
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _image_source_for_key(key: str) -> str | None:
    if key.startswith("dummy"):
        return None
    if key in {"exterior_image", "exterior_image_1"}:
        return "observation/exterior_image_1_left"
    if key == "exterior_image_2":
        return "observation/exterior_image_2_left"
    if key == "wrist_image":
        return "observation/wrist_image_left"
    return None


def _state_source_for_key(key: str) -> str | None:
    if key in {"joint_position", "right_arm"}:
        return "observation/joint_position"
    if key in {"gripper", "right_gripper"}:
        return "observation/gripper_position"
    return None


def _droid_obs_to_raw_obs(droid_obs: dict[str, Any], processor) -> dict[str, Any]:
    """Convert RoboArena's flat DROID observation into serve_policy raw_obs."""
    raw_obs: dict[str, Any] = {}
    if "embodiment_type" in droid_obs:
        raw_obs["embodiment_type"] = droid_obs["embodiment_type"]

    proc = resolve_processor(processor, raw_obs)

    images: dict[str, np.ndarray] = {}
    image_meta_by_key = {m["key"]: m for m in proc.shape_meta["images"]}
    for key, meta in image_meta_by_key.items():
        source_key = _image_source_for_key(key)
        if source_key is None:
            shape = tuple(_shape_dim(meta))
            images[key] = np.zeros(shape, dtype=np.uint8)
            continue

        # Some configs mention a second exterior image; RoboArena usually sends
        # one. Reusing exterior_image_1 matches the training-time random swap.
        if source_key not in droid_obs and key == "exterior_image_2":
            source_key = "observation/exterior_image_1_left"

        if source_key not in droid_obs:
            raise ValueError(
                f"image key '{key}' maps to '{source_key}', but that field is missing"
            )
        images[key] = _to_chw_uint8(droid_obs[source_key])

    state: dict[str, np.ndarray] = {}
    for meta in proc.shape_meta["state"]:
        key = meta["key"]
        source_key = _state_source_for_key(key)
        if source_key is None or source_key not in droid_obs:
            raise ValueError(
                f"state key '{key}' maps to '{source_key}', but that field is missing"
            )
        value = np.asarray(droid_obs[source_key], dtype=np.float32).reshape(-1)
        expected_dim = int(_shape_dim(meta))
        if value.shape[0] != expected_dim:
            raise ValueError(
                f"state key '{key}' has dim={value.shape[0]}, expected {expected_dim}"
            )
        if key in {"gripper", "right_gripper"}:
            # DroidLerobotDataset stores gripper as 1 - raw_gripper. Mirror that
            # convention before feeding observations into the G05 processor.
            value = 1.0 - value
        state[key] = value

    prompt = droid_obs.get("prompt", "")
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")

    raw_obs.update(
        {
            "images": images,
            "state": state,
            "task": str(prompt),
            "frequency": float(droid_obs.get("frequency", _DEFAULT_FREQUENCY)),
        }
    )
    return raw_obs


def _summarize_action(action: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in action.items():
        if key.startswith("_"):
            continue
        arr = _as_numpy(value)
        summary[key] = {
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
        }
    return summary


def _first_present(action: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, np.ndarray] | None:
    for key in keys:
        if key in action:
            return key, _strip_batch_dim(action[key])
    return None


def _hold_current_gripper(current_gripper_pos: np.ndarray, horizon: int) -> np.ndarray:
    current = np.asarray(current_gripper_pos, dtype=np.float32).reshape(-1)
    if current.shape != (1,):
        raise ValueError(f"current gripper position must have shape (1,), got {current.shape}")
    value = np.clip(current[0], 0.0, 1.0)
    return np.full((horizon, 1), value, dtype=np.float32)


def _action_dict_to_roboarena(
    action: dict[str, Any],
    *,
    absent_keys: set[str],
    current_joint_pos: np.ndarray,
    current_gripper_pos: np.ndarray,
    action_space: str,
) -> np.ndarray:
    """Convert G05 postprocessed action dict into RoboArena ``[N, 8]`` actions."""
    arm_entry = _first_present(action, ("joint_position", "right_arm"))
    grip_entry = _first_present(action, ("gripper", "right_gripper"))
    hold_gripper = "right_gripper" in absent_keys

    _, arm_abs = arm_entry

    horizon = arm_abs.shape[0]
    if grip_entry is not None and not hold_gripper:
        _, grip_abs = grip_entry
        horizon = min(horizon, grip_abs.shape[0])
        grip_abs = grip_abs[:horizon]
    arm_abs = arm_abs[:horizon]

    if action_space == "joint_position":
        arm_out = arm_abs.astype(np.float32, copy=False)
    else:
        current = np.asarray(current_joint_pos, dtype=np.float32).reshape(7)
        prev = np.concatenate([current[None, :], arm_abs[:-1]], axis=0)
        arm_out = ((arm_abs - prev) / _DROID_MAX_JOINT_DELTA).astype(np.float32)

    if hold_gripper:
        grip_out = _hold_current_gripper(current_gripper_pos, horizon)
    else:
        # G05 predicts in the flipped gripper convention used by the DROID
        # dataset; RoboArena/OpenPI expects open=0, close=1.
        grip_out = np.clip(1.0 - grip_abs, 0.0, 1.0).astype(np.float32, copy=False)
    actions = np.concatenate([arm_out, grip_out], axis=-1).astype(np.float32, copy=False)

    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"RoboArena actions must be [N, 8], got shape={actions.shape}")
    return actions


async def _handler(
    websocket,
    *,
    inferencer: PolicyInferencer,
    processor,
    action_space: str,
) -> None:
    client = websocket.remote_address
    logger.info("RoboArena client connected: %s", client)

    metadata = {
        "action_dim": 8,
        "action_space": action_space,
        "server": "g05-roboarena",
        "needs_wrist": True,
        "n_ext": 1,
    }
    await websocket.send(packb_roboarena(metadata))

    try:
        async for msg in websocket:
            t0 = time.monotonic()
            try:
                droid_obs = unpackb(msg)
                if isinstance(droid_obs, dict) and droid_obs.get("endpoint") == "reset":
                    await websocket.send(packb_roboarena({"__reset__": True}))
                    continue

                raw_obs = _droid_obs_to_raw_obs(droid_obs, processor)
                obs_dict = build_obs_dict(raw_obs, processor)
                actions_list = await asyncio.to_thread(inferencer.infer, [obs_dict])
                action = actions_list[0]

                cot_text = action.pop("_cot_text", None)
                absent_keys = set(action.pop("_absent_keys", set()))
                logger.info(
                    "G05 action keys/shapes: %s absent_keys=%s",
                    _summarize_action(action),
                    sorted(absent_keys),
                )

                current_joint_pos = np.asarray(
                    droid_obs.get("observation/joint_position", np.zeros(7)),
                    dtype=np.float32,
                )
                current_gripper_pos = np.asarray(
                    droid_obs.get("observation/gripper_position", np.zeros(1)),
                    dtype=np.float32,
                )
                roboarena_actions = _action_dict_to_roboarena(
                    action,
                    absent_keys=absent_keys,
                    current_joint_pos=current_joint_pos,
                    current_gripper_pos=current_gripper_pos,
                    action_space=action_space,
                )

                response = {
                    "actions": roboarena_actions,
                    "server_timing": {"infer_ms": (time.monotonic() - t0) * 1000.0},
                }
                if cot_text is not None:
                    response["cot_text"] = cot_text
                await websocket.send(packb_roboarena(response))
            except Exception as exc:
                logger.exception("RoboArena inference failed")
                await websocket.send(str(exc))
            finally:
                logger.debug("RoboArena request: %.1fms", (time.monotonic() - t0) * 1000.0)
    except websockets.exceptions.ConnectionClosed:
        logger.info("RoboArena client disconnected: %s", client)


async def serve_roboarena(
    policy,
    processor,
    host: str,
    port: int,
    *,
    device: str,
    action_space: str,
) -> None:
    inferencer = PolicyInferencer(policy, processor, device=device)
    bound_handler = functools.partial(
        _handler,
        inferencer=inferencer,
        processor=processor,
        action_space=action_space,
    )

    async with websockets.serve(bound_handler, host, port, compression=None, max_size=None):
        logger.info(
            "RoboArena policy server listening on ws://%s:%d (device=%s, action_space=%s)",
            host,
            port,
            device,
            action_space,
        )
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="G05 RoboArena/OpenPI policy server")
    parser.add_argument("--ckpt_path", required=True, help="Path to model_state_dict.pt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--action_space",
        default="joint_position",
        choices=["joint_position", "joint_velocity"],
        help="RoboArena action protocol exposed by this server.",
    )
    args, remaining = parser.parse_known_args()

    overrides = [item for item in remaining if "=" in item]

    run_dir = find_run_dir(args.ckpt_path)
    cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

    eval_embodiment = cfg.get("eval_embodiment", None)
    if eval_embodiment and "embodiment_datasets" in cfg.data:
        filter_embodiment(cfg, eval_embodiment)

    from g05.utils.logging.logging_config import setup_logging

    setup_logging(log_level=logging.INFO, is_main_process=True)
    logger.info("Found run dir: %s", run_dir)

    policy, processor = setup(cfg, device=args.device)
    logger.info("Model and processor loaded on %s.", args.device)

    asyncio.run(
        serve_roboarena(
            policy,
            processor,
            args.host,
            args.port,
            device=args.device,
            action_space=args.action_space,
        )
    )


if __name__ == "__main__":
    main()
