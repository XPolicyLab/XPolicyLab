#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

ROOT_PATH = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT_PATH / "src"
for candidate in [str(SRC_ROOT), str(ROOT_PATH)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio
import numpy as np
import torch
import tyro

OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"


def _get_libero_search_roots() -> list[Path]:
    candidates: list[Path] = []

    libero_home = os.environ.get("LIBERO_HOME", "").strip()
    if libero_home:
        candidates.append(Path(libero_home).expanduser())

    candidates.extend(
        [
            ROOT_PATH / "LIBERO",
            ROOT_PATH / "third_party" / "LIBERO",
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = str(candidate.resolve())
        except OSError:
            normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(candidate)
    return unique_candidates


def _import_libero_runtime():
    from libero.libero import benchmark as libero_benchmark, get_libero_path as libero_get_libero_path
    from libero.libero.envs import OffScreenRenderEnv as libero_offscreen_render_env

    return libero_benchmark, libero_get_libero_path, libero_offscreen_render_env


LIBERO_SEARCH_ROOTS = _get_libero_search_roots()
LIBERO_IMPORT_ERROR = None
benchmark = None
get_libero_path = None
OffScreenRenderEnv = None

try:
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_runtime()
except ImportError as exc:  # pragma: no cover - depends on LIBERO runtime setup
    LIBERO_IMPORT_ERROR = exc

    for search_root in LIBERO_SEARCH_ROOTS:
        if not (search_root / "libero").exists():
            continue

        search_root_str = str(search_root)
        if search_root_str not in sys.path:
            sys.path.insert(0, search_root_str)

        try:
            benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_runtime()
            LIBERO_IMPORT_ERROR = None
            break
        except ImportError as retry_exc:
            LIBERO_IMPORT_ERROR = retry_exc


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_ACTION_DIM = 7


@dataclass
class EvalArgs:
    ckpt_path: Union[str, Path] = ""
    task_suite_name: str = "libero_goal"
    stats_key: str = "franka"
    task_id: int | None = None
    seed: int = 7
    num_trials_per_task: int = 50
    num_steps_wait: int = 10
    resize_size: int = 224
    render_resolution: int = LIBERO_ENV_RESOLUTION
    image_history_interval: int = 15
    # When omitted, default to the checkpoint's own n_action_steps/chunk_size,
    # which matches StarVLA-style chunk execution.
    infer_horizon: int | None = None
    dtype: str = "bfloat16"  # bfloat16 | float32
    video_dir: Path = ROOT_PATH / "evaluation" / "Libero" / "output"
    fps: int = 10
    save_videos: bool = True
    save_actions: bool = False
    rotate_images_180: bool = True
    gripper_mode: str = "libero_open_prob"  # libero_open_prob | sign | raw
    gripper_threshold: float = 0.5
    decode_image_flag: bool = False
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    debug: bool = False
    ws_url: str = ""

    policy_type: str | None = None
    qwen3_vl_pretrained_path: str | None = None
    qwen3_vl_processor_path: str | None = None
    cosmos_tokenizer_path_or_name: str | None = None
    da3_model_path_or_name: str | None = None
    da3_code_root: str | None = None
    disable_3d_teacher_for_eval: bool = True


def ensure_libero_available() -> None:
    if benchmark is None or get_libero_path is None or OffScreenRenderEnv is None:
        search_hint = ", ".join(str(path) for path in LIBERO_SEARCH_ROOTS)
        raise ImportError(
            "LIBERO is not available in the current environment. "
            "Please install LIBERO and its simulator dependencies before running evaluation. "
            f"Python executable: {sys.executable}. "
            f"Search roots checked: {search_hint or '(none)'}. "
            "If LIBERO is already checked out locally, set LIBERO_HOME=/path/to/LIBERO "
            "or add that checkout root to PYTHONPATH."
        ) from LIBERO_IMPORT_ERROR


def resolve_ckpt_dir(ckpt_path: Union[str, Path]) -> Path:
    ckpt_str = str(ckpt_path)
    if not ckpt_str.strip():
        raise ValueError(
            "ckpt_path is required for local LIBERO evaluation. "
            "Pass --args.ws_url for split websocket mode."
        )

    from huggingface_hub import snapshot_download

    local_dir = Path(ckpt_str).expanduser()
    if local_dir.exists():
        if (local_dir / "config.json").exists():
            return local_dir.resolve()
        nested_pretrained_dir = local_dir / "pretrained_model"
        if (nested_pretrained_dir / "config.json").exists():
            return nested_pretrained_dir.resolve()
        return local_dir.resolve()
    snapshot_dir = snapshot_download(repo_id=ckpt_str)
    return Path(snapshot_dir)


def resolve_runtime_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        if device == "cpu":
            logging.warning("Requested bfloat16 on CPU; falling back to float32 for evaluation.")
            return torch.float32
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def apply_runtime_config_overrides(config, args: EvalArgs) -> None:
    if args.policy_type is not None:
        from lerobot.policies.names import canonical_policy_type

        if canonical_policy_type(config.type) != canonical_policy_type(args.policy_type):
            raise ValueError(
                f"Checkpoint policy type is {config.type!r}, but --args.policy_type={args.policy_type!r}. "
                "Use a matching checkpoint or leave --args.policy_type unset."
            )

    if args.qwen3_vl_pretrained_path is not None and hasattr(config, "qwen3_vl_pretrained_path"):
        config.qwen3_vl_pretrained_path = args.qwen3_vl_pretrained_path
    if args.qwen3_vl_processor_path is not None and hasattr(config, "qwen3_vl_processor_path"):
        config.qwen3_vl_processor_path = args.qwen3_vl_processor_path
    if args.cosmos_tokenizer_path_or_name is not None and hasattr(config, "cosmos_tokenizer_path_or_name"):
        config.cosmos_tokenizer_path_or_name = args.cosmos_tokenizer_path_or_name
    if args.da3_model_path_or_name is not None and hasattr(config, "da3_model_path_or_name"):
        config.da3_model_path_or_name = args.da3_model_path_or_name
    if args.da3_code_root is not None and hasattr(config, "da3_code_root"):
        config.da3_code_root = args.da3_code_root

    from lerobot.policies.names import is_wsa_base

    if is_wsa_base(config.type) and args.disable_3d_teacher_for_eval and hasattr(config, "lambda_3d"):
        config.lambda_3d = 0.0


def get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown LIBERO task suite: {task_suite_name}")


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("_")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3])))) / den


def preprocess_env_images(obs: dict, rotate_images_180: bool) -> tuple[np.ndarray, np.ndarray]:
    head = obs["agentview_image"]
    wrist = obs["robot0_eye_in_hand_image"]
    if rotate_images_180:
        head = head[::-1, ::-1]
        wrist = wrist[::-1, ::-1]
    return np.ascontiguousarray(head), np.ascontiguousarray(wrist)


def build_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ),
        axis=0,
    ).astype(np.float32)


def convert_gripper_action(raw_gripper: float, mode: str, threshold: float) -> float:
    if mode == "libero_open_prob":
        return float(1.0 - 2.0 * (raw_gripper > threshold))
    if mode == "sign":
        return float(1.0 if raw_gripper >= 0.0 else -1.0)
    if mode == "raw":
        return float(np.clip(raw_gripper, -1.0, 1.0))
    raise ValueError(f"Unsupported gripper_mode: {mode}")


def action_to_env(action: np.ndarray, gripper_mode: str, gripper_threshold: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < LIBERO_ACTION_DIM:
        raise ValueError(f"Expected at least {LIBERO_ACTION_DIM} action dims, got {action.shape}")

    env_action = action[:LIBERO_ACTION_DIM].copy()
    env_action[6] = convert_gripper_action(float(env_action[6]), gripper_mode, gripper_threshold)
    return env_action


def build_policy_and_transforms(args: EvalArgs):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.utils import load_json
    from lerobot.policies.WSA_Base.transform_wsa_base import (
        Qwen3_VLProcessorTransformFn as WSABaseProcessorTransformFn,
    )
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.names import is_wsa_base
    from lerobot.transforms.core import (
        NormalizeTransformFn,
        ResizeImagesWithPadFn,
        UnNormalizeTransformFn,
    )

    ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)
    apply_runtime_config_overrides(config, args)

    if not is_wsa_base(config.type):
        raise ValueError(f"LIBERO evaluation currently supports WSA_Base checkpoints only, got {config.type!r}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    dtype = resolve_runtime_dtype(args.dtype, device)

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(config=config, pretrained_name_or_path=ckpt_dir)
    policy.to(device=device, dtype=dtype).eval()

    stats = load_json(ckpt_dir / "stats.json")[args.stats_key]
    stat_keys = ["min", "max", "mean", "std"]
    state_stat = {"observation.state": {k: np.asarray(stats["observation.state"][k]) for k in stat_keys}}
    action_stat = {"action": {k: np.asarray(stats["action"][k]) for k in stat_keys}}

    resize_fn = ResizeImagesWithPadFn(height=args.resize_size, width=args.resize_size)
    normalize_state_fn = NormalizeTransformFn(
        selected_keys=["observation.state"],
        mode="mean_std",
        norm_stats=state_stat,
    )
    unnormalize_action_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        mode="mean_std",
        norm_stats=action_stat,
    )

    processor_path = (
        args.qwen3_vl_processor_path
        or getattr(config, "qwen3_vl_processor_path", None)
        or getattr(config, "qwen3_vl_pretrained_path", None)
    )
    if processor_path is None:
        raise ValueError("Failed to resolve a Qwen3-VL processor path for WSABase LIBERO evaluation.")

    processor_fn = WSABaseProcessorTransformFn(
        pretrained_model_name_or_path=processor_path,
        max_length=int(getattr(config, "tokenizer_max_length", 48)),
    )

    return policy, config, dtype, resize_fn, normalize_state_fn, unnormalize_action_fn, processor_fn


def build_remote_policy_client(args: EvalArgs):
    from evaluation.Libero.libero_remote_client import LiberoRemoteClient

    client = LiberoRemoteClient(ws_url=args.ws_url, image_history_interval=args.image_history_interval)
    metadata = client.metadata
    infer_horizon = int(args.infer_horizon or metadata.get("infer_horizon") or 1)
    if infer_horizon <= 0:
        raise ValueError(f"infer_horizon must be positive, got {infer_horizon}")

    logging.info("Connected to split policy server at %s", args.ws_url)
    logging.info("Policy server metadata:\n%s", json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    return client, metadata, infer_horizon


def get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def update_image_history(
    action_plan: deque[np.ndarray],
    head_history: list[np.ndarray],
    wrist_history: list[np.ndarray],
    head_img: np.ndarray,
    wrist_img: np.ndarray,
    interval: int,
) -> None:
    if len(action_plan) <= interval:
        head_history.append(np.ascontiguousarray(head_img))
        wrist_history.append(np.ascontiguousarray(wrist_img))

        while len(head_history) > interval + 1:
            head_history.pop(0)
            wrist_history.pop(0)

    if not head_history:
        head_history.append(np.ascontiguousarray(head_img))
        wrist_history.append(np.ascontiguousarray(wrist_img))


def build_image_history_pair(history: list[np.ndarray], interval: int) -> np.ndarray:
    past_idx = max(len(history) - interval - 1, 0)
    return np.stack([history[past_idx], history[-1]], axis=0)


def maybe_append_history(
    action_plan: deque[np.ndarray],
    head_history: list[np.ndarray],
    wrist_history: list[np.ndarray],
    head_img: np.ndarray,
    wrist_img: np.ndarray,
    interval: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    update_image_history(
        action_plan=action_plan,
        head_history=head_history,
        wrist_history=wrist_history,
        head_img=head_img,
        wrist_img=wrist_img,
        interval=interval,
    )
    image_head_with_history = torch.as_tensor(build_image_history_pair(head_history, interval), dtype=torch.float32) / 255.0
    image_wrist_with_history = (
        torch.as_tensor(build_image_history_pair(wrist_history, interval), dtype=torch.float32) / 255.0
    )
    return image_head_with_history, image_wrist_with_history


def prepare_policy_inputs(
    head_history: torch.Tensor,
    wrist_history: torch.Tensor,
    state: np.ndarray,
    task_description: str,
    resize_fn,
    normalize_state_fn,
    processor_fn,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    dummy_history = torch.ones_like(head_history)
    sample = {
        f"{OBS_IMAGES}.image0": head_history.permute(0, 3, 1, 2),
        f"{OBS_IMAGES}.image1": wrist_history.permute(0, 3, 1, 2),
        f"{OBS_IMAGES}.image2": dummy_history.permute(0, 3, 1, 2),
        OBS_STATE: torch.from_numpy(state),
        "task": task_description,
    }

    sample = resize_fn(sample)
    sample[f"{OBS_IMAGES}.image0_mask"] = torch.tensor(True)
    sample[f"{OBS_IMAGES}.image1_mask"] = torch.tensor(True)
    sample[f"{OBS_IMAGES}.image2_mask"] = torch.tensor(False)
    sample = processor_fn(sample)
    sample = normalize_state_fn(sample)

    inputs = {}
    for key, value in sample.items():
        if key == "task":
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if value.dtype == torch.bool:
            inputs[key] = value.reshape(1).to(device)
        elif value.dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
            inputs[key] = value[None].to(device)
        elif value.is_floating_point():
            inputs[key] = value[None].to(device=device, dtype=dtype)
        else:
            inputs[key] = value[None].to(device)

    return inputs


def predict_action_chunk(
    policy,
    inputs: dict[str, torch.Tensor],
    unnormalize_action_fn,
    infer_horizon: int,
    decode_image_flag: bool,
) -> np.ndarray:
    with torch.no_grad():
        action_pred, _ = policy.predict_action_chunk(inputs, decode_image=decode_image_flag)
    action_pred = action_pred[0, :infer_horizon]
    action_pred = unnormalize_action_fn({"action": action_pred})["action"]
    return action_pred.detach().cpu().numpy()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_single_episode(
    *,
    args: EvalArgs,
    env,
    initial_state,
    task_description: str,
    max_steps: int,
    infer_horizon: int,
    policy,
    resize_fn,
    normalize_state_fn,
    unnormalize_action_fn,
    processor_fn,
    device: str,
    dtype: torch.dtype,
) -> tuple[bool, list[np.ndarray], np.ndarray]:
    env.reset()
    obs = env.set_init_state(initial_state)

    policy.reset()
    action_plan: deque[np.ndarray] = deque()
    replay_images: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    head_history: list[np.ndarray] = []
    wrist_history: list[np.ndarray] = []

    done = False
    t = 0
    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        head_img, wrist_img = preprocess_env_images(obs, rotate_images_180=args.rotate_images_180)
        replay_images.append(head_img.copy())

        image_head_with_history, image_wrist_with_history = maybe_append_history(
            action_plan=action_plan,
            head_history=head_history,
            wrist_history=wrist_history,
            head_img=head_img,
            wrist_img=wrist_img,
            interval=args.image_history_interval,
        )

        if not action_plan:
            state = build_state(obs)
            inputs = prepare_policy_inputs(
                head_history=image_head_with_history,
                wrist_history=image_wrist_with_history,
                state=state,
                task_description=task_description,
                resize_fn=resize_fn,
                normalize_state_fn=normalize_state_fn,
                processor_fn=processor_fn,
                device=device,
                dtype=dtype,
            )
            predicted_chunk = predict_action_chunk(
                policy=policy,
                inputs=inputs,
                unnormalize_action_fn=unnormalize_action_fn,
                infer_horizon=infer_horizon,
                decode_image_flag=args.decode_image_flag,
            )
            action_plan.extend(predicted_chunk)

        model_action = np.asarray(action_plan.popleft(), dtype=np.float32)
        env_action = action_to_env(
            model_action,
            gripper_mode=args.gripper_mode,
            gripper_threshold=args.gripper_threshold,
        )
        executed_actions.append(env_action.copy())

        obs, _, done, _ = env.step(env_action.tolist())
        t += 1
        if done:
            break

    action_array = np.stack(executed_actions) if executed_actions else np.zeros((0, LIBERO_ACTION_DIM), dtype=np.float32)
    return bool(done), replay_images, action_array


def run_single_episode_remote(
    *,
    args: EvalArgs,
    env,
    initial_state,
    task_description: str,
    max_steps: int,
    infer_horizon: int,
    remote_client,
) -> tuple[bool, list[np.ndarray], np.ndarray]:
    env.reset()
    obs = env.set_init_state(initial_state)

    action_plan: deque[np.ndarray] = deque()
    replay_images: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    head_history: list[np.ndarray] = []
    wrist_history: list[np.ndarray] = []

    done = False
    t = 0
    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        head_img, wrist_img = preprocess_env_images(obs, rotate_images_180=args.rotate_images_180)
        replay_images.append(head_img.copy())
        update_image_history(
            action_plan=action_plan,
            head_history=head_history,
            wrist_history=wrist_history,
            head_img=head_img,
            wrist_img=wrist_img,
            interval=args.image_history_interval,
        )

        if not action_plan:
            state = build_state(obs)
            response = remote_client.infer_step(
                head_history=head_history,
                wrist_history=wrist_history,
                state=state,
                prompt=task_description,
                timestep=len(executed_actions),
            )
            predicted_chunk = np.asarray(response["actions"], dtype=np.float32)
            if predicted_chunk.ndim == 1:
                predicted_chunk = predicted_chunk[None]
            predicted_chunk = predicted_chunk[:infer_horizon]
            if predicted_chunk.size == 0:
                raise RuntimeError("Remote policy server returned an empty action chunk.")
            action_plan.extend(predicted_chunk)

        model_action = np.asarray(action_plan.popleft(), dtype=np.float32)
        env_action = action_to_env(
            model_action,
            gripper_mode=args.gripper_mode,
            gripper_threshold=args.gripper_threshold,
        )
        executed_actions.append(env_action.copy())

        obs, _, done, _ = env.step(env_action.tolist())
        t += 1
        if done:
            break

    action_array = np.stack(executed_actions) if executed_actions else np.zeros((0, LIBERO_ACTION_DIM), dtype=np.float32)
    return bool(done), replay_images, action_array


def evaluate_suite(args: EvalArgs) -> None:
    ensure_libero_available()

    logging.info("Arguments:\n%s", json.dumps(asdict(args), indent=2, default=str))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    remote_client = None
    remote_metadata = None
    policy = None
    config = None
    dtype = None
    resize_fn = None
    normalize_state_fn = None
    unnormalize_action_fn = None
    processor_fn = None
    device = "remote"
    try:
        if args.ws_url.strip():
            remote_client, remote_metadata, infer_horizon = build_remote_policy_client(args)
        else:
            policy, config, dtype, resize_fn, normalize_state_fn, unnormalize_action_fn, processor_fn = (
                build_policy_and_transforms(args)
            )
            device = str(config.device)
            infer_horizon = int(args.infer_horizon or getattr(config, "n_action_steps", getattr(config, "chunk_size", 1)))
            if infer_horizon <= 0:
                raise ValueError(f"infer_horizon must be positive, got {infer_horizon}")

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        max_steps = get_max_steps(args.task_suite_name)

        task_indices = [args.task_id] if args.task_id is not None else list(range(task_suite.n_tasks))
        if args.task_id is not None and not (0 <= args.task_id < task_suite.n_tasks):
            raise ValueError(f"task_id must be in [0, {task_suite.n_tasks}), got {args.task_id}")

        args.video_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Starting LIBERO evaluation for suite=%s, task_id=%s. Results will be saved to %s",
            args.task_suite_name,
            args.task_id if args.task_id is not None else "all",
            args.video_dir,
        )

        task_summaries: list[dict] = []
        total_episodes = 0
        total_successes = 0

        for task_id in task_indices:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            num_trials = min(args.num_trials_per_task, len(initial_states))
            if num_trials < args.num_trials_per_task:
                logging.warning(
                    "Task %s only provides %s initial states; clamping num_trials_per_task from %s to %s.",
                    task_id,
                    len(initial_states),
                    args.num_trials_per_task,
                    num_trials,
                )

            env, task_description = get_libero_env(task, args.render_resolution, args.seed)
            try:
                task_name = sanitize_name(task_description) or f"task_{task_id:02d}"
                task_dir = args.video_dir / f"{task_id:02d}_{task_name}"
                task_dir.mkdir(parents=True, exist_ok=True)

                logging.info(
                    "Evaluating suite=%s task %s/%s (task_id=%s, task_name=%s): %s",
                    args.task_suite_name,
                    task_id + 1,
                    task_suite.n_tasks,
                    task_id,
                    task_name,
                    task_description,
                )

                task_successes = 0
                for episode_idx in range(num_trials):
                    if remote_client is not None:
                        success, replay_images, action_array = run_single_episode_remote(
                            args=args,
                            env=env,
                            initial_state=initial_states[episode_idx],
                            task_description=task_description,
                            max_steps=max_steps,
                            infer_horizon=infer_horizon,
                            remote_client=remote_client,
                        )
                    else:
                        success, replay_images, action_array = run_single_episode(
                            args=args,
                            env=env,
                            initial_state=initial_states[episode_idx],
                            task_description=task_description,
                            max_steps=max_steps,
                            infer_horizon=infer_horizon,
                            policy=policy,
                            resize_fn=resize_fn,
                            normalize_state_fn=normalize_state_fn,
                            unnormalize_action_fn=unnormalize_action_fn,
                            processor_fn=processor_fn,
                            device=device,
                            dtype=dtype,
                        )

                    task_successes += int(success)
                    total_successes += int(success)
                    total_episodes += 1

                    suffix = "success" if success else "failure"
                    if args.save_videos and replay_images:
                        imageio.mimwrite(
                            task_dir / f"episode_{episode_idx:03d}_{suffix}.mp4",
                            [np.asarray(frame) for frame in replay_images],
                            fps=args.fps,
                        )
                    if args.save_actions:
                        np.save(task_dir / f"episode_{episode_idx:03d}_{suffix}.npy", action_array)

                    logging.info(
                        "[suite=%s task=%02d (%s) episode=%03d] %s | task success rate: %.2f%% | overall success rate: %.2f%%",
                        args.task_suite_name,
                        task_id,
                        task_name,
                        episode_idx,
                        suffix,
                        100.0 * task_successes / max(episode_idx + 1, 1),
                        100.0 * total_successes / max(total_episodes, 1),
                    )
            finally:
                env.close()

            task_summary = {
                "task_suite_name": args.task_suite_name,
                "task_id": int(task_id),
                "task_name": task_name,
                "task_description": task_description,
                "num_trials": int(num_trials),
                "successes": int(task_successes),
                "success_rate": float(task_successes / max(num_trials, 1)),
                "task_dir": str(task_dir),
            }
            task_summaries.append(task_summary)
            task_summary_path = task_dir / "summary.json"
            write_json(task_summary_path, task_summary)
            logging.info(
                "Saved task summary for suite=%s task=%02d (%s) to %s",
                args.task_suite_name,
                task_id,
                task_name,
                task_summary_path,
            )

        overall_summary = {
            "task_suite_name": args.task_suite_name,
            "task_id": args.task_id,
            "seed": int(args.seed),
            "ckpt_path": str(args.ckpt_path),
            "inference_mode": "remote" if remote_client is not None else "local",
            "deployment_mode": "split_ws" if remote_client is not None else "local",
            "ws_url": args.ws_url,
            "stats_key": args.stats_key,
            "infer_horizon": int(infer_horizon),
            "total_episodes": int(total_episodes),
            "total_successes": int(total_successes),
            "success_rate": float(total_successes / max(total_episodes, 1)),
            "task_summaries": task_summaries,
        }
        if remote_metadata is not None:
            overall_summary["remote_server_metadata"] = remote_metadata
        overall_summary_path = args.video_dir / "summary.json"
        write_json(overall_summary_path, overall_summary)
        logging.info(
            "Finished LIBERO evaluation for suite=%s task_id=%s (%s). Success rate: %.2f%% (%s/%s). Summary saved to %s",
            args.task_suite_name,
            args.task_id if args.task_id is not None else "all",
            "split websocket mode" if remote_client is not None else "local mode",
            100.0 * overall_summary["success_rate"],
            total_successes,
            total_episodes,
            overall_summary_path,
        )
    finally:
        if remote_client is not None:
            remote_client.close()


def main(args: EvalArgs) -> None:
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    evaluate_suite(args)


if __name__ == "__main__":
    tyro.cli(main)
