import json
import inspect
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from recoverwam.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from recoverwam.libero.eval_helpers import (
    compose_libero_camera_views,
    hash_libero_trial_seed,
    reset_libero_episode,
    resolve_reset_mode,
    resolve_trial_seed_mode,
    resolve_max_steps,
)
from recoverwam.datasets.lerobot.processors.recoverwam_processor import RecoverWAMProcessor
from recoverwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))
    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    msg = "Failed to locate dataset_stats.json. Tried explicit EVALUATION.dataset_stats_path and checkpoint parent directories. Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    raise FileNotFoundError(msg)


def _normalize_proprio(proprio: np.ndarray, processor: RecoverWAMProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: RecoverWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, but num_output_cameras={processor.num_output_cameras}."
        )
    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    image_shapes = [meta["shape"] for meta in image_meta]
    rgb = compose_libero_camera_views(
        imgs,
        concatenation=str(concatenation),
        image_shapes=image_shapes,
        num_cameras=int(processor.num_output_cameras),
        preprocess_mode=str(cfg.EVALUATION.get("camera_preprocess_mode", "legacy_center_crop")),
    )
    actual_h, actual_w = (int(rgb.shape[0]), int(rgb.shape[1]))
    expected_h, expected_w = (int(height), int(width))
    assert actual_h == expected_h and actual_w == expected_w, (
        f"Input image size mismatch after per-camera resize + concat: got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) from data.train.video_size={[expected_h, expected_w]}; shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    return (x, proprio, imgs)


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: RecoverWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if any(
        (
            bool(cfg.EVALUATION.get(k, False))
            for k in ("visualize_future_video", "save_one_step_video", "use_action_ensembler")
        )
    ):
        raise ValueError(
            "Paper LIBERO release uses plain action inference without video capture or ensembling"
        )


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: RecoverWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    capture_one_step_video: bool = False,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )
    inference_seed = None if cfg.get("seed") is None else int(cfg.seed)
    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": None
        if cfg.EVALUATION.get("sigma_shift") is None
        else float(cfg.EVALUATION.get("sigma_shift")),
        "seed": inference_seed,
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    video_seed_offset = int(cfg.EVALUATION.get("video_seed_offset", 0))
    if video_seed_offset != 0:
        if inference_seed is None:
            raise ValueError("EVALUATION.video_seed_offset requires a non-null evaluation seed.")
        if "video_seed" not in inspect.signature(model.infer_action).parameters:
            raise ValueError("The selected model does not support an independent video seed.")
        infer_kwargs["video_seed"] = inference_seed + video_seed_offset
    predicted_future_frames = None
    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    with torch.no_grad():
        pred = model.infer_action(**infer_kwargs)
    action = pred["action"]
    action = _denormalize_action(action, processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return (action, imgs, predicted_future_frames)


def _get_max_steps(task_suite_name: str) -> int:
    return resolve_max_steps(task_suite_name)


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: RecoverWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    episode_seed: int | None = None,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    reset_mode = resolve_reset_mode(cfg)
    obs = reset_libero_episode(env, initial_state, reset_mode, episode_seed=episode_seed)
    replay_images = []
    save_rollout_videos = bool(cfg.EVALUATION.get("save_rollout_videos", True))
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    t = 0
    done = False
    pbar = tqdm(
        total=max_steps + num_steps_wait,
        desc=f"Episode {episode_idx + 1}",
        disable=bool(cfg.EVALUATION.get("disable_progress_bar", False)),
    )
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                capture_one_step_video=False,
            )
            pending_actions = action_chunk[:replan_steps].tolist()
            if save_rollout_videos:
                replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            if save_rollout_videos:
                replay_images.append(imgs.copy())
        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
        t += 1
    pbar.close()
    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return (bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr)


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: RecoverWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    fresh_environment_per_episode = bool(cfg.EVALUATION.get("fresh_environment_per_episode", False))
    if fresh_environment_per_episode:
        env = None
        task_description = str(task.language)
    else:
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    trial_seed_mode = resolve_trial_seed_mode(cfg)
    trial_seed_id = cfg.EVALUATION.get("trial_seed_id")
    trial_seed_start = int(cfg.EVALUATION.get("trial_seed_start", 1))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "trial_seed_mode": trial_seed_mode,
        "trial_seed_id": None if trial_seed_id is None else int(trial_seed_id),
        "trial_seed_hash_scheme": "sha256-v1" if trial_seed_mode == "task_episode_hash" else None,
        "trial_seeds": [],
        "fresh_environment_per_episode": fresh_environment_per_episode,
    }
    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        episode_seed = None
        if trial_seed_mode == "task_episode_hash":
            episode_seed = hash_libero_trial_seed(
                str(cfg.EVALUATION.task_suite_name),
                int(cfg.EVALUATION.task_id),
                trial_idx,
                int(trial_seed_id),
            )
        elif trial_seed_mode == "benchmark_seed_sequence":
            episode_seed = trial_seed_start + trial_idx
        results["trial_seeds"].append(episode_seed)
        episode_env = env
        if fresh_environment_per_episode:
            episode_env, _ = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
        try:
            success, replay_images, predicted_future_video_clips, episode_mean_psnr = (
                run_single_episode(
                    env=episode_env,
                    initial_state=initial_states[trial_idx],
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    episode_idx=trial_idx,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                    episode_seed=episode_seed,
                )
            )
        finally:
            if fresh_environment_per_episode:
                episode_env.close()
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if bool(cfg.EVALUATION.get("save_rollout_videos", True)):
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
    if env is not None:
        env.close()
    return results
