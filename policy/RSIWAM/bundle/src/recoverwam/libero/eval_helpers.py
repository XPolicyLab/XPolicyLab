import hashlib
import logging
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as transforms_F

_STANDARD_SUITE_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}
_LIBERO_PRO_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
_PI0_TEXT_LATENT_OOD_SUITE_MAX_STEPS = {
    "libero_spatial_ood": 330,
    "libero_object_ood": 420,
    "libero_goal_ood": 450,
}
_RESET_MODES = {"task_init_state", "random"}
_TRIAL_SEED_MODES = {"sequential", "task_episode_hash", "benchmark_seed_sequence"}
_TASK_SUITE_PATTERN = re.compile("^[A-Za-z0-9_-]+$")


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"LIBERO camera image must be [H,W,3], got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    resized_w, resized_h = resized.size
    left = max((resized_w - width) // 2, 0)
    top = max((resized_h - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _training_direct_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Apply the tensor resize used by the LIBERO training processor."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"LIBERO camera image must be [H,W,3], got {array.shape}")
    tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    else:
        tensor = tensor.to(dtype=torch.float32)
        if float(tensor.max().item()) > 1.0:
            tensor.div_(255.0)
    resized = transforms_F.resize(
        tensor,
        size=[int(height), int(width)],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resized.permute(1, 2, 0).numpy()


def compose_libero_camera_views(
    images: dict,
    *,
    concatenation: str,
    image_shapes: list,
    num_cameras: int | None = None,
    preprocess_mode: str = "legacy_center_crop",
) -> np.ndarray:
    """Compose simulator cameras under a selected evaluation preprocessing mode."""
    preprocess_mode = str(preprocess_mode).strip()
    if preprocess_mode not in {"legacy_center_crop", "training_exact"}:
        raise ValueError(
            f"Invalid LIBERO camera preprocess mode: {preprocess_mode!r}. Expected legacy_center_crop or training_exact."
        )
    resize = _training_direct_resize if preprocess_mode == "training_exact" else _center_crop_resize

    def finalize(array: np.ndarray) -> np.ndarray:
        if preprocess_mode == "training_exact":
            return array * 255.0
        return array

    if num_cameras is None:
        num_cameras = len(image_shapes)
    num_cameras = int(num_cameras)
    if num_cameras not in {1, 2}:
        raise ValueError(f"LIBERO eval supports one or two cameras, got {num_cameras}")
    if len(image_shapes) < num_cameras:
        raise ValueError(f"Only {len(image_shapes)} image shapes for {num_cameras} cameras")

    def _shape_to_hw(shape, camera_idx: int) -> tuple[int, int]:
        if len(shape) != 3:
            raise ValueError(f"image_shapes[{camera_idx}] must be [C,H,W], got {shape}")
        return (int(shape[1]), int(shape[2]))

    primary_h, primary_w = _shape_to_hw(image_shapes[0], 0)
    primary = resize(images["image"], width=primary_w, height=primary_h)
    if num_cameras == 1:
        return finalize(primary)
    wrist_h, wrist_w = _shape_to_hw(image_shapes[1], 1)
    wrist = resize(images["wrist_image"], width=wrist_w, height=wrist_h)
    if concatenation == "horizontal":
        return finalize(np.concatenate([primary, wrist], axis=1))
    if concatenation == "vertical":
        return finalize(np.concatenate([primary, wrist], axis=0))
    if concatenation == "pretrain_tshape":
        primary = resize(primary, width=304, height=240)
        wrist = resize(wrist, width=112, height=120)
        right = np.zeros((240, 112, 3), dtype=primary.dtype)
        right[:120] = wrist
        return finalize(np.concatenate([primary, right], axis=1))
    raise ValueError(
        f"Invalid concat_multi_camera: {concatenation!r}. Expected horizontal, vertical, or pretrain_tshape."
    )


def load_trusted_task_init_states(task_suite, task_id: int):
    """Load trusted official LIBERO init files under PyTorch 2.6+.

    LIBERO benchmark implementations call ``torch.load`` without an explicit
    ``weights_only`` argument.  These files contain numpy objects rather than
    model weights, so PyTorch 2.6+'s new default rejects them.  The override is
    scoped to this single benchmark call and should only be used with trusted
    local benchmark data.
    """
    original_torch_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _trusted_load
    try:
        return task_suite.get_task_init_states(int(task_id))
    finally:
        torch.load = original_torch_load


def resolve_reset_mode(cfg) -> str:
    """Resolve the episode reset protocol without changing legacy defaults."""
    reset_mode = str(cfg.EVALUATION.get("reset_mode", "task_init_state")).strip()
    if reset_mode not in _RESET_MODES:
        raise ValueError(
            f"Invalid LIBERO reset mode: {reset_mode!r}. Expected task_init_state or random."
        )
    return reset_mode


def reset_libero_episode(env, initial_state, reset_mode: str, episode_seed: int | None = None):
    """Reset one episode according to the selected benchmark protocol."""
    if episode_seed is not None:
        env.seed(int(episode_seed))
    if reset_mode == "random":
        return env.reset()
    if reset_mode == "task_init_state":
        env.reset()
        return env.set_init_state(initial_state)
    raise ValueError(
        f"Invalid LIBERO reset mode: {reset_mode!r}. Expected task_init_state or random."
    )


def resolve_trial_seed_mode(cfg) -> str:
    """Select sequential resets or the paper's deterministic trial seed hash."""
    mode = str(cfg.EVALUATION.get("trial_seed_mode", "sequential")).strip()
    if mode not in _TRIAL_SEED_MODES:
        raise ValueError(
            f"Invalid LIBERO trial seed mode: {mode!r}. Expected sequential, task_episode_hash, or benchmark_seed_sequence."
        )
    if mode == "task_episode_hash":
        if resolve_reset_mode(cfg) != "random":
            raise ValueError("task_episode_hash requires EVALUATION.reset_mode=random.")
        if cfg.EVALUATION.get("trial_seed_id") is None:
            raise ValueError("task_episode_hash requires EVALUATION.trial_seed_id.")
    if mode == "benchmark_seed_sequence":
        start = int(cfg.EVALUATION.get("trial_seed_start", 1))
        if start < 0:
            raise ValueError(
                "benchmark_seed_sequence requires a non-negative EVALUATION.trial_seed_start."
            )
    return mode


def hash_libero_trial_seed(
    task_suite_name: str, task_id: int, episode_index: int, seed_id: int
) -> int:
    """Hash one paper-protocol trial tuple into a simulator seed.

    The paper specifies a deterministic hash over ``(task, episode-index,
    seed-id)`` but does not publish a concrete hash function.  This repository
    pins a portable SHA-256-based v1 scheme so reruns and machines use exactly
    the same initial states.
    """
    task_id = int(task_id)
    episode_index = int(episode_index)
    seed_id = int(seed_id)
    if task_id < 0 or episode_index < 0 or seed_id < 0:
        raise ValueError("task_id, episode_index, and seed_id must all be non-negative.")
    payload = f"libero-trial-seed-v1\x00{str(task_suite_name).strip()}\x00{task_id}\x00{episode_index}\x00{seed_id}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 2147483647


def build_trial_initial_states(
    task_suite, task_id: int, num_trials: int, reset_mode: str, init_state_start: int = 0
) -> list:
    """Build per-trial state inputs, avoiding init files for random resets."""
    if reset_mode not in _RESET_MODES:
        raise ValueError(
            f"Invalid LIBERO reset mode: {reset_mode!r}. Expected task_init_state or random."
        )
    num_trials = int(num_trials)
    init_state_start = int(init_state_start)
    if num_trials < 1:
        raise ValueError(f"num_trials must be positive, got {num_trials}")
    if init_state_start < 0:
        raise ValueError(f"init_state_start must be non-negative, got {init_state_start}")
    if reset_mode == "random":
        return [None] * num_trials
    initial_states = list(load_trusted_task_init_states(task_suite, task_id))
    if not initial_states:
        raise ValueError(f"No init states found for LIBERO task {task_id}")
    if init_state_start == 0 and len(initial_states) < num_trials:
        while len(initial_states) < num_trials:
            initial_states.extend(initial_states[: num_trials - len(initial_states)])
    stop = init_state_start + num_trials
    if len(initial_states) < stop:
        raise ValueError(
            f"LIBERO task {task_id} exposes {len(initial_states)} init states, but indices [{init_state_start}, {stop}) were requested."
        )
    return initial_states[init_state_start:stop]


def _resolve_action_horizon(cfg) -> int:
    explicit = cfg.EVALUATION.get("action_horizon")
    if explicit is not None:
        action_horizon = int(explicit)
    else:
        data_horizon = cfg.data.train.get("action_horizon")
        if data_horizon is not None:
            action_horizon = int(data_horizon)
        else:
            action_horizon = int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"Action horizon must be positive, got {action_horizon}")
    return action_horizon


def _load_model_checkpoint(model, ckpt, video_checkpoint_override=None) -> None:
    kwargs = {}
    if video_checkpoint_override is not None:
        kwargs["video_checkpoint_override"] = video_checkpoint_override
    model.load_checkpoint(ckpt, **kwargs)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)


def resolve_max_steps(task_suite_name: str) -> int:
    task_suite_name = str(task_suite_name).strip()
    if task_suite_name in _STANDARD_SUITE_MAX_STEPS:
        return _STANDARD_SUITE_MAX_STEPS[task_suite_name]
    if task_suite_name in _PI0_TEXT_LATENT_OOD_SUITE_MAX_STEPS:
        return _PI0_TEXT_LATENT_OOD_SUITE_MAX_STEPS[task_suite_name]
    for prefix, max_steps in _LIBERO_PRO_SUITE_MAX_STEPS.items():
        if task_suite_name.startswith(f"{prefix}_"):
            return max_steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def validate_task_file(path: str | Path) -> list[tuple[str, int]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation task file not found: {resolved}")
    tasks: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid task-file row {line_number}: expected suite,task_id; got {line!r}"
            )
        suite_name, raw_task_id = parts
        if not _TASK_SUITE_PATTERN.fullmatch(suite_name):
            raise ValueError(f"Invalid task suite at row {line_number}: {suite_name!r}")
        try:
            task_id = int(raw_task_id)
        except ValueError as exc:
            raise ValueError(f"Invalid task id at row {line_number}: {raw_task_id!r}") from exc
        if task_id < 0:
            raise ValueError(f"Task id must be non-negative at row {line_number}: {task_id}")
        task = (suite_name, task_id)
        if task in seen:
            raise ValueError(
                f"Duplicate evaluation task at row {line_number}: {suite_name},{task_id}"
            )
        seen.add(task)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"Evaluation task file is empty: {resolved}")
    return tasks


def resolve_task_bddl_path(task, bddl_root: str | Path) -> str:
    """Return a string path so LIBERO-Plus can parse logical view suffixes."""
    return str(Path(bddl_root) / str(task.problem_folder) / str(task.bddl_file))
