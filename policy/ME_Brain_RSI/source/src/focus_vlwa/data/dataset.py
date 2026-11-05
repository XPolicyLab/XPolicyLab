"""Focus-VLWA post-training dataset pipeline."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from focus_vlwa.configs.model import FocusVLWAConfig
from focus_vlwa.data import head_history
from focus_vlwa.data.normalization import NormStats, normalize_quantile
from focus_vlwa.model.observation import FocusVLWAObservation

if TYPE_CHECKING:
    from focus_vlwa.inference.tokenizer import FocusVLWATokenizer


_DELTA_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False])
_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_COORDINATE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def _field(sample: dict[str, Any], path: str, default: Any = None) -> Any:
    if path in sample:
        return sample[path]
    value: Any = sample
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item() if value.size else default
    return value


def _prompt(sample: dict[str, Any]) -> str:
    raw = _field(sample, "prompt_pack", _field(sample, "prompt", _field(sample, "task", "")))
    raw = _scalar(raw, "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    text = str(raw).strip()
    try:
        choices = json.loads(text)
    except json.JSONDecodeError:
        choices = [text]
    if isinstance(choices, str):
        choices = [choices]
    choices = [str(choice).strip() for choice in choices if str(choice).strip()]
    if not choices:
        raise ValueError("Dataset sample does not contain a task prompt")
    selected = choices[int(np.random.randint(len(choices)))]

    def round_coordinate(match: re.Match[str]) -> str:
        x = int(int(match.group(1)) / 100 + 0.5) * 100
        y = int(int(match.group(2)) / 100 + 0.5) * 100
        return f"[{x},{y}]"

    return _COORDINATE.sub(round_coordinate, selected)


def _image(image: Any, *, pad: bool = False, head: bool = False) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = array.astype(np.uint8)
    if head:
        return np.asarray(head_history.full_frame(array))
    if array.shape[:2] != (224, 224):
        if pad:
            from focus_vlwa.data.image_tools import resize_with_pad_pil
            return resize_with_pad_pil(array, 224, 224)
        array = np.asarray(Image.fromarray(array).resize((224, 224), Image.Resampling.BILINEAR))
    return array


def _pad(array: np.ndarray, dimension: int) -> np.ndarray:
    if array.shape[-1] >= dimension:
        return array[..., :dimension]
    return np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, dimension - array.shape[-1])])


def _as_numpy(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def resolve_repo_ids(pattern: str) -> list[str]:
    from focus_vlwa.data.lerobot import resolve_repo_ids as resolve
    return resolve(pattern)


class FocusVLWAPostTrainingDataset(Dataset):
    """Apply the released checkpoint transforms to post-training samples."""

    def __init__(
        self,
        dataset: str,
        norm_stats: dict[str, NormStats],
        tokenizer: FocusVLWATokenizer,
        model_config: FocusVLWAConfig,
    ):
        from focus_vlwa.data.lerobot import LeRobotShards
        self.source = LeRobotShards(resolve_repo_ids(dataset), model_config)
        self.norm_stats = norm_stats
        self.tokenizer = tokenizer
        self.model_config = model_config

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        state = _as_numpy(_field(sample, "observation.state")).reshape(-1)
        actions = _as_numpy(_field(sample, "action")).copy()
        if actions.ndim == 1:
            actions = actions[None]
        if actions.shape[0] != self.model_config.action_horizon:
            raise ValueError(
                f"Expected {self.model_config.action_horizon} action steps, got shape {actions.shape}. "
                "The released dataset stores precomputed action chunks."
            )
        action_dimension = min(14, actions.shape[-1], state.shape[-1])
        mask = _DELTA_MASK[:action_dimension]
        actions[..., :action_dimension] -= np.where(mask, state[:action_dimension], 0.0)

        normalized_state = normalize_quantile(state, self.norm_stats["state"])
        normalized_actions = normalize_quantile(actions, self.norm_stats["actions"])
        tokens, token_mask = self.tokenizer.tokenize(_prompt(sample), normalized_state)
        images = {name: _image(
            _field(sample, f"observation.images.{name}"),
            pad="wrist" in name,
            head=name == "cam_high",
        ) for name in _CAMERAS}

        direct_history = _field(sample, "hist_images")
        if direct_history is None:
            raise ValueError("Full head-history samples require hist_images and hist_mask")
        if _field(sample, "hist_cells") is not None:
            raise ValueError("History cells are not supported; provide full head frames")
        history_data = head_history.pack_history(direct_history, _field(sample, "hist_mask"))

        action_mask = _field(sample, "action_mask")
        if action_mask is None:
            action_mask = np.ones(actions.shape[0], dtype=np.float32)
        world_state = _field(sample, "world_state", _field(sample, "s1"))
        world_state_mask = _field(sample, "world_state_mask", _field(sample, "s1_mask"))
        event_action = _field(sample, "event_action")
        event_action_mask = _field(sample, "event_action_mask")
        if self.model_config.use_world_model:
            if any(value is None for value in (world_state, world_state_mask, event_action, event_action_mask)):
                raise ValueError("World-model training requires state/event targets and masks")
        if event_action is not None:
            event_action = _as_numpy(event_action).copy()
            event_action[..., :action_dimension] -= np.where(mask, state[:action_dimension], 0.0)
            event_action = normalize_quantile(event_action, self.norm_stats["actions"])

        return {
            "images": images,
            "state": _pad(normalized_state, self.model_config.action_dim),
            "tokens": tokens,
            "token_mask": token_mask,
            "actions": _pad(normalized_actions, self.model_config.action_dim),
            "action_mask": _as_numpy(action_mask),
            "world_state": None if world_state is None else _as_numpy(world_state),
            "world_state_mask": None if world_state_mask is None else _as_numpy(world_state_mask),
            "event_action": None if event_action is None else _pad(event_action, self.model_config.action_dim),
            "event_action_mask": None if event_action_mask is None else _as_numpy(event_action_mask),
            "history_images": history_data["hist_images"],
            "history_mask": history_data["hist_mask"],
        }


def _stack_optional(items: list[dict[str, Any]], key: str, dtype: torch.dtype) -> torch.Tensor | None:
    values = [item[key] for item in items]
    if any(value is None for value in values):
        return None
    return torch.as_tensor(np.stack(values), dtype=dtype)


def collate_focus_vlwa_batch(items: list[dict[str, Any]]) -> tuple[FocusVLWAObservation, torch.Tensor]:
    """Build the typed model input and action target from dataset samples."""
    images = {
        target: torch.as_tensor(np.stack([item["images"][source] for item in items]), dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div(255.0)
        .mul(2.0)
        .sub(1.0)
        for target, source in {
            "base_0_rgb": "cam_high",
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }.items()
    }
    batch_size = len(items)
    history_images = _stack_optional(items, "history_images", torch.float32)
    if history_images is not None:
        history_images = history_images.div(255.0).mul(2.0).sub(1.0)
    observation = FocusVLWAObservation(
        images=images,
        image_masks={key: torch.ones(batch_size, dtype=torch.bool) for key in images},
        state=torch.as_tensor(np.stack([item["state"] for item in items])),
        tokenized_prompt=torch.as_tensor(np.stack([item["tokens"] for item in items]), dtype=torch.long),
        tokenized_prompt_mask=torch.as_tensor(np.stack([item["token_mask"] for item in items]), dtype=torch.bool),
        action_mask=torch.as_tensor(np.stack([item["action_mask"] for item in items]), dtype=torch.float32),
        world_state=_stack_optional(items, "world_state", torch.float32),
        world_state_mask=_stack_optional(items, "world_state_mask", torch.float32),
        event_action=_stack_optional(items, "event_action", torch.float32),
        event_action_mask=_stack_optional(items, "event_action_mask", torch.float32),
        history_images=history_images,
        history_mask=_stack_optional(items, "history_mask", torch.float32),
    )
    actions = torch.as_tensor(np.stack([item["actions"] for item in items]), dtype=torch.float32)
    return observation, actions
