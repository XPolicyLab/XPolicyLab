from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal

import torch
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig
from ..utils.normalizer import LinearNormalizer, NormMode
from flexpi.utils.pytorch_utils import dict_apply
from flexpi.utils.logging_config import get_logger
from .base_processor import BaseProcessor, _admissible_widths

logger = get_logger(__name__)

class FlexPiProcessor(BaseProcessor):
    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,

        action_state_transforms: Optional[List[Any]],

        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]],

        action_state_merger,

        # image transform
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None,

        # instruction transform
        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,

        tokenizer: Optional[Any] = None,
        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,
        norm_skip_dims_mask: Optional[Dict[str, Dict[str, List[bool]]]] = None,
        action_norm_clamp: Optional[float] = None,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        # image
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self.norm_skip_dims_mask = norm_skip_dims_mask
        self.action_norm_clamp = action_norm_clamp
        self._normalizer = None

        self.tokenizer = tokenizer
        if delta_action_dim_mask is None:
            self.delta_action_dim_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(delta_action_dim_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`delta_action_dim_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )

            self.delta_action_dim_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["raw_shape"]
                mask = delta_action_dim_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`delta_action_dim_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.delta_action_dim_mask[key] = torch.as_tensor(mask, dtype=torch.bool)

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("normalizer has not been set. Please call set_normalizer_from_stats() first.")
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
            skip_dims_mask=self.norm_skip_dims_mask,
            action_clamp=self.action_norm_clamp,
        )

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """
        Args:
            data: Dict[str, str] | List[str], lerobot sample in raw mcap

        Returns:
            List[str], processed instructions
        """
        # if single instruction, convert to list
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = f"{low_level_instruction}"
        else:
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"

        return instruction

    def action_state_transform(self, batch):
        # `raw_shape` may be a list of admissible widths ([32, 34] for mixed
        # yam/mm2 + AgiBot runs); the chain must still EMIT exactly `shape`.
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape in _admissible_widths(meta_shape), \
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape in _admissible_widths(meta_shape), \
                f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        if self.action_state_transforms is not None:
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)

        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.

        Args:
            Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index

        Returns:
            Sample: Dict[str, Any], which can collated:
                - "input_ids": torch.Tensor -> [max_image_text_tokens,]
                - "attention_mask": torch.Tensor -> [max_image_text_tokens,]
                - "pixel_values": torch.Tensor -> [num_input_cameras, C, H, W]
                  (uniform-shape path only — emitted when every cam comes out at
                  the same (H, W) after transforms; this is bit-identical to the
                  legacy single-stacked-tensor flow)
                - "per_cam_rgb": Dict[str, torch.Tensor] -> {cam_key: [T, C, H, W]}
                  (heterogeneous-shape path — emitted INSTEAD of "pixel_values"
                  when per-cam transforms target different (H, W) per cam, e.g.
                  YAM head 256×320 + wrists 224×224. Mutually exclusive with
                  "pixel_values" — downstream datasets detect the presence of
                  per_cam_rgb and bypass the stacked-tensor view + per-cam resize.)
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action: Optional, deepcopy of input action for open loop eval, which is left untouched
                - "idx": int, sample index
        """
        sample = {}
        # 1. instruction
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]

        # 2. image
        processed_images = []
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]
            assert image.ndim == 4, f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"

            # Apply transforms efficiently on the merged batch.
            # Accept either a flat list (applied to all cams) or a per-cam
            # mapping. The DictConfig branch is required because Hydra leaves
            # nested DictConfig nodes intact under default _convert_=none —
            # `isinstance(transforms, dict)` alone returns False for
            # DictConfig and silently falls into the flat-list branch,
            # which then iterates the keys (strings) and crashes with
            # "'str' object is not callable".
            transforms = self.train_transforms if self.is_train else self.val_transforms
            if isinstance(transforms, (dict, DictConfig)):
                current_transforms = transforms[key]
            else:
                current_transforms = transforms
            for trans in current_transforms:
                image = trans(image)

            # Spatial dims [C, H, W] must match shape_meta.images[*].shape exactly.
            # The temporal dim T can vary: `num_obs_steps` (legacy) or any
            # shorter value when BaseLerobotDataset.image_sample_stride > 1
            # decoded only the frames the model uses. Downstream consumers
            # (Robot{,PerCam,PerCamDepth}VideoDataset) are idempotent w.r.t. T.
            assert list(image.shape[1:]) == list(shape), \
                f"Expected per-frame shape {shape}, got {tuple(image.shape[1:])} for key {key}"
            assert image.shape[0] >= 1, \
                f"Empty temporal dim for key {key}: {tuple(image.shape)}"

            processed_images.append((key, image))

        # Detect heterogeneous per-cam shapes (audit-fix #5). With per-cam
        # transforms targeting different output sizes (e.g. YAM head 256×320,
        # wrists 224×224), `torch.stack` would crash on unequal tensor shapes.
        # When shapes diverge, emit a `per_cam_rgb` dict keyed by cam name;
        # downstream datasets pick this up and skip their per-cam resize loop.
        # The uniform-shape case stays bit-identical to the legacy path: stack
        # into `pixel_values` with the same num_output_cameras handling.
        first_shape = tuple(processed_images[0][1].shape)
        uniform_shape = all(
            tuple(img.shape) == first_shape for _, img in processed_images
        )

        if uniform_shape:
            pixel_values = torch.stack(
                [img for _, img in processed_images], dim=0
            )  # [num_input_cameras, T, C, H, W]
            if self.num_output_cameras > pixel_values.shape[0]:
                out = torch.zeros((self.num_output_cameras,) + pixel_values.shape[1:], device=pixel_values.device, dtype=pixel_values.dtype)
                out[0: pixel_values.shape[0]] = pixel_values
                sample["pixel_values"] = out
            elif self.num_output_cameras < pixel_values.shape[0]:
                logger.warning(f"num_output_cameras {self.num_output_cameras} is less than the number of cameras in data {pixel_values.shape[0]}, "
                               f"truncating the input to the first {self.num_output_cameras} cameras.")
                sample["pixel_values"] = pixel_values[:self.num_output_cameras]
            else:
                sample["pixel_values"] = pixel_values
        else:
            # Heterogeneous: emit a per-cam dict. Downstream wrappers detect
            # the presence of `per_cam_rgb` and consume it directly without
            # going through `pixel_values.view(...)`. Per-cam padding via
            # num_output_cameras has no meaningful uniform-tensor analogue
            # here — datasets that need cam-count padding use the dataset
            # layer's `synthetic_zero_cams` knob instead.
            if self.num_output_cameras != len(processed_images):
                raise ValueError(
                    f"Heterogeneous per-cam transforms produced "
                    f"{len(processed_images)} cams, but num_output_cameras="
                    f"{self.num_output_cameras}. Per-cam padding via "
                    f"num_output_cameras is not supported on the heterogeneous "
                    f"path; use the dataset's synthetic_zero_cams instead."
                )
            sample["per_cam_rgb"] = {key: img for key, img in processed_images}

        # Copy action before transform for open-loop evaluation,
        # disabled for training dataset as it may cause collating key problem.
        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        # 3. action & state
        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.delta_action_dim_mask.items():
                    cur_action = data["action"][key]
                    cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                    cur_dim_mask = dim_mask.to(device=cur_action.device)
                    pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                    cur_action[pad_delta_mask] = 0.0
        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"] # [action_horizon, action_dim]
            sample["action_is_pad"] = data["action_is_pad"] # [action_horizon,]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"] # [action_dim,]
            assert sample["action"].shape[-1] == self.action_output_dim
            # sample["action"][sample["action_is_pad"], :-1] = 0.0 # NOTE: we assume use delta_eef_pose + gripper， so pad action is 0


        # TODO: rename all "state" into "proprio"
        sample["proprio"] = data["state"] # [num_obs_steps, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"] # [num_obs_steps,]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"] # [proprio_dim,]
        assert sample["proprio"].shape[-1] == self.proprio_output_dim

        sample["idx"] = data["idx"]

        # sample = self.tokenizer(sample)

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the data for the policy model.

        Args:
            data: Dict[str, Any], lerobot sample in raw mcap

        Returns:
            data: Dict[str, Any], processed data including unnormalized action
        """
        assert "action" in data, "Action is required in postprocess"
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)

        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data
