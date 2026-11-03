from typing import Dict, Any, Optional, List

import torch
import numpy as np
from copy import deepcopy
from ..utils.normalizer import LinearNormalizer, NormMode
from wam.utils.pytorch_utils import dict_apply
from wam.utils.logging_config import get_logger
from .base_processor import BaseProcessor

logger = get_logger(__name__)

class WAMProcessor(BaseProcessor):
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
        vlm_current_image_position: Optional[int] = None,
        vlm_use_processed_images: bool = False,
        normalization_passthrough_dims: Optional[
            Dict[str, Dict[str, List[int]]]
        ] = None,
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
        self.normalization_passthrough_dims = normalization_passthrough_dims
        self._normalizer = None

        self.tokenizer = tokenizer
        self.vlm_current_image_position = vlm_current_image_position
        self.vlm_use_processed_images = bool(vlm_use_processed_images)
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
                expected_dim = meta["shape"]
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
            passthrough_dims=self.normalization_passthrough_dims,
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
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
                    
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
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

    def _transform_images(self, data: Dict[str, Any]) -> List[torch.Tensor]:
        processed_images = []
        transforms = (
            self.train_transforms if self.is_train else self.val_transforms
        )
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]
            if image.ndim != 4:
                raise ValueError(
                    "Expected image [T,C,H,W], got "
                    f"{tuple(image.shape)} for camera {key!r}."
                )
            current_transforms = (
                transforms[key] if isinstance(transforms, dict) else transforms
            )
            for transform in current_transforms:
                image = transform(image)
            expected_frame_shape = tuple(shape)
            if tuple(image.shape[1:]) != expected_frame_shape:
                raise ValueError(
                    f"Expected frame shape {expected_frame_shape}, got "
                    f"{tuple(image.shape[1:])} after transforms for key {key}."
                )
            processed_images.append(image)
        return processed_images

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._preprocess_images(
            data,
            vlm_current_image_position=self.vlm_current_image_position,
            include_pixel_values=True,
        )

    def preprocess_vlm_current(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess non-image fields and retain only the current VLM views.

        This entry point is used by latent-cache training.  The cached VAE
        tensors already provide the current/anchor/recent video inputs, so the
        source dataset only has to decode one current frame per camera for the
        understanding branch.
        """

        return self._preprocess_images(
            data,
            vlm_current_image_position=0,
            include_pixel_values=False,
        )

    def preprocess_without_images(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Keep actions, state and instructions when encoder artifacts supply images."""
        return self._preprocess_images(
            data, vlm_current_image_position=None, include_pixel_values=False
        )

    def _preprocess_images(
        self,
        data: Dict[str, Any],
        *,
        vlm_current_image_position: Optional[int],
        include_pixel_values: bool,
    ) -> Dict[str, Any]:
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
        if (
            vlm_current_image_position is not None
            and not self.vlm_use_processed_images
        ):
            sample["vlm_current_images"] = torch.stack(
                [
                    data["images"][meta["key"]][vlm_current_image_position].clone()
                    for meta in self.shape_meta["images"]
                ],
                dim=0,
            )
        processed_images = []
        needs_processed_images = include_pixel_values or (
            vlm_current_image_position is not None
            and self.vlm_use_processed_images
        )
        if needs_processed_images:
            processed_images = self._transform_images(data)
        if (
            vlm_current_image_position is not None
            and self.vlm_use_processed_images
        ):
            sample["vlm_current_images"] = torch.stack(
                [
                    image[vlm_current_image_position].clone()
                    for image in processed_images
                ],
                dim=0,
            )
        if include_pixel_values:
            pixel_values = torch.stack(processed_images, dim=0) # [num_input_cameras, T, C, H, W]

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

        # Copy action before transform for open-loop evaluation, 
        # disabled for training dataset as it may cause collating key problem.
        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        # 3. action & state
        data = self.action_state_transform(data)
        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.delta_action_dim_mask.items():
                    cur_action = data["action"][key]
                    cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                    cur_dim_mask = dim_mask.to(device=cur_action.device)
                    pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                    cur_action[pad_delta_mask] = 0.0
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
