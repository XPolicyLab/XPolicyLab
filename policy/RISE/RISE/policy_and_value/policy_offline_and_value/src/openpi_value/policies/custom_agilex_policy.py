"""Policy transforms for the CustomAgilex robot."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi_value.models.model as _model
import openpi_value.transforms as transforms

@dataclasses.dataclass(frozen=True)
class CustomAgilexInputs(transforms.DataTransformFn):
    """Inputs for the CustomAgilex policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    required_rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }


    # * Not required cameras, can be ignored if not in the dataloader
    optional_rename_map = {
        "his_-100_top_head": "base_-100_rgb",
        "his_-100_hand_left": "left_wrist_-100_rgb",
        "his_-100_hand_right": "right_wrist_-100_rgb",

        # * future frames
        
        "fut_1_top_head": "base_1_rgb",
        "fut_1_hand_left": "left_wrist_1_rgb",  # Fixed: was "his_1_hand_left"
        "fut_1_hand_right": "right_wrist_1_rgb",  # Fixed: was "his_1_hand_right"
    }

    all_rename_map = {**required_rename_map, **optional_rename_map}

    EXTRA_CAMERAS = tuple(optional_rename_map.keys())
    
    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0

        # * ['hand_left', 'hand_right', 'his_-100_cam_hand_left', 'his_-100_cam_hand_right', 'his_-100_top_head', 'top_head']

        in_images = data["images"]

        # * ALL in_images keys must be in set(EXPECTED_CAMERAS + EXTRA_CAMERAS)
        # * but in_images keys can be a subset of EXPECTED_CAMERAS + EXTRA_CAMERAS
        if set(in_images) - set(self.EXPECTED_CAMERAS) - set(self.EXTRA_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        image_masks = {}
        for camera in self.EXPECTED_CAMERAS + self.EXTRA_CAMERAS:
            if camera in in_images:
                img = in_images[camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                # images[self.rename_map[camera]] = img
                images[self.all_rename_map[camera]] = img
                image_masks[self.all_rename_map[camera]] = np.True_

            elif camera not in in_images and camera in self.EXTRA_CAMERAS:
                # images[self.all_rename_map[camera]] = np.zeros_like(img)
                continue  # * optional camera can be skipped
            else:
                raise ValueError(f"Camera {camera} not found in data")

        # Create image mask based on available cameras
        # image_mask = {self.required_rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}


        # filter unnormal state / action value, set to 0
        state = np.where(state > np.pi, 0, state)
        state = np.where(state < -np.pi, 0, state)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Add actions if present
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            actions = np.where(actions > np.pi, 0, actions)
            actions = np.where(actions < -np.pi, 0, actions)
            if mask_padding:
                # Create action mask for padding
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[:, self.action_dim:] = False
                inputs["action_mask"] = action_mask
            
            inputs["actions"] = actions.squeeze()


        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # * Custom
        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]
        
        if "frame_index_progress" in data:
            inputs["frame_index_progress"] = data["frame_index_progress"]
        
        if "is_failure_data" in data:
            inputs["is_failure_data"] = data["is_failure_data"]

        if "is_infer_data" in data:
            inputs["is_infer_data"] = data["is_infer_data"]

        if "episode_length" in data:
            inputs["episode_length"] = data["episode_length"]

        if "inferred_action" in data:
            inputs["inferred_action"] = data["inferred_action"]

        if "noise" in data:
            inputs["noise"] = data["noise"]


        # * if 'action_advantage' in repack_transform, will always be created (default 1)
        # * if not in repack_transform, will NOT be created.
        if "action_advantage" in data:
            action_advantage = data["action_advantage"]

            if action_advantage is not None:
                if type(action_advantage) is np.ndarray:
                    action_advantage = torch.from_numpy(action_advantage)
                elif type(action_advantage) is torch.Tensor:
                    action_advantage = action_advantage.detach().clone()
                else:
                    NotImplementedError(f"Unsupported type for action_advantage: {type(action_advantage)}")
                
            else:
                action_advantage = torch.tensor(1.)
            
            inputs["action_advantage"] = action_advantage

        if "action_advantage_original" in data:
            action_advantage_original = data["action_advantage_original"]
        
            if type(action_advantage_original) is np.ndarray:
                action_advantage_original = torch.from_numpy(action_advantage_original)
            elif type(action_advantage_original) is torch.Tensor:
                action_advantage_original = action_advantage_original.detach().clone()
            else:
                NotImplementedError(f"Unsupported type for action_advantage_original: {type(action_advantage_original)}")
            
            inputs["action_advantage_original"] = action_advantage_original

        if "image_original" in data:
            inputs["image_original"] = data["image_original"]

        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CustomAgilexOutputs(transforms.DataTransformFn):
    """Outputs for the CustomAgilex policy."""

    def __call__(self, data: dict) -> dict:
        # Return the first 14 dimensions of actions (13 joints + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :14])} 