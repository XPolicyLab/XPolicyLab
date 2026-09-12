from collections.abc import Callable
from collections import deque

import numpy as np
import torch

from evals.common.simulator import ObservationMapping, SimulatorAdapter
from worldscape_policy.checkpoint.transforms import NativeCheckpointTransform
from worldscape_policy.data.augmentation import NativeVideoAugmentation
from worldscape_policy.types import ObservationBatch, WorldActionOutput

POLICY_VIDEO_NATIVE_WIDTH = 320
POLICY_VIDEO_NATIVE_HEIGHT = 160
DEFAULT_VLM_HISTORY_NUM_FRAMES = 8


class RoboTwin2Adapter(SimulatorAdapter):
    """Map RoboTwin 2 observations and actions to the WorldScape API."""

    def __init__(
        self,
        *,
        camera_keys: tuple[str, ...] = (
            "observation.head_camera.rgb",
            "observation.left_camera.rgb",
            "observation.right_camera.rgb",
        ),
        state_keys: tuple[str, ...] = ("joint_action.vector",),
        head_camera_key: str | None = "observation.head_camera.rgb",
        embodiment_id: int = 0,
        action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        checkpoint_transform: NativeCheckpointTransform | None = None,
        image_width: int = POLICY_VIDEO_NATIVE_WIDTH,
        image_height: int = POLICY_VIDEO_NATIVE_HEIGHT,
        vlm_history_num_frames: int = DEFAULT_VLM_HISTORY_NUM_FRAMES,
        preserve_camera_resolution: bool = False,
        action_horizon: int = 24,
    ) -> None:
        if vlm_history_num_frames < 1:
            raise ValueError("vlm_history_num_frames must be positive")
        if action_horizon not in {24, 48}:
            raise ValueError("RoboTwin2 action_horizon must be 24 or 48")
        vlm_video_transform = NativeVideoAugmentation(
            training=False,
            width=image_width,
            height=image_height,
        )
        self._vlm_video_transform = vlm_video_transform
        super().__init__(
            ObservationMapping(
                camera_keys=camera_keys,
                state_keys=state_keys,
                head_camera_key=head_camera_key,
                embodiment_id=embodiment_id,
            ),
            action_transform=action_transform,
            image_size=(image_height, image_width),
            image_resize_interpolation="area",
            preserve_camera_resolution=preserve_camera_resolution,
            head_video_transform=vlm_video_transform,
        )
        self.checkpoint_transform = checkpoint_transform
        self.vlm_history_num_frames = int(vlm_history_num_frames)
        self.action_horizon = int(action_horizon)
        self._vlm_anchor_history: deque[torch.Tensor] = deque(
            maxlen=self.vlm_history_num_frames
        )

    def set_checkpoint_transform(
        self, transform: NativeCheckpointTransform
    ) -> None:
        self.checkpoint_transform = transform

    def observation(
        self,
        value,
        *,
        device: torch.device | str = "cpu",
        vlm_history_frames: np.ndarray | None = None,
    ) -> ObservationBatch:
        batch = super().observation(value, device=device)
        if batch.proprioception.shape[-1] != 14:
            raise ValueError(
                "RoboTwin2 joint state must have width 14, got "
                f"{batch.proprioception.shape[-1]}"
            )
        if batch.images.shape[2] != 3:
            raise ValueError("RoboTwin2 requires exactly three camera views")
        if batch.images.shape[0] != 1:
            raise ValueError("RoboTwin2 evaluation supports batch size 1 only")
        if vlm_history_frames is not None:
            raw_history = np.asarray(vlm_history_frames)
            if (
                raw_history.dtype != np.uint8
                or raw_history.ndim != 4
                or raw_history.shape[-1] != 3
                or raw_history.shape[0] != self.vlm_history_num_frames
            ):
                raise ValueError(
                    "Explicit RoboTwin2 VLM history must be uint8 "
                    f"[{self.vlm_history_num_frames},H,W,3], got "
                    f"dtype={raw_history.dtype} shape={raw_history.shape}"
                )
            transformed = self._vlm_video_transform(raw_history)
            history = (
                torch.from_numpy(np.ascontiguousarray(transformed))
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .to(device=batch.images.device)
            )
        else:
            frame_count = int(batch.images.shape[1])
            current_head_frame = batch.head_view[0, -1]
            if frame_count == 1:
                self._vlm_anchor_history.append(
                    current_head_frame.detach().cpu().clone()
                )
            elif frame_count != 9:
                raise ValueError(
                    "RoboTwin2 observation history must contain 1 or 9 frames, "
                    f"got {frame_count}"
                )
            else:
                if not self._vlm_anchor_history:
                    self._vlm_anchor_history.append(
                        current_head_frame.detach().cpu().clone()
                    )
                self._vlm_anchor_history.append(
                    current_head_frame.detach().cpu().clone()
                )
            anchors = list(self._vlm_anchor_history)
            anchors = (
                [anchors[0]] * (self.vlm_history_num_frames - len(anchors))
                + anchors
            )
            history = torch.stack(
                anchors[-self.vlm_history_num_frames :], dim=0
            )
            history = history.unsqueeze(0).to(device=batch.images.device)
        batch.vlm_history_images = history
        batch.vlm_history_mask = torch.ones(
            history.shape[:2], dtype=torch.bool, device=history.device
        )
        if self.checkpoint_transform is not None:
            state = self.checkpoint_transform.apply_state(
                {"state.vector": batch.proprioception}
            )
            max_state_dim = self.checkpoint_transform.embodiment.max_state_dim
            if state.shape[-1] > max_state_dim:
                raise ValueError(
                    f"Normalized RoboTwin2 state width {state.shape[-1]} exceeds "
                    f"checkpoint maximum {max_state_dim}"
                )
            batch.proprioception = torch.nn.functional.pad(
                state,
                (0, max_state_dim - state.shape[-1]),
            )
            batch.validate()
        return batch

    def reset(self) -> None:
        """Clear episode-scoped VLM anchor history."""

        self._vlm_anchor_history.clear()

    def action(self, output: WorldActionOutput) -> np.ndarray:
        action = output.require_action().detach().to(device="cpu", dtype=torch.float32)
        if (
            action.ndim != 3
            or action.shape[0] != 1
            or action.shape[1] != self.action_horizon
        ):
            raise ValueError(
                "RoboTwin2 policy output must have shape "
                f"[1,{self.action_horizon},D], got "
                f"{tuple(action.shape)}"
            )
        if self.checkpoint_transform is not None:
            action = self.checkpoint_transform.unapply({"action": action[0]})[
                "action.vector"
            ]
            result = np.asarray(action.numpy(), dtype=np.float32)
        else:
            if action.shape[-1] != 14:
                raise ValueError(
                    "RoboTwin2 action must have width 14 without a checkpoint transform"
                )
            result = np.asarray(action[0].numpy(), dtype=np.float32)
        if result.shape != (self.action_horizon, 14):
            raise ValueError(
                "RoboTwin2 absolute qpos chunk must be "
                f"[{self.action_horizon},14], got {result.shape}"
            )
        if self._action_transform is not None:
            result = np.asarray(self._action_transform(result))
        return result

__all__ = ["RoboTwin2Adapter"]
