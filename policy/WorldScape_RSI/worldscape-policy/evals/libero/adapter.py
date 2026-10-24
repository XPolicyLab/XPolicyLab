from collections.abc import Callable

import numpy as np

from evals.common.simulator import ObservationMapping, SimulatorAdapter


class LiberoAdapter(SimulatorAdapter):
    """Map LIBERO observations and actions to the WorldScape API."""

    def __init__(
        self,
        *,
        camera_keys: tuple[str, ...] = (
            "agentview_image",
            "robot0_eye_in_hand_image",
        ),
        state_keys: tuple[str, ...] = (
            "robot0_joint_pos",
            "robot0_gripper_qpos",
        ),
        head_camera_key: str | None = None,
        embodiment_id: int = 0,
        action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            ObservationMapping(
                camera_keys=camera_keys,
                state_keys=state_keys,
                head_camera_key=head_camera_key,
                embodiment_id=embodiment_id,
            ),
            action_transform=action_transform,
        )


__all__ = ["LiberoAdapter"]
