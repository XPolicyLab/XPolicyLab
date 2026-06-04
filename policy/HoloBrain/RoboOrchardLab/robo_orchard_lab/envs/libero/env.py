# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robo_orchard_core.datatypes import (
    BatchFrameTransform,
    BatchTransform3D,
)
from robo_orchard_core.envs.task import TaskInfo
from robo_orchard_core.kinematics.chain import KinematicChain
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.math import (
    matrix_to_quaternion,
    pose_diff,
    quaternion_to_axis_angle,
)
from robosuite import macros
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.environments.base import MujocoEnv
from robosuite.utils.binding_utils import MjSim
from typing_extensions import Literal

from robo_orchard_lab.envs.base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_lab.envs.libero.obs import (
    get_camera_data,
    get_joints,
    get_robot_tf,
)

if TYPE_CHECKING:
    from robosuite.robots.robot import Robot
    from robosuite.robots.single_arm import SingleArm

macros.IMAGE_CONVENTION = "opencv"

LiberoObsType: TypeAlias = dict[str, Any]
LiberoRewardType: TypeAlias = float | np.ndarray
LiberoSuiteName: TypeAlias = Literal[
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]

libero_suite_names = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)

ArrayLike: TypeAlias = np.ndarray | list[float] | torch.Tensor

__all__ = [
    "get_libero_task",
    "LiberoEnvStepReturn",
    "LiberoEnv",
    "LiberoEnvCfg",
    "LiberoEvalEnv",
    "LiberoEvalEnvCfg",
    "LiberoSuiteName",
]


@dataclass
class LiberoEnvStepReturn(EnvStepReturn[LiberoObsType, LiberoRewardType]):
    observations: LiberoObsType
    terminated: bool
    """done flag"""
    rewards: LiberoRewardType
    """The rewards is a [0,1] value indicating whether the task was successful.
    """
    truncated: bool | None
    """For evaluation environments, whether the episode was truncated due to
    reaching the step limit. Otherwise always None."""

    info: dict[str, Any]


@dataclass
class ActionInfo:
    """Information about the last action taken in the environment."""

    goal_eef: BatchFrameTransform
    """The goal end-effector pose after taking the last action."""
    osc_arm_action: torch.Tensor
    """The OSC arm action taken."""
    osc_gripper_action: torch.Tensor
    """The OSC gripper action taken."""


@dataclass
class LiberoTask:
    suite: benchmark.Benchmark
    task: benchmark.Task
    task_bddl_file: str
    hdf5_path: str


def get_libero_task(suite_name: str, task_id: int) -> LiberoTask:
    """Get the Libero task information.

    Args:
        suite_name (str): The name of the Libero task suite.
        task_id (int): The ID of the task within the suite.

    Returns:
        LiberoTask: The Libero task information.
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite: benchmark.Benchmark = benchmark_dict[suite_name]()
    task: benchmark.Task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    hdf5_path = os.path.join(
        get_libero_path("datasets"),
        task_bddl_file.split("bddl_files/")[-1].replace(".bddl", "_demo.hdf5"),
    )
    return LiberoTask(
        suite=task_suite,
        task=task,
        task_bddl_file=task_bddl_file,
        hdf5_path=hdf5_path,
    )


class LiberoEnv(EnvBase):
    cfg: LiberoEnvCfg

    def __init__(self, cfg: LiberoEnvCfg) -> None:
        self.cfg = cfg

        libero_task = get_libero_task(cfg.suite_name, cfg.task_id)

        self._hdf5_path = libero_task.hdf5_path
        self._task_suite = libero_task.suite
        self._task = libero_task.task
        self._task_bddl_file = libero_task.task_bddl_file
        self._last_action: np.ndarray | None = None
        self._last_obs: dict | None = None

        env = OffScreenRenderEnv(
            bddl_file_name=libero_task.task_bddl_file,
            camera_heights=cfg.camera_heights,
            camera_widths=cfg.camera_widths,
            camera_depths=cfg.camera_depths,
        )
        self._env = env

        self._check_env_config()
        self._control_dim: int = env.robots[0].controller.control_dim  # type: ignore # noqa
        self._robot_chain = KinematicChain.from_content(
            data=self.get_robot_xml(), format="mjcf", dtype=torch.float64
        )

    def _check_env_config(self):
        robots: list[Robot] = self._env.robots
        assert len(robots) == 1, (
            "LiberoEnv currently only supports single-arm robots. "
            f"Got {len(robots)} robots."
        )
        robot: SingleArm = robots[0]  # type: ignore
        controller = robot.controller
        if not isinstance(controller, OperationalSpaceController):
            raise ValueError(
                "get_arm_action currently only supports "
                "OperationalSpaceController."
            )
        assert controller.impedance_mode == "fixed", (
            "Only fixed impedance mode is supported."
        )
        assert controller.use_delta, "Only delta OSC is supported."

        assert controller.position_limits is None, (
            "LiberoEnv currently only supports unlimited position limits."
        )
        assert controller.orientation_limits is None, (
            "LiberoEnv currently only supports unlimited orientation limits."
        )
        assert controller.control_dim == 6, (
            "LiberoEnv currently only supports 6-DoF OSC control."
        )

    def step(self, action: ArrayLike) -> LiberoEnvStepReturn:
        """Take a step in the environment.

        Libero uses OSC (operational space control) actions, which are
        7-dimensional vectors representing the desired end-effector
        movements in the x, y, z directions and rotations around those axes,
        as well as the gripper open/close command (-1 = open, +1 = close).

        See :class:`robosuite.controllers.osc.OperationalSpaceController`
        and :class:`robosuite.manipulator.Manipulator` for more details.

        This is the default action type for LiberoEnv, but users can also
        choose to use the target end-effector pose as action by setting
        `use_action_type` to "orchard_osc_target_eef" in the config.
        In that case, the action should be a 7-dimensional vector representing
        the target end-effector pose (x, y, z, q_w, q_x, q_y, q_z) in world
        frame, followed by gripper command.

        Note: Different from step function in libero env, we force update
        the observations after each step to ensure that the observations
        are always up-to-date with the simulation state, as we observe that
        the observation from step is delayed. The value is very close to
        the one before the fix, but this ensures consistency of LiberoEnv,
        because some method of LiberoEnv directly access the simulation state.

        """
        action = self._convert_action_if_needed(action)
        self._last_action = action

        _, reward, done, info = self._env.step(action)
        return LiberoEnvStepReturn(
            observations=self._get_obs(force_update=True),
            rewards=reward,
            terminated=bool(done),
            truncated=None,
            info=info,
        )

    def _convert_action_if_needed(self, action: ArrayLike) -> np.ndarray:
        """Convert the action to numpy array if needed."""
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        if isinstance(action, list):
            action = np.array(action)

        if self.cfg.use_action_type == "libero":
            assert isinstance(action, np.ndarray), (
                "Action must be a numpy array."
            )
            return action

        assert self.cfg.use_action_type == "orchard_osc_target_eef", (
            "Only 'libero' and 'orchard_osc_target_eef' action "
            "types are supported."
        )

        # Convert the target pose to the OSC delta action expected by Libero.
        if action.shape[-1] != 8:
            raise ValueError(
                "For 'orchard_osc_target_eef' action type, "
                "the action must be a 8-dimensional vector: "
                "[x, y, z, q_w, q_x, q_y, q_z , gripper_command]. "
                f"Got action with shape {action.shape}."
            )
        source_pose = self._last_obs["tf_world"][self.eef_name]  # type: ignore

        target_pose = BatchTransform3D(
            xyz=torch.from_numpy(action[:3]).reshape(1, 3),
            quat=torch.from_numpy(action[3:7]).reshape(1, 4),
        )
        gripper_action = action[7:]
        arm_action_torch = self.get_arm_osc_delta_pose_from_target_pose(
            target_pose=target_pose, source_pose=source_pose
        )
        arm_action = arm_action_torch[0].numpy()

        return np.concatenate([arm_action, gripper_action], axis=0)

    def get_sim_state(self) -> np.ndarray:
        """Get the current simulator state as a flattened tensor."""
        sim: MjSim = self._env.sim
        return sim.get_state().flatten().copy()

    def _get_obs(self, force_update=True):
        """Get the current observation from the environment.

        Args:
            force_update (bool, optional): Whether to force update the
                observations from the simulator. Defaults to True.

        """
        # force update observations and controller states
        obs = self._env.env._get_observations(force_update=force_update)
        if self.cfg.format_datatypes:
            obs = self._format_observation(obs)
        self._last_obs = obs
        return obs

    def _format_observation(self, obs: LiberoObsType) -> LiberoObsType:
        """Format the observation dict to use robo_orchard datatypes.

        The following keys are reformatted:
        - Camera data: replace with any rgb/depth camera keys containing
            BatchCameraData.
        - Robot joint data: replaced with 'joints' key containing
          BatchJointsState.
        - Robot frame transforms: replaced with 'tf_world' key containing
          BatchFrameTransform.

        """
        formatted_obs: LiberoObsType = obs.copy()

        # format camera data, and replace key.
        camera_data = get_camera_data(obs, self._env)
        formatted_obs.update(camera_data)

        # format robot joint data: remove old keys and add 'joints' key
        # 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin',
        # 'robot0_joint_vel', 'robot0_gripper_qpos', 'robot0_gripper_qvel'

        joint_keys = [
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ]
        for key in joint_keys:
            formatted_obs.pop(key, None)
        formatted_obs["joints"] = get_joints(self._env)

        # format robot frame transforms: remove eef keys and add tf_world
        # 'robot0_eef_pos', 'robot0_eef_quat',
        for key in ["robot0_eef_pos", "robot0_eef_quat"]:
            formatted_obs.pop(key, None)
        formatted_obs["tf_world"] = get_robot_tf(self._robot_chain, self._env)

        return formatted_obs

    def reset(
        self, seed: int | None = None, init_state: Any | None = None
    ) -> tuple[LiberoObsType, dict]:
        """Reset the environment.

        Arguments:
            seed (int | None, optional): Seed for the environment's random
                number generator. Defaults to None.
            init_state (Any | None, optional): Initial state to set after
                reset. Defaults to None.

        """
        if seed is not None:
            self._env.seed(seed)
        _ = self._env.reset()
        if init_state is not None:
            _ = self._env.set_init_state(init_state)
        else:
            self._env.env.sim.forward()
            self._env.check_success()
            self._env._post_process()
        self._last_action = None
        return self._get_obs(force_update=True), {}

    @property
    def num_envs(self) -> int:
        # always 1 for simplicity.
        return 1

    @property
    def task_info(self) -> TaskInfo:
        """Get the task information."""
        return TaskInfo(
            description=self._task.language,
            goal_condition=None,
            instructions=self._task.language,
        )

    def unwrapped_env(self) -> OffScreenRenderEnv:
        """Get the original Libero environment."""
        return self._env.env

    def get_robot_xml(self) -> str:
        """Get the robot XML used in the environment.

        Currently, Libero environments only support single-arm robots.

        Returns:
            str: robot XML strings.
        """
        robots: list[Robot] = self._env.robots
        robot: Robot = robots[0]
        return robot.robot_model.get_xml()  # type: ignore # noqa

    @property
    def eef_name(self) -> str:
        """Get the end-effector name of the robot.

        Currently, Libero environments only support single-arm robots.

        Note:
            Libero uses the end-effector name defined in the robot controller,
            while the robot model may have a different end-effector name.
            This is a BUG in Libero, which we work around here by
            accessing the controller's end-effector name directly, as this
            is what Libero actually uses in the environment!

        Returns:
            str: The end-effector name of the robot.
        """
        robots: list[Robot] = self._env.robots
        robot: Robot = robots[0]
        return robot.controller.eef_name  # type: ignore

    def get_arm_osc_delta_pose_from_target_pose(
        self, target_pose: BatchTransform3D, source_pose: BatchTransform3D
    ) -> torch.Tensor:
        """Convert a target end-effector pose to an OSC delta action.

        This method requires that the input target_pose and source_pose
        have the same batch size, and the output OSC action will have
        the same batch size as well.
        """
        if target_pose.batch_size != source_pose.batch_size:
            raise ValueError(
                "Batch size of target_pose and source_pose must be the same."
            )
        diff_p, diff_q = pose_diff(
            ta=target_pose.xyz,
            qa=target_pose.quat,
            tb=source_pose.xyz,
            qb=source_pose.quat,
        )
        diff_rot_v = quaternion_to_axis_angle(diff_q)
        delta_pose = torch.cat([diff_p, diff_rot_v], dim=-1)
        delta_pose_np = delta_pose.numpy()
        # unscale the delta pose to osc action
        robot: SingleArm = self._env.robots[0]  # type: ignore
        controller: OperationalSpaceController = robot.controller  # type: ignore
        osc_action = (
            delta_pose_np - controller.action_output_transform
        ) / controller.action_scale + controller.action_input_transform
        return torch.from_numpy(osc_action)

    def get_last_action(self) -> ActionInfo | None:
        """Get information about the last action taken in the environment."""
        if self._last_action is None:
            return None
        robots: list[Robot] = self._env.robots
        robot: SingleArm = robots[0]  # type: ignore
        if not isinstance(robot.controller, OperationalSpaceController):
            raise ValueError(
                "get_arm_action currently only supports "
                "OperationalSpaceController."
            )

        arm_goal_pos = robot.controller.goal_pos.copy()
        arm_goal_ori_mat = robot.controller.goal_ori.copy()
        arm_dim = robot.controller.control_dim
        osc_gripper_action = self._last_action[arm_dim:]
        osc_arm_action = self._last_action[:arm_dim]
        eef_frame = self.eef_name
        goal_pose = BatchFrameTransform(
            xyz=torch.from_numpy(arm_goal_pos).unsqueeze(0),
            quat=matrix_to_quaternion(
                torch.from_numpy(arm_goal_ori_mat).unsqueeze(0),
                normalize_output=True,
            ),
            child_frame_id=eef_frame,
            parent_frame_id="world",
        )

        return ActionInfo(
            goal_eef=goal_pose,
            osc_arm_action=torch.from_numpy(osc_arm_action).unsqueeze(0),
            osc_gripper_action=torch.from_numpy(osc_gripper_action).unsqueeze(
                0
            ),
        )

    def close(self) -> None:
        self._env.close()

    @property
    def control_timestep(self) -> float:
        """Get the average time interval between steps in the environment.

        In Libero, this value is around 1/20 seconds, corresponding to
        the control frequency of 20 Hz.

        Returns:
            float: The time interval between steps in seconds.
        """
        env: MujocoEnv = self._env.env  # type: ignore
        return env.control_timestep  # type: ignore

    def __del__(self):
        try:
            del self._env
        except Exception:
            pass


class LiberoEvalEnv(LiberoEnv):
    """Libero evaluation environment.

    The evaluation environment is used for benchmarking the performance
    of agents on the Libero tasks. The environment is configured to use a
    fixed set of initial states for each task, and the episode length is
    limited to a predefined number of steps based on the task suite.
    """

    cfg: LiberoEvalEnvCfg

    step_limits: dict[str, int] = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    """Step limits for different Libero task suites.

    The value is copied from OpenVLA's Libero evaluation script.
    """

    def __init__(self, cfg: LiberoEvalEnvCfg) -> None:
        super().__init__(cfg)
        self._total_steps = 0

    @property
    def step_limit(self) -> int:
        return self.step_limits[self.cfg.suite_name]

    def step(self, action: ArrayLike) -> LiberoEnvStepReturn:
        ret = super().step(action)
        self._total_steps += 1

        ret.truncated = self._total_steps >= self.step_limit
        return ret

    def reset(self, seed: int | None = None) -> tuple[LiberoObsType, dict]:
        # for benchmarking purpose, we fix the a set of initial states
        init_states = self._task_suite.get_task_init_states(self.cfg.task_id)
        init_state_id = self.cfg.episode_idx
        super().reset(seed=seed, init_state=init_states[init_state_id])
        # step a few times to let the env stabilize
        for _ in range(self.cfg.num_steps_wait):
            ret = self.step(_generate_dummy_action())
        self._total_steps = 0
        self._last_action = None
        return ret.observations, ret.info


LiberoEnvType = TypeVar("LiberoEnvType", bound=LiberoEnv)


class LiberoEnvCfg(EnvBaseCfg[LiberoEnvType]):
    """Configuration for the RoboTwin environment."""

    class_type: ClassType[LiberoEnvType] = LiberoEnv  # type: ignore

    suite_name: LiberoSuiteName
    task_id: int

    camera_heights: int = 256
    camera_widths: int = 256
    camera_depths: bool = False

    format_datatypes: bool = False
    """whether to format observation as robo_orchard datatypes.

    If True, the observation dict will be reformatted in the
    following way:
    - Camera data: replace with any rgb/depth camera keys containing
    BatchCameraData.
    - Robot joint data: replaced with 'joints' key containing
    BatchJointsState.
    - Robot frame transforms: replaced with 'tf_world' key containing
    BatchFrameTransform.

    """

    use_action_type: Literal["libero", "orchard_osc_target_eef"] = "libero"
    """The type of action to use in the environment:
    - "libero": use the original Libero action type defined in the robot
    controller.
    - "orchard_osc_target_eef": use the target end-effector pose as the action,
    which will be converted to OSC delta pose internally in the environment.

    """

    def __post_init__(self):
        if self.use_action_type == "orchard_osc_target_eef":
            if self.format_datatypes is False:
                raise ValueError(
                    "format_datatypes must be True when using "
                    "orchard_osc_target_eef action type, because the target "
                    "eef pose in the action info requires the observation to "
                    "be formatted with BatchFrameTransform."
                )


class LiberoEvalEnvCfg(LiberoEnvCfg[LiberoEvalEnv]):
    """Configuration for the RoboTwin evaluation environment."""

    class_type: ClassType[LiberoEvalEnv] = LiberoEvalEnv

    episode_idx: int
    """Index of the evaluation episode.

    This is used to select the initial state from the predefined set
    of initial states for evaluation.
    """

    num_steps_wait: int = 15
    """Number of steps to wait after setting the initial state."""


def _generate_dummy_action() -> list[float]:
    """Generate a dummy action for testing purposes.

    Libero uses a 7-dimensional action space, where the first 6 dimensions
    correspond to delta movements in the x, y, z directions and rotations
    around those axes, and the 7th dimension controls the gripper (open/close).
    The gripper action is represented as -1 for open and +1 for close.

    We return zeros for the first 6 dimensions (no movement) and -1 for
    the gripper to indicate that the gripper should remain open.

    This method is used when initializing the environment to ensure that
    the initial state of the environment is stable.

    Returns:
        list[float]: A list of zeros representing the dummy action.
    """
    # (-1 = open, +1 = close)
    return [0, 0, 0, 0, 0, 0, -1]
