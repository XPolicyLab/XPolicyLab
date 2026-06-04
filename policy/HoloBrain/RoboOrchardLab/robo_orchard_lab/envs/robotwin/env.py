# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
import copy
import functools
import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias, cast

import gymnasium as gym
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.logging import LoggerManager
from typing_extensions import Literal

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Robot,
    RobotDescriptionFormat,
)
from robo_orchard_lab.envs.base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_lab.envs.robotwin.kinematics import (
    RoboTwinEEF,
    RoboTwinJointsToEEF,
)
from robo_orchard_lab.envs.robotwin.obs import (
    get_joints,
    get_observation_cams,
)
from robo_orchard_lab.envs.sapien import sapien_pose_to_orchard
from robo_orchard_lab.envs.state import EnvStateScope, StatefulEnvMixin
from robo_orchard_lab.utils.state import (
    State,
    validate_recovery_state,
)
from robo_orchard_lab.utils.video import (
    VideoBackendUnavailableError,
    VideoPixelFormat,
    VideoWriter,
    VideoWriterError,
)

if TYPE_CHECKING:
    from envs._base_task import (  # pyright: ignore[reportMissingImports]
        Base_Task,
    )

EVAL_SEED_BASE = 100000
EVAL_INSTRUCTION_NUM = 100
_logger_manager = LoggerManager()
_logger_manager_logger = _logger_manager.get_logger()
if _logger_manager_logger.handlers:
    _logger_manager_logger.propagate = False
logger = _logger_manager.get_child(__name__)

InstructionType: TypeAlias = Literal["seen", "unseen"]
RoboTwinObsType: TypeAlias = dict[str, Any] | None
__all__ = ["RoboTwinEnvStepReturn", "RoboTwinEnv", "RoboTwinEnvCfg"]

LEFT_EEF_FROM_JOINT_FRAME_ID = "left_eef_from_joint"
RIGHT_EEF_FROM_JOINT_FRAME_ID = "right_eef_from_joint"
COMBINED_DUAL_ARM_OBS_ROBOT_KEY = "left"
ROBOTWIN_VIDEO_FPS = 10
ROBOTWIN_VIDEO_PIXEL_FORMAT = VideoPixelFormat.RGB24
ROBOTWIN_ENV_STATE_SCHEMA_VERSION = 1


@dataclass
class RoboTwinEnvStepReturn(EnvStepReturn[RoboTwinObsType, bool]):
    observations: RoboTwinObsType
    terminated: bool
    rewards: bool
    """The rewards is a boolean indicating whether the task was successful."""
    truncated: bool
    """Whether the episode was truncated due to reaching the step limit."""


# State.config owns task_name, start seed, and episode_id; this payload keeps
# only post-reset runtime values that cannot be derived from config.
class _RoboTwinPostResetStatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    scope: Literal[EnvStateScope.POST_RESET]
    offset_seed: int = Field(ge=0)
    task_config: dict[str, Any]
    instructions: Any = None
    eval_chosen_instruction: str | None = None
    post_reset_state_available: bool = True
    episode_finalized: bool = False


class RoboTwinEnv(EnvBase[RoboTwinObsType, bool], StatefulEnvMixin):
    """RoboTwin environment wrapped with the orchard env interface.

    This class adapts RoboTwin tasks to the ``robo_orchard_core`` env API.
    To use it, RoboTwin must be installed and the ``RoboTwin_PATH``
    environment variable must point to the RoboTwin package. The current
    wrapper supports only RoboTwin's combined dual-arm layout with one shared
    robot base.

    Public interface:

    - ``reset(...)``: create or recreate the RoboTwin task and return the
      initial observation.
    - ``step(action)``: execute one RoboTwin action and return
      ``RoboTwinEnvStepReturn``.
    - ``close(...)``: close the current RoboTwin task.
    - ``finalize_episode()``: finalize episode-local artifacts without
      closing the reusable RoboTwin runtime.
    - ``unwrapped_env()``: return the underlying RoboTwin ``Base_Task``.
    - ``get_robot_urdf()``: return the supported combined dual-arm URDF.
      RoboTwin stores this dual-arm descriptor under the ``"left"`` key by
      convention; that is a RoboTwin compatibility contract, not an env bug.
    - ``get_obs_robots()``: return observation-facing robot metadata.
    - ``current_seed`` / ``instructions`` / ``num_envs``: runtime properties
      exposed by the wrapper.

    Typical usage::

        env = RoboTwinEnv(
            RoboTwinEnvCfg(
                task_name="place_object_basket",
                check_expert=False,
                check_task_init=False,
                action_type="qpos",
            )
        )
        obs, info = env.reset()
        action = np.zeros(14, dtype=np.float32)
        step_ret = env.step(action)
        env.close()

    The example above uses ``action_type="qpos"``. See ``step()`` for the
    exact action layout for both ``"qpos"`` and ``"ee"`` modes.
    """

    supported_state_scopes = frozenset({EnvStateScope.POST_RESET})

    def __init__(self, cfg: RoboTwinEnvCfg):
        self.cfg = cfg
        self._task = cast("Base_Task", None)
        self._resolved_start_seed = self.cfg.resolve_start_seed(self.cfg.seed)
        self._offset_seed = 0
        self._instructions: Any = None
        self._eval_chosen_instruction: str | None = None
        self._episode_finalized = True
        self._post_reset_state_available = False
        self._joints_to_eef_transform: RoboTwinJointsToEEF | None = None
        self._cached_obs_robots: dict[str, Robot] | None = None
        self._video_writer = VideoWriter(
            pixel_format=ROBOTWIN_VIDEO_PIXEL_FORMAT,
            fps=ROBOTWIN_VIDEO_FPS,
        )

    def _check_and_update_seed(self):
        instructions = None
        task = None
        from description.utils.generate_episode_instructions import (  # pyright: ignore[reportMissingImports]
            generate_episode_descriptions,
        )

        if self.cfg.check_expert:
            logger.debug(
                "Checking RoboTwin expert trajectory: task=%s seed=%s",
                self.cfg.task_name,
                self.current_seed,
            )
            requested_seed = self.current_seed
            task, success = self._check_expert_traj()
            retry_num = 0
            while not success:
                retry_num += 1
                if retry_num >= 50:
                    raise RuntimeError(
                        f"Failed to create task {self.cfg.task_name} "
                        f"with expert trajectory after {retry_num} retries. "
                        "Please check the task configuration!"
                    )

                failed_seed = self.current_seed
                self._offset_seed += 1
                logger.debug(
                    "RoboTwin expert trajectory check failed: "
                    "task=%s seed=%s retry_seed=%s",
                    self.cfg.task_name,
                    failed_seed,
                    self.current_seed,
                )
                task, success = self._check_expert_traj()
            if retry_num > 0:
                logger.info(
                    "RoboTwin expert trajectory resolved after retry: "
                    "task=%s requested_seed=%s actual_seed=%s retries=%s",
                    self.cfg.task_name,
                    requested_seed,
                    self.current_seed,
                    retry_num,
                )

            assert task is not None
            instructions = generate_episode_descriptions(
                self.cfg.task_name,
                [task.info["info"]],
                max_descriptions=self.cfg.max_instruction_num,
            )[0]
        else:
            if self.cfg.check_task_init:
                logger.debug(
                    "Checking RoboTwin task init: task=%s seed=%s",
                    self.cfg.task_name,
                    self.current_seed,
                )
                task, success = self._check_expert_traj()
                if task is None:
                    raise RuntimeError(
                        f"Failed to create task {self.cfg.task_name} "
                        f"with seed {self.cfg.seed}. Please try a different "
                        "seed or check the task configuration."
                    )
                instructions = generate_episode_descriptions(
                    self.cfg.task_name,
                    [task.info["info"]],
                    max_descriptions=self.cfg.max_instruction_num,
                )[0]
            else:
                task = self._create_task()
                instructions = None

        return task, instructions

    @property
    def current_seed(self) -> int:
        """The actual RoboTwin runtime seed for the current episode."""
        return self._resolved_start_seed + self._offset_seed

    @property
    def start_seed(self) -> int:
        """The caller-facing start seed configured on the env."""
        return self.cfg.seed

    @property
    def resolved_start_seed(self) -> int:
        """The eval-mode-normalized runtime start seed."""
        return self._resolved_start_seed

    @property
    def offset_seed(self) -> int:
        """The env-local retry offset from ``resolved_start_seed``."""
        return self._offset_seed

    @property
    def instructions(self) -> dict | None | str:
        """The instructions for the environment.

        This property is only valid if the environment is initialized
        with `check_expert=True` or `check_task_init=True`.

        If in eval_mode, return the instruction from the task, usually
        a string, otherwise the returned instruction is a dictionary
        containing multiple instructions, with maximum number specified
        by `max_instruction_num`.

        """
        if self.cfg.eval_mode:
            if self._eval_chosen_instruction is None:
                assert self._instructions is not None
                # random pick one in unseen instructions
                eval_instruction_type: InstructionType = "unseen"
                self._eval_chosen_instruction = np.random.choice(
                    self._instructions[eval_instruction_type]
                )

            return self._eval_chosen_instruction

        else:
            return self._instructions

    def _get_state(self) -> State:
        if not self._post_reset_state_available:
            raise RuntimeError(
                "RoboTwinEnv state is only available after reset() and "
                "before the first step()."
            )
        if self._task is None:
            raise RuntimeError("RoboTwinEnv has no active task to capture.")

        state_payload: dict[str, object] = _RoboTwinPostResetStatePayload(
            schema_version=ROBOTWIN_ENV_STATE_SCHEMA_VERSION,
            scope=EnvStateScope.POST_RESET,
            offset_seed=self._offset_seed,
            task_config=copy.deepcopy(
                self.cfg.get_task_config_for_seed(self.current_seed)
            ),
            instructions=copy.deepcopy(self._instructions),
            eval_chosen_instruction=copy.deepcopy(
                self._eval_chosen_instruction
            ),
            post_reset_state_available=self._post_reset_state_available,
            episode_finalized=self._episode_finalized,
        ).model_dump(mode="json")
        return State(
            class_type=type(self),
            config=copy.deepcopy(self.cfg),
            state=state_payload,
            hierarchical_save=None,
        )

    def _set_state(self, state: State) -> None:
        self._restore_post_reset_state(state)

    def reset_from_state(self, state: State) -> tuple[RoboTwinObsType, dict]:
        """Restore a post-reset State and return ``reset(...)`` output."""

        payload = self._restore_post_reset_state(state, activate=False)
        obs = self._format_obs(self._task.get_obs())
        info = self._get_info()
        self._post_reset_state_available = payload.post_reset_state_available
        self._episode_finalized = payload.episode_finalized
        return obs, info

    def _restore_post_reset_state(
        self,
        state: State,
        *,
        activate: bool = True,
    ) -> _RoboTwinPostResetStatePayload:
        validate_recovery_state(
            state,
            require_class_type=True,
            require_config=True,
            context="RoboTwinEnv state",
        )
        state_class_type = state.class_type
        if state_class_type is not type(self):
            raise TypeError(
                "RoboTwinEnv state class_type must match the target env. "
                f"Got {state_class_type} for {type(self).__name__}."
            )
        if not isinstance(state.config, RoboTwinEnvCfg):
            raise TypeError(
                "RoboTwinEnv state config must be RoboTwinEnvCfg. "
                f"Got {type(state.config).__name__}."
            )

        payload = _RoboTwinPostResetStatePayload.model_validate(
            state.state
        ).model_copy(deep=True)
        cfg = copy.deepcopy(state.config)

        task: Base_Task | None = None
        try:
            with in_robotwin_workspace():
                task = create_task_from_name(cfg.task_name)
                task.setup_demo(**payload.task_config)  # type: ignore
            if task is None:
                raise RuntimeError("RoboTwin task creation returned None.")
            self._assert_supported_robot_layout(task)
        except Exception:
            if task is not None:
                try:
                    task.close_env(clear_cache=True)
                except Exception:
                    logger.exception(
                        "Failed to close staged RoboTwin task after State "
                        "restore validation failed."
                    )
            raise

        self.close(clear_cache=True)
        self.cfg = cfg
        self._resolved_start_seed = self.cfg.resolve_start_seed(self.cfg.seed)
        self._offset_seed = payload.offset_seed
        self._instructions = payload.instructions
        self._eval_chosen_instruction = payload.eval_chosen_instruction
        self._task = task
        self._joints_to_eef_transform = None
        self._cached_obs_robots = None
        if activate:
            self._post_reset_state_available = (
                payload.post_reset_state_available
            )
            self._episode_finalized = payload.episode_finalized
        else:
            self._post_reset_state_available = False
            self._episode_finalized = True
        return payload

    def _create_task(self) -> Base_Task:
        with in_robotwin_workspace():
            task = create_task_from_name(self.cfg.task_name)
            task_config = self.cfg.get_task_config_for_seed(
                runtime_seed=self.current_seed
            )
            task.setup_demo(**task_config)  # type: ignore
            return task

    def _check_expert_traj(self) -> tuple[Base_Task | None, bool]:
        """Check whether current config can success if using expert trajectory.

        Returns:
            tuple[Base_Task | None, bool]: A tuple containing the task and a
                boolean indicating whether the task was successful.

        """
        with in_robotwin_workspace():
            task = create_task_from_name(self.cfg.task_name)
            config = self.cfg.get_task_config_for_seed(
                runtime_seed=self.current_seed
            )
            config["render_freq"] = 0
            try:
                task.setup_demo(**config)  # type: ignore
                task.play_once()  # type: ignore
            except Exception as e:
                logger.debug(
                    "RoboTwin expert trajectory check failed while playing "
                    "task config: task=%s seed=%s error=%s",
                    self.cfg.task_name,
                    self.current_seed,
                    e,
                )
                return task, False
            finally:
                task.close_env()

        success: bool = task.plan_success and task.check_success()  # type: ignore
        return task, success

    def step(self, action: list[float] | np.ndarray) -> RoboTwinEnvStepReturn:
        """Take a step in the environment.

        Args:
            action (list[float] | np.ndarray): The action to take in the
                environment. The exact semantics depend on
                `self.cfg.action_type`.

                - If `self.cfg.action_type == "qpos"`, the action must be a
                  1-D sequence in RoboTwin joint-control order:
                  `[left_arm_joint_targets..., left_gripper,
                  right_arm_joint_targets..., right_gripper]`.
                  The arm-joint counts come from the current RoboTwin robot
                  embodiment, so the expected total length is
                  `len(left_arm_joints_name) + 1 +
                  len(right_arm_joints_name) + 1`. The two gripper values use
                  RoboTwin's normalized gripper convention.

                - If `self.cfg.action_type == "ee"`, the action must be a
                  1-D sequence in RoboTwin end-effector-control order:
                  `[left_xyz(3), left_quat(4), left_gripper,
                  right_xyz(3), right_quat(4), right_gripper]`,
                  where each quaternion follows RoboTwin's
                  `[qw, qx, qy, qz]` convention. For the currently supported
                  combined dual-arm layout this means 16 values in total.

                The wrapper validates this expected width exactly before
                forwarding the action to RoboTwin.

        Returns:
            RoboTwinEnvStepReturn: The step result after taking the action.
                This function always returns a step result. Episode end is
                reported via `terminated` and `truncated` instead of
                returning None. `rewards` is a boolean indicating whether
                the task has succeeded.

        Raises:
            RuntimeError: If no active episode is available for stepping.
                This includes newly constructed, closed, and finalized env
                states. Call ``reset()`` or restore a non-finalized ``State``
                with ``reset_from_state()`` before stepping again.
        """
        if self._episode_finalized:
            raise RuntimeError(
                "RoboTwinEnv has no active episode. "
                "Call reset() or reset_from_state() before step()."
            )

        action_array = np.asarray(action)
        if action_array.ndim != 1:
            raise ValueError(
                "Action should be a 1-D array, "
                f"but got {action_array.ndim} dimensions."
            )

        if self.cfg.action_type == "qpos":
            expected_action_dim = len(
                self._task.robot.get_left_arm_jointState()
            ) + len(self._task.robot.get_right_arm_jointState())
        elif self.cfg.action_type == "ee":
            expected_action_dim = 16
        else:
            raise ValueError(
                f"Unsupported RoboTwin action_type: {self.cfg.action_type!r}."
            )

        # RoboTwin silently slices extra dimensions for qpos actions, so the
        # wrapper validates the exact width before forwarding the command.
        if action_array.shape[0] != expected_action_dim:
            raise ValueError(
                "Action width does not match RoboTwin action_type "
                f"{self.cfg.action_type!r}: expected {expected_action_dim}, "
                f"got {action_array.shape[0]}."
            )
        # the take_action method will do internal check if reach step limit
        # or task is successful. Either case, the task will not take further
        # actions.
        self._task.take_action(
            action_array,
            action_type=self.cfg.action_type,
        )
        self._post_reset_state_available = False

        # when reach step limit, truncated is True
        # Note that step_lim is None for default unlimited steps.
        # It will be set in evaluation mode.
        if (
            self._task.step_lim is not None
            and self._task.take_action_cnt >= self._task.step_lim
        ):
            truncated = True
        else:
            truncated = False

        # robotwin env does not have a concept of done.
        # when a task is evaluated as success, the task does not
        # take further actions anymore. We consider the episode
        # is done when the task is successful.
        if self._task.eval_success:
            terminated = True
        else:
            terminated = False

        raw_obs = self._task.get_obs()
        self._write_video_frame(raw_obs)

        return RoboTwinEnvStepReturn(
            observations=self._format_obs(raw_obs),
            rewards=self._task.eval_success,
            terminated=terminated,
            truncated=truncated,
            info=self._get_info(),
        )

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        seed: int | None = None,
        offset_seed: int | None = None,
        task_name: str | None = None,
        clear_cache: bool = False,
        return_obs: bool = True,
        video_dir: str | None = None,
        episode_id: int | None = None,
    ) -> tuple[RoboTwinObsType, dict]:
        """Reset the environment.

        If the environment has not been reset before, or the seed is
        different from the previous one, or the task_name is different
        from the previous one, the environment will be re-created
        and check the seed. The config ``seed`` remains the caller-facing
        start seed while the env tracks runtime retries through
        ``offset_seed``.

        Warning:
            RoboTwin does not use local RandomGenerator, when the environment
            is re-created, the seed will be set to the one in the config
            for both numpy and torch. This may affect the randomness of other
            parts of the code!
            This is a BUG in RoboTwin!

        Args:
            env_ids (Sequence[int] | None, optional): Not supported.
                Defaults to None.
            seed (int | None, optional): The seed to reset the
                environment start point. If None, the seed in the config will
                be used. If an int is provided, it replaces the caller-facing
                start seed. Default is None.
            offset_seed (int | None, optional): Runtime offset from the
                resolved start seed. If None, the existing env offset is
                reused unless ``seed`` also changes, in which case the offset
                resets to 0. Default is None.
            task_name (str | None, optional): The task name to reset the
                environment. If None, the task name in the config will be used.
                Default is None.
            clear_cache (bool, optional): Whether to clear the cache
                when closing the environment. Default is False.
            return_obs (bool, optional): Whether to format and return the
                initial observation. Default is True.
            video_dir (str | None, optional): Directory where the env writes
                the episode video. The env controls the final file name using
                ``episode_{episode_id}_seed_{actual_seed}.mp4`` because the
                actual RoboTwin runtime seed is only known after reset.
                Default is None.
            episode_id (int | None, optional): Episode identifier forwarded to
                RoboTwin as ``now_ep_num``. When ``video_dir`` is set, this
                value is also used in the generated video file name. If None,
                the existing ``self.cfg.episode_id`` is reused. Default is
                None.

        Returns:
            tuple[RoboTwinObsType, dict]:
                A tuple containing the initial observation and
                environment info after reset.

        """
        if env_ids is not None:
            raise NotImplementedError(
                "RoboTwinEnv does not support env_ids in reset()."
            )

        if isinstance(seed, str):
            raise TypeError(
                "RoboTwinEnv.reset() seed must be an int or None. "
                f"Got {seed!r}."
            )
        if isinstance(offset_seed, str):
            raise TypeError(
                "RoboTwinEnv.reset() offset_seed must be an int or None. "
                f"Got {offset_seed!r}."
            )

        self.close(clear_cache=clear_cache)
        start_seed = None
        if seed is not None:
            start_seed = self.cfg.calculate_seed(seed)
        if episode_id is not None:
            self.cfg.episode_id = episode_id

        seed_changes = start_seed is not None and start_seed != self.cfg.seed
        if offset_seed is not None:
            next_offset_seed = self._resolve_offset_seed(offset_seed)
        elif seed_changes:
            next_offset_seed = 0
        else:
            next_offset_seed = self._offset_seed
        offset_seed_changes = next_offset_seed != self._offset_seed
        task_name_changes = (
            task_name is not None and task_name != self.cfg.task_name
        )
        # check if task is not initialized or seed/task_name changes
        if (
            self._task is None
            or seed_changes
            or offset_seed_changes
            or task_name_changes
        ):
            # when need to create new env:
            # * when no existing env
            # * when seed changes
            if seed_changes:
                assert start_seed is not None
                self.cfg.seed = start_seed
                self._resolved_start_seed = self.cfg.resolve_start_seed(
                    self.cfg.seed
                )
            self._offset_seed = next_offset_seed
            if task_name_changes:
                assert task_name is not None
                self.cfg.task_name = task_name
            with in_robotwin_workspace():
                task, instructions = self._check_and_update_seed()
            assert task is not None
            self._task = task
            self._instructions = instructions

        with in_robotwin_workspace():
            task_config = self.cfg.get_task_config_for_seed(
                runtime_seed=self.current_seed
            )
            self._task.setup_demo(**task_config)  # type: ignore
        self._assert_supported_robot_layout()
        # Reset the FK helper before formatting the first observation so the
        # returned post-reset EEF edges always reflect the current episode.
        self._joints_to_eef_transform = None
        self._cached_obs_robots = None

        self._eval_chosen_instruction = None

        episode_video_path = None
        if video_dir is not None:
            episode_video_path = os.path.join(
                video_dir,
                f"episode_{self.cfg.episode_id}_seed_{self.current_seed}.mp4",
            )

        self._stop_video_recording()
        raw_obs = self._task.get_obs()
        if episode_video_path is not None:
            frame = self._extract_video_frame(raw_obs)
            if frame is None:
                logger.warning(
                    "Skip RoboTwin episode video recording because the head "
                    "camera RGB frame is unavailable."
                )
            else:
                try:
                    writer = self._get_video_writer()
                    writer.open(episode_video_path)
                    writer.write_frame(frame)
                except VideoBackendUnavailableError:
                    self._stop_video_recording()
                    logger.warning(
                        "Skip RoboTwin episode video recording because "
                        "ffmpeg is not available in PATH."
                    )
                except VideoWriterError:
                    self._stop_video_recording()
                    logger.exception(
                        "Failed to start RoboTwin episode video recording at "
                        "%s.",
                        episode_video_path,
                    )
        try:
            obs = self._format_obs(raw_obs) if return_obs else None
            info = self._get_info()
        except Exception:
            self._stop_video_recording()
            raise
        self._post_reset_state_available = True
        self._episode_finalized = False

        return obs, info

    def _joints2ee_pose(self, joints: np.ndarray) -> RoboTwinEEF:
        """Convert joint positions to world-frame end-effector transforms.

        Args:
            joints (np.ndarray): The joint positions of the robot.

        Returns:
            RoboTwinEEF: Left and right end-effector transforms in world
                frame. ``left_eef.parent_frame_id`` and
                ``right_eef.parent_frame_id`` are both ``"world"``.

        """
        joints_np = np.asarray(joints, dtype=np.float32)
        if joints_np.ndim == 1:
            joints_np = joints_np[None, :]
        if joints_np.ndim != 2 or joints_np.shape[0] != 1:
            raise ValueError(
                "Expected joints to have shape (D,) or (1, D), got "
                f"{tuple(joints_np.shape)}."
            )

        left_joint_count = len(self._task.robot.left_arm_joints_name)
        right_joint_count = len(self._task.robot.right_arm_joints_name)
        total_arm_joint_count = left_joint_count + right_joint_count
        joint_dim = joints_np.shape[-1]
        if joint_dim == total_arm_joint_count + 2:
            left_arm_joints = joints_np[:, :left_joint_count]
            right_start = left_joint_count + 1
        elif joint_dim == total_arm_joint_count:
            left_arm_joints = joints_np[:, :left_joint_count]
            right_start = left_joint_count
        else:
            raise ValueError(
                "Expected RoboTwin joints to contain left/right arm joints "
                "with optional gripper values, got shape "
                f"{tuple(joints_np.shape)}."
            )
        right_arm_joints = joints_np[
            :,
            right_start : right_start + right_joint_count,
        ]

        return self._get_joints_to_eef_transform().transform(
            left_arm_joints=torch.from_numpy(left_arm_joints),
            right_arm_joints=torch.from_numpy(right_arm_joints),
        )

    def _get_joints_to_eef_transform(self) -> RoboTwinJointsToEEF:
        """Get the cached RoboTwin joint-to-EEF forward-kinematics helper.

        The helper is built lazily from the supported combined dual-arm URDF
        and the current ``world -> robot_base`` transform, then cached for the
        rest of the episode.
        """
        if self._joints_to_eef_transform is not None:
            return self._joints_to_eef_transform

        urdf_map = self.get_robot_urdf()
        urdf_content = urdf_map["left"]

        robot_base_tf = self._get_tf().get_tf("world", "robot_base")
        if not isinstance(robot_base_tf, BatchFrameTransform):
            raise RuntimeError(
                "Expected supported RoboTwin layouts to expose a single "
                "world->robot_base BatchFrameTransform."
            )

        self._joints_to_eef_transform = RoboTwinJointsToEEF(
            urdf_content=urdf_content,
            robot_base_xyz=robot_base_tf.xyz[0].tolist(),
            robot_base_quat=robot_base_tf.quat[0].tolist(),
        )
        return self._joints_to_eef_transform

    def _get_info(self) -> dict[str, Any]:
        info = {
            "seed": self.current_seed,
            "start_seed": self.start_seed,
            "resolved_start_seed": self.resolved_start_seed,
            "offset_seed": self.offset_seed,
            "task": self.cfg.task_name,
        }
        info.update(self._task.info)
        return info

    def _resolve_offset_seed(self, offset_seed: int | None) -> int:
        if offset_seed is None:
            return self._offset_seed
        if isinstance(offset_seed, str):
            raise TypeError(
                "RoboTwinEnv.reset() offset_seed must be an int or None. "
                f"Got {offset_seed!r}."
            )
        if offset_seed < 0:
            raise ValueError(f"offset_seed must be >= 0, got {offset_seed}.")
        return offset_seed

    def _assert_supported_robot_layout(
        self,
        task: Base_Task | None = None,
    ) -> None:
        """Validate that the current RoboTwin robot layout is supported.

        RoboTwinEnv currently supports only the combined dual-arm layout with
        one shared robot base pose. Unsupported layouts are rejected at the
        env boundary during ``reset()`` and by robot-structure helper methods.
        """
        if task is None:
            task = self._task
        if task is None:
            raise RuntimeError("RoboTwinEnv has no active task to validate.")
        if task.robot.is_dual_arm is False:
            raise NotImplementedError(
                "RoboTwinEnv currently only supports a combined dual-arm "
                "robot layout. Separate left/right URDF layouts are not "
                "supported."
            )

        left_base_tf = sapien_pose_to_orchard(
            task.robot.left_entity_origion_pose
        )
        right_base_tf = sapien_pose_to_orchard(
            task.robot.right_entity_origion_pose
        )
        if left_base_tf != right_base_tf:
            raise NotImplementedError(
                "RoboTwinEnv currently only supports a combined dual-arm "
                "robot with a shared robot base. Separate left/right robot "
                "base poses are not supported."
            )

    def finalize_episode(self) -> None:
        """Finalize episode-local artifacts without closing the runtime.

        This method is idempotent and safe to call when no episode is active.
        It stops the current episode video writer but keeps the reusable
        RoboTwin task runtime open. After this call, ``step()`` rejects the
        finalized episode until ``reset()`` succeeds or ``reset_from_state()``
        restores a non-finalized episode state.
        """

        self._episode_finalized = True
        self._stop_video_recording()

    def close(self, clear_cache: bool = True):
        """Close the environment."""
        self._episode_finalized = True
        self._post_reset_state_available = False
        self._stop_video_recording()
        self._joints_to_eef_transform = None
        self._cached_obs_robots = None
        if self._task is not None:
            self._task.close_env(clear_cache=clear_cache)
            if self._task.render_freq > 0:
                self._task.viewer.close()

    def _get_joint_state_names(self: RoboTwinEnv) -> list[str]:
        ret_names = []
        ret_names.extend(self._task.robot.left_arm_joints_name)
        ret_names.append(self._task.robot.left_gripper_name["base"])
        ret_names.extend(self._task.robot.right_arm_joints_name)
        ret_names.append(self._task.robot.right_gripper_name["base"])
        return ret_names

    def _get_obs(self) -> dict[str, Any]:
        """Get the current observation from the environment.

        Note that in current RoboTwin implementation, the joints of the robot
        are provided in the "joint_action" key of the observation, and it
        actually represents the joint target positions! This is a design
        flaw in RoboTwin, and we leave it as is to be consistent with RoboTwin!

        """
        ret = self._task.get_obs()
        return self._format_obs(ret)

    @staticmethod
    def _pose_vector_to_tf(
        pose_vector: Sequence[float] | np.ndarray,
        *,
        child_frame_id: str,
    ) -> BatchFrameTransform:
        """Convert a RoboTwin EE pose vector to a world-frame transform.

        Args:
            pose_vector (Sequence[float] | np.ndarray): RoboTwin EE pose in
                ``[x, y, z, qw, qx, qy, qz]`` order.
            child_frame_id (str): Child frame name for the returned transform.

        Returns:
            BatchFrameTransform: ``world -> child_frame_id`` transform.
        """
        pose_np = np.asarray(pose_vector, dtype=np.float32)
        if pose_np.shape != (7,):
            raise ValueError(
                "Expected RoboTwin endpose to contain 7 values "
                f"(xyz + quaternion), got shape {tuple(pose_np.shape)}."
            )
        return BatchFrameTransform(
            xyz=torch.from_numpy(pose_np[:3]).unsqueeze(0),
            quat=torch.from_numpy(pose_np[3:]).unsqueeze(0),
            parent_frame_id="world",
            child_frame_id=child_frame_id,
        )

    def _get_endpose_frame_ids(self) -> tuple[str, str]:
        """Return RoboTwin's runtime end-effector frame names.

        Raw ``ret["endpose"]`` values come from RoboTwin's EE pose helpers.
        Reuse the corresponding EE child-link names reported by the runtime
        robot object, namely ``self._task.robot.left_ee.child_link`` and
        ``self._task.robot.right_ee.child_link``, instead of hardcoding
        embodiment-specific frame IDs locally.
        """
        try:
            return (
                self._task.robot.left_ee.child_link.get_name(),
                self._task.robot.right_ee.child_link.get_name(),
            )
        except AttributeError as exc:
            raise RuntimeError(
                "Failed to infer RoboTwin end-effector frame IDs from the "
                "runtime robot object."
            ) from exc

    def _get_eef_tf_edges(
        self, ret: dict[str, Any]
    ) -> list[BatchFrameTransform]:
        """Build extra world-frame EEF edges from RoboTwin observations.

        When ``joint_action["vector"]`` is available, this method adds
        joint-derived world-frame EEF transforms and renames their child
        frames to ``left_eef_from_joint`` / ``right_eef_from_joint`` to avoid
        colliding with RoboTwin's runtime EE frame names. When raw
        ``ret["endpose"]`` is available, it adds another pair of world-frame
        transforms that keep the runtime EE child-link names reported by the
        RoboTwin robot object.
        """
        tf_edges: list[BatchFrameTransform] = []

        joint_action = ret.get("joint_action")
        if isinstance(joint_action, dict) and "vector" in joint_action:
            joint_eef = self._joints2ee_pose(joint_action["vector"])
            tf_edges.extend(
                [
                    joint_eef.left_eef.model_copy(
                        update={"child_frame_id": LEFT_EEF_FROM_JOINT_FRAME_ID}
                    ),
                    joint_eef.right_eef.model_copy(
                        update={
                            "child_frame_id": RIGHT_EEF_FROM_JOINT_FRAME_ID
                        }
                    ),
                ]
            )

        endpose = ret.get("endpose")
        if isinstance(endpose, dict) and endpose:
            left_endpose_frame_id, right_endpose_frame_id = (
                self._get_endpose_frame_ids()
            )
            tf_edges.extend(
                [
                    self._pose_vector_to_tf(
                        endpose["left_endpose"],
                        child_frame_id=left_endpose_frame_id,
                    ),
                    self._pose_vector_to_tf(
                        endpose["right_endpose"],
                        child_frame_id=right_endpose_frame_id,
                    ),
                ]
            )

        return tf_edges

    def _format_obs(self, ret: dict[str, Any]) -> dict[str, Any]:
        """Format raw RoboTwin observations into orchard-compatible ones.

        The returned ``ret["tf"]`` graph always includes ``world ->
        robot_base``. When raw joint targets or RoboTwin end poses are
        available, it also includes additional world-frame end-effector edges:
        the joint-derived edges use ``*_eef_from_joint`` child frame IDs,
        while the raw RoboTwin end poses keep the runtime EE frame IDs
        reported by the RoboTwin robot object. The returned observation also
        includes ``ret["robots"]`` with observation-facing robot metadata
        derived from the supported combined dual-arm URDF descriptor.
        """
        eef_tf_edges = self._get_eef_tf_edges(ret)
        ret["instructions"] = self.instructions
        if self.cfg.format_datatypes:
            ret["joints"] = get_joints(
                ret, joint_names=self._get_joint_state_names()
            )
            ret.pop("joint_action", None)
            ret["cameras"] = get_observation_cams(ret)
            ret.pop("observation")
        ret["tf"] = self._get_tf()
        ret["robots"] = self.get_obs_robots()
        if eef_tf_edges:
            ret["tf"].add_tf(eef_tf_edges)
        return ret

    @staticmethod
    def _extract_video_frame(raw_obs: dict[str, Any]) -> np.ndarray | None:
        observation = raw_obs.get("observation")
        if not isinstance(observation, dict):
            return None
        head_camera = observation.get("head_camera")
        if not isinstance(head_camera, dict):
            return None
        frame = head_camera.get("rgb")
        if frame is None:
            return None

        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[2] != 3:
            return None
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        return np.ascontiguousarray(frame_np)

    def _write_video_frame(self, raw_obs: dict[str, Any]) -> None:
        writer = self._get_video_writer()
        if writer.is_closed:
            return

        frame = self._extract_video_frame(raw_obs)
        if frame is None:
            return

        try:
            writer.write_frame(frame)
        except VideoBackendUnavailableError:
            self._stop_video_recording()
            logger.warning(
                "Skip RoboTwin episode video recording because ffmpeg is "
                "not available in PATH."
            )
        except VideoWriterError:
            self._stop_video_recording()
            logger.exception("Failed to write RoboTwin episode video frame.")

    def _stop_video_recording(self) -> None:
        writer = self._video_writer
        if writer is None or writer.is_closed:
            return
        try:
            writer.close()
        except VideoWriterError:
            logger.exception("Failed to finalize RoboTwin episode video.")

    def _get_video_writer(self) -> VideoWriter:
        writer = self._video_writer
        if writer is None:
            writer = VideoWriter(
                pixel_format=ROBOTWIN_VIDEO_PIXEL_FORMAT,
                fps=ROBOTWIN_VIDEO_FPS,
            )
            self._video_writer = writer
        return writer

    @property
    def num_envs(self) -> int:
        # always 1 because RoboTwin does not support multi-envs
        return 1

    @property
    def action_space(self) -> gym.Space:
        """The action space of the environment.

        Actually RoboTwin does not implement the action space!
        Call this method will raise an error!

        Returns:
            gym.Space: The action space of the environment.
        """
        return self._task.action_space

    @property
    def observation_space(self) -> gym.Space:
        """The observation space of the environment.

        Actually RoboTwin does not implement the observation space!
        Call this method will raise an error!

        Returns:
            gym.Space: The observation space of the environment.
        """
        return self._task.observation_space

    def unwrapped_env(self) -> Base_Task:
        """Get the original RoboTwin environment."""
        return self._task

    def _get_tf(self) -> BatchFrameTransformGraph:
        """Get the frame transforms in the environment.

        For supported RoboTwin layouts this graph contains one static
        ``world -> robot_base`` edge. ``reset()`` rejects layouts with
        separate left/right robot bases before any observations are returned.

        Returns:
            BatchFrameTransformGraph: The static robot base transform graph.
        """
        self._assert_supported_robot_layout()
        left_base_tf = sapien_pose_to_orchard(
            self._task.robot.left_entity_origion_pose
        )
        return BatchFrameTransformGraph(
            tf_list=[
                BatchFrameTransform(
                    xyz=left_base_tf.xyz,
                    quat=left_base_tf.quat,
                    timestamps=left_base_tf.timestamps,
                    parent_frame_id="world",
                    child_frame_id="robot_base",
                )
            ],
            static_tf=[True],
        )

    def get_robot_urdf(self) -> dict[str, bytes]:
        """Get the supported combined dual-arm URDF content of the robot.

        Returns:
            dict[str, bytes]: A compatibility mapping containing the combined
                dual-arm URDF content under the ``"left"`` key. RoboTwin
                itself uses ``"left"`` as the compatibility slot for the
                shared dual-arm descriptor, so this key name is preserved
                here intentionally and is not an env-layer bug.
        """
        self._assert_supported_robot_layout()

        assert self._task.robot.left_urdf_path is not None
        with in_robotwin_workspace():
            with open(self._task.robot.left_urdf_path, "rb") as f:
                urdf_content = f.read()

        return {"left": urdf_content}

    def get_obs_robots(self) -> dict[str, Robot]:
        """Return observation-facing robot metadata for the current layout.

        The current RoboTwin env surface exposes a single combined dual-arm
        robot descriptor under the ``"left"`` key. This key matches
        RoboTwin's existing ``get_robot_urdf()`` convention for the shared
        dual-arm URDF and is kept intentionally for consistency, not because
        the env mistakes the robot for a single-arm ``left`` embodiment.
        Future layouts may return more than one descriptor, but the public
        observation contract is already plural.
        """
        if self._cached_obs_robots is not None:
            return self._cached_obs_robots.copy()

        urdf_map = self.get_robot_urdf()
        urdf_content = urdf_map.get("left")
        if not isinstance(urdf_content, bytes):
            raise RuntimeError(
                "Expected supported RoboTwin layouts to expose a combined "
                "dual-arm URDF under the 'left' key."
            )

        robot = Robot(
            index=0,
            name=COMBINED_DUAL_ARM_OBS_ROBOT_KEY,
            content=urdf_content.decode("utf-8"),
            content_format=RobotDescriptionFormat.URDF,
        )
        robot.update_md5()
        self._cached_obs_robots = {
            COMBINED_DUAL_ARM_OBS_ROBOT_KEY: robot,
        }
        return self._cached_obs_robots.copy()


class RoboTwinEnvCfg(EnvBaseCfg[RoboTwinEnv]):
    """Configuration for the RoboTwin environment."""

    class_type: ClassType[RoboTwinEnv] = RoboTwinEnv

    task_name: str
    """The name of the task to run, e.g., 'place_object_scale'."""

    seed: int = 0
    """The caller-facing start seed for the environment.

    In eval mode the env resolves this start seed into RoboTwin's reserved
    runtime seed range, but the config field itself remains unchanged.
    """

    episode_id: int = 0
    """Episode identifier forwarded to RoboTwin as ``now_ep_num``.

    The value may be updated per reset by passing ``episode_id`` to
    ``RoboTwinEnv.reset()``. When episode video recording is enabled, the env
    also uses this identifier in its output file-name convention together with
    the actual runtime seed selected during reset.
    """

    action_type: Literal["qpos", "ee"] = "qpos"
    """The RoboTwin action representation to use in the environment.

    `"qpos"` uses joint target positions. `"ee"` uses RoboTwin's
    end-effector action representation.
    """

    check_expert: bool = False
    """Whether to check the expert trajectory for the task.

    If true, the environment will attempt to run the task with the current
    runtime seed
    to check if the task can be completed successfully using the expert
    trajectory. If it fails, the env will increment its runtime
    ``offset_seed`` and retry until it finds a seed that can be completed by
    the expert trajectory.

    This mode is stronger than ``check_task_init``: it not only executes
    RoboTwin's ``play_once()`` initialization path, but also treats expert
    success as a requirement and may rewrite the env runtime offset to the
    first valid seed that passes the check.

    This field is used to make sure that the environment can be recorded
    successfully using the expert trajectory for imitation learning.

    ``check_expert`` and ``check_task_init`` are mutually exclusive. For
    evaluation, this is the recommended mode: use expert-verified seeds by
    setting ``check_expert=True`` and ``check_task_init=False``.

    """

    check_task_init: bool = True
    """Whether to check the task initialization.

    If true, the environment will call `play_once()` to execute the task
    with expert trajectory to check if the task can be initialized
    successfully.

    Compared with ``check_expert``, this mode is weaker and is meant as a
    RoboTwin warm-up path: it runs the same ``play_once()`` initialization
    flow once so task-specific runtime attributes are created, but it does
    not search for a new seed when the current one is unstable or cannot be
    solved by the expert trajectory.

    This field should be set to True because some task attributes that
    required for interaction may be initialized in the `play_once()` method,
    such as `place_object_scale` task.

    This should be a BUG in RoboTwin and will significantly affect the
    performance of the environment initialization.

    ``check_task_init`` and ``check_expert`` can not both be True. For
    evaluation, prefer ``check_expert`` instead of this flag because
    evaluation should use a seed that is known to be expert-solvable.

    """

    eval_mode: bool = False
    """Whether for evaluation.

    If true, the environment will use unseen texture_type.

    Evaluation also requires expert-verified seeds, so ``__post_init__``
    forces ``check_expert=True`` and ``check_task_init=False`` when
    ``eval_mode=True``. In other words, callers usually do not need to set
    those two flags manually for evaluation; enabling ``eval_mode`` is the
    recommended entrypoint.
    """

    max_instruction_num: int = 10
    """The maximum number of instructions to generate for the env."""

    format_datatypes: bool = False
    """whether to format obs as robo_orchard datatypes.

    If true, the observation will be formatted as:
        - "joints": dict of joint name to joint position. This key will
            replace the original "joint_action" key.
        - "cameras": dict of camera name to camera image. This key will
            replace the original "observation" key.
        - other keys in the original observation will be kept.

    The default is False for compatibility with original RoboTwin code.
    We highly recommend to set this field to True for better usability!
    """

    task_config_path: str | None = None
    """Path to the task configuration file.

    If not provided, the path will be set to
    `<RoboTwin_PATH>/task_config/_config_template.yml` for RoboTwin2.0.

    Note that we only support RoboTwin2.0 for now.
    """

    task_config_overrides: list[tuple[str, Any]] | None = None
    """Final overrides applied to the resolved task config.

    Each item is a `(path, value)` pair where `path` uses `/` to address
    nested dictionary keys, for example `("data_type/rgb", True)`.
    These overrides are applied after `_update_task_config()` finishes.
    """

    def __post_init__(self):
        if self.task_config_path is None:
            robo_twin_root = config_robotwin_path()
            self.task_config_path = os.path.join(
                robo_twin_root, "task_config", "_config_template.yml"
            )

        task_config_path = self.task_config_path
        if not os.path.exists(task_config_path):
            raise FileNotFoundError(
                f"Task configuration file {task_config_path} does not exist."
            )

        # check that check_expert or check_task_init can not be both True
        if self.check_expert and self.check_task_init:
            raise ValueError(
                "check_expert and check_task_init can not be both True."
            )

        if self.eval_mode and self.check_expert is False:
            logger.info(
                "Set check_expert from False to True for eval_mode."
                "This is to make sure the environment can successfully "
                "be initialized and completed using expert trajectory."
            )
            self.check_expert = True
            self.check_task_init = False

        if self.eval_mode and self.max_instruction_num != EVAL_INSTRUCTION_NUM:
            logger.info(
                f"Set max_instruction_num from "
                f"{self.max_instruction_num} to "
                f"{EVAL_INSTRUCTION_NUM} for eval_mode."
            )
            self.max_instruction_num = EVAL_INSTRUCTION_NUM

    def calculate_seed(self, seed: int) -> int:
        """Normalize the caller-facing start seed.

        This compatibility helper preserves the existing public method name
        while returning a caller-space start seed, not the actual runtime seed.
        """
        return seed

    def resolve_start_seed(self, seed: int) -> int:
        """Resolve a caller-facing start seed into a RoboTwin runtime seed.

        In eval mode, start seeds below ``EVAL_SEED_BASE`` are mapped into
        RoboTwin's reserved evaluation seed range.

        Args:
            seed (int): The caller-facing start seed.

        Returns:
            int: The resolved runtime start seed used in RoboTwin.
        """
        seed = self.calculate_seed(seed)

        if self.eval_mode and seed < EVAL_SEED_BASE:
            seed = EVAL_SEED_BASE * (1 + seed)

        if seed >= EVAL_SEED_BASE and self.eval_mode is False:
            raise ValueError(
                f"Seed {seed} is >= {EVAL_SEED_BASE} but eval_mode is "
                "False. This is reserved for RoboTwin evaluation mode."
            )

        return seed

    @property
    def embodiment_config_path(self) -> str:
        """Path to the embodiment configuration file."""
        robo_twin_root = config_robotwin_path()
        return os.path.join(
            robo_twin_root, "task_config", "_embodiment_config.yml"
        )

    @property
    def camera_config_path(self) -> str:
        """Path to the camera configuration file."""
        robo_twin_root = config_robotwin_path()
        return os.path.join(
            robo_twin_root, "task_config", "_camera_config.yml"
        )

    def get_task_config(self) -> dict[str, Any]:
        return self.get_task_config_for_seed(
            runtime_seed=self.resolve_start_seed(self.seed)
        )

    def get_task_config_for_seed(self, runtime_seed: int) -> dict[str, Any]:
        """Return the resolved task configuration for `setup_demo()`.

        The returned config combines the YAML template, derived RoboTwin
        fields from `_update_task_config()`, and the final
        `task_config_overrides` patches.
        """
        assert self.task_config_path is not None
        with (
            open(self.task_config_path, "r", encoding="utf-8") as f,
        ):
            task_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            ret = self._update_task_config(
                task_config,
                runtime_seed=runtime_seed,
            )
            self._apply_task_config_overrides(ret)

            ret["task_name"] = self.task_name
            return ret

    def _apply_task_config_overrides(
        self, task_config: dict[str, Any]
    ) -> None:
        """Apply final task-config overrides in place.

        This helper treats each item in `task_config_overrides` as a patch to
        the fully resolved task-config dictionary returned by
        `_update_task_config()`.

        Path rules:
            - Each override is a `(path, value)` pair.
            - `path` uses `/` as the separator for nested dictionary keys.
            - Only dictionary traversal is supported; list indices are not.
            - Every path segment must already exist in `task_config`.

        Guard rails:
            - Paths that correspond to env-managed fields or fields whose
              values are derived earlier in `get_task_config()` are rejected.
            - Invalid paths raise `ValueError`.
            - Missing intermediate or leaf keys raise `KeyError`.

        Args:
            task_config (dict[str, Any]): The resolved task config to patch.
        """
        reserved_paths = {
            "task_name",
            "seed",
            "now_ep_num",
            "eval_mode",
            "is_test",
            "camera/head_camera_type",
            "embodiment",
        }
        for path, value in self.task_config_overrides or []:
            if path in reserved_paths:
                raise ValueError(
                    f"Task config override path {path!r} is not supported "
                    "because it affects env-managed or derived fields."
                )

            keys = path.split("/")
            if not path or any(key == "" for key in keys):
                raise ValueError(
                    f"Invalid task config override path {path!r}."
                )

            target: Any = task_config
            for key in keys[:-1]:
                if not isinstance(target, dict):
                    raise KeyError(
                        f"Task config override path {path!r} does not "
                        f"resolve to a nested dict at {key!r}."
                    )
                if key not in target:
                    raise KeyError(
                        f"Task config override path {path!r} is missing "
                        f"segment {key!r}."
                    )
                target = target[key]

            if not isinstance(target, dict):
                raise KeyError(
                    f"Task config override path {path!r} does not resolve "
                    "to a dict parent."
                )

            leaf_key = keys[-1]
            if leaf_key not in target:
                raise KeyError(
                    f"Task config override path {path!r} is missing leaf "
                    f"key {leaf_key!r}."
                )
            target[leaf_key] = value

    def _update_task_config(
        self,
        task_args: dict[str, Any],
        *,
        runtime_seed: int,
    ) -> dict[str, Any]:
        """Update the task configuration.

        The function reads additional configuration files for task arguments
        such as embodiment and camera settings, and updates the task arguments
        accordingly. The returned dictionary is used for `setup_demo()`.

        """
        embodiment_type: list[str] = task_args.get("embodiment")  # type: ignore
        with open(self.embodiment_config_path, "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(embodiment_type: str) -> str:
            robot_file = embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise ValueError("No embodiment files")
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        with open(self.camera_config_path, "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = task_args["camera"]["head_camera_type"]
        task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
        task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            task_args["left_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["right_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            raise NotImplementedError(
                "RoboTwinEnv currently only supports a combined dual-arm "
                "robot layout. Task configs that specify separate left/right "
                "embodiments are not supported."
            )
        else:
            raise RuntimeError("embodiment items should be 1 or 3")

        task_args["left_embodiment_config"] = get_embodiment_config(
            task_args["left_robot_file"]
        )
        task_args["right_embodiment_config"] = get_embodiment_config(
            task_args["right_robot_file"]
        )
        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = (
                str(embodiment_type[0]) + "+" + str(embodiment_type[1])
            )
        task_args["embodiment_name"] = embodiment_name

        # update attributes in self

        task_args["seed"] = runtime_seed
        task_args["now_ep_num"] = self.episode_id
        task_args["eval_mode"] = self.eval_mode
        task_args["is_test"] = self.eval_mode

        return task_args


@functools.lru_cache(maxsize=1)
def config_robotwin_path() -> str:
    robo_twin_path = os.environ.get("RoboTwin_PATH", default=None)
    if robo_twin_path is None:
        raise ValueError(
            "RoboTwin_PATH environment variable is not set. "
            "Please set it to the path of the RoboTwin package."
        )
    if robo_twin_path not in sys.path:
        sys.path.append(robo_twin_path)
    return robo_twin_path


@contextmanager
def in_robotwin_workspace():
    """Context manager to temporarily change the `cwd` to the RoboTwin root."""
    robotwin_root = config_robotwin_path()
    original_cwd = os.getcwd()
    os.chdir(robotwin_root)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def create_task_from_name(task_name: str) -> Base_Task:
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as _:
        raise ImportError(
            f"Failed to import environment class {task_name} from "
            f"module {envs_module.__name__}. "
            "Please ensure the class name matches the task name and "
            "is defined in the module."
        )
    return env_instance
