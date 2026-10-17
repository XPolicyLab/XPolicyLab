# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for variable-batch planning on top of a single
:class:`~curobo.batch_motion_planner.BatchMotionPlanner`.

v2's ``BatchMotionPlanner.plan_pose`` requires input tensors at exactly
``max_batch_size`` (the IK stage pads internally, but TrajOpt does not). That
makes v1-style "plan any N up to max" call sites raise
``RuntimeError: output with shape ... doesn't match the broadcast shape ...``.

This module adds :class:`VariableBatchPlanner`, a thin wrapper that pads the
call to ``max_batch_size`` with a **known-trivial** dummy problem and slices
the result back to the caller's actual ``B``. Both the extra compute and the
extra memory are negligible:

- Empty collision scene (no obstacles to evaluate).
- Current state = robot retract configuration.
- Goal pose = forward kinematics of the retract configuration (i.e. a
  zero-displacement ``start == goal`` problem the optimizer solves in one
  iteration).

So pad-slot solves converge instantly; they never fail, never pull solver
iterations off the real problems, and (measured on a 4090) cost ~4 % more GPU
time than a planner built natively at the smaller batch size — while using a
single planner instance.

Caller contract:

1. Build and warm up a :class:`BatchMotionPlanner` with
   ``multi_env=True`` and the max batch size you will ever use.
2. Wrap it: ``vbp = VariableBatchPlanner(planner)``.
3. Before each call, load the real per-slot scenes into slots
   ``0 .. B-1`` via ``planner.scene_collision_checker.load_collision_model``
   (same as before).
4. Call ``vbp.plan_pose(goal_B, current_state_B, ...)`` where
   ``current_state_B.position.shape == (B, dof)`` and
   ``goal_B.position.shape == (B, 1, L, G, 3)``.
5. Read the result as if it were a native ``B`` problem:
   ``result.success.shape == (B, 1)`` and so on.

The wrapper tracks the previous ``B`` and only reloads empty scenes into
pad slots when ``B`` shrinks — no overhead on steady-state calls.
"""
from __future__ import annotations

from typing import Optional

import torch

from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.motion_planner import MotionPlannerCfg  # noqa: F401 (typing / docs)
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState


class VariableBatchPlanner:
    """Pad-and-slice wrapper around :class:`BatchMotionPlanner`.

    See module docstring for the padding contract. The wrapper is stateful:
    it tracks the last called ``B`` so it can reset pad-slot scenes only
    when the batch shrinks. Scene loads for the *real* slots
    (``0 .. B-1``) stay the caller's responsibility.
    """

    def __init__(self, planner: BatchMotionPlanner):
        self._planner = planner
        self._max_batch = planner.batch_size         # = MotionPlannerCfg.max_batch_size

        # Precompute the dummy problem once. The retract joint config (active
        # joints only) is both the "current state" and — via FK — the goal
        # pose for every pad slot. Store on the planner's device/dtype.
        retract_pos = planner.default_joint_state.position.view(1, -1)  # (1, dof)
        self._retract_position = retract_pos
        self._retract_joint_names = list(planner.kinematics.joint_names)

        retract_js = JointState.from_position(retract_pos, joint_names=self._retract_joint_names)
        retract_kin = planner.compute_kinematics(retract_js)
        # tool_poses.to_dict() → {link_name: Pose(position=(1,3), quaternion=(1,4))}
        retract_goal = GoalToolPose.from_poses(
            retract_kin.tool_poses.to_dict(),
            ordered_tool_frames=planner.tool_frames,
            num_goalset=1,
        )
        # Shape: (1, 1, L, 1, 3/4). Keep on GPU; we repeat on demand.
        self._retract_goal_position = retract_goal.position
        self._retract_goal_quaternion = retract_goal.quaternion
        self._tool_frames = list(planner.tool_frames)

        self._empty_scene_template: Scene = Scene()
        # Track the last B so we only reload pad slots on shrink.
        self._last_B: Optional[int] = None

    # -- Pass-throughs so callers can poke at the underlying planner -------

    @property
    def planner(self) -> BatchMotionPlanner:
        return self._planner

    @property
    def max_batch_size(self) -> int:
        return self._max_batch

    @property
    def kinematics(self):
        return self._planner.kinematics

    @property
    def tool_frames(self):
        return self._planner.tool_frames

    # -- Scene reload helper ----------------------------------------------

    def _load_empty_into_pad_slots(self, B_actual: int) -> None:
        """Point pad slots (B_actual..max-1) at an empty scene."""
        for slot_idx in range(B_actual, self._max_batch):
            self._planner.scene_collision_checker.load_collision_model(
                Scene(), env_idx=slot_idx,
            )

    # -- The one interesting method ---------------------------------------

    def plan_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        **plan_pose_kwargs,
    ):
        """Variable-B ``plan_pose`` with automatic pad/slice.

        Shapes on entry:
        - ``current_state.position``: ``(B, dof)``, ``B ∈ [1, max_batch_size]``.
        - ``goal_tool_poses.position``: ``(B, 1, L, G, 3)``.

        Shapes on exit (same as a native ``B`` call):
        - ``result.success``: ``(B, 1)``.
        - ``result.interpolated_trajectory.position``: ``(B, 1, max_H, dof_full)``.
        - ``result.interpolated_last_tstep``: ``(B, 1)``.
        """
        B_actual = current_state.position.shape[0]
        if B_actual > self._max_batch:
            raise ValueError(
                f"B_actual={B_actual} exceeds max_batch_size={self._max_batch}"
            )
        if B_actual == self._max_batch:
            # Full batch: no padding needed. If the *previous* call was
            # smaller, the pad slots still hold empty scenes — caller is
            # expected to have reloaded real scenes into all slots before
            # calling at full B. We cannot reload on their behalf because
            # we don't know which scenes go where.
            self._last_B = B_actual
            return self._planner.plan_pose(goal_tool_poses, current_state, **plan_pose_kwargs)

        # --- Padding path: B_actual < max_batch_size -----------------------

        # Only reload empty scenes into pad slots when B shrinks (or first call).
        if self._last_B is None or self._last_B > B_actual or self._last_B == self._max_batch:
            self._load_empty_into_pad_slots(B_actual)

        pad_n = self._max_batch - B_actual

        # Pad current_state: [real_B, retract, retract, ...]
        padded_pos = torch.cat(
            [current_state.position, self._retract_position.expand(pad_n, -1)], dim=0,
        )
        padded_cs = JointState.from_position(padded_pos, joint_names=self._retract_joint_names)

        # Pad goal: [real_B, retract_FK, retract_FK, ...]
        padded_goal_pos = torch.cat(
            [goal_tool_poses.position,
             self._retract_goal_position.expand(pad_n, -1, -1, -1, -1)], dim=0,
        ).contiguous()
        padded_goal_quat = torch.cat(
            [goal_tool_poses.quaternion,
             self._retract_goal_quaternion.expand(pad_n, -1, -1, -1, -1)], dim=0,
        ).contiguous()
        padded_goal = GoalToolPose(
            tool_frames=list(goal_tool_poses.tool_frames),
            position=padded_goal_pos,
            quaternion=padded_goal_quat,
        )

        result = self._planner.plan_pose(padded_goal, padded_cs, **plan_pose_kwargs)

        # Slice back to B_actual. Mutate in place so the returned object
        # behaves exactly like a native B-size result.
        result.success = result.success[:B_actual]
        if getattr(result, "interpolated_trajectory", None) is not None:
            traj = result.interpolated_trajectory
            if traj.position is not None:
                traj.position = traj.position[:B_actual]
            if traj.velocity is not None:
                traj.velocity = traj.velocity[:B_actual]
            if traj.acceleration is not None:
                traj.acceleration = traj.acceleration[:B_actual]
            if traj.jerk is not None:
                traj.jerk = traj.jerk[:B_actual]
        if getattr(result, "interpolated_last_tstep", None) is not None:
            result.interpolated_last_tstep = result.interpolated_last_tstep[:B_actual]
        if getattr(result, "js_solution", None) is not None:
            js = result.js_solution
            if js.position is not None:
                js.position = js.position[:B_actual]
            if js.velocity is not None:
                js.velocity = js.velocity[:B_actual]
            if js.acceleration is not None:
                js.acceleration = js.acceleration[:B_actual]

        self._last_B = B_actual
        return result
