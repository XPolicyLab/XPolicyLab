"""A small kinematic bimanual world for tests and offline development.

Flanges track commanded poses with bounded speed; boxes rest on a table, attach
to a gripper that closes around them, and drop to their support when released.
The grasp point cannot pass through the table or the top of an unheld box, which
produces the same "blocked" stalls a real contact does. Perception is ground
truth from the world state (MockPerceiver) with calibrated synthetic cameras.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np

from knowin_agent.core.geometry import Pose, matrix_to_quat, rot_x, rot_z, wrap180
from knowin_agent.perception.types import AxisTipEstimate, ObjectEstimate, OpeningEstimate, PerceptionError, SupportPlaneEstimate
from knowin_agent.runtime.backend import (ARMS, ArmCommand, ArmState, Budget, CameraFrame, Observation, RobotProfile,
                                 StepOutcome, default_orientation)

TABLE_Z = 0.75
MAX_OPENING_M = 0.08


@dataclass
class Box:
    name: str
    center: np.ndarray  # geometric centre
    size: tuple  # length (along yaw), width, height
    yaw_deg: float = 0.0
    held_by: Optional[str] = None
    grip_offset: Optional[np.ndarray] = None  # object centre in flange frame while held
    grip_rotation: Optional[np.ndarray] = None
    color: tuple = (200, 60, 60)

    @property
    def top(self) -> float:
        return float(self.center[2] + self.size[2] / 2)

    def contains_xy(self, point, margin=0.0) -> bool:
        c, s = math.cos(math.radians(-self.yaw_deg)), math.sin(math.radians(-self.yaw_deg))
        dx, dy = point[0] - self.center[0], point[1] - self.center[1]
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= self.size[0] / 2 + margin and abs(ly) <= self.size[1] / 2 + margin


def mock_profile() -> RobotProfile:
    return RobotProfile(
        name="mock_bimanual",
        cameras={"head": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"},
        wrist_camera={"left": "left_wrist", "right": "right_wrist"},
        grasp_offset=(0.12, 0.0, 0.0),
        fingertip_beyond_grasp_m=0.01,
        table_z=TABLE_Z,
        control_hz=25.0,
        workspace_min=(-0.8, -0.8, TABLE_Z - 0.01),
        workspace_max=(0.8, 0.8, TABLE_Z + 0.8),
        orientation_fn=default_orientation,
        notes="mock world: arms start above the table pointing down",
    )


def _look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 1, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x, y, z, eye
    return t


WRIST_MOUNT = np.eye(4)
WRIST_MOUNT[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # optical z along the tool axis
WRIST_MOUNT[:3, 3] = [0.03, 0.0, 0.06]


class MockBackend:
    def __init__(self, objects: Optional[list[Box]] = None, *, ticks_limit: Optional[int] = 3000,
                 instruction: str = "mock task", success: Optional[Callable[["MockBackend"], bool]] = None,
                 image_size=(64, 48), speed_m: float = 0.025, speed_rad: float = 0.25):
        self.profile = mock_profile()
        self.objects = {b.name: b for b in (objects or [])}
        self._instruction = instruction
        self._success = success
        self.ticks = 0
        self.ticks_limit = ticks_limit
        self.speed_m, self.speed_rad = speed_m, speed_rad
        self.image_size = image_size
        down = default_orientation(0.0, 90.0, 0.0)
        self.poses = {"left": Pose.make([-0.25, 0.0, TABLE_Z + 0.35], matrix_to_quat(down)),
                      "right": Pose.make([0.25, 0.0, TABLE_Z + 0.35], matrix_to_quat(down))}
        self.grippers = {arm: 1.0 for arm in ARMS}
        self.commands_log: list[dict] = []
        self.head_extrinsics = _look_at([0.0, -0.55, TABLE_Z + 0.55], [0.0, 0.05, TABLE_Z])
        self.head_intrinsics = np.array([[60.0, 0, 32.0], [0, 60.0, 24.0], [0, 0, 1]])
        self._initial = (copy.deepcopy(self.objects), dict(self.poses))

    supports_reset = True

    def reset(self) -> Observation:
        """Restore the initial scene (objects, arm poses, grippers); the tick budget starts over."""
        self.objects = copy.deepcopy(self._initial[0])
        self.poses = dict(self._initial[1])
        self.grippers = {arm: 1.0 for arm in ARMS}
        self.ticks = 0
        return self.observe()

    # -- protocol ---------------------------------------------------------------------------

    def instruction(self) -> str:
        return self._instruction

    def budget(self) -> Budget:
        return Budget(self.ticks, self.ticks_limit)

    def evaluation(self) -> dict:
        ok = bool(self._success(self)) if self._success else False
        return {"score": 100.0 if ok else 0.0, "success": ok, "ticks": self.ticks}

    def close(self) -> None:
        pass

    def observe(self) -> Observation:
        arms = {}
        for arm in ARMS:
            held = [b for b in self.objects.values() if b.held_by == arm]
            opening = self.grippers[arm]
            if self.grippers[arm] < 0.5:
                opening = min(1.0, float(held[0].size[1]) / MAX_OPENING_M) if held else 0.0
            arms[arm] = ArmState(ee_pose=self.poses[arm], joints=np.zeros(6), gripper_command=self.grippers[arm],
                                 gripper_opening=opening)
        w, h = self.image_size
        rgb = np.full((h, w, 3), 128, dtype=np.uint8)
        cams = {"head": CameraFrame("head", rgb, self.head_intrinsics, self.head_extrinsics)}
        for arm in ARMS:
            ext = self.poses[arm].matrix() @ WRIST_MOUNT
            cams[f"{arm}_wrist"] = CameraFrame(f"{arm}_wrist", rgb, self.head_intrinsics, ext)
        return Observation(tick=self.ticks, arms=arms, cameras=cams, instruction=self._instruction)

    def step(self, commands: Mapping[str, ArmCommand]) -> StepOutcome:
        if set(commands) != set(ARMS):
            raise ValueError("mock backend needs a command for both arms every tick")
        self.commands_log.append({arm: commands[arm] for arm in ARMS})
        for arm in ARMS:
            self._track(arm, commands[arm])
            self._gripper(arm, float(commands[arm].gripper))
        self.ticks += 1
        over = self.ticks_limit is not None and self.ticks >= self.ticks_limit
        done = bool(self._success(self)) if self._success else False
        return StepOutcome(self.observe(), episode_over=over or done,
                           reason="task_ended" if done else ("action_limit" if over else None))

    # -- world dynamics -----------------------------------------------------------------------

    def _grasp_point(self, pose: Pose) -> np.ndarray:
        return pose.transform(self.profile.grasp_offset)

    def _floor_at(self, point, arm) -> float:
        floor = TABLE_Z + self.profile.fingertip_beyond_grasp_m
        for box in self.objects.values():
            if box.held_by is None and box.contains_xy(point, margin=-0.004) and self.grippers[arm] < 0.5:
                floor = max(floor, box.top)
            elif box.held_by is None and box.contains_xy(point, margin=0.01) and self.grippers[arm] >= 0.5:
                floor = max(floor, box.center[2] - box.size[2] / 2 + self.profile.fingertip_beyond_grasp_m)
        held = [b for b in self.objects.values() if b.held_by == arm]
        for box in held:
            for other in self.objects.values():
                if other is not box and other.held_by is None and other.contains_xy(point, margin=0.01):
                    floor = max(floor, other.top + box.size[2] / 2)
            floor = max(floor, TABLE_Z + box.size[2] / 2)
        return floor

    def _track(self, arm: str, command: ArmCommand) -> None:
        current = self.poses[arm]
        delta = command.ee_pose.position - current.position
        dist = float(np.linalg.norm(delta))
        pos = current.position + delta * min(1.0, self.speed_m / max(dist, 1e-12))
        from knowin_agent.core.geometry import slerp

        _, angle = current.errors_to(command.ee_pose)
        quat = slerp(current.quat, command.ee_pose.quat, min(1.0, self.speed_rad / max(angle, 1e-12)))
        candidate = Pose.make(pos, quat)
        grasp = self._grasp_point(candidate)
        floor = self._floor_at(grasp, arm)
        if grasp[2] < floor:
            candidate = candidate.with_position(candidate.position + [0, 0, floor - grasp[2]])
        self.poses[arm] = candidate
        for box in self.objects.values():
            if box.held_by == arm:
                box.center = candidate.transform(box.grip_offset)

    def _gripper(self, arm: str, value: float) -> None:
        previous = self.grippers[arm]
        self.grippers[arm] = value
        grasp = self._grasp_point(self.poses[arm])
        if value < 0.5 <= previous:
            for box in self.objects.values():
                if box.held_by is None and box.contains_xy(grasp, margin=0.006) and \
                        box.center[2] - box.size[2] / 2 - 0.002 <= grasp[2] <= box.top + 0.002:
                    box.held_by = arm
                    flange = self.poses[arm]
                    box.grip_offset = flange.rotation.T @ (box.center - flange.position)
                    break
        elif value >= 0.5 > previous:
            for box in self.objects.values():
                if box.held_by == arm:
                    box.held_by = None
                    support = TABLE_Z
                    for other in self.objects.values():
                        if other is not box and other.held_by is None and other.contains_xy(box.center, margin=0.0):
                            support = max(support, other.top)
                    box.center = np.array([box.center[0], box.center[1], support + box.size[2] / 2])


def demo_world(ticks_limit: Optional[int] = 3000) -> MockBackend:
    """Two blocks on the table; success when the red block rests on the blue one."""
    objects = [Box("red block", np.array([-0.15, 0.10, TABLE_Z + 0.02]), (0.06, 0.03, 0.04)),
               Box("blue block", np.array([0.15, 0.12, TABLE_Z + 0.02]), (0.04, 0.04, 0.04), yaw_deg=30.0,
                   color=(60, 60, 200))]

    def stacked(world: MockBackend) -> bool:
        red, blue = world.objects["red block"], world.objects["blue block"]
        return red.held_by is None and abs(red.center[2] - blue.top - 0.02) < 0.003 and \
            float(np.linalg.norm(red.center[:2] - blue.center[:2])) < 0.015

    return MockBackend(objects, instruction="Stack the red block on the blue block.", success=stacked,
                       ticks_limit=ticks_limit)


class MockPerceiver:
    """Ground-truth perception over a MockBackend world (for tests and dry runs)."""

    def __init__(self, backend: MockBackend):
        self.backend = backend

    def _find(self, text: str) -> list[Box]:
        words = [w for w in text.lower().replace("_", " ").split() if w not in ("the", "a", "an", "object")]
        hits = [b for b in self.backend.objects.values() if all(w in b.name.lower() for w in words)]
        if not hits:
            raise PerceptionError(f"no object matching {text!r} is visible")
        return hits

    def _estimate(self, box: Box, obs: Observation, camera: str, text: str) -> ObjectEstimate:
        length, width, height = box.size
        long_axis = wrap180(box.yaw_deg if length >= width else box.yaw_deg + 90)
        long_axis = long_axis - 180 if long_axis > 90 else long_axis
        grasp = None if abs(length - width) <= 0.15 * max(length, width) else wrap180(long_axis + 90)
        if grasp is not None and grasp > 90:
            grasp -= 180
        elif grasp is not None and grasp <= -90:
            grasp += 180
        return ObjectEstimate(label=text, camera=camera, obs_id=obs.obs_id, center_xyz=[float(v) for v in box.center],
                              top_z=box.top, height_m=float(height), length_m=float(max(length, width)),
                              width_m=float(min(length, width)), long_axis_deg=float(long_axis), grasp_yaw_deg=grasp,
                              score=1.0, pixel_centroid=[0.0, 0.0], mask_pixels=100, method="mock_ground_truth")

    def locate(self, obs, text, camera="head", *, hint_px=None, max_objects=1, support_z=None):
        return [self._estimate(b, obs, camera, text) for b in self._find(text)[:max_objects]]

    def inspect_shape(self, obs, text, camera, **_):
        return self._estimate(self._find(text)[0], obs, camera, text)

    def inspect_axis_tip(self, obs, text, camera, *, gripper_point=None):
        box = self._find(text)[0]
        rotation = rot_z(box.yaw_deg)
        if box.held_by is not None:
            rotation = self.backend.poses[box.held_by].rotation @ box.grip_rotation if box.grip_rotation is not None \
                else self.backend.poses[box.held_by].rotation
        axis = rotation[:, 0]
        ends = [box.center + axis * box.size[0] / 2, box.center - axis * box.size[0] / 2]
        if gripper_point is not None:
            ends.sort(key=lambda p: -np.linalg.norm(p - gripper_point))
        else:
            ends.sort(key=lambda p: p[2])
        tip, base = ends
        return AxisTipEstimate(text, camera, obs.obs_id, [float(v) for v in (tip - base) / np.linalg.norm(tip - base)],
                               [float(v) for v in tip], [float(v) for v in base], float(box.size[0]), "mock_ground_truth")

    def inspect_support_plane(self, obs, camera, *, around_xyz=None, pixel=None, radius_m=0.08, exclude_text=None):
        center = [0.0, 0.0, TABLE_Z] if around_xyz is None else [float(around_xyz[0]), float(around_xyz[1]), TABLE_Z]
        return SupportPlaneEstimate(camera, obs.obs_id, TABLE_Z, [0.0, 0.0, 1.0], 1.0, center, "mock_ground_truth")

    def inspect_opening(self, obs, text, camera):
        box = self._find(text)[0]
        return OpeningEstimate(text, camera, obs.obs_id, [float(box.center[0]), float(box.center[1]), box.top],
                               float(min(box.size[:2]) / 2), box.top, [0.0, 0.0, 1.0], "mock_ground_truth")
