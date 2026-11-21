"""Reflexes (skills/reflexes): the colour tracker, intercept's controller, and how fast a
guard stops a motion."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import skills.reflexes as reflexes  # noqa: E402
from skills.reflexes import ColorTracker, PatchTracker, Reflexes, intercept_controller, step_toward  # noqa: E402
from harness_policy.geometry import DOWN, Mask, down_with_yaw  # noqa: E402
from harness_policy.runtime.events import Escalation  # noqa: E402
from harness_policy.services import policy_server  # noqa: E402
from test_run_policy import Policy, Worker, session  # noqa: E402


def frame(u, v, size=10):
    image = np.full((480, 640, 3), 120, dtype=np.uint8)
    image[v - size:v + size, u - size:u + size] = (200, 30, 30)
    return SimpleNamespace(images={"cam_head": image})


def square(u, v, size=10):
    mask = np.zeros((480, 640), dtype=bool)
    mask[v - size:v + size, u - size:u + size] = True
    return mask


class ColorTrackerTests(unittest.TestCase):
    def test_follows_the_colour_and_loses_it_out_of_the_window(self):
        tracker = ColorTracker(frame(100, 200).images["cam_head"], square(100, 200))
        for k in range(1, 10):
            u, v = tracker.find(frame(100 + 8 * k, 200).images["cam_head"])
            self.assertAlmostEqual(u, 100 + 8 * k - 0.5, delta=1.0)
            self.assertAlmostEqual(v, 199.5, delta=1.0)
        self.assertIsNone(tracker.find(frame(400, 400).images["cam_head"]))  # jumped out of the window

    def test_a_tiny_mask_is_refused(self):
        with self.assertRaises(ValueError):
            ColorTracker(frame(100, 200).images["cam_head"], square(100, 200, size=1))

    def test_takes_the_masks_locate_keeps(self):
        mask = Mask(square(100, 200), [90, 190, 110, 210], 0.9)
        tracker = ColorTracker(frame(100, 200).images["cam_head"], mask)
        self.assertAlmostEqual(tracker.uv[0], 99.5, delta=0.6)
        self.assertEqual(tracker.find(frame(108, 200).images["cam_head"])[1], 199.5)


def scene(u, v, size=15, belt=True):
    """A light belt (210) across the image, and on it a light object (215) with a dark band:
    the object's colour alone matches the whole belt."""
    image = np.full((480, 640, 3), 120, dtype=np.uint8)
    if belt:
        image[150:260] = 210
    image[v - size:v + size, u - size:u + size] = 215
    image[v - 3:v + 3, u - size:u + size] = 60
    return SimpleNamespace(images={"cam_head": image})


class PatchTrackerTests(unittest.TestCase):
    def test_follows_a_light_object_on_a_light_belt(self):
        tracker = PatchTracker(scene(200, 200).images["cam_head"], square(200, 200, size=15))
        colour = ColorTracker(scene(200, 200).images["cam_head"], square(200, 200, size=15))
        for k in range(1, 10):
            u, v = tracker.find(scene(200 + 7 * k, 200).images["cam_head"])
            self.assertAlmostEqual(u, 200 + 7 * k - 0.5, delta=1.5)
            self.assertAlmostEqual(v, 199.5, delta=1.5)
            colour.find(scene(200 + 7 * k, 200).images["cam_head"])
        self.assertGreater(abs(colour.uv[0] - (200 + 63)), 10)  # the colour drifted with the belt

    def test_an_object_covered_by_the_arm_is_not_seen(self):
        tracker = PatchTracker(scene(200, 200).images["cam_head"], square(200, 200, size=15))
        covered = scene(205, 200).images["cam_head"]
        covered[170:240, 175:240] = 90  # the arm over it
        self.assertIsNone(tracker.find(covered))
        self.assertEqual(tracker.find(scene(210, 200).images["cam_head"])[1], 199.5)

    def test_a_patch_without_texture_is_refused(self):
        with self.assertRaises(ValueError):
            PatchTracker(np.full((480, 640, 3), 120, dtype=np.uint8), square(200, 200))


class FitVelocityTests(unittest.TestCase):
    def test_a_straight_line_fit_ignores_one_noisy_sighting(self):
        history = [(0.1 * k, 0.1 * 0.1 * k, -0.05) for k in range(8)]
        history[4] = (0.4, 0.04 + 0.01, -0.05)
        vx, vy = reflexes.fit_velocity(history)
        self.assertAlmostEqual(vx, 0.1, delta=0.02)
        self.assertAlmostEqual(vy, 0.0, places=9)


class StepTowardTests(unittest.TestCase):
    def test_bounded_steps_reach_the_target(self):
        start = [0.0, 0.0, 1.2, *DOWN]
        target = [0.3, 0.1, 0.9, *down_with_yaw(1.2)]
        pose, steps = start, 0
        while steps < 200 and pose != step_toward(pose, target):
            nxt = step_toward(pose, target)
            self.assertLessEqual(np.linalg.norm(np.subtract(nxt[:3], pose[:3])), reflexes.MAX_STEP_M + 1e-9)
            turn = 2 * np.arccos(min(1.0, abs(float(np.dot(nxt[3:], pose[3:])))))
            self.assertLessEqual(turn, reflexes.MAX_TURN_RAD + 1e-6)
            pose, steps = nxt, steps + 1
        np.testing.assert_allclose(pose[:3], target[:3], atol=1e-9)
        self.assertAlmostEqual(abs(float(np.dot(pose[3:], target[3:]))), 1.0, places=9)
        self.assertGreater(steps, 10)


class InterceptControllerTests(unittest.TestCase):
    def test_follows_comes_down_closes_and_finishes(self):
        # Pixels map to the plane as (u / 1000, v / 1000); the object moves +x 2 px a frame.
        plane = mock.patch.object(reflexes, "pixel_to_plane",
                                  lambda observation, camera, u, v, z: np.array([u / 1000.0, v / 1000.0, z]))
        plane.start()
        self.addCleanup(plane.stop)
        tracker = ColorTracker(frame(300, 200).images["cam_head"], square(300, 200))
        commands = []
        controller, state = intercept_controller(
            tracker, "cam_head", 0.8, DOWN, "left_arm",
            lambda tip, orientation: [tip[0], tip[1], tip[2] + 0.15, *orientation], lead_seconds=0.0)
        ee = [0.0, 0.0, 1.2, *DOWN]
        for tick in range(300):
            t = tick / 25.0
            view = {"t": t, "left_ee": list(ee[:3]), "left_ee_pose": list(ee),
                    "observation": frame(300 + 2 * (tick // 5), 200)}
            command = controller(view)
            if command is None:
                break
            commands.append(command)
            if "left_eef" in command:  # the arm reaches its target at once
                ee = command["left_eef"]
        self.assertEqual(state["phase"], "closed")
        poses = [[0.0, 0.0, 1.2]] + [c["left_eef"][:3] for c in commands if "left_eef" in c]
        steps = [np.linalg.norm(np.subtract(b, a)) for a, b in zip(poses, poses[1:])]
        self.assertLessEqual(max(steps), reflexes.MAX_STEP_M + 1e-9)  # no tick jumps
        heights = [p[2] for p in poses[1:]]
        # Followed above it: came down to the hover height before the jaws were over it.
        self.assertTrue(any(abs(h - (0.8 + reflexes.HOVER_M + 0.15)) < reflexes.MAX_STEP_M for h in heights))
        self.assertAlmostEqual(min(heights), 0.8 + 0.15)  # came down to the grasp height
        self.assertEqual(sum(1 for c in commands if c.get("left_gripper") == 0.0), reflexes.CLOSE_TICKS)
        self.assertGreater(commands[-1]["left_eef"][0], 0.3)  # still following while it closed

    def test_coasts_on_the_fitted_motion_while_the_arm_hides_it(self):
        # Seen for 4 frames moving +x 2 px a frame (0.02 m/s with the mock plane), then hidden.
        plane = mock.patch.object(reflexes, "pixel_to_plane",
                                  lambda observation, camera, u, v, z: np.array([u / 1000.0, v / 1000.0, z]))
        plane.start()
        self.addCleanup(plane.stop)
        tracker = ColorTracker(frame(300, 200).images["cam_head"], square(300, 200))
        controller, state = intercept_controller(
            tracker, "cam_head", 0.8, DOWN, "left_arm",
            lambda tip, orientation: [tip[0], tip[1], tip[2] + 0.15, *orientation], lead_seconds=0.0)
        hidden = SimpleNamespace(images={"cam_head": np.full((480, 640, 3), 120, dtype=np.uint8)})
        ee, command = [0.3, 0.2, 1.2, *DOWN], {}
        for tick in range(200):
            seen = tick < 20
            image = frame(300 + 2 * (tick // 5), 200) if seen else SimpleNamespace(**vars(hidden))
            command = controller({"t": tick / 25.0, "left_ee": list(ee[:3]), "left_ee_pose": list(ee),
                                  "observation": image})
            if command is None:
                break
            ee = command.get("left_eef", ee)
        self.assertEqual(state["phase"], "closed")
        self.assertGreaterEqual(state["sightings"], reflexes.MIN_TRACK)
        # It closed where the fitted motion put the object, past the last place it was seen.
        self.assertGreater(ee[0], 0.306)

    def test_gives_up_when_the_object_is_never_seen(self):
        tracker = ColorTracker(frame(300, 200).images["cam_head"], square(300, 200))
        controller, state = intercept_controller(tracker, "cam_head", 0.8, DOWN, "left_arm",
                                                 lambda tip, orientation: [*tip, *orientation])
        empty = SimpleNamespace(images={"cam_head": np.full((480, 640, 3), 120, dtype=np.uint8)})
        results = [controller({"t": k / 25.0, "left_ee": [0, 0, 1], "left_ee_pose": [0, 0, 1, *DOWN],
                               "observation": SimpleNamespace(**vars(empty))})
                   for k in range(reflexes.LOST_FRAMES)]
        self.assertIsNone(results[-1])
        self.assertEqual(state["phase"], "lost")


class InterceptTests(unittest.TestCase):
    """intercept as a program calls it: the object and its mask from the last locate."""

    def test_catches_with_the_mask_of_locate_and_lifts(self):
        plane = mock.patch.object(reflexes, "pixel_to_plane",
                                  lambda observation, camera, u, v, z: np.array([u / 1000.0, v / 1000.0, z]))
        plane.start()
        self.addCleanup(plane.stop)

        class Arm(Reflexes):
            def __init__(self):
                self._eef_scene = [{"id": "o1", "position": [0.3, 0.2, 0.70], "angle_deg": 0.0}]
                self._eef_masks = (frame(300, 200), {"o1": {"cam_head": Mask(square(300, 200), [290, 190, 310, 210], 0.9)}})
                self.ee = [0.2, 0.1, 1.1, *DOWN]
                self.lifts = []

            def link6_for_tip(self, tip, orientation):
                return [tip[0], tip[1], tip[2] + 0.15, *orientation]

            def run_controller(self, fn, max_seconds=None, cameras=None, camera_hz=None):
                for tick in range(int(max_seconds * 25)):
                    command = fn({"t": tick / 25.0, "left_ee": self.ee[:3], "left_ee_pose": list(self.ee),
                                  "observation": frame(300 + 2 * (tick // 5), 200)})
                    if command is None:
                        break
                    self.ee = command.get("left_eef", self.ee)
                return {"ticks": tick}

            def get_state(self):
                return {"left_ee_pose": list(self.ee)}

            def move_eef(self, arm, pose, max_seconds=None):
                self.lifts.append(pose)
                return {"lifted": True}

        arm = Arm()
        result = arm.intercept("left_arm", "o1", max_seconds=12.0)
        self.assertTrue(result["caught"])
        self.assertGreater(result["sightings"], 5)
        # A flat object (z 0.70, below the lowest grasp) is grasped at pick's lowest height.
        self.assertAlmostEqual(arm.lifts[0][2], reflexes.MIN_GRASP_Z + 0.15 + reflexes.LIFT_M, places=6)

    def test_an_object_not_located_is_refused(self):
        arm = Reflexes()
        arm._eef_scene, arm._eef_masks = [], (None, {})
        with self.assertRaises(ValueError):
            arm.intercept("left_arm", "o9")


class ControllerJointStepTests(unittest.TestCase):
    """An eef target whose IK solution swings a joint further than the shield allows in one
    tick (near the straight-down wrist singularity) is approached over several ticks."""

    def controller(self, solution):
        from skills.reactive import Reactive

        class Arm(Reactive):
            groups = {"left_arm": {"joints": [0, 1, 2, 3, 4, 5], "gripper": 6}}
            profile = {"shield": {"max_joint_jump_rad": 0.3}}  # as the worker's profile carries it

            def ik(self, arm, pose):
                return list(solution)

            def _base_command(self, state):
                return list(state["joint_position"])

        return Arm()

    def test_a_far_solution_is_approached_within_the_jump_limit(self):
        arm = self.controller([0.1, 0.0, 0.0, 0.75, 0.0, -0.05])
        joints = [0.0] * 7
        for _ in range(4):
            target = arm._merge_command({"joint_position": joints}, {"left_eef": [0.3, 0.0, 1.0, *DOWN]})
            self.assertLessEqual(max(abs(a - b) for a, b in zip(target[:6], joints[:6])), 0.24 + 1e-9)
            joints = target
        self.assertAlmostEqual(joints[3], 0.75)
        self.assertAlmostEqual(joints[0], 0.1)

    def test_a_near_solution_is_taken_as_it_is(self):
        arm = self.controller([0.05, 0.0, 0.0, -0.1, 0.0, 0.0])
        target = arm._merge_command({"joint_position": [0.0] * 7}, {"left_eef": [0.3, 0.0, 1.0, *DOWN]})
        self.assertEqual(target[:6], [0.05, 0.0, 0.0, -0.1, 0.0, 0.0])


class GuardLatencyTests(unittest.TestCase):
    def setUp(self):
        self.locks = tempfile.TemporaryDirectory()
        self.directory = mock.patch.object(policy_server, "POLICY_LOCK_DIR", self.locks.name)
        self.directory.start()

    def tearDown(self):
        self.directory.stop()
        self.locks.cleanup()

    def test_a_guard_stops_on_the_first_tick_its_observation_satisfies(self):
        worker = Worker(step_limit=100)
        s = session(worker, Policy())
        s.reactive.when(lambda o: o["step"] >= 7, "stop", name="seven")
        with self.assertRaises(Escalation) as raised:
            s._execute("test", {}, lambda state, tick: list(worker.joints), 2.0, None, None)
        event = raised.exception.trace.event
        self.assertEqual((event.kind, event.source), ("GUARD_TRIGGERED", "seven"))
        self.assertEqual(worker.step, 7)  # no command after the observation that satisfied it


if __name__ == "__main__":
    unittest.main()
