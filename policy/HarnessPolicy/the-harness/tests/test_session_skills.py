"""Skills v2: both arms at once, run_policy slices, pick/place targets and misses,
push, press, watch_scene, the IK branch-flip guard and the kernel's API names."""
import base64
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from harness_policy.runtime.events import Escalation  # noqa: E402
from harness_policy.runtime.kernel import restore_api  # noqa: E402
from harness_policy.services import policy_server as session_module  # noqa: E402
from harness_policy.geometry import DOWN, FORWARD, link6_for_tip  # noqa: E402
from harness_policy.runtime.session import session_class  # noqa: E402

RoboDojoSession = session_class()
from test_session_motion import FakeWorker, session  # noqa: E402

BASES = {"left_arm": [-0.3, -0.45, 0.765], "right_arm": [0.3, -0.45, 0.765]}


class TwoArmWorker(FakeWorker):
    """FakeWorker whose planner starts each arm's path from that arm's own joints;
    link6 of the right arm is at (j7, j8, j9)."""

    def call(self, name, **arguments):
        if name == "plan":
            offset = 0 if arguments["arm"] == "left_arm" else 7
            goal = arguments["pose"][:3] + [0.0, 0.0, 0.0] if arguments["pose"] else list(arguments["joints"])
            start = self.joints[offset:offset + 6]
            return {"success": True, "waypoints": [[s + (g - s) * k / 4 for s, g in zip(start, goal)]
                                                   for k in range(1, 5)]}
        return super().call(name, **arguments)


def with_profile(s, **extra):
    s.profile = {"gripper_empty_opening": 0.185, "arm_bases": BASES, **extra}
    s._held = {}
    return s


def link6_ik(arm, pose):
    """The fake worker's kinematics: link6 position = the arm's first three joints."""
    return list(pose[:3]) + [0.0, 0.0, 0.0]


class TogetherTests(unittest.TestCase):
    def test_home_moves_both_arms_in_one_execution(self):
        worker = TwoArmWorker()
        s = with_profile(session(worker), return_to_start=True)
        s._start_joints = list(worker.joints)
        worker.joints[0:3] = [0.4, -0.2, 0.9]
        worker.joints[7:10] = [-0.3, 0.1, 0.8]
        s.state = worker.state()
        moved = []
        original = worker.call

        def recording(name, **arguments):
            result = original(name, **arguments)
            if name == "tick":
                moved.append((worker.joints[0] != 0.4, worker.joints[7] != -0.3))
            return result

        worker.call = recording
        traces = s.home()
        self.assertEqual([t.event.kind for t in traces], ["COMPLETED", "COMPLETED"])
        self.assertEqual([t.parameters["arm"] for t in traces], ["left_arm", "right_arm"])
        self.assertEqual(moved[0], (True, True))  # both arms move from the first tick
        self.assertEqual(len(s._events), 1)  # one execution
        self.assertEqual([round(v, 6) for v in worker.joints[0:6] + worker.joints[7:13]], [0.0] * 12)
        self.assertEqual(traces[0].ticks, traces[1].ticks)

    def test_wrap_up_homes_both_arms_together(self):
        worker = TwoArmWorker()
        s = with_profile(session(worker), return_to_start=True)
        s._start_joints = list(worker.joints)
        worker.joints[7:10] = [-0.3, 0.1, 0.8]
        s.state = worker.state()
        self.assertEqual(s._wrap_up(), ["COMPLETED"] * 4)
        # The two grippers open, then one execution homes both arms together.
        self.assertEqual([event["skill"] for event in s._events], ["open_gripper", "open_gripper", "home"])

    def test_an_arm_without_a_path_is_reported_after_the_other_moved(self):
        worker = TwoArmWorker()
        s = with_profile(session(worker))
        s._start_joints = list(worker.joints)
        worker.joints[0:3] = [0.4, -0.2, 0.9]
        worker.joints[7:10] = [-0.3, 0.1, 0.8]
        s.state = worker.state()
        s.plan_path = lambda arm, pose=None, joints=None: (None if arm == "right_arm" else
                                                           [[0.0] * 6])
        with self.assertRaises(Escalation) as raised:
            s.home()
        self.assertEqual(raised.exception.trace.event.kind, "SKILL_ERROR")
        kinds = [t.event.kind for t in raised.exception.traces]
        self.assertEqual(kinds, ["COMPLETED", "SKILL_ERROR"])
        self.assertAlmostEqual(worker.joints[0], 0.0)  # the left arm still went home

    def test_a_blocked_arm_ends_the_execution_with_its_event(self):
        worker = TwoArmWorker(floor=0.95)  # the left arm cannot get below link6 z = 0.95
        s = with_profile(session(worker))
        with self.assertRaises(Escalation) as raised:
            s._together("clear_view", {"left_arm": ("pose", [0.1, -0.1, 0.9] + list(DOWN)),
                                       "right_arm": ("pose", [0.2, -0.1, 1.0] + list(DOWN))}, 0.6)
        event = raised.exception.trace.event
        self.assertEqual((event.kind, event.detail["arm"]), ("BLOCKED", "left_arm"))
        self.assertEqual([t.event.kind for t in raised.exception.traces], ["BLOCKED", "COMPLETED"])
        self.assertAlmostEqual(worker.joints[9], 1.0)


def png_url(image):
    import cv2

    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode()


class PolicyWorker:
    def __init__(self):
        self.joints = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
        self.step, self.commands = 0, []

    def state(self):
        return {"step": self.step, "step_limit": 1000, "ended": False, "joint_position": list(self.joints),
                "left_ee_pose": [0.0, 0.0, 1.0, *DOWN], "right_ee_pose": [0.0, 0.0, 1.0, *DOWN]}

    def call(self, name, **arguments):
        import numpy as np

        if name == "policy_obs":
            return {"vision": {"cam_head": png_url(np.zeros((4, 4, 3), dtype=np.uint8))},
                    "state": {}, "instruction": "Stack the bowls.", "additional_info": {"frequency": 25},
                    "data_format_version": "v1.0", "robot_state": self.state()}
        if name == "tick":
            self.joints = list(arguments["command"])
            self.commands.append(self.joints[0])
            self.step += 1
            return {"executed": True, "shield": {"verdict": "pass", "reasons": [], "command": self.joints},
                    "state": self.state()}
        if name == "plan":
            return {"success": True, "waypoints": [list(arguments["joints"])]}
        raise AssertionError(name)


class ChunkPolicy:
    """Chunks of 16 actions; chunk n moves left joint 0 to n + 0.01 k."""

    def __init__(self):
        self.calls, self.chunks = [], 0

    def call(self, name, obs=None):
        self.calls.append(name)
        if name == "get_action":
            self.chunks += 1
            return [{"left_arm_joint_state": [self.chunks + 0.01 * k] + [0.0] * 5, "left_ee_joint_state": [1.0],
                     "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [1.0]} for k in range(16)]
        return None


def policy_session(worker, policy):
    s = session(worker)
    s._policy_clients = {"g05": policy}
    s.policies = {"g05": {"url": "ws://127.0.0.1:19999"}}
    s.state = worker.state()
    return s


class PolicySliceTests(unittest.TestCase):
    def setUp(self):
        self.locks = tempfile.TemporaryDirectory()
        self.directory = mock.patch.object(session_module, "POLICY_LOCK_DIR", self.locks.name)
        self.directory.start()

    def tearDown(self):
        self.directory.stop()
        self.locks.cleanup()

    def test_slices_execute_what_one_continuous_run_does(self):
        whole_worker, whole_policy = PolicyWorker(), ChunkPolicy()
        policy_session(whole_worker, whole_policy).run_policy("g05", max_seconds=40 / 25.0)
        sliced_worker, sliced_policy = PolicyWorker(), ChunkPolicy()
        s = policy_session(sliced_worker, sliced_policy)
        for _ in range(8):
            trace = s.run_policy("g05", max_seconds=5 / 25.0)
        self.assertEqual(sliced_worker.commands, whole_worker.commands)
        self.assertEqual(sliced_policy.calls.count("get_action"), 3)  # chunks at ticks 1, 17 and 33
        # Running out of the time given ends nominally, with the limit in the detail.
        self.assertEqual((trace.event.kind, trace.event.detail), ("COMPLETED", {"max_seconds": 0.2}))

    def test_reset_or_moved_arms_drop_the_rest_of_the_chunk(self):
        worker, policy = PolicyWorker(), ChunkPolicy()
        s = policy_session(worker, policy)
        s.run_policy("g05", max_seconds=5 / 25.0)
        s.run_policy("g05", max_seconds=1 / 25.0, reset=True)
        self.assertEqual(policy.chunks, 2)  # a fresh chunk after the reset
        self.assertEqual(worker.commands[-1], 2.0)
        s.move_joints("left_arm", [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        s.run_policy("g05", max_seconds=1 / 25.0)
        self.assertEqual(policy.chunks, 3)  # the arm moved: the rest was stale
        s.run_policy("g05", max_seconds=1 / 25.0)
        self.assertEqual((policy.chunks, worker.commands[-1]), (3, 3.01))  # continues that chunk


class PickPlaceTargetTests(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.s = with_profile(session(self.worker))
        self.s.get_observation = lambda cameras=None: "obs"
        self.s.ik = lambda arm, pose: [0.0] * 6
        original = self.worker.call

        def fingers(name, **arguments):
            result = original(name, **arguments)
            if name == "tick" and arguments["command"][6] == 0.0:
                self.worker.joints[6] = 0.5  # closing holds the object
                result["state"] = self.worker.state()
            return result

        self.worker.call = fingers
        self.plans = []

        def plan(target, arm, coarse):
            self.plans.append((target, [round(v, 3) for v in coarse]))
            tip = [coarse[0], coarse[1], 0.8]
            return ({"arm": arm, "approach": "top_down", "tip": tip, "pregrasp": link6_for_tip(
                [tip[0], tip[1], 0.9], DOWN), "grasp": link6_for_tip(tip, DOWN)}, None, "cam_left_wrist")

        self.s._grasp_from_view = plan

    def test_pick_takes_a_numpy_position(self):
        import numpy as np

        self.s._find = lambda target, near=None, radius=None: self.fail("a position needs no search")
        result = self.s.pick(np.array([-0.1, -0.12, 0.79]))
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.plans, [("object", [-0.1, -0.12, 0.79])])

    def test_pick_near_chooses_among_identical_objects(self):
        # localize_3d's best is the other block; both are nearer the left arm.
        self.s.localize_3d = lambda observation, target: {"position": [-0.25, -0.1, 0.78]}
        self.s._instances = lambda target: [[-0.25, -0.1, 0.78], [-0.05, -0.15, 0.78]]
        self.s._at_any_fingers = lambda position: False
        result = self.s.pick("red block", near=[-0.06, -0.14])
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.plans[0], ("red block", [-0.05, -0.15, 0.78]))
        missing = self.s.pick("red block", near=[0.4, 0.0])
        self.assertEqual((missing["ok"], missing["reason"]), (False, "not found within 0.10 m of [0.4, 0.0]"))

    def test_unsupported_targets_are_errors_not_prompts(self):
        from harness_policy.geometry import Mask
        import numpy as np

        with self.assertRaises(TypeError):
            self.s.place(Mask(np.zeros((2, 2)), [0, 0, 1, 1], 1.0), "left_arm")
        with self.assertRaises(ValueError):
            self.s.place(np.zeros(5), "left_arm")
        self.assertEqual(self.s._as_point({"position": [0.1, 0.2, 0.8]}, "place"), [0.1, 0.2, 0.8])
        self.assertEqual(self.s._as_point((0.1, 0.2), "place"), [0.1, 0.2, 0.765])

    def test_place_takes_a_numpy_position(self):
        import numpy as np

        self.s.pick(np.array([-0.1, -0.12, 0.79]))
        self.s._instances = lambda target: self.fail("a position is not pointed at")
        result = self.s.place(np.array([0.2, -0.1, 0.80]), "left_arm")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["surface"], [0.2, -0.1, 0.8])

    def held(self, below):
        self.s._held = {"left_arm": {"target": "block", "below_tip": below}}
        self.s._gripper_commands["left_arm"] = 0.0

    def test_place_warns_when_it_landed_on_the_table(self):
        # The object's bottom is 0.035 below the fingertips; it meets the table (link6
        # cannot sink below 0.953) 0.022 below the located target top 0.79: beside a
        # stack or inside a container, which only a look tells apart.
        self.worker.floor = 0.953
        self.held(0.035)
        result = self.s.place([0.2, -0.1, 0.79], "left_arm")
        self.assertTrue(result["ok"])
        self.assertTrue(result["contact"])
        self.assertIn("missed", result["warning"])
        self.assertAlmostEqual(result["below_surface_m"], 0.022, places=3)
        self.assertNotIn("left_arm", self.s._held)  # released there

    def test_place_on_the_target_is_ok(self):
        self.worker.floor = 0.79 + 0.035 + 0.15  # the object's bottom meets the target's top
        self.held(0.035)
        result = self.s.place([0.2, -0.1, 0.79], "left_arm")
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["contact"])

    def test_place_approaches_from_lower_when_high_is_out_of_reach(self):
        self.held(0.03)
        surface = [0.2, -0.1, 0.85]
        ceiling = surface[2] + 0.03 + 0.07  # no IK above 7 cm over the object's bottom
        self.s.ik = lambda arm, pose: None if self.s.fingertip_of(pose)[2] > ceiling else [0.0] * 6
        tips = []
        move = self.s.move_eef

        def recording(arm, pose, **kwargs):
            tips.append(round(self.s.fingertip_of(pose)[2], 4))
            return move(arm, pose, **kwargs)

        self.s.move_eef = recording
        result = self.s.place(surface, "left_arm")
        self.assertTrue(result["ok"], result)
        self.assertEqual(tips[0], round(surface[2] + 0.03 + 0.06, 4))

    def test_place_skips_the_other_arms_look_when_it_cannot_reach(self):
        self.held(0.03)
        self.s.ik = lambda arm, pose: [0.0] * 6 if arm == "left_arm" else None
        self.s._instances = lambda target: [[0.2, -0.1, 0.8]]
        self.s._at_fingers = lambda position, arm: False
        self.s._locate_from = lambda arm, target, surface: self.fail("the right arm cannot reach: no look")
        self.assertTrue(self.s.place("bowl", "left_arm")["ok"])


class ReachTests(unittest.TestCase):
    def setUp(self):
        self.s = with_profile(session(FakeWorker()))

    def test_reaches_accepts_the_tilted_approaches_place_uses(self):
        def ik(arm, pose):
            w, x, y, z = pose[3:]
            down = 2 * (x * z - w * y) < -0.99  # approach axis straight down
            forward = abs(pose[3] - FORWARD[0]) < 1e-3 and abs(pose[6] - FORWARD[3]) < 1e-3
            return None if down or forward else [0.0] * 6

        self.s.ik = ik
        self.assertTrue(self.s._reaches("left_arm", [-0.1, 0.05, 0.8]))
        self.assertIsNotNone(self.s._place_approach("left_arm", [-0.1, 0.05, 0.8], 0.02, 0.01))

    def test_without_an_arm_for_both_the_one_reaching_the_object_nearer_the_goal(self):
        # Both arms reach the object; neither reaches the goal at x = 0.9.
        self.s._reaches = lambda arm, position: position[0] < 0.5
        self.assertEqual(self.s._arm_for([-0.05, -0.1, 0.8], [0.9, -0.1, 0.8]), "right_arm")
        # Only the right arm reaches the object: it picks, wherever the goal is.
        self.s._reaches = lambda arm, position: arm == "right_arm" and position[0] < 0.5
        self.assertEqual(self.s._arm_for([-0.15, -0.1, 0.8], [-0.9, -0.1, 0.8]), "right_arm")


class PushTests(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.s = with_profile(session(self.worker))
        self.s.ik = link6_ik
        self.commanded = []
        original = self.worker.call

        def recording(name, **arguments):
            result = original(name, **arguments)
            if name == "tick":
                self.commanded.append(self.s.fingertip_of(self.worker.state()["left_ee_pose"]))
            return result

        self.worker.call = recording

    def test_pushes_along_a_straight_line_at_the_height(self):
        result = self.s.push("left_arm", [0.0, -0.1], [0.1, -0.1], speed=0.05)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["pushed_m"], 0.1, delta=0.005)
        on_line = [tip for tip in self.commanded if abs(tip[2] - 0.78) < 1e-6]
        self.assertGreaterEqual(len(on_line), 50)  # 0.1 m at 0.05 m/s: 2 s = 50 ticks
        self.assertTrue(all(abs(tip[1] + 0.1) < 1e-6 for tip in on_line))
        self.assertEqual(self.worker.joints[6], 0.0)  # closed fingers
        self.assertAlmostEqual(self.commanded[-1][2], 0.78 + 0.06)  # lifted after

    def test_reports_how_far_a_blocked_push_got(self):
        original = self.worker.call

        def wall(name, **arguments):
            if name == "tick":
                command = list(arguments["command"])
                command[0] = min(command[0], 0.05)  # something stops the arm at x = 0.05
                arguments = {**arguments, "command": command}
            return original(name, **arguments)

        self.worker.call = wall
        result = self.s.push("left_arm", [0.0, -0.1], [0.1, -0.1])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "blocked")
        self.assertAlmostEqual(result["pushed_m"], 0.05, delta=0.005)


def pressing_session(worker):
    """A session whose left arm starts with link6 at (-0.1, -0.3, 1.1)."""
    worker.joints[0:3] = [-0.1, -0.3, 1.1]
    s = with_profile(session(worker))
    s.ik = link6_ik
    s._start_joints = list(worker.joints)
    return s


class PressTests(unittest.TestCase):
    def test_presses_count_contacts_and_go_deeper_than_contact(self):
        worker = FakeWorker(floor=0.80 + 0.15 - 0.004)  # the button gives 4 mm below its top at 0.80
        s = pressing_session(worker)
        lowest = []
        original = worker.call

        def recording(name, **arguments):
            if name == "tick":
                lowest.append(arguments["command"][2])
            return original(name, **arguments)

        worker.call = recording
        result = s.press([-0.1, -0.1, 0.80], times=2)
        self.assertEqual((result["ok"], result["presses"], result["arm"]), (True, 2, "left_arm"), result)
        self.assertEqual(len(result["contacts"]), 2)
        self.assertAlmostEqual(result["contacts"][0], 0.796, delta=0.002)
        self.assertLess(min(lowest), 0.796 + 0.15 - 0.007)  # commanded past the contact
        self.assertEqual(worker.joints[0:3], [-0.1, -0.3, 1.1])  # back at the start pose
        self.assertEqual(worker.joints[6], 0.0)  # pressed with closed fingers

    def test_later_presses_go_straight_down_past_the_first_contact(self):
        worker = FakeWorker(floor=0.80 + 0.15 - 0.004)
        s = pressing_session(worker)
        result = s.press([-0.1, -0.1, 0.80], times=1)
        one = worker.step
        worker.step = 0
        result = s.press([-0.1, -0.1, 0.80], times=6)
        self.assertEqual((result["ok"], result["presses"]), (True, 6), result)
        self.assertTrue(all(abs(z - 0.796) < 0.002 for z in result["contacts"]), result["contacts"])
        # Five more presses without the slow probe: about 15 ticks each, not a probe's 40.
        self.assertLess(worker.step - one, 5 * 20)

    def test_the_gripper_closes_on_the_way_and_the_last_press_leaves_at_once(self):
        worker = FakeWorker(floor=0.80 + 0.15 - 0.004)
        s = pressing_session(worker)
        commands = []
        executions = []
        original = worker.call

        def recording(name, **arguments):
            if name == "tick":
                commands.append(list(arguments["command"]))
            return original(name, **arguments)

        worker.call = recording
        run = s._execute

        def counting(skill, *args, **kwargs):
            executions.append(skill)
            return run(skill, *args, **kwargs)

        s._execute = counting
        result = s.press([-0.1, -0.1, 0.80], times=1)
        self.assertTrue(result["ok"], result)
        self.assertEqual(commands[0][6], 0.0)  # closing from the first tick of the approach
        self.assertNotIn("close_gripper", executions)
        self.assertNotIn("hold", executions)  # a single press: no wait for the button
        # Above the button, then the probe: it starts 4 cm above the target, not 6.
        self.assertAlmostEqual(max(c[2] for c in commands[:40] if c[2] < 0.80 + 0.15 + 0.05), 0.80 + 0.15 + 0.04,
                               places=3)

    def test_a_scene_id_presses_that_object(self):
        worker = FakeWorker(floor=0.80 + 0.15 - 0.004)
        s = pressing_session(worker)
        s._eef_scene = [{"id": "o1", "name": "red button", "position": [-0.1, -0.1, 0.80]}]
        s._find = mock.Mock(side_effect=AssertionError("looked up as a name"))
        result = s.press("o1", times=1)
        self.assertEqual((result["ok"], result["presses"]), (True, 1), result)

    def test_a_low_located_object_is_aimed_at_from_the_head_camera(self):
        worker = FakeWorker(floor=0.80 + 0.15 - 0.004)
        s = pressing_session(worker)
        s._eef_scene = [{"id": "o1", "name": "red button", "position": [-0.109, -0.083, 0.765]}]
        mask = mock.Mock(centroid=(320, 240))
        s._eef_masks = ("observation", {"o1": {"cam_head": mask}})
        commanded = []
        original = worker.call

        def recording(name, **arguments):
            if name == "tick":
                commanded.append(arguments["command"][0:3])
            return original(name, **arguments)

        worker.call = recording
        flat = mock.Mock(return_value=[-0.1, -0.1, 0.785])
        with mock.patch("skills.manipulation.pixel_to_plane", flat):
            result = s.press([-0.109, -0.083, 0.765], times=1)  # the triangulated, biased point
        self.assertTrue(result["ok"], result)
        flat.assert_called_once_with("observation", "cam_head", 320, 240, 0.765 + 0.02)
        lowest = min(commanded, key=lambda command: command[2])  # link6 at its lowest: the press
        self.assertAlmostEqual(lowest[0], -0.1, places=3)
        self.assertAlmostEqual(lowest[1], -0.1, places=3)

    def test_meeting_the_table_stops_without_pushing(self):
        worker = FakeWorker(floor=0.765 + 0.005 + 0.15)  # nothing but the table under the fingertips
        s = pressing_session(worker)
        result = s.press([-0.1, -0.1, 0.785], times=3)
        self.assertEqual((result["ok"], result["presses"]), (False, 0), result)
        self.assertIn("met the table", result["reason"])

    def test_nothing_to_press_stops(self):
        s = pressing_session(FakeWorker())
        result = s.press([-0.1, -0.1, 0.80], times=3)
        self.assertEqual((result["ok"], result["presses"]), (False, 0))
        self.assertIn("no contact", result["reason"])


class SceneWorker(FakeWorker):
    """A 40x40 head image: a bright square at (10, 10) until step 5, and a square at
    (30, 30) hidden for steps 2 and 3 only (an arm passing over it)."""

    def image(self):
        import numpy as np

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        if self.step < 5:
            image[5:15, 5:15] = 255
        if self.step not in (2, 3):
            image[25:35, 25:35] = 200
        return image

    def observation(self):
        import numpy as np

        transform = np.eye(4)
        transform[:3, 3] = [0.0, 0.0, 0.0]
        return {"state": self.state(), "images": {"cam_head": png_url(self.image())},
                "intrinsics": {"cam_head": [[40.0, 0, 20], [0, 40.0, 20], [0, 0, 1]]},
                "camera_to_world": {"cam_head": transform.tolist()}}

    def call(self, name, **arguments):
        if name == "observe":
            return self.observation()
        result = super().call(name, **{k: v for k, v in arguments.items() if k != "cameras"})
        if name == "tick" and arguments.get("cameras"):
            result["observation"] = self.observation()
        return result


class WatchSceneTests(unittest.TestCase):
    def test_frames_held_still_with_departures_and_a_contact_sheet(self):
        worker = SceneWorker()
        s = with_profile(session(worker))
        s.point_to = lambda observation, prompt, camera="cam_head", tiles=None: (
            [(10, 10)] if prompt == "cube" else [(30, 30)])
        before = list(worker.joints)
        result = s.watch_scene(0.4, every=0.04, track=["cube", "ball"])
        self.assertEqual(worker.joints, before)  # nothing moved
        self.assertEqual(result["trace"].event.kind, "COMPLETED")
        self.assertEqual([frame["step"] for frame in result["frames"]], list(range(11)))
        self.assertEqual([frame["t"] for frame in result["frames"]][:3], [0.0, 0.04, 0.08])
        departures = {item["name"]: item["left_at"] for item in result["departures"]}
        self.assertEqual(departures, {"cube": 0.2, "ball": None})  # the passing arm does not count
        self.assertEqual(result["departures"][0]["name"], "cube")
        self.assertEqual(len(s._attachments), 1)
        self.assertIn("watch_scene cam_head", s._attachments[0]["caption"])

    def test_unknown_camera(self):
        with self.assertRaises(ValueError):
            with_profile(session(SceneWorker())).watch_scene(1.0, cameras=["cam_top"])


class BranchFlipTests(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.worker.joints[0:6] = [0.0, 0.5, 0.9, -1.2, 0.3, 0.1]
        self.s = with_profile(session(self.worker))
        self.s.state = self.worker.state()  # link6 at (0.0, 0.5, 0.9), DOWN

    def answer(self, joints):
        self.worker.call = lambda name, **arguments: {"success": True, "joints": list(joints)}

    def test_a_full_turn_is_turned_back(self):
        self.answer([0.0, 0.5, 0.9, -1.2 - 2 * math.pi, 0.3, 0.1 + 2 * math.pi])
        joints = self.s.ik("left_arm", [0.01, 0.5, 0.9, *DOWN])
        self.assertEqual([round(v, 6) for v in joints], [0.0, 0.5, 0.9, -1.2, 0.3, 0.1])

    def test_limits_keep_a_turn_that_cannot_be_undone(self):
        self.s.profile["action_low"] = [-10.0] * 14
        self.s.profile["action_high"] = [-7.0] * 3 + [10.0] * 11
        self.answer([-7.8, 0.5, 0.9, -1.2, 0.3, 0.1])  # -7.8 + 2 pi would exceed the upper limit -7
        far = self.s.ik("left_arm", [0.5, 0.0, 1.0, *FORWARD])  # a far target: a large swing is allowed
        self.assertAlmostEqual(far[0], -7.8)

    def test_a_branch_flip_for_a_small_move_is_no_solution(self):
        self.answer([0.0, 0.5, 0.9, 1.8, 0.3, 0.1])  # joint 3 swings 3 rad
        self.assertIsNone(self.s.ik("left_arm", [0.01, 0.5, 0.9, *DOWN]))
        self.assertIsNotNone(self.s.ik("left_arm", [0.3, 0.2, 1.0, *DOWN]))  # far: a real motion


class ApiNameTests(unittest.TestCase):
    def test_rebound_and_deleted_names_are_restored(self):
        def pick(target):
            return "skill"

        api = {"pick": pick, "TABLE_Z": 0.765, "place": print}
        namespace = dict(api, __name__="__program__")
        exec("def pick(m):\n    return 'mine'\ndel place\nx = 1\nTABLE_Z = 0.765", namespace)
        self.assertEqual(restore_api(namespace, api), ["pick", "place"])
        self.assertIs(namespace["pick"], pick)
        self.assertIs(namespace["place"], print)
        self.assertEqual(namespace["x"], 1)
        self.assertEqual(restore_api(namespace, api), [])


if __name__ == "__main__":
    unittest.main()
