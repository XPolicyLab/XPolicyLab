from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.runtime.events import Escalation
from harness_policy.runtime.reactive import ReactiveLayer
from harness_policy.geometry import DOWN, link6_for_tip
from harness_policy.runtime.session import session_class

RoboDojoSession = session_class()

GROUPS = [{"name": "left_arm", "joints": [0, 1, 2, 3, 4, 5], "gripper": 6},
          {"name": "right_arm", "joints": [7, 8, 9, 10, 11, 12], "gripper": 13}]


class FakeWorker:
    """Joints follow the command at once, except joint 2 of the left arm, which cannot go
    below `floor` (contact). link6 of an arm is at (j0, j1, j2) of that arm."""

    def __init__(self, floor=None):
        self.floor = floor
        self.joints = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
        self.step = 0

    def state(self):
        left, right = self.joints[0:3], self.joints[7:10]
        return {"step": self.step, "ended": False, "joint_position": list(self.joints),
                "left_ee_pose": left + list(DOWN), "right_ee_pose": right + list(DOWN)}

    def call(self, name, **arguments):
        if name == "plan":
            goal = arguments["pose"][:3] + [0.0, 0.0, 0.0] if arguments["pose"] else arguments["joints"]
            start = self.joints[0:6]
            return {"success": True, "waypoints": [[s + (g - s) * k / 4 for s, g in zip(start, goal)]
                                                   for k in range(1, 5)]}
        if name == "tick":
            command = list(arguments["command"])
            if self.floor is not None:
                command[2] = max(command[2], self.floor)
            self.joints = command
            self.step += 1
            return {"executed": True, "shield": {"verdict": "pass", "reasons": [], "command": command},
                    "state": self.state()}
        raise AssertionError(name)


def session(worker):
    s = object.__new__(RoboDojoSession)
    s.env, s.episode_id, s.hz, s.default_max_seconds = worker, "e", 25.0, 30.0
    s.reactive = ReactiveLayer(0.005)
    s.groups = {group["name"]: group for group in GROUPS}
    s.handlers, s.policies, s._policy_clients = {}, {}, {}
    s._gripper_commands, s._attachments, s._events, s._live = {}, [], [], None
    s.tip_offset, s.table_z = 0.150, 0.765
    s.state = worker.state()
    return s


class ArrivalTests(unittest.TestCase):
    def test_reaching_the_target_completes(self):
        trace = session(FakeWorker()).move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN))
        self.assertEqual(trace.event.kind, "COMPLETED")

    def test_stopping_short_is_blocked_with_the_gap(self):
        with self.assertRaises(Escalation) as raised:
            session(FakeWorker(floor=0.93)).move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN))
        event = raised.exception.trace.event
        self.assertEqual(event.kind, "BLOCKED")
        self.assertAlmostEqual(event.detail["gap_m"], 0.03, places=4)
        self.assertEqual(event.detail["reached_link6"], [0.1, -0.1, 0.93])

    def test_a_handler_can_accept_blocked(self):
        s = session(FakeWorker(floor=0.93))
        s.handlers["BLOCKED"] = lambda trace: True
        trace = s.move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN))
        self.assertEqual(trace.event.kind, "BLOCKED")
        self.assertTrue(s._events[-1]["handled"])



class PushedWorker(FakeWorker):
    """A steady push on joint 1 of either arm (the other arm leaning on it): each tick the
    joint ends PUSH below its command, so an arm commanded to its measured joints gives
    way PUSH a tick and stays there."""
    PUSH = 0.002

    def call(self, name, **arguments):
        result = super().call(name, **arguments)
        if name == "tick":
            self.joints = list(self.joints)  # the reported command stays as sent
            for index in (1, 8):
                self.joints[index] -= self.PUSH
            result["state"] = self.state()
        return result


class ContactWorker(FakeWorker):
    """Contact stops left joint 2 at `contact` while the command goes on below it."""

    def __init__(self, contact):
        super().__init__()
        self.contact = contact

    def call(self, name, **arguments):
        result = super().call(name, **arguments)
        if name == "tick":
            self.joints = list(self.joints)
            self.joints[2] = max(self.joints[2], self.contact)
            result["state"] = self.state()
        return result


class IdleArmTests(unittest.TestCase):
    def test_an_idle_arm_holds_its_place_while_the_other_moves(self):
        worker = PushedWorker()
        s = session(worker)
        s.move_joints("right_arm", [0.2, 0.3, 0.1, 0.0, 0.0, 0.0])
        parked, step = worker.joints[8], worker.step
        for k in range(4):
            s.move_joints("left_arm", [0.1 * (k % 2), 0.2, 0.1, 0.0, 0.0, 0.0])
        self.assertGreaterEqual(worker.step - step, 15)
        # Commanded to its measured joints it would have given way PUSH every tick.
        self.assertLess(parked - worker.joints[8], 2 * PushedWorker.PUSH)

    def test_an_arm_contact_held_back_does_not_push_on(self):
        worker = ContactWorker(contact=0.93)
        s = session(worker)
        s.handlers["BLOCKED"] = lambda trace: True
        s.move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN))
        commands, call = [], worker.call
        worker.call = lambda name, **a: (commands.append(a["command"]) if name == "tick" else None,
                                         call(name, **a))[1]
        s.move_joints("right_arm", [0.2, 0.3, 0.1, 0.0, 0.0, 0.0])
        self.assertTrue(commands)
        self.assertTrue(all(command[2] == 0.93 for command in commands))

    def test_an_interrupted_motion_holds_where_the_arm_stopped(self):
        worker = LaggingWorker()
        s = session(worker)
        s.move_joints("left_arm", [0.1, 0.1, 0.1, 0.8, 0.8, 0.8], until=lambda view: view["t"] >= 0.08)
        stopped = list(worker.joints[3:6])
        commands, call = [], worker.call
        worker.call = lambda name, **a: (commands.append(a["command"]) if name == "tick" else None,
                                         call(name, **a))[1]
        s.hold(0.2)
        self.assertTrue(commands)
        self.assertTrue(all(command[3:6] == stopped for command in commands))


class LaggingWorker(FakeWorker):
    """The last three arm joints lag behind their command (link6 is at j0..j2 and does
    not depend on them), as a redundant wrist would."""

    def call(self, name, **arguments):
        if name == "tick":
            before = list(self.joints)
            result = super().call(name, **arguments)
            for index in (3, 4, 5):
                self.joints[index] = before[index] + 0.5 * (self.joints[index] - before[index])
            result["state"] = self.state()
            return result
        return super().call(name, **arguments)


class SettleTests(unittest.TestCase):
    def test_move_eef_stops_settling_when_link6_is_there(self):
        worker = LaggingWorker()
        s = session(worker)
        s.plan_path = lambda arm, pose=None, joints=None: [[0.1, -0.1, 0.9, 0.4, 0.4, 0.4]]
        trace = s.move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN))
        self.assertEqual(trace.event.kind, "COMPLETED")
        # One tick to the waypoint, one to find link6 there; not ARRIVAL_SETTLE_TICKS more.
        self.assertLessEqual(worker.step, 3)

class ImageWorker(FakeWorker):
    """Serves a 4x4 head image whose bright pixel column moves one step per tick, and
    IK that maps a link6 position to the first three joints."""

    def __init__(self):
        super().__init__()
        self.observed = 0

    def _observation(self):
        import base64

        import cv2
        import numpy as np

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, min(self.step, 3)] = 255
        url = "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode()
        return {"state": self.state(), "images": {"cam_head": url}, "intrinsics": {"cam_head": np.eye(3).tolist()},
                "camera_to_world": {"cam_head": np.eye(4).tolist()}}

    def call(self, name, **arguments):
        if name == "observe":
            self.observed += 1
            return self._observation()
        if name == "ik":
            return {"success": True, "joints": list(arguments["pose"][:3]) + [0.0, 0.0, 0.0]}
        result = super().call(name, **{k: v for k, v in arguments.items() if k != "cameras"})
        if name == "tick" and arguments.get("cameras"):
            result["observation"] = self._observation()
        return result


class VisualControllerTests(unittest.TestCase):
    def test_controller_sees_fresh_images_and_commands_link6(self):
        worker = ImageWorker()
        s = session(worker)
        s.ik = lambda arm, pose: worker.call("ik", arm=arm, pose=pose)["joints"]
        seen = []

        def servo(obs):
            import numpy as np

            column = int(np.argmax(obs["observation"].images["cam_head"][..., 0].sum(axis=0)))
            seen.append(column)
            if column >= 3:
                return None
            return {"left_eef": [0.1 * column, 0.0, 0.9, *DOWN]}

        s.reactive.budget_seconds = 1.0
        trace = s.run_controller(servo, cameras=["cam_head"], camera_hz=25, max_seconds=2.0)
        self.assertEqual(trace.event.kind, "COMPLETED")
        self.assertEqual(seen[:4], [0, 1, 2, 3])
        self.assertAlmostEqual(worker.joints[0], 0.2)  # last command before the column reached 3

    def test_unreachable_targets_hold_then_fault(self):
        worker = ImageWorker()
        s = session(worker)
        reachable = iter([True, False, False, True] + [False] * 10)
        s.ik = lambda arm, pose: [0.1, 0.0, 0.9, 0.0, 0.0, 0.0] if next(reachable) else None
        s.reactive.budget_seconds = 1.0
        with self.assertRaises(Escalation) as raised:
            s.run_controller(lambda obs: {"left_eef": [0.1, 0.0, 0.9, *DOWN]}, max_seconds=2.0)
        self.assertEqual(raised.exception.trace.event.kind, "REACTIVE_FAULT")
        self.assertIn("5 times in a row", raised.exception.trace.event.detail["reason"])
        self.assertAlmostEqual(worker.joints[0], 0.1)  # the arm held where the last reachable target put it

    def test_hz_holds_the_command_between_calls(self):
        s = session(FakeWorker())
        calls = []

        def fn(obs):
            calls.append(obs["t"])
            return None if len(calls) > 3 else {"left_gripper": 0.5}

        s.reactive.budget_seconds = 1.0
        s.run_controller(fn, hz=5, max_seconds=2.0)
        self.assertEqual(len(calls), 4)
        self.assertAlmostEqual(calls[1] - calls[0], 0.2)


class FaultInjectionTests(unittest.TestCase):
    """App. H: a dropped bridge or sensor ends the execution; no command follows."""

    def _failing(self, method, after):
        worker = FakeWorker()
        calls = {"tick": 0, "sent_after_failure": 0}
        failed = [False]
        original = worker.call

        def call(name, **arguments):
            if name == "tick":
                if failed[0]:
                    calls["sent_after_failure"] += 1
                calls["tick"] += 1
            if name == method and calls["tick"] >= after:
                failed[0] = True
                raise ConnectionError(f"{method}: bridge disconnected")
            return original(name, **arguments)

        worker.call = call
        return worker, calls

    def test_bridge_disconnect_stops_the_motion(self):
        worker, calls = self._failing("tick", after=3)
        s = session(worker)
        with self.assertRaises(ConnectionError):
            s.move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN), speed=0.05)
        self.assertEqual(calls["sent_after_failure"], 0)
        self.assertEqual(calls["tick"], 3)

    def test_sensor_dropout_is_an_error_not_a_stale_view(self):
        worker, calls = self._failing("state", after=0)
        s = session(worker)
        with self.assertRaises(ConnectionError):
            s.get_state()
        self.assertEqual(calls["tick"], 0)


class GripperTests(unittest.TestCase):
    def test_close_reports_whether_something_is_held(self):
        worker = FakeWorker()
        s = session(worker)
        s.profile = {"gripper_empty_opening": 0.185}
        original = worker.call

        def blocked_fingers(name, **arguments):
            result = original(name, **arguments)
            if name == "tick":
                worker.joints[6] = max(worker.joints[6], 0.5)  # an object stops the fingers
                result["state"] = worker.state()
            return result

        worker.call = blocked_fingers
        trace = s.close_gripper("left_arm")
        self.assertEqual(trace.event.detail["holding"], True)
        self.assertTrue(s.holding("left_arm"))
        worker.call = original
        worker.joints[6] = 0.185
        s.state = worker.state()
        self.assertFalse(s.holding("left_arm"))


class PickPlaceTests(unittest.TestCase):
    """pick and place on the fake worker, with planning stubbed: retries, the held
    object's depth, and failures absorbed instead of ending the program."""

    def setUp(self):
        self.worker = FakeWorker()
        self.s = session(self.worker)
        self.s.profile = {"gripper_empty_opening": 0.185,
                          "arm_bases": {"left_arm": [-0.3, -0.45, 0.765], "right_arm": [0.3, -0.45, 0.765]}}
        self.s._held = {}
        self.s.get_observation = lambda cameras=None: "obs"
        self.closes = 0
        original = self.worker.call

        def fingers(name, **arguments):
            result = original(name, **arguments)
            if name == "tick" and arguments["command"][6] == 0.0:
                # The first close grasps nothing; later closes hold the object.
                self.worker.joints[6] = 0.185 if self.closes == 0 else 0.5
                result["state"] = self.worker.state()
            return result

        self.worker.call = fingers
        grasp = {"arm": "left_arm", "approach": "top_down", "tip": [0.1, -0.1, 0.8],
                 "pregrasp": link6_for_tip([0.1, -0.1, 0.9], DOWN), "grasp": link6_for_tip([0.1, -0.1, 0.8], DOWN)}
        self.plans = []

        def plan(target, arm, coarse):
            self.plans.append((arm, [round(v, 3) for v in coarse]))
            return dict(grasp), None, "cam_left_wrist"

        self.s._find = lambda target, near=None, radius=None: [-0.1, -0.1, 0.8]  # nearer the left arm
        self.s._grasp_from_view = plan
        close = self.s.close_gripper

        def counting_close(arm, max_seconds=1.5):
            trace = close(arm, max_seconds)
            self.closes += 1
            return trace

        self.s.close_gripper = counting_close

    def test_pick_retries_until_the_object_is_held(self):
        result = self.s.pick("block")
        self.assertEqual((result["ok"], result["attempts"], result["arm"]), (True, 2, "left_arm"))
        self.assertEqual(self.plans, [("left_arm", [-0.1, -0.1, 0.8])] * 2)  # nearer arm, same object
        self.assertAlmostEqual(self.s._held["left_arm"]["below_tip"], 0.8 - 0.765)

    def test_place_lowers_the_object_bottom_to_the_clearance(self):
        self.s.pick("block")
        targets = []
        move = self.s.move_eef

        def recording(arm, pose, **kwargs):
            targets.append(self.s.fingertip_of(pose)[2])
            return move(arm, pose, **kwargs)

        self.s.move_eef = recording
        self.s.ik = lambda arm, pose: [0.0] * 6
        result = self.s.place([0.2, -0.1, 0.80], "left_arm", clearance=0.01)
        self.assertTrue(result["ok"])
        # Above, the approach start, then a probe past the estimated release height.
        release = 0.80 + (0.8 - 0.765) + 0.01
        self.assertAlmostEqual(min(targets), release - 0.04, places=3)
        self.assertIn(round(release + 0.03, 3), [round(t, 3) for t in targets])
        self.assertNotIn("left_arm", self.s._held)
        self.assertFalse(self.s.holding("left_arm"))  # opened: holds nothing

    def test_failures_inside_do_not_end_the_program(self):
        from harness_policy.runtime.events import Escalation

        def unreachable(*args, **kwargs):
            raise Escalation(self.s._failed("move_eef", {}, "no path"))

        self.s.move_eef = unreachable
        result = self.s.pick("block", attempts=2)
        self.assertEqual((result["ok"], result["reason"]), (False, "pregrasp not reached"))


    def test_destination_selects_an_arm_that_reaches_both(self):
        # The object is nearer the left arm, but only the right arm reaches the destination.
        self.s._reaches = lambda arm, position: arm == "right_arm" or position[0] < 0.2
        result = self.s.pick("block", destination=[0.4, -0.1, 0.8])
        self.assertEqual(result["arm"], "right_arm")
        self.assertEqual(self.plans[0][0], "right_arm")

    def test_place_does_not_park_the_other_arm_after_it_stepped_back(self):
        self.s.pick("block")
        self.s.ik = lambda arm, pose: [0.0] * 6
        self.s._instances = lambda target: [[0.2, -0.1, 0.8]]
        self.s._locate_from = lambda arm, target, surface: ([0.21, -0.1, 0.8], False)
        parks = []
        self.s._park = lambda arm, **kwargs: parks.append(arm)
        result = self.s.place("bowl", "left_arm")
        self.assertTrue(result["ok"])
        self.assertEqual(result["surface"], [0.21, -0.1, 0.8])
        self.assertEqual(parks, [])

    def test_place_reports_when_only_the_other_arm_reaches(self):
        self.s.pick("block")
        self.s.ik = lambda arm, pose: None if arm == "left_arm" else [0.0] * 6
        result = self.s.place([0.5, -0.1, 0.8], "left_arm")
        self.assertFalse(result["ok"])
        self.assertIn("right_arm can", result["reason"])
        self.assertIn("left_arm", self.s._held)  # still holding



class MovingPickTests(unittest.TestCase):
    """An object on a conveyor at 0.1 m/s along +x: pick intercepts it."""

    def test_intercepts_and_holds_a_moving_object(self):
        worker = FakeWorker()
        s = session(worker)
        s.profile = {"gripper_empty_opening": 0.185,
                     "arm_bases": {"left_arm": [-0.3, -0.45, 0.765], "right_arm": [0.3, -0.45, 0.765]}}
        s._held = {}
        s.hz = 25.0
        start = [-0.3, -0.1, 0.80]

        def where(step):
            return [start[0] + 0.1 * step / 25.0, start[1], start[2]]

        s._find = lambda target, near=None, radius=None: where(worker.step)
        s._grasp_from_view = lambda target, arm, coarse: (None, "no grasp near the object", "cam_left_wrist")

        def looking(target, arm, coarse):
            s.hold(1.0)  # the wrist look takes a second; the object moves on
            return None, "no grasp near the object", "cam_left_wrist"

        s._grasp_from_view = looking
        s.get_observation = lambda cameras=None: "obs"
        s._mask_near = lambda observation, camera, target, position: None  # no mask: grasp at the located point
        def ik(arm, pose):
            # Only top-down poses (approach axis -z): the fake worker reports DOWN.
            w, x, y, z = pose[3:]
            down = 2 * (x * z - w * y) < -0.99
            return list(pose[:3]) + [0.0, 0.0, 0.0] if down else None  # link6 = (j0, j1, j2)

        s.ik = ik
        original = worker.call

        def fingers(name, **arguments):
            result = original(name, **arguments)
            if name == "tick":
                tip = s.fingertip_of(worker.state()["left_ee_pose"])
                closed = arguments["command"][6] < 0.5
                if closed and math.dist(tip[:2], where(worker.step)[:2]) < 0.01:
                    worker.joints[6] = 0.5  # the fingers close on the object
                result["state"] = worker.state()
            return result

        import math
        worker.call = fingers
        result = s.pick("toy car", attempts=1)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["arm"], result["approach"]), ("left_arm", "intercept"))
        self.assertAlmostEqual(result["tries"][0]["velocity"][0], 0.1, delta=0.01)

class OutlineHeightTests(unittest.TestCase):
    """_measure: the outline's depth from how much it shrinks when the camera backs off."""

    def test_height_from_the_shrinking_outline(self):
        import numpy as np
        from harness_policy.geometry import Mask

        s = session(FakeWorker())
        # A camera looking straight down from z = 1.10 (camera z axis = world -z), and
        # an outline 0.05 m across at height 0.85: 0.25 m below the near view, 0.35 m
        # below the far view (SILHOUETTE_BACK_M = 0.10).
        focal, size = 1000.0, 800

        def view(camera_z):
            transform = np.eye(4)
            transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
            transform[:3, 3] = [0.0, 0.0, camera_z]
            half = int(round(focal * 0.025 / (camera_z - 0.85)))
            array = np.zeros((size, size), dtype=bool)
            array[400 - half:400 + half, 400 - half:400 + half] = True
            obs = type("Obs", (), {})()
            obs.camera_to_world = {"cam_left_wrist": transform}
            obs.intrinsics = {"cam_left_wrist": np.array([[focal, 0, 400], [0, focal, 400], [0, 0, 1.0]])}
            obs.images = {"cam_left_wrist": np.zeros((size, size, 3), dtype=np.uint8)}
            return obs, Mask(array, [0, 0, 1, 1], 1.0)

        near, near_mask = view(1.10)
        far, far_mask = view(1.20)
        s._look = lambda arm, position, tilts=None: (near, "cam_left_wrist", 0.0, [0.0, 0.0, 1.25, *DOWN])
        s._mask_near = lambda observation, camera, target, position: near_mask
        s.ik = lambda arm, pose: [0.0] * 6
        s.move_eef = lambda arm, pose, **kwargs: type("T", (), {"ok": True})()
        s.get_observation = lambda cameras=None: far
        s.segment_point = lambda observation, xy, camera="cam_head", labels=None: [far_mask]
        _obs, _camera, mask, height, tilt = s._measure("left_arm", "mug", [0.0, 0.0, 0.80])
        self.assertIs(mask, near_mask)
        self.assertEqual(tilt, 0.0)
        self.assertAlmostEqual(height, 0.85, delta=0.01)



class GripperInImageTests(unittest.TestCase):
    def test_pixels_around_a_holding_gripper_are_excluded(self):
        import numpy as np

        worker = FakeWorker()
        s = session(worker)
        s.move_eef("left_arm", [0.0, 0.0, 1.05] + list(DOWN))  # fingertips at (0, 0, 0.90)
        s._gripper_commands = {"left_arm": 0.0}
        worker.joints[6] = 0.5  # closed on something
        s.state = worker.state()
        s.profile = {"gripper_empty_opening": 0.185}
        transform = np.eye(4)
        transform[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking straight down
        transform[:3, 3] = [0.0, 0.0, 1.40]  # 0.5 m above the fingertips
        obs = type("Obs", (), {})()
        obs.camera_to_world = {"cam_head": transform}
        obs.intrinsics = {"cam_head": np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])}
        # HELD_NEAR_M = 0.10 at 0.5 m: 100 px around the fingertips' pixel (320, 240).
        self.assertTrue(s._on_gripper_in_image(obs, "cam_head", (380.0, 240.0)))
        self.assertFalse(s._on_gripper_in_image(obs, "cam_head", (440.0, 240.0)))

class SolidCenterTests(unittest.TestCase):
    def points(self, xy):
        import numpy as np
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([xy, np.full(len(xy), 0.78)])

    def test_solid_ring_and_arc(self):
        import numpy as np
        grid = np.array([(x, y) for x in np.linspace(-0.03, 0.03, 25) for y in np.linspace(-0.015, 0.015, 13)])
        self.assertTrue(RoboDojoSession._solid_center(self.points(grid)))
        angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        ring = np.column_stack([0.04 * np.cos(angles), 0.03 * np.sin(angles)])
        self.assertTrue(RoboDojoSession._solid_center(self.points(ring)))  # squeezed from outside
        ends = grid[np.abs(grid[:, 0]) > 0.02]  # two ends with a gap at the center
        self.assertFalse(RoboDojoSession._solid_center(self.points(ends)))



class WrapUpTests(unittest.TestCase):
    def test_arms_return_to_their_start_joints_when_the_profile_asks(self):
        worker = FakeWorker()
        s = session(worker)
        s._start_joints = list(worker.joints)
        s.move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN))
        s.profile = {"return_to_start": True}
        # Both grippers open first (RoboDojo scores most tasks only with every gripper open).
        self.assertEqual(s._wrap_up(), ["COMPLETED"] * 4)
        self.assertEqual([round(v, 6) for v in worker.joints[0:6]], [0.0] * 6)

    def test_nothing_moves_without_the_profile_flag(self):
        worker = FakeWorker()
        s = session(worker)
        s._start_joints = [0.0] * 14
        s.move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN))
        s.profile = {}
        self.assertEqual(s._wrap_up(), [])
        self.assertEqual(worker.joints[0:3], [0.1, -0.1, 0.9])

class ClearViewTests(unittest.TestCase):
    def test_arms_move_out_to_the_sides(self):
        worker = FakeWorker()
        s = session(worker)
        s.profile = {"arm_bases": {"left_arm": [-0.3, -0.45, 0.765], "right_arm": [0.3, -0.45, 0.765]}}
        targets = []
        original = worker.call

        def recording(name, **arguments):
            if name == "plan":
                targets.append((arguments["arm"], [round(v, 3) for v in arguments["pose"][:3]]))
            return original(name, **arguments)

        worker.call = recording
        s.clear_view()
        tips = {arm: s.fingertip_of(list(pose) + [0.707, 0.0, 0.0, 0.707]) for arm, pose in targets}
        self.assertEqual([arm for arm, _ in targets], ["left_arm", "right_arm"])
        for arm, x in (("left_arm", -0.55), ("right_arm", 0.55)):
            self.assertAlmostEqual(tips[arm][0], x, places=2)
            self.assertAlmostEqual(tips[arm][1], -0.25, places=2)
            self.assertAlmostEqual(tips[arm][2], 1.015, places=2)


class SpeedTests(unittest.TestCase):
    def test_speed_sets_the_duration(self):
        # link6 travels |(0.1, -0.1, 0.9)| = 0.911 m at 0.1 m/s: 9.1 s = 228 ticks at 25 Hz.
        worker = FakeWorker()
        trace = session(worker).move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN), speed=0.1)
        self.assertEqual(trace.event.kind, "COMPLETED")
        self.assertIn(trace.ticks, (228, 229))
        self.assertAlmostEqual(worker.joints[2], 0.9)

    def test_fast_paths_respect_the_joint_step(self):
        worker = FakeWorker()
        before = list(worker.joints)
        steps = []
        original = worker.call

        def recording(name, **arguments):
            if name == "tick":
                steps.append(max(abs(a - b) for a, b in zip(arguments["command"], before)))
                before[:] = arguments["command"]
            return original(name, **arguments)

        worker.call = recording
        session(worker).move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN), speed=50.0)
        self.assertLessEqual(max(steps), 0.12 + 1e-9)
        self.assertEqual(len(steps), 8)  # 0.9 rad on joint 2 at 0.12 rad per tick

    def test_motions_are_timed_within_the_shield_bounds(self):
        # 3.0 rad/s at 25 Hz: at most 0.108 rad per tick after the 0.9 margin;
        # 10 rad/s^2: at most 0.0144 rad per tick per tick.
        worker = FakeWorker()
        s = session(worker)
        s.joint_speed_limit, s.joint_acceleration_limit = 3.0, 10.0
        start = list(worker.joints)
        commands = []
        original = worker.call

        def recording(name, **arguments):
            if name == "tick":
                commands.append(list(arguments["command"]))
            return original(name, **arguments)

        worker.call = recording
        trace = s.move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN), speed=50.0)
        self.assertEqual(trace.event.kind, "COMPLETED")
        self.assertAlmostEqual(worker.joints[2], 0.9)
        points = [start] + commands
        steps = [[b - a for a, b in zip(p, q)] for p, q in zip(points, points[1:])]
        self.assertLessEqual(max(abs(d) for step in steps for d in step), 0.108 + 1e-9)
        changes = [abs(b - a) for p, q in zip([[0.0] * 14] + steps, steps) for a, b in zip(p, q)]
        self.assertLessEqual(max(changes), 0.0144 + 1e-9)

    def test_paths_within_the_bounds_keep_their_timing(self):
        s = session(FakeWorker())
        s.joint_speed_limit, s.joint_acceleration_limit = 3.0, 10.0
        path = [[0.001 * k * k] + [0.0] * 5 for k in range(1, 6)] + [[0.025 + 0.009 * k] + [0.0] * 5 for k in range(1, 4)]
        self.assertIs(s._within_limits([0.0] * 6, path), path)

    def test_speed_must_be_positive(self):
        with self.assertRaises(ValueError):
            session(FakeWorker()).move_eef("left_arm", [0.1, -0.1, 0.9] + list(DOWN), speed=0)


if __name__ == "__main__":
    unittest.main()
