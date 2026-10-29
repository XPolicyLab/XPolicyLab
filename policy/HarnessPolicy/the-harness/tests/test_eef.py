"""The eef skill: Z1's end-effector tools by object id on bounded executions."""

from pathlib import Path
import math
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.embodiments import load_embodiment  # noqa: E402
from harness_policy.geometry import DOWN, TABLE_Z, Mask, at_border, project  # noqa: E402
from harness_policy.runtime.events import Event, ExecutionTrace  # noqa: E402
from harness_policy.runtime.session import session_class  # noqa: E402
from skills import eef  # noqa: E402

BASES = {"left_arm": [-0.3, -0.45, 0.765, 1, 0, 0, 0], "right_arm": [0.3, -0.45, 0.765, 1, 0, 0, 0]}


def item(oid, x, y, z=0.80, length=0.05, width=0.04, angle=0.0, name="block"):
    return {"id": oid, "name": name, "position": [x, y, z], "triangulated": True, "score": 0.9,
            "length": length, "width": width, "angle_deg": angle, "container": False, "held_by": None,
            "seen_in": ["cam_head", "cam_left_wrist"], "reach": {"left_arm": "down", "right_arm": "down"}}


def trace(skill="move_eef", kind="COMPLETED"):
    return ExecutionTrace(skill, {}, 0, 1, Event(kind, skill))


class Stub:
    """A session with every skill's methods, motions recorded instead of executed: each
    move_eef puts link6 where it was sent."""

    def __new__(cls, reach=None, holding=True):
        s = object.__new__(session_class())
        s.profile = {"arm_bases": BASES}
        s.tip_offset, s.table_z = 0.150, TABLE_Z
        s.state = {"step": 0, "joint_position": [0.0] * 14,
                   "left_ee_pose": [-0.3, -0.25, 1.05, *DOWN], "right_ee_pose": [0.3, -0.25, 1.05, *DOWN]}
        s._gripper_commands, s._attachments, s._events = {}, [], []
        s.moves = []
        reach = reach or (lambda x, y: {"left_arm": "down", "right_arm": "down"})
        s._eef_reach = reach

        def move_eef(arm, pose, speed=None, **_):
            s.moves.append((arm, s.fingertip_of(pose)))
            s.state[f"{arm.split('_')[0]}_ee_pose"] = list(pose)
            return trace()

        s.move_eef = move_eef
        s.open_gripper = lambda arm, *a, **k: (s._gripper_commands.__setitem__(arm, 1.0), trace("open_gripper"))[1]
        s.close_gripper = lambda arm, *a, **k: (s._gripper_commands.__setitem__(arm, 0.0), trace("close_gripper"))[1]
        s.hold = lambda seconds: trace("hold")
        s.holding = lambda arm: holding
        s._eef_align = lambda obj, arm, plan: (plan["grasp_z"], "")
        s._eef_rose = lambda obj, arm, lift: None
        s._eef_air_offset = lambda arm, grip: grip
        return s


class GraspTests(unittest.TestCase):
    def test_across_the_width_when_the_jaws_span_it(self):
        s = Stub()
        obj = item("o1", 0.0, -0.2, z=0.80, width=0.04, angle=30.0)
        s.__dict__["_eef_scene"] = [obj]
        plan = s._eef_grasp(obj, "left_arm", "auto")
        self.assertTrue(plan["how"].startswith("across its width"))
        self.assertAlmostEqual(plan["yaw_deg"], -60.0)
        self.assertAlmostEqual(plan["grasp_z"], TABLE_Z + (0.80 - TABLE_Z) / 2)
        self.assertEqual(plan["pitch_deg"], 0.0)

    def test_a_wide_object_by_the_edge_with_room(self):
        s = Stub()
        bowl = item("o1", 0.0, -0.2, z=0.81, length=0.16, width=0.14, angle=0.0, name="bowl")
        near = item("o2", 0.0, -0.08, name="cup")  # beyond the far edge (y = -0.13)
        s.__dict__["_eef_scene"] = [bowl, near]
        plan = s._eef_grasp(bowl, "left_arm", "auto")
        self.assertTrue(plan["how"].startswith("by its edge"))
        self.assertAlmostEqual(plan["y"], -0.27)  # the near edge: the far one has the cup beside it
        self.assertAlmostEqual(plan["grasp_z"], max(0.81, eef.WALL_TOP_MIN) - 0.01)

    @staticmethod
    def spatula():
        """Footprint points of a spatula lying flat: a 9 x 9 cm blade and a 3 cm handle."""
        import numpy as np

        blade = [(x, y) for x in np.arange(-0.045, 0.045, 0.003) for y in np.arange(-0.30, -0.21, 0.003)]
        handle = [(x, y) for x in np.arange(-0.015, 0.015, 0.003) for y in np.arange(-0.21, -0.07, 0.003)]
        return np.array([(x, y, 0.79) for x, y in blade + handle])

    def test_a_wide_flat_object_by_a_thin_part(self):
        s = Stub()
        spatula = item("o1", 0.0, -0.21, z=0.79, length=0.23, width=0.09, angle=90.0, name="spatula")
        s.__dict__["_eef_scene"] = [spatula]
        s._eef_footprint = lambda obj: self.spatula()
        plan = s._eef_grasp(spatula, "left_arm", "auto")
        self.assertTrue(plan["how"].startswith("across a thin part"), plan)
        self.assertLess(abs(plan["x"]), 0.015)  # on the handle
        self.assertGreater(plan["y"], -0.20)
        self.assertLess(abs(plan["yaw_deg"]), 10.0)  # the jaws close across the handle
        bowl = dict(spatula, container=True)  # a container is still taken by its wall
        self.assertTrue(s._eef_grasp(bowl, "left_arm", "auto")["how"].startswith("by its edge"))
        self.assertTrue(s._eef_grasp(bowl, "left_arm", "part")["how"].startswith("across a thin part"))

    def test_beyond_reach_is_refused_without_moving(self):
        s = Stub(reach=lambda x, y: {"left_arm": None, "right_arm": "down"})
        s.__dict__["_eef_scene"] = [item("o1", 0.5, 0.1)]
        result = s.eef_pick("left", "o1")
        self.assertFalse(result["ok"])
        self.assertIn("beyond", result["reason"])
        self.assertEqual(s.moves, [])

    def test_pick_comes_down_slowly_and_records_the_grip(self):
        s = Stub()
        s.__dict__["_eef_scene"] = [item("o1", 0.0, -0.2, z=0.80)]
        result = s.eef_pick("left_arm", "o1")
        self.assertTrue(result["ok"], result)
        heights = [round(tip[2], 4) for _, tip in s.moves]
        grasp_z = result["grasp_point"][2]
        self.assertEqual(heights[-5:-1], [round(grasp_z + d, 4) for d in (0.03, 0.02, 0.01, 0.0)])
        self.assertAlmostEqual(heights[-1], grasp_z + eef.DEFAULT_LIFT_M, places=3)
        self.assertEqual(s._eef_grips()["left_arm"][:2], (0.0, 0.0))
        self.assertEqual(s.__dict__["_eef_scene"][0]["held_by"], "left_arm")

    def test_without_a_measured_gripper_the_rise_decides(self):
        """Under the XPolicyLab contract the gripper value is the last command: whether the
        object rose with the jaws (SAM 3) decides, the wrist camera's look only when the
        rise cannot be measured."""
        for rose, look, held in ((True, False, True), (False, True, False), (None, True, True), (None, False, False)):
            with self.subTest(rose=rose, look=look):
                s = Stub(holding=look)
                s.profile["gripper_feedback"] = "commanded"
                s._eef_rose = lambda obj, arm, lift, rose=rose: rose
                s.__dict__["_eef_scene"] = [item("o1", 0.0, -0.2)]
                result = s.eef_pick("left_arm", "o1")
                self.assertEqual(result["ok"], held, result)
                self.assertIn("SAM 3" if rose is not None else "wrist camera", result["check"])

    def test_failed_picks_are_not_repeated(self):
        s = Stub(holding=False)
        s.__dict__["_eef_scene"] = [item("o1", 0.0, -0.2)]
        self.assertFalse(s.eef_pick("left_arm", "o1")["ok"])
        moves = len(s.moves)
        again = s.eef_pick("left_arm", "o1")
        self.assertIn("same pick just failed", again["reason"])
        self.assertEqual(len(s.moves), moves)


class ReleaseTests(unittest.TestCase):
    def test_landing_in_a_container_onto_an_object_or_at_a_point(self):
        s = Stub()
        basket = item("o1", 0.0, -0.1, z=0.82, length=0.20, width=0.16, name="basket")
        block = item("o2", 0.1, -0.2, z=0.79)
        s.__dict__["_eef_scene"] = [basket, block]
        x, y, surface, words = s._eef_landing("left_arm", "o1", None, None, None)
        self.assertEqual((x, y), (0.0, -0.1))
        self.assertAlmostEqual(surface, TABLE_Z + eef.CONTAINER_FLOOR_M)
        self.assertIn("into o1", words)
        self.assertEqual(s._eef_landing("left_arm", None, "o2", None, None)[2], 0.79)
        self.assertEqual(s._eef_landing("left_arm", None, None, 0.1, -0.3)[:3], (0.1, -0.3, TABLE_Z))

    def test_a_container_beyond_straight_down_reach_takes_the_nearest_spot_inside(self):
        def reach(x, y):
            return {"left_arm": "down" if y < -0.10 else "tilted", "right_arm": "down"}

        s = Stub(reach=reach)
        s.__dict__["_eef_scene"] = [item("o1", -0.3, -0.05, length=0.30, width=0.30, name="box")]
        x, y, _surface, _words = s._eef_landing("left_arm", "o1", None, None, None)
        self.assertAlmostEqual(x, -0.3)
        self.assertAlmostEqual(y, -0.11)  # 6 cm toward the arm's base: the first spot it reaches straight down

    def test_release_sets_the_object_down_with_the_grip_offset(self):
        s = Stub()
        held = item("o1", 0.0, -0.2)
        held["held_by"] = "left_arm"
        s.__dict__["_eef_scene"] = [held]
        s._eef_grips()["left_arm"] = (0.01, 0.0, 0.03)
        s._gripper_commands["left_arm"] = 0.0
        result = s.eef_release("left_arm", x=0.1, y=-0.3, withdraw=False)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["release_z"], TABLE_Z + 0.03 + eef.RELEASE_GAP_M)
        lowest = min(s.moves, key=lambda move: move[1][2])[1]
        self.assertAlmostEqual(lowest[0], 0.09, places=4)  # 0.1 minus the grip offset
        self.assertAlmostEqual(lowest[2], result["release_z"], places=4)
        self.assertEqual(s._gripper_commands["left_arm"], 1.0)
        self.assertNotIn("left_arm", s._eef_grips())
        self.assertIsNone(held["held_by"])

    def test_release_refuses_empty_jaws(self):
        s = Stub(holding=False)
        result = s.eef_release("right", x=0.1, y=-0.3)
        self.assertFalse(result["ok"])
        self.assertIn("holds nothing", result["reason"])


class PushTests(unittest.TestCase):
    def test_push_starts_behind_and_slides_in_short_pieces(self):
        s = Stub()
        s.__dict__["_eef_scene"] = [item("o1", 0.0, -0.2, length=0.06, width=0.06)]
        result = s.eef_push("right_arm", "o1", 0.0, -0.1)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["start"][1], -0.2 - 0.03 - eef.PUSH_GAP_M, places=3)
        self.assertAlmostEqual(result["end"][1], -0.1 - 0.03, places=3)
        slides = [tip for _, tip in s.moves if abs(tip[2] - eef.PUSH_Z) < 1e-6]
        steps = [math.dist(a[:2], b[:2]) for a, b in zip(slides, slides[1:])]
        self.assertTrue(steps and max(steps) <= eef.PUSH_SEGMENT_M + 1e-6)
        self.assertEqual(s._gripper_commands["right_arm"], 0.0)

    def test_push_range(self):
        s = Stub()
        s.__dict__["_eef_scene"] = [item("o1", 0.0, -0.2)]
        self.assertFalse(s.eef_push("left_arm", "o1", 0.0, -0.199)["ok"])
        self.assertEqual(s.moves, [])


class MoveTipTests(unittest.TestCase):
    def test_free_motion_by_the_jaws_centre(self):
        s = Stub()
        result = s.eef_move_tip("right", 0.2, -0.2, 0.95, yaw_deg=90.0, gripper=0)
        self.assertTrue(result["ok"], result)
        self.assertEqual(s._gripper_commands["right_arm"], 0.0)
        arm, tip = s.moves[-1]
        self.assertEqual(arm, "right_arm")
        self.assertEqual([round(v, 4) for v in tip], [0.2, -0.2, 0.95])
        self.assertLess(result["arrival_m"], 1e-6)


class TurnReportTests(unittest.TestCase):
    """After every program the located names are measured again (Z1's every-turn tracking)."""

    def stub(self, objects):
        import numpy as np

        s = Stub()
        image = np.zeros((48, 64, 3), np.uint8)
        mask = np.zeros((48, 64), bool)
        mask[10:20, 10:20] = True
        observation = type("O", (), {"images": {"cam_head": image}})()
        s.get_observation = lambda cameras=None: observation
        s.calls = []

        def scene_from(obs, names):
            s.calls.append(list(names))
            s.__dict__["_eef_scene"] = [dict(o) for o in objects]
            return s.__dict__["_eef_scene"], {"cam_head": [("o1 block", Mask(mask, [10, 10, 20, 20], 0.9))]}, []

        s._eef_scene_from = scene_from
        return s

    def test_nothing_before_the_first_locate(self):
        self.assertIsNone(self.stub([])._eef_turn_report())

    def test_the_scene_is_measured_again_relabelled_and_outlined(self):
        s = self.stub([item("o1", 0.0, -0.2), item("o2", 0.1, -0.2, name="cup")])
        s.__dict__["_eef_names"] = ["block", "cup"]
        s.state["step"] = 40
        report = s._eef_turn_report()
        self.assertEqual(s.calls, [["block", "cup"]])
        self.assertIn("Scene this turn", report["text"])
        self.assertIn("o1 'block'", report["text"])
        self.assertIn("o2 'cup'", report["text"])
        self.assertTrue(report["images"]["cam_head"].startswith("data:image/jpeg;base64,"))
        s._eef_turn_report()  # nothing moved since: the measurement is reused
        self.assertEqual(len(s.calls), 1)

    def test_grip_offset_and_a_slip(self):
        held = item("o1", -0.29, -0.25, z=0.95)
        held["held_by"] = "left_arm"
        s = self.stub([held])
        s.__dict__["_eef_names"] = ["block"]
        s._gripper_commands["left_arm"] = 0.0
        s._eef_grips()["left_arm"] = (0.01, 0.0, 0.03)
        self.assertIn("Grip, left_arm holds o1 'block'", s._eef_turn_report()["text"])
        s.__dict__["_eef_scene"] = [held]
        s.state["step"] = 80
        s._eef_scene_from = lambda obs, names: ([item("o1", 0.1, -0.2, z=0.79)], {}, [])
        self.assertIn("may have slipped out", s._eef_turn_report()["text"])
        # Seen by one camera only, its height is assumed: no slip is claimed from it.
        s.__dict__["_eef_scene"] = [held]
        s.state["step"] = 120
        one_view = dict(item("o1", 0.1, -0.2, z=eef.FLAT_Z), triangulated=False, seen_in=["cam_left_wrist"])
        s._eef_scene_from = lambda obs, names: ([one_view], {}, [])
        self.assertNotIn("may have slipped out", s._eef_turn_report()["text"])

    def test_jaws_opened_since_hold_nothing(self):
        s = self.stub([item("o1", 0.0, -0.2)])
        s.__dict__["_eef_names"] = ["block"]
        s._eef_grips()["left_arm"] = (0.0, 0.0, 0.02)
        s._gripper_commands["left_arm"] = 1.0
        self.assertNotIn("Grip", s._eef_turn_report()["text"])
        self.assertNotIn("left_arm", s._eef_grips())


class SceneTests(unittest.TestCase):
    """locate's scene: names matched to the same pixels are one object; an object seen by
    one camera has its height measured from above before the jaws close."""

    @staticmethod
    def mask(rows, cols, score=0.9):
        import numpy as np

        array = np.zeros((48, 64), bool)
        array[rows[0]:rows[1], cols[0]:cols[1]] = True
        return Mask(array, [cols[0], rows[0], cols[1], rows[1]], score)

    def scene(self, measured):
        s = Stub()
        s._eef_measure = lambda observation, name: ([dict(entry, masks=dict(entry["masks"]))
                                                     for entry in measured.get(name, [])], "")
        return s._eef_scene_from(None, list(measured))

    def test_names_matched_to_the_same_pixels_are_one_object(self):
        tiara = dict(item("", -0.35, -0.14, name="tiara"), score=0.6, masks={"cam_head": self.mask((10, 20), (10, 30))})
        crown = dict(item("", -0.35, -0.14, name="crown"), score=0.9, masks={"cam_head": self.mask((10, 20), (11, 30))})
        objects, outlines, _ = self.scene({"tiara": [tiara], "crown": [crown]})
        self.assertEqual([(o["id"], o["name"], o["aliases"]) for o in objects], [("o1", "crown", ["tiara"])])
        self.assertEqual(len(outlines["cam_head"]), 1)
        self.assertIn("also what 'tiara' matched", Stub()._eef_line(objects[0]))

    def test_an_object_in_a_holder_stays_separate(self):
        holder = dict(item("", 0.0, -0.2, name="holder"), masks={"cam_head": self.mask((10, 30), (10, 40))})
        coin = dict(item("", 0.0, -0.2, name="coin"), masks={"cam_head": self.mask((12, 18), (20, 26))})
        objects, _, _ = self.scene({"holder": [holder], "coin": [coin]})
        self.assertEqual([o["name"] for o in objects], ["holder", "coin"])

    def test_a_thin_part_grasp_does_not_follow_the_centre(self):
        s = Stub()
        del s._eef_align  # the real one
        s.get_observation = lambda cameras=None: None
        spatula = item("o1", -0.05, -0.17, z=0.79, length=0.23, width=0.09, name="spatula")
        s._eef_measure = lambda observation, name: ([dict(spatula, position=[-0.068, -0.172, 0.79])], "")
        plan = {"x": 0.065, "y": -0.161, "yaw_deg": 0.0, "grasp_z": 0.78, "pitch_deg": 0.0,
                "how": "across a thin part of it (a handle, a band)"}
        grasp_z, note = s._eef_align(spatula, "left_arm", plan)
        self.assertEqual((plan["x"], plan["y"]), (0.065, -0.161))  # still on the handle
        self.assertIn("stays on the thin part", note)
        whole = dict(plan, how="across its width")  # a whole-object grasp still follows
        s._eef_align(dict(spatula), "left_arm", whole)
        self.assertAlmostEqual(whole["x"], 0.065 - 0.018)

    def test_a_height_seen_by_one_camera_is_measured_from_above(self):
        s = Stub()
        del s._eef_align  # the real one
        s.get_observation = lambda cameras=None: None
        obj = dict(item("o1", 0.0, -0.2, z=eef.FLAT_Z), triangulated=False, seen_in=["cam_head"])
        s._eef_measure = lambda observation, name: ([dict(obj, position=[0.0, -0.2, 0.84], triangulated=True)], "")
        plan = s._eef_grasp(obj, "left_arm", "auto")
        grasp_z, note = s._eef_align(obj, "left_arm", plan)
        self.assertAlmostEqual(grasp_z, TABLE_Z + (0.84 - TABLE_Z) / 2)
        self.assertIn("measured from above", note)
        self.assertEqual(obj["position"][2], 0.84)


class LocateTriangulationTests(unittest.TestCase):
    """locate triangulates only views the image border does not cut, and no point higher
    than any object or held one (two instances' rays can cross in the air). The head
    camera is the robot's; a wrist camera looks straight down from where a test puts it."""

    CALIBRATION = load_embodiment("arx_x5")["calibration"]

    def observation(self, wrist_origin):
        import numpy as np

        wrist = np.eye(4)
        wrist[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # optical axis down, image up = +y
        wrist[:3, 3] = wrist_origin
        head = self.CALIBRATION["cam_head"]
        return type("O", (), {
            "images": {"cam_head": "cam_head", "cam_left_wrist": "cam_left_wrist"},
            "camera_to_world": {"cam_head": np.array(head["camera_to_world"]), "cam_left_wrist": wrist},
            "intrinsics": {"cam_head": np.array(head["intrinsics"]),
                           "cam_left_wrist": np.array(self.CALIBRATION["cam_left_wrist"]["intrinsics"])}})()

    @staticmethod
    def disk(observation, camera, point, radius=14):
        """The mask of a disk around a point's projection, clipped by the 640 x 480 image."""
        import numpy as np

        (u, v), = project(observation, camera, [point])[0]
        rows, cols = np.mgrid[0:480, 0:640]
        array = (cols - u) ** 2 + (rows - v) ** 2 <= radius ** 2
        r, c = np.nonzero(array)
        return Mask(array, [c.min(), r.min(), c.max(), r.max()], 0.9)

    @staticmethod
    def measure(observation, masks, tip_z=None):
        s = Stub()
        if tip_z is not None:
            s.state["left_ee_pose"][2] = tip_z + s.tip_offset
        s.segmenter = type("S", (), {"segment": staticmethod(lambda image, name: masks.get(image, []))})()
        found, _ = s._eef_measure(observation, "button")
        return found

    def test_a_view_cut_by_the_image_border_is_not_triangulated(self):
        point = [-0.15, -0.17, 0.78]
        # Straight above the button, then 12.9 cm toward +y: the button sits at the
        # bottom border of the wrist image (v = 472), as the near row does in the arms'
        # start pose.
        for shift, cut in ((0.0, False), (0.1286, True)):
            observation = self.observation([point[0], point[1] + shift, 1.0])
            masks = {camera: [self.disk(observation, camera, point)] for camera in ("cam_head", "cam_left_wrist")}
            self.assertEqual(at_border(masks["cam_left_wrist"][0]), cut)
            obj, = self.measure(observation, masks)
            if cut:
                self.assertFalse(obj["triangulated"])
                self.assertEqual(obj["seen_in"], ["cam_head"])
                self.assertAlmostEqual(obj["position"][2], eef.FLAT_Z, places=3)
            else:
                self.assertTrue(obj["triangulated"])
                self.assertEqual(obj["seen_in"], ["cam_head", "cam_left_wrist"])
                self.assertAlmostEqual(obj["position"][2], point[2], delta=0.003)
            self.assertLess(math.dist(obj["position"][:2], point[:2]), 0.006)

    def test_rays_meeting_above_any_object_are_not_a_position(self):
        import numpy as np

        point = [0.10, -0.15, 0.78]
        head = np.array(self.CALIBRATION["cam_head"]["camera_to_world"])[:3, 3]
        # Where the head camera's ray through the button is 1.15 m high: a wrist mask
        # there is another instance of the name whose ray happens to cross it.
        air = head + (np.array(point) - head) * (1.15 - head[2]) / (point[2] - head[2])
        observation = self.observation([air[0], air[1], air[2] + 0.2])
        masks = {"cam_head": [self.disk(observation, "cam_head", point)],
                 "cam_left_wrist": [self.disk(observation, "cam_left_wrist", air)]}
        obj, = self.measure(observation, masks)  # the jaws at z 0.90
        self.assertFalse(obj["triangulated"])
        self.assertAlmostEqual(obj["position"][2], eef.FLAT_Z, places=3)
        self.assertLess(math.dist(obj["position"][:2], point[:2]), 0.006)
        # With the jaws lifted to z 1.20 an object held there can be that high.
        obj, = self.measure(observation, masks, tip_z=1.20)
        self.assertTrue(obj["triangulated"])
        self.assertAlmostEqual(obj["position"][2], 1.15, delta=0.003)


class HelperTests(unittest.TestCase):
    def test_distinct_drops_overlapping_masks(self):
        import numpy as np

        a = np.zeros((10, 10), bool)
        a[2:6, 2:6] = True
        b = a.copy()
        b[2:5, 2:6] = True
        c = np.zeros((10, 10), bool)
        c[7:9, 7:9] = True
        kept = eef._distinct([Mask(a, [2, 2, 6, 6], 0.9), Mask(b, [2, 2, 6, 5], 0.5), Mask(c, [7, 7, 9, 9], 0.4)])
        self.assertEqual([m.score for m in kept], [0.9, 0.4])

    def test_arm_names(self):
        self.assertEqual(eef._arm_name("left"), "left_arm")
        self.assertEqual(eef._arm_name("right_arm"), "right_arm")
        with self.assertRaises(ValueError):
            eef._arm_name("middle")

    def test_unknown_object_ids(self):
        s = Stub()
        with self.assertRaisesRegex(ValueError, "locate first"):
            s.eef_pick("left", "o9")


if __name__ == "__main__":
    unittest.main()
