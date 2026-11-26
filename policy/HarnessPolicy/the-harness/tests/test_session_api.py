import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.runtime.library import Library
from harness_policy.geometry import DOWN, down_with_yaw, fingertip_of
from harness_policy.runtime.session import session_class

RoboDojoSession = session_class()


class FakeWorker:
    def __init__(self, left_pose):
        self.left_pose = left_pose

    def call(self, name, **arguments):
        assert name == "state", name
        return {"step": 0, "left_ee_pose": list(self.left_pose), "right_ee_pose": [0.3, 0, 1, 1, 0, 0, 0]}


def session(left_pose=(0.1, -0.2, 0.95, *DOWN), memory=()):
    s = object.__new__(RoboDojoSession)
    s.env, s.episode_id = FakeWorker(left_pose), "e"
    s.profile = {"arm_bases": {"left_arm": [-0.3, -0.45, 0.765, 1.0, 0.0, 0.0, 0.0],
                               "right_arm": [0.3, -0.45, 0.765]}}
    s.memory = [dict(entry) for entry in memory]
    s.tip_offset, s.table_z = 0.150, 0.765
    return s


class TransformTests(unittest.TestCase):
    def assertClose(self, got, want, places=6):
        self.assertEqual(len(got), len(want))
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=places)

    def test_a_point_ahead_of_link6_is_the_fingertip(self):
        pose = [0.1, -0.2, 0.95, *down_with_yaw(0.6)]
        s = session(pose)
        self.assertClose(s.transform([0.150, 0, 0], "left_link6", "world"), fingertip_of(pose))

    def test_round_trip_and_arm_base(self):
        s = session()
        point = [0.05, 0.1, 0.8]
        local = s.transform(point, "world", "left_arm_base")
        self.assertClose(local, [0.35, 0.55, 0.035])
        self.assertClose(s.transform(local, "left_arm_base", "world"), point)
        self.assertClose(s.transform(point, "world", "right_arm_base"), [-0.25, 0.55, 0.035])

    def test_poses_keep_orientation(self):
        s = session()
        pose = [0.1, -0.2, 0.95, *DOWN]
        self.assertClose(s.transform(pose, "world", "left_link6"), [0, 0, 0, 1, 0, 0, 0])

    def test_unknown_frame(self):
        with self.assertRaises(ValueError):
            session().transform([0, 0, 0], "world", "gripper")


class RecallTests(unittest.TestCase):
    def test_ranks_by_overlap_and_limits(self):
        s = session(memory=[
            {"name": "thin_coin_grasp", "text": "Coins lie flat: close the jaws 5 mm above the table."},
            {"name": "stack_blocks_speed", "text": "Scan the table once with the head camera, then move fast."},
            {"name": "unrelated", "text": "Bowls wider than 9 cm need a rim grasp."},
        ])
        results = s.recall("grasp a flat coin from the holder", k=2)
        self.assertEqual([r["name"] for r in results], ["thin_coin_grasp", "unrelated"])
        self.assertEqual(s.recall("xylophone"), [])
        self.assertEqual(session().recall("anything"), [])


class PredictionTests(unittest.TestCase):
    def _data(self, frames=9, actions=32):
        import base64

        import cv2
        import numpy as np

        image = np.zeros((8, 6, 3), dtype=np.uint8)
        image[..., 0] = 200  # red in RGB
        url = "data:image/jpeg;base64," + base64.b64encode(
            cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()).decode()
        step = {"left_ee_pose": [0.0] * 7, "left_ee_joint_state": [1.0],
                "right_ee_pose": [0.0] * 7, "right_ee_joint_state": [1.0]}
        return {"frames": [url] * frames, "actions": [dict(step) for _ in range(actions)], "seconds": 1.5}

    def test_frames_decode_to_rgb(self):
        from skills.wam import Prediction

        prediction = Prediction(self._data())
        self.assertEqual((len(prediction.frames), len(prediction.actions)), (9, 32))
        self.assertEqual(prediction.frames[0].shape, (8, 6, 3))
        self.assertGreater(int(prediction.frames[0][..., 0].mean()), 150)

    def test_horizon_trims_actions_and_frames_in_proportion(self):
        from skills.wam import Prediction

        prediction = Prediction(self._data(), horizon=8)
        self.assertEqual((len(prediction.frames), len(prediction.actions)), (2, 8))
        self.assertEqual(len(Prediction(self._data(), horizon=100).actions), 32)


class LibraryMemoryTests(unittest.TestCase):
    def test_memory_belongs_to_the_notes_layer(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "memory").mkdir()
            (Path(root) / "memory" / "coin.md").write_text("Close the jaws low on coins.\n")
            full = Library(root)
            self.assertEqual(full.memory_entries(), [{"name": "coin", "text": "Close the jaws low on coins."}])
            self.assertEqual(Library(root, ("skills", "reactive")).memory_entries(), [])
            before = full.version()
            (Path(root) / "memory" / "coin.md").write_text("changed\n")
            self.assertNotEqual(Library(root).version(), before)
            self.assertNotIn("changed", Library(root).notes())


if __name__ == "__main__":
    unittest.main()


class FinishAndVerifyTests(unittest.TestCase):
    def test_finish_ends_the_program(self):
        from harness_policy.runtime.events import ProgramFinished

        with self.assertRaises(ProgramFinished) as raised:
            session().finish("stacked")
        self.assertEqual(raised.exception.summary, "stacked")

    def test_verify_asks_the_verifier_and_counts_usage(self):
        class Views:
            images = {"cam_head": "head", "cam_left_wrist": "wrist"}

        class Verifier:
            def __init__(self):
                self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                self.asked = None

            def ask(self, images, question, system=None):
                self.asked = (images, question)
                self.usage.update(calls=1, input_tokens=10, output_tokens=2, cost_usd=0.01)
                return "YES the blocks are stacked"

        s = session()
        s.vision, s.pointer, s.verifier = None, None, Verifier()
        s.get_observation = lambda cameras=None: Views()
        holds, answer = s.verify("the blocks are stacked")
        self.assertTrue(holds)
        self.assertEqual(s.verifier.asked, (["head", "wrist"], "Claim: the blocks are stacked"))
        self.assertEqual(s._drain_usage()["verifier"]["calls"], 1)
        self.assertEqual(s._drain_usage(), {})  # reset after each drain

        s.verifier = None
        self.assertIsNone(s.verify("anything")[0])


class GraspPlanningTests(unittest.TestCase):
    """plan_grasp on a synthetic footprint (mask_to_plane is replaced)."""

    def plan(self, footprint, reachable, **options):
        import numpy as np
        from unittest import mock

        s = session()
        s.groups = {"left_arm": {}, "right_arm": {}}
        asked = []

        def ik(arm, pose):
            asked.append((arm, pose))
            return [0.0] * 6 if reachable(pose) else None

        s.ik = ik
        points = np.array(footprint, dtype=float)
        with mock.patch("harness_policy.geometry.mask_to_plane", return_value=points):
            return s.plan_grasp(None, None, z=0.8, **options), asked

    @staticmethod
    def box(cx, cy, sx, sy, z=0.8):
        return [[cx + sx * (i / 9 - 0.5), cy + sy * (j / 9 - 0.5), z] for i in range(10) for j in range(10)]

    def test_top_down_when_graspable_and_reachable(self):
        grasp, asked = self.plan(self.box(-0.2, -0.1, 0.04, 0.04), lambda pose: True)
        self.assertEqual((grasp["approach"], grasp["arm"], grasp["reachable"]), ("top_down", "left_arm", True))
        self.assertAlmostEqual(grasp["pregrasp"][2] - grasp["grasp"][2], 0.10)
        self.assertTrue(all(arm == "left_arm" for arm, _pose in asked))

    def test_pitched_grasp_before_side_grasp(self):
        # link6 sits 0.15 * sin(pitch) above the tip and the pregrasp 0.10 * sin(pitch)
        # above that: with IK below 0.93 only the 30 degree pitch (0.875 / 0.925) and the
        # side grasp are reachable; the pitched one comes first.
        grasp, _asked = self.plan(self.box(0.3, 0.0, 0.05, 0.05), lambda pose: pose[2] < 0.93)
        self.assertEqual((grasp["approach"], grasp["arm"]), ("pitched_30", "right_arm"))
        # Approach from the right base (0.3, -0.45): +y, tilted 30 degrees down.
        self.assertAlmostEqual(grasp["grasp"][0], grasp["tip"][0])
        self.assertAlmostEqual(grasp["grasp"][1], grasp["tip"][1] - 0.150 * math.cos(math.radians(30)))
        self.assertAlmostEqual(grasp["grasp"][2], grasp["tip"][2] + 0.150 * math.sin(math.radians(30)))
        self.assertAlmostEqual(grasp["pregrasp"][2] - grasp["grasp"][2], 0.10 * math.sin(math.radians(30)))
        self.assertAlmostEqual(abs(grasp["closing_direction"][0]), 1.0)

    def test_side_grasp_when_top_down_is_out_of_reach(self):
        # Top-down puts link6 0.15 m above the tip; pretend only low link6 poses have IK.
        grasp, _asked = self.plan(self.box(0.3, 0.0, 0.05, 0.05), lambda pose: pose[2] < 0.81)
        self.assertEqual((grasp["approach"], grasp["arm"], grasp["reachable"]), ("side", "right_arm", True))
        # The right arm's base is at (0.3, -0.45): the approach is +y, link6 behind the tip.
        self.assertAlmostEqual(grasp["yaw"], math.pi / 2)
        self.assertAlmostEqual(grasp["grasp"][1], grasp["tip"][1] - 0.150)
        self.assertAlmostEqual(grasp["grasp"][2], grasp["tip"][2])
        self.assertAlmostEqual(grasp["pregrasp"][1], grasp["grasp"][1] - 0.10)
        self.assertAlmostEqual(abs(grasp["closing_direction"][0]), 1.0)
        self.assertAlmostEqual(grasp["width_m"], 0.05)

    def test_part_grasp_across_the_band_of_a_ring(self):
        # An open ring 12 cm across with a 1 cm band: too wide whole, graspable across the band.
        ring = [[0.3 + r * math.cos(a), 0.0 + r * math.sin(a), 0.8]
                for a in [i * math.radians(5) for i in range(0, 60)] for r in (0.055, 0.06, 0.065)]
        grasp, _asked = self.plan(ring, lambda pose: True)
        self.assertEqual(grasp["approach"], "top_down_part")
        radius = math.hypot(grasp["tip"][0] - 0.3, grasp["tip"][1])
        self.assertAlmostEqual(radius, 0.06, delta=0.01)  # on the band, not the empty middle
        radial = ((grasp["tip"][0] - 0.3) / radius, grasp["tip"][1] / radius)
        closing = grasp["closing_direction"]
        self.assertGreater(abs(radial[0] * closing[0] + radial[1] * closing[1]), 0.9)

    def test_tall_objects_prefer_the_side(self):
        grasp, _asked = self.plan(self.box(0.3, 0.0, 0.04, 0.04), lambda pose: True, prefer_side=True)
        self.assertEqual(grasp["approach"], "side")

    def test_too_wide_without_a_thin_part_pinches_the_edge_nearest_the_arm(self):
        # A 12 cm disc-like footprint (a bowl from above): the wall at the edge facing
        # the right arm's base (0.3, -0.45), fingers closing radially.
        grasp, _asked = self.plan(self.box(0.3, 0.0, 0.12, 0.12), lambda pose: True)
        self.assertEqual(grasp["approach"], "top_down_rim")
        self.assertTrue(grasp["graspable"])
        self.assertAlmostEqual(grasp["tip"][0], 0.3, delta=0.01)
        self.assertAlmostEqual(grasp["tip"][1], -0.06 + 0.012, delta=0.01)

    def test_nothing_feasible_returns_the_top_down_grasp(self):
        grasp, _asked = self.plan(self.box(0.3, 0.0, 0.12, 0.12), lambda pose: False)
        self.assertEqual(grasp["approach"], "top_down")
        self.assertFalse(grasp["graspable"])

    def test_graspable_width_follows_the_embodiment(self):
        s = session()
        self.assertAlmostEqual(s._graspable_width(), 0.080)
        s.profile = {**s.profile, "gripper_open_width_m": 0.0953}
        self.assertAlmostEqual(s._graspable_width(), 0.0873)

    def test_top_down_only_skips_reachability(self):
        grasp, asked = self.plan(self.box(0.3, 0.0, 0.03, 0.06), lambda pose: False, top_down=True)
        self.assertEqual(grasp["approach"], "top_down")
        self.assertNotIn("reachable", grasp)
        self.assertEqual(asked, [])
        self.assertAlmostEqual(grasp["width_m"], 0.03)


class Localize3DTests(unittest.TestCase):
    """Two cameras looking down at the table from different sides; each view lists the
    two blocks in a different order, so pairing by order would mix them up."""

    def setUp(self):
        import numpy as np

        def camera(position):
            # Looks at the origin from `position`; OpenCV axes (z forward, y down).
            forward = -np.asarray(position, dtype=float)
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            transform = np.eye(4)
            transform[:3, :3] = np.column_stack([right, down, forward])
            transform[:3, 3] = position
            return transform

        class Views:
            images = {"cam_head": np.zeros((480, 640, 3), np.uint8), "cam_left_wrist": np.zeros((480, 640, 3), np.uint8)}
            intrinsics = {name: np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]) for name in images}
            camera_to_world = {"cam_head": camera([0.0, -0.8, 1.6]), "cam_left_wrist": camera([-0.5, -0.3, 1.2])}

        from harness_policy.geometry import project

        self.views = Views()
        self.blocks = [np.array([0.1, 0.0, 0.785]), np.array([-0.15, 0.1, 0.785])]
        self.pixels = {name: [tuple(project(self.views, name, [b])[0][0]) for b in self.blocks]
                       for name in Views.images}
        self.pixels["cam_left_wrist"].reverse()

    def test_views_are_paired_by_geometry_not_order(self):
        s = session()
        s.point_to = lambda observation, prompt, camera="cam_head", tiles=None: list(self.pixels[camera])
        found = s.localize_3d(self.views, "block", cameras=("cam_head", "cam_left_wrist"))
        self.assertFalse(found["single_view"])
        for a, b in zip(found["position"], self.blocks[0]):  # the head camera's first block
            self.assertAlmostEqual(a, b, places=4)

    def test_estimates_at_the_fingers_rank_last(self):
        from harness_policy.geometry import link6_for_tip

        # The left fingertips sit on the head camera's first block: that estimate is
        # passed over for the other block.
        s = session(left_pose=link6_for_tip(list(self.blocks[0]), DOWN, 0.150))
        s.state = s.env.call("state")
        s.point_to = lambda observation, prompt, camera="cam_head", tiles=None: list(self.pixels[camera])
        found = s.localize_3d(self.views, "block", cameras=("cam_head", "cam_left_wrist"))
        for a, b in zip(found["position"], self.blocks[1]):
            self.assertAlmostEqual(a, b, places=4)

    def test_one_view_of_a_held_object_ranks_last(self):
        import numpy as np
        from harness_policy.geometry import pixel_ray
        from harness_policy.geometry import link6_for_tip

        # The head camera sees a lifted block right below the fingertips and a block on
        # the table; seen from one view only, the lifted block's ray would land on the
        # table far from the fingers.
        tip = np.array([0.0, 0.0, 0.95])
        s = session(left_pose=link6_for_tip(list(tip), DOWN, 0.150))
        s.state = s.env.call("state")
        from harness_policy.geometry import project
        held = tuple(project(self.views, "cam_head", [tip - [0, 0, 0.02]])[0][0])
        table_block = self.pixels["cam_head"][1]
        s.point_to = lambda observation, prompt, camera="cam_head", tiles=None: (
            [held, table_block] if camera == "cam_head" else [])
        found = s.localize_3d(self.views, "block", cameras=("cam_head", "cam_left_wrist"))
        self.assertTrue(found["single_view"])
        self.assertAlmostEqual(found["position"][0], self.blocks[1][0], places=3)
        self.assertAlmostEqual(found["position"][1], self.blocks[1][1], places=3)

    def test_one_view_falls_back_to_the_table_plane(self):
        s = session()
        s.point_to = lambda observation, prompt, camera="cam_head", tiles=None: (
            self.pixels[camera][:1] if camera == "cam_head" else [])
        found = s.localize_3d(self.views, "block", cameras=("cam_head", "cam_left_wrist"))
        self.assertTrue(found["single_view"])
        self.assertAlmostEqual(found["position"][2], 0.785)


class PointingTests(unittest.TestCase):
    def test_small_objects_are_found_in_enlarged_tiles(self):
        import numpy as np

        class Pointer:
            def __init__(self):
                self.sizes = []

            def point(self, image, prompt):
                height, width = np.asarray(image).shape[:2]
                self.sizes.append((width, height))
                # Only an enlarged tile shows the coin; it sits at (100, 50) in the full image.
                return [(2 * (100 - 0), 2 * (50 - 0))] if (width, height) == (800, 600) and len(self.sizes) == 2 else []

        s = session()
        s.pointer = Pointer()
        views = type("Views", (), {"images": {"cam_head": np.zeros((480, 640, 3), np.uint8)}})()
        self.assertEqual(s.point_to(views, "coin"), [(100.0, 50.0)])
        self.assertEqual(s.pointer.sizes[:2], [(640, 480), (800, 600)])
        s.pointer.sizes.clear()
        self.assertEqual(s.point_to(views, "coin", camera="cam_head", tiles=False), [])

    def test_molmo_points_are_scaled_to_pixels(self):
        from harness_policy.services.clients import molmo_points

        text = 'The cups: <points coords="1 500 250 2 1000 0000" alt="cups">cups</points>'
        self.assertEqual(molmo_points(text, 640, 480), [(320.0, 120.0), (640.0, 0.0)])
        # A reply of the served model for two blocks (image number 1, then two points).
        reply = '<points coords="1 1 433 424 2 614 465">wooden block.</points>'
        self.assertEqual([(round(x, 2), round(y, 2)) for x, y in molmo_points(reply, 640, 480)],
                         [(277.12, 203.52), (392.96, 223.2)])
        self.assertEqual(molmo_points('<point x="25.0" y="50.0" alt="cup">cup</point>', 640, 480), [(160.0, 240.0)])
        self.assertEqual(molmo_points("There is no cup.", 640, 480), [])

    def test_point_to_uses_the_pointer_when_configured(self):
        class Model:
            def __init__(self, name):
                self.name = name

            def point(self, image, prompt):
                return [(self.name, prompt)]

        class Views:
            images = {"cam_head": "image"}

        s = session()
        s.vision, s.pointer = Model("vlm"), Model("molmo")
        self.assertEqual(s.point_to(Views(), "red cup"), [("molmo", "red cup")])
        s.pointer = None
        self.assertEqual(s.point_to(Views(), "red cup"), [("vlm", "red cup")])

    def test_empty_system_message_is_left_out(self):
        from harness_policy.runtime.chat import ChatModel

        chat = ChatModel({"backend": "openai_chat", "base_url": "http://127.0.0.1:1/v1", "model": "m"})
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual([m["role"] for m in chat.payload("", messages)["messages"]], ["user"])
        self.assertEqual([m["role"] for m in chat.payload("be brief", messages)["messages"]], ["system", "user"])
