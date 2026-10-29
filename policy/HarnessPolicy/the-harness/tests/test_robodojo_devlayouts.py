import importlib.util
import os
from pathlib import Path
import unittest

from benches.robodojo import devlayouts
from benches.robodojo.general import validate_layout_source


def layout(pos, ori=(1.0, 0.0, 0.0, 0.0), index=1):
    return {"Rigid": {"bowl": [{"label": "bowl0", "category_idx": index,
                                "default_pos": list(pos), "default_ori": list(ori)}]}}


def conveyor_layout():
    slots = [[-0.3, -0.3], [-0.14, -0.14], [0.02, 0.02], [0.2, 0.2]]

    def item(label, x, y, xlim=slots, ylim=(-0.3, -0.2)):
        return {"label": label, "relative_plane": "conveyor/conveyor/0", "xlim": xlim, "ylim": list(ylim),
                "default_pos": [x, y, 0.78], "default_ori": [0.7071, -0.7071, 0.0, 0.0]}

    return {"Dynamic": {"conveyor": [{"default_pos": [-2.08, 0.175, 0.05]}]},
            "Rigid": {"phone": [item("target1", -1.935, -0.10), item("other0", -1.775, -0.05),
                                item("target0", 0.245, 0.04, xlim=[1.88, 1.88], ylim=(-0.13, -0.13))],
                      "cup": [item("other1", -1.615, -0.08), item("other2", -1.435, -0.07)]}}


class ConveyorDevLayoutTests(unittest.TestCase):
    def test_slots_are_shuffled_and_poses_stay_on_the_belt(self):
        import numpy as np

        np.random.seed(3)
        placed = devlayouts._sample_conveyor(np, conveyor_layout())
        items = {i["label"]: i for group in placed["Rigid"].values() for i in group}
        slot_x = sorted(round(items[l]["default_pos"][0], 3) for l in ("target1", "other0", "other1", "other2"))
        self.assertEqual(slot_x, [-1.935, -1.775, -1.615, -1.435])
        for label in ("target1", "other0", "other1", "other2"):
            self.assertLess(abs(items[label]["default_pos"][1] - (-0.075)), 0.06)
            self.assertAlmostEqual(np.linalg.norm(items[label]["default_ori"]), 1.0, places=4)
        # The reference object near the robot keeps its place; only its yaw changes.
        self.assertEqual([round(v, 3) for v in items["target0"]["default_pos"]], [0.245, 0.04, 0.78])

    def test_different_seeds_give_different_layouts(self):
        import numpy as np

        np.random.seed(0)
        a = devlayouts._sample_conveyor(np, conveyor_layout())
        np.random.seed(1)
        b = devlayouts._sample_conveyor(np, conveyor_layout())
        self.assertIsNone(devlayouts._coincident_eval(a, [b]))


class DevLayoutTests(unittest.TestCase):
    def test_layout_source_options(self):
        self.assertEqual(validate_layout_source("eval", None), ("eval", None))
        source, directory = validate_layout_source("procedural", "dev")
        self.assertEqual(source, "procedural")
        self.assertTrue(directory.is_absolute())
        # Without a directory, procedural layouts use the checkout's.
        self.assertEqual(validate_layout_source("procedural", None, Path("ckpt") / "harness_dev_layouts"),
                         ("procedural", (Path("ckpt") / "harness_dev_layouts").absolute()))
        self.assertEqual(validate_layout_source("procedural", "dev", Path("other"))[1], Path("dev").absolute())
        for bad in (("procedural", None), ("eval", "dev"), ("random", None)):
            with self.assertRaises(ValueError):
                validate_layout_source(*bad)

    def test_seed_is_stable_and_task_specific(self):
        self.assertEqual(devlayouts.rng_seed("stack_bowls", 3, 0), devlayouts.rng_seed("stack_bowls", 3, 0))
        self.assertNotEqual(devlayouts.rng_seed("stack_bowls", 3, 0), devlayouts.rng_seed("stack_blocks", 3, 0))
        self.assertNotEqual(devlayouts.rng_seed("stack_bowls", 3, 0), devlayouts.rng_seed("stack_bowls", 3, 1))
        for bad in (-1, 1.0, True, "0"):
            with self.assertRaises(ValueError):
                devlayouts.validate_dev_seed(bad)

    @unittest.skipUnless(importlib.util.find_spec("shapely"), "shapely is in the harness-policy environment only")
    def test_region_slots_lines_and_boxes(self):
        slots = devlayouts._region([[-0.2, -0.2], [0, 0], [0.2, 0.2]], [-0.13, -0.13])
        self.assertEqual(slots.geom_type, "MultiPoint")
        self.assertEqual(len(slots.geoms), 3)
        self.assertEqual(devlayouts._region([0.0, 0.0], [-0.2, -0.1]).geom_type, "LineString")
        self.assertAlmostEqual(devlayouts._region([[-0.3, -0.1], [0.1, 0.3]], [0.0, 0.1]).area, 0.04)

    def test_coincidence_needs_same_asset_pose_and_yaw(self):
        evals = [layout((0.1, 0.1, 0.79))]
        self.assertEqual(devlayouts._coincident_eval(layout((0.105, 0.1, 0.79)), evals), 0)
        self.assertIsNone(devlayouts._coincident_eval(layout((0.13, 0.1, 0.79)), evals))
        self.assertIsNone(devlayouts._coincident_eval(layout((0.1, 0.1, 0.79), index=2), evals))
        yawed = (0.9914, 0.0, 0.0, 0.1305)  # 15 degrees about z
        self.assertIsNone(devlayouts._coincident_eval(layout((0.1, 0.1, 0.79), yawed), evals))

    @unittest.skipUnless(os.environ.get("HARNESS_ROBODOJO_ROOT"), "needs a RoboDojo checkout")
    def test_generation_is_deterministic_and_new(self):
        checkout = Path(os.environ["HARNESS_ROBODOJO_ROOT"])
        first = devlayouts.generate_dev_layout(checkout, "arx_x5", "stack_bowls", 0)
        second = devlayouts.generate_dev_layout(checkout, "arx_x5", "stack_bowls", 0)
        self.assertEqual(first, second)
        evals = [__import__("json").loads(p.read_text())
                 for p in devlayouts.eval_layout_files(checkout, "arx_x5", "stack_bowls")]
        self.assertIsNone(devlayouts._coincident_eval(first, evals))


if __name__ == "__main__":
    unittest.main()
