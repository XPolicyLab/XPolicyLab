"""Embodiment profiles: every profile loads and implies a worker profile; export refuses a
profile that is not calibrated on its rig."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.embodiments import EMBODIMENTS_ROOT, load_embodiment, profile  # noqa: E402
from harness_policy.export.xpolicylab import check_embodiment  # noqa: E402


class EmbodimentTests(unittest.TestCase):
    def test_every_profile_loads(self):
        names = sorted(path.parent.name for path in EMBODIMENTS_ROOT.glob("*/embodiment.toml"))
        self.assertIn("piper", names)
        for name in names:
            data = load_embodiment(name)
            worker = profile(data)
            joints = sum(len(group["joints"]) + 1 for group in worker["groups"])
            self.assertEqual(len(data["action_low"]), joints, name)
            self.assertEqual(len(data["action_high"]), joints, name)
            self.assertTrue(all(low <= high for low, high in zip(data["action_low"], data["action_high"])), name)

    def test_export_needs_a_calibrated_profile(self):
        self.assertEqual(check_embodiment("arx_x5")["name"], "arx_x5")  # simulated: nothing to measure
        with self.assertRaises(ValueError):
            check_embodiment("piper")

    def test_another_robot_needs_its_own_kinematics(self):
        text = (EMBODIMENTS_ROOT / "piper" / "embodiment.toml").read_text().replace("calibrated = false",
                                                                                    "calibrated = true")
        with tempfile.TemporaryDirectory() as directory:
            calibrated = Path(directory) / "embodiment.toml"
            calibrated.write_text(text)
            with self.assertRaisesRegex(ValueError, "kinematics"):
                check_embodiment(calibrated)
            calibrated.write_text(text.replace('return_to_start = true', 'return_to_start = true\n'
                                               'kinematics_service = "piper-kinematics"'))
            self.assertEqual(check_embodiment(calibrated)["kinematics_service"], "piper-kinematics")

    def test_piper_joint3_stops_at_zero(self):
        data = load_embodiment("piper")
        for joint in (2, 9):  # joint3 of each arm
            self.assertEqual((data["action_low"][joint], data["action_high"][joint]), (-2.697, 0.0))

    def test_export_takes_the_embodiment(self):
        from harness_policy.cli import parser

        args = parser().parse_args(["export", "xpolicylab", "--name", "X", "--out", "x"])
        self.assertEqual(args.embodiment, "arx_x5")
        args = parser().parse_args(["export", "xpolicylab", "--name", "X", "--out", "x", "--embodiment", "piper"])
        self.assertEqual(args.embodiment, "piper")


if __name__ == "__main__":
    unittest.main()
