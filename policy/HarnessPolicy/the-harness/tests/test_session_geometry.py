from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.geometry import DOWN, FINGERTIP_OFFSET, FORWARD, down_with_yaw, fingertip_of, link6_for_tip


def rotation(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


class DownWithYawTests(unittest.TestCase):
    def test_fingers_close_along_the_yaw_direction_and_approach_stays_down(self):
        import math
        for yaw in (0.0, 0.4, -1.2, math.pi / 2):
            matrix = rotation(down_with_yaw(yaw))
            approach = [row[0] for row in matrix]
            fingers = [row[1] for row in matrix]
            for got, want in zip(approach, [0, 0, -1]):
                self.assertAlmostEqual(got, want, places=6)
            # link6 +y maps to -(cos, sin) or +(cos, sin): either finger closes on the same line.
            self.assertAlmostEqual(abs(fingers[0] * math.cos(yaw) + fingers[1] * math.sin(yaw)), 1.0, places=6)
            self.assertAlmostEqual(fingers[2], 0.0, places=6)
        for got, want in zip(down_with_yaw(0.0), DOWN):
            self.assertAlmostEqual(got, want, places=6)


class Link6ForTipTests(unittest.TestCase):
    def test_down_puts_link6_above_the_tip(self):
        pose = link6_for_tip([0.1, -0.05, 0.76], DOWN)
        for got, want in zip(pose[:3], [0.1, -0.05, 0.76 + FINGERTIP_OFFSET]):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(len(pose), 7)

    def test_fingertip_of_inverts_link6_for_tip(self):
        for orientation in (DOWN, FORWARD, down_with_yaw(0.7)):
            tip = [0.12, -0.03, 0.8]
            for got, want in zip(fingertip_of(link6_for_tip(tip, orientation)), tip):
                self.assertAlmostEqual(got, want, places=6)

    def test_forward_puts_link6_behind_the_tip(self):
        pose = link6_for_tip([-0.2, 0.2, 0.85], FORWARD)
        for got, want in zip(pose[:3], [-0.2, 0.2 - FINGERTIP_OFFSET, 0.85]):
            self.assertAlmostEqual(got, want, places=3)


if __name__ == "__main__":
    unittest.main()
