"""Property tests for the shield (App. H): random and adversarial command
sequences, including a faulty controller that feeds on its own output, must never
make the shield forward a command that violates a limit, and the shield forwards
what it accepts unchanged."""

import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.shield import PASS, REJECTED, Shield, ShieldProfile

# Two planar two-link arms (joints 0-1 and 3-4) with grippers 2 and 5. Forward
# kinematics is nonlinear, so a joint-space step between two safe poses can pass
# through an unsafe one.
LOWER = [-3.0, -2.5, 0.0, -3.0, -2.5, 0.0]
UPPER = [3.0, 2.5, 1.0, 3.0, 2.5, 1.0]
PROFILE = {"groups": {"left": {"joints": [0, 1], "gripper": 2}, "right": {"joints": [3, 4], "gripper": 5}},
           "max_joint_jump_rad": 0.3, "workspace": {"x": [-0.2, 0.9], "y": [-0.6, 0.6], "z": [0.0, 1.0]},
           "max_cartesian_jump_m": 0.25}
TOLERANCE = 1e-3


def fk(command):
    def arm(a, b, base_x):
        return [base_x + 0.5 * math.cos(a) + 0.4 * math.cos(a + b), 0.5 * math.sin(a) + 0.4 * math.sin(a + b), 0.5]
    return {"left": arm(command[0], command[1], 0.0), "right": arm(command[3], command[4], 0.1)}


def inside(point):
    return all(PROFILE["workspace"][axis][0] <= value <= PROFILE["workspace"][axis][1]
               for axis, value in zip("xyz", point))


def safe_state(rng):
    while True:
        state = [rng.uniform(lo, hi) for lo, hi in zip(LOWER, UPPER)]
        if all(inside(p) for p in fk(state).values()):
            return state


def adversarial(rng, current):
    choice = rng.random()
    if choice < 0.1:
        return [rng.choice([float("nan"), float("inf"), -float("inf"), 1e300, -1e300]) for _ in current]
    if choice < 0.3:
        return [rng.uniform(lo - 5, hi + 5) for lo, hi in zip(LOWER, UPPER)]
    if choice < 0.5:
        return [rng.choice([lo, hi]) for lo, hi in zip(LOWER, UPPER)]
    if choice < 0.7:
        return [c + rng.uniform(-0.2, 0.2) for c in current]
    return [c + rng.gauss(0, 1.0) for c in current]


def shield():
    return Shield(ShieldProfile.from_dict(PROFILE, LOWER, UPPER))


def outside(point):
    """How far a point is outside the workspace (0 inside)."""
    return math.sqrt(sum(max(PROFILE["workspace"][axis][0] - value, value - PROFILE["workspace"][axis][1], 0.0) ** 2
                         for axis, value in zip("xyz", point)))


class ShieldPropertyTests(unittest.TestCase):
    def assertSafe(self, result, request, previous, current):
        """A rejected command has a reason; an accepted one is forwarded unchanged, is
        within the joint limits and the workspace (or no further out than the measured
        `current`), and does not jump from both `previous` and `current`."""
        if result.verdict == REJECTED:
            self.assertIsNone(result.command)
            self.assertEqual(len(result.reasons), 1)
            return
        self.assertEqual(result.verdict, PASS)
        self.assertEqual(result.command, [float(v) for v in request])
        command = result.command
        self.assertTrue(all(math.isfinite(v) for v in command))
        for value, measured, lo, hi in zip(command, current, LOWER, UPPER):
            self.assertLessEqual(max(lo - value, value - hi), max(lo - measured, measured - hi, 0.0) + TOLERANCE)
        for group in PROFILE["groups"].values():
            joints = group["joints"]
            self.assertLessEqual(min(max(abs(command[i] - previous[i]) for i in joints),
                                     max(abs(command[i] - current[i]) for i in joints)),
                                 PROFILE["max_joint_jump_rad"])
        last, measured, after = fk(previous), fk(current), fk(command)
        for name, point in after.items():
            self.assertLessEqual(outside(point), outside(measured[name]), f"{name} at {point} leaves the workspace")
            self.assertLessEqual(min(math.dist(point, last[name]), math.dist(point, measured[name])),
                                 PROFILE["max_cartesian_jump_m"])

    def test_random_and_adversarial_commands(self):
        rng = random.Random(0)
        guard = shield()
        for _ in range(20000):
            current = safe_state(rng)
            guard.reset()  # an unrelated state each time: jumps are measured from it
            request = adversarial(rng, current)
            self.assertSafe(guard.check(request, current, fk), request, current, current)

    def test_faulty_controller_never_leaves_the_safe_set(self):
        rng = random.Random(1)
        guard = shield()
        for _episode in range(200):
            current = safe_state(rng)
            guard.reset()  # a new episode from a new state
            accepted = current
            target = [rng.uniform(lo - 1, hi + 1) for lo, hi in zip(LOWER, UPPER)]
            for _tick in range(100):
                # A controller that chases a far, partly unsafe target with gain > 1 and spikes.
                command = [c + 1.7 * (t - c) + (rng.gauss(0, 2) if rng.random() < 0.05 else 0.0)
                           for c, t in zip(current, target)]
                result = guard.check(command, current, fk)
                self.assertSafe(result, command, accepted, current)
                if result.command is not None:
                    accepted = current = result.command

    def test_a_noisy_smooth_policy_passes_unchanged(self):
        # A learned policy's targets: a fast smooth motion (up to 2.5 rad/s at 25 Hz) with
        # noise from tick to tick, while the arm stays where it was. Nothing is rejected
        # or changed, however far the targets get from the arm.
        rng = random.Random(3)
        guard = shield()
        arm = [0.5, 0.3, 0.5, 0.5, 0.3, 0.5]
        farthest = 0.0
        for tick in range(40):
            command = list(arm)
            command[0] = 0.5 * math.cos(tick / 5) + rng.gauss(0, 0.01)
            command[1] = arm[1] + 0.03 * math.sin(tick / 4) + rng.gauss(0, 0.01)
            command[2] = float(tick % 2)  # the gripper may open or close at once
            result = guard.check(command, arm, fk)
            self.assertEqual(result.verdict, PASS, result.reasons)
            self.assertEqual(result.command, command)
            farthest = max(farthest, abs(command[0] - arm[0]))
        self.assertGreater(farthest, 3 * PROFILE["max_joint_jump_rad"])

    def test_a_hold_after_a_lead_is_accepted(self):
        # A policy or an interrupted move left its commands well ahead of the arm; a hold
        # where the arm is, and the next skill's first waypoint from there, still pass.
        guard = shield()
        arm = [0.2, 0.3, 0.5, 0.5, 0.3, 0.5]
        command = list(arm)
        for _tick in range(6):
            command = list(command)
            command[0] -= 0.08
            self.assertEqual(guard.check(command, arm, fk).verdict, PASS)
        self.assertGreater(abs(command[0] - arm[0]), PROFILE["max_joint_jump_rad"])
        for _tick in range(3):
            self.assertEqual(guard.check(list(arm), arm, fk).verdict, PASS)
        waypoint = list(arm)
        waypoint[1] += 0.05
        self.assertEqual(guard.check(waypoint, arm, fk).verdict, PASS)

    def test_an_arm_that_tracks_imperfectly_never_locks_a_smooth_controller(self):
        # The arm follows each accepted command only part of the way, with noise, and
        # falls far behind a fast smooth controller. Nothing is rejected.
        rng = random.Random(4)

        def target(phase, tick):
            return [0.4 * math.sin(phase + tick / 6), 0.3, 0.5, 0.4 * math.cos(phase + tick / 7), 1.0, 0.5]

        for _episode in range(50):
            guard = shield()
            phase = rng.uniform(0, 6.28)
            arm = target(phase, 0)
            for tick in range(200):
                command = target(phase, tick)
                result = guard.check(command, arm, fk)
                self.assertEqual(result.verdict, PASS, (tick, result.reasons))
                arm = [a + rng.uniform(0.1, 0.5) * (c - a) + rng.gauss(0, 0.002) for a, c in zip(arm, command)]
                arm[2], arm[5] = command[2], command[5]

    def test_contact_past_a_limit_holds_and_moves_back_but_not_further(self):
        guard = shield()
        arm = [0.2, 0.3, 0.5, 0.5, 0.3, 0.5]
        arm[4] = UPPER[4] + 0.01  # contact pushed the joint past its limit
        self.assertEqual(guard.check(list(arm), arm, fk).verdict, PASS)
        back = list(arm)
        back[4] = UPPER[4] - 0.05
        self.assertEqual(guard.check(back, arm, fk).verdict, PASS)
        further = list(arm)
        further[4] = UPPER[4] + 0.03
        result = guard.check(further, arm, fk)
        self.assertEqual((result.verdict, result.reasons[0]["check"]), (REJECTED, "joint_limit"))

    def test_contact_outside_the_workspace_holds_and_moves_back_but_not_further(self):
        arm = [0.2, 0.3, 0.5, 0.5, 0.3, 0.5]
        angle = 0.2
        while outside(fk(arm[:0] + [angle] + arm[1:])["left"]) < 0.01:
            angle += 0.001  # the hand rises in y past the box
        arm[0] = angle
        out = outside(fk(arm)["left"])
        self.assertLess(out, 0.02)
        guard = shield()
        self.assertEqual(guard.check(list(arm), arm, fk).verdict, PASS)
        back = list(arm)
        back[0] -= 0.1
        self.assertEqual(guard.check(back, arm, fk).verdict, PASS)
        guard = shield()
        further = list(arm)
        further[0] += 0.05
        self.assertGreater(outside(fk(further)["left"]), out)
        result = guard.check(further, arm, fk)
        self.assertEqual((result.verdict, result.reasons[0]["check"]), (REJECTED, "workspace"))

    def test_a_jump_is_rejected_and_not_remembered(self):
        guard = shield()
        arm = [0.5, 0.3, 0.5, 0.5, 0.3, 0.5]
        self.assertEqual(guard.check(list(arm), arm, fk).verdict, PASS)
        flip = list(arm)
        flip[1] = -arm[1]  # the elbow on the other branch
        result = guard.check(flip, arm, fk)
        self.assertEqual(result.verdict, REJECTED)
        self.assertEqual(result.reasons[0]["check"], "joint_jump")
        near = list(arm)
        near[0] += 0.05  # judged from the last accepted command, not from the rejected one
        self.assertEqual(guard.check(near, arm, fk).verdict, PASS)

    def test_a_cartesian_jump_without_a_joint_jump_is_rejected(self):
        profile = ShieldProfile.from_dict({**PROFILE, "max_cartesian_jump_m": 0.1}, LOWER, UPPER)
        command = [0.2, 0.3, 0.5, 0.5, 0.3, 0.5]
        moved = list(command)
        moved[0] += 0.25  # 0.25 rad of the shoulder moves the hand about 0.2 m
        result = Shield(profile).check(moved, command, fk)
        self.assertEqual(result.reasons[0]["check"], "cartesian_jump")

    def test_out_of_limits_is_rejected_not_clamped(self):
        guard = shield()
        current = [0.5, 0.3, 0.5, 0.5, 0.3, 0.5]
        command = list(current)
        command[2] = 1.2
        result = guard.check(command, current, fk)
        self.assertEqual((result.verdict, result.reasons[0]["check"]), (REJECTED, "joint_limit"))
        command[2] = 1.0 + TOLERANCE / 2  # a measured gripper resting at its limit
        self.assertEqual(guard.check(command, current, fk).verdict, PASS)

    def test_invalid_profiles_are_rejected(self):
        for change in ({"max_joint_jump_rad": 0}, {"max_cartesian_jump_m": -1},
                       {"joint_limit_tolerance_rad": -1}, {"workspace": {"x": [1, 0], "y": [0, 1], "z": [0, 1]}}):
            with self.assertRaises(ValueError):
                ShieldProfile.from_dict({**PROFILE, **change}, LOWER, UPPER)


if __name__ == "__main__":
    unittest.main()
