"""Observation-triggered, per-arm joint5 correction for stand_up_bottles."""

import numpy as np


class BottleJoint5Correction:
    # Use the same float32 representation as packed observations/predictions so
    # values exactly at a cutoff don't cross it due to scalar promotion.
    GRIPPER_CUTOFF = np.float32(0.6)
    JOINT5_CUTOFF = np.float32(-1.1)
    JOINT5_OFFSET = np.float32(0.05)
    JOINT5_MIN = np.float32(-1.25)
    ARM_COLUMNS = ((4, 6), (11, 13))

    def __init__(self):
        self.active = [False, False]

    def observe(self, state):
        """Only a newly received real observation can arm/rearm correction."""
        for arm, (joint5, gripper) in enumerate(self.ARM_COLUMNS):
            if state[gripper] > self.GRIPPER_CUTOFF:
                self.active[arm] = False
            elif state[gripper] < self.GRIPPER_CUTOFF and state[joint5] < self.JOINT5_CUTOFF:
                self.active[arm] = True

    def apply(self, actions):
        """Stop at the first predicted joint5 above cutoff, keeping the tail raw."""
        corrected = actions.copy()
        for arm, (joint5, _) in enumerate(self.ARM_COLUMNS):
            if not self.active[arm]:
                continue
            for step, value in enumerate(actions[:, joint5]):
                if value > self.JOINT5_CUTOFF:
                    self.active[arm] = False
                    break
                corrected[step, joint5] = max(value - self.JOINT5_OFFSET, self.JOINT5_MIN)
        return corrected
