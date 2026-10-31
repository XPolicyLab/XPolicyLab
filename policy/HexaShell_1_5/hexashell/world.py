"""Physical facts of RoboDojo's two-arm table scene.

These are properties of the simulator, not design choices: the table, where the two arm bases
stand, the gripper geometry and the joint ranges the simulator actually lets the arms reach.
The arm-base poses were re-fitted from our own recordings.
"""

from __future__ import annotations

import math

ARMS = ("left", "right")

#: Height of the table top (world z, metres); the arm bases stand on it.
TABLE_Z = 0.765
#: Extent of the table top in world x and y (metres).
TABLE_X = (-0.70, 0.70)
TABLE_Y = (-0.60, 0.50)

#: Each arm's base on the table: position and heading (rotation about world z, radians).
#: Checked with bench/fit_bases.py on 169 recorded readings (2026-10-23): the chain then matches
#: the reported link6 poses within 0.01 mm and 0.001 degrees.
ARM_BASES = {
    "left": ((-0.30, -0.45, 0.765), 1.5710982518),
    "right": ((0.30, -0.45, 0.765), 1.5710982518),
}

#: Joint ranges the simulator enforces (radians), joint1..joint6, with a small margin.
JOINT_RANGES = (
    (-3.1, 3.1),
    (-0.2, 3.1),
    (0.0, 3.1),
    (-1.75, 1.75),
    (-1.52, 1.52),
    (-3.1, 3.1),
)

#: Grasp point: this far along link6's x axis (between the finger pads).
GRASP_REACH = 0.145
#: Fingertips: this far along link6's x axis.
TIP_REACH = 0.158
#: The gripper reading RoboDojo reports is the command it applied, which it keeps within 20 % of
#: the fingers' real position.  While squeezing (command 0) an object of width w, the reading
#: sits about 0.2 below the real position, and w ~ GAP0 + GAP_PER_UNIT * reading.
GAP0 = 0.0023
GAP_PER_UNIT = 0.1091
#: Finger joint travel per unit of opening is 0.054 m per finger (gripper_scale -0.01 .. 0.044);
#: the pads touch at an opening of about 0.18.  Free (not squeezing) gap ~ 0.108 g - 0.0193.
FREE_GAP_PER_UNIT = 0.108
FREE_GAP_AT_ZERO = -0.0193


def finger_gap(reading: float, squeezing: bool) -> float:
    """Metres between the finger pads for a gripper reading (an estimate)."""
    if squeezing:
        return max(0.0, GAP0 + GAP_PER_UNIT * reading)
    return max(0.0, FREE_GAP_AT_ZERO + FREE_GAP_PER_UNIT * reading)

#: One RoboDojo action: 10 physics steps.
SECONDS_PER_ACTION = 0.04

DEG = math.pi / 180.0
