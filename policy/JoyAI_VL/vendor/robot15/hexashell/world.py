"""Physical facts of RoboDojo's two-arm table scene.

These are properties of the simulator, not design choices: the table, where the two arm bases
stand, the gripper geometry and the joint ranges the simulator actually lets the arms reach.
The arm-base poses were re-fitted from our own recordings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


ASSETS = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True, eq=False)
class GripperModel:
    """How a gripper's reading (0 closed .. 1 open) relates to its fingers."""

    #: metres between the finger pads for a reading, while squeezing (True) or not
    gap: Callable[[float, bool], float]
    #: a squeezing gripper reading at most this has closed fully: there is nothing between the pads
    closed: float
    #: a squeezing gripper reading at least this has hardly closed: the fingers are blocked
    jammed: float
    #: the reading runs ahead of the fingers while they open (it is the command, not the fingers)
    leads: bool
    #: the gap between the pads when fully open (m)
    open_gap_m: float


@dataclass(frozen=True, eq=False)
class Rig:
    """Two arms of the X5 family beside (or on) a table.  Compared by identity, so it can key caches."""

    name: str
    #: arm name -> (base position (x, y, z), heading about world z in radians) in the world
    bases: dict
    #: the URDF with each arm's chain, and the chain's last link (the flange) per arm
    urdf: Path
    tips: dict
    #: joint ranges the motions keep to (radians), joint1..joint6
    joint_ranges: tuple
    #: the grasp point (between the pads) and the fingertips: this far along the flange's x axis
    grasp_reach: float
    tip_reach: float
    #: the table top: height and extent in world x and y (metres)
    table_z: float
    table_x: tuple
    table_y: tuple
    gripper: GripperModel
    #: simulated seconds one step of ``wait`` holds still
    step_s: float
    arms: tuple = ("left", "right")

    def on_table(self, z: float, x: tuple | None = None, y: tuple | None = None) -> "Rig":
        """The same rig with the table top at height ``z`` (and, given, another extent)."""
        return dataclasses.replace(self, table_z=float(z), table_x=tuple(x or self.table_x),
                                   table_y=tuple(y or self.table_y))


ROBODOJO = Rig(
    name="robodojo",
    bases=dict(ARM_BASES),
    urdf=ASSETS / "X5A.urdf",
    tips={"left": "link6", "right": "link6"},
    joint_ranges=JOINT_RANGES,
    grasp_reach=GRASP_REACH,
    tip_reach=TIP_REACH,
    table_z=TABLE_Z,
    table_x=TABLE_X,
    table_y=TABLE_Y,
    # RoboDojo reports the opening it applied, which runs up to 20 % ahead of the fingers; in R0 a
    # reading of 0.78 or more after closing meant the fingers had not closed on anything
    gripper=GripperModel(gap=finger_gap, closed=0.01, jammed=0.78, leads=True, open_gap_m=0.089),
    step_s=SECONDS_PER_ACTION,
    arms=ARMS,
)
