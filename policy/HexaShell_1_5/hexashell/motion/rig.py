"""What the motion code needs to know about a two-arm table rig: the arms' chains and where they
stand, the gripper's readings, the table, the length of a step.

:data:`ROBODOJO` is RoboDojo's scene (the constants of :mod:`hexashell.world`).  A benchmark with the
same kind of arms on another table makes its own rig (RoboTwin: :mod:`hexashell.benches.robotwin.rig`);
a rig whose table height is measured at the start of an episode is made per episode with
:meth:`Rig.on_table`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import world as W

ASSETS = Path(__file__).resolve().parents[1] / "assets"


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
    bases=dict(W.ARM_BASES),
    urdf=ASSETS / "X5A.urdf",
    tips={"left": "link6", "right": "link6"},
    joint_ranges=W.JOINT_RANGES,
    grasp_reach=W.GRASP_REACH,
    tip_reach=W.TIP_REACH,
    table_z=W.TABLE_Z,
    table_x=W.TABLE_X,
    table_y=W.TABLE_Y,
    # RoboDojo reports the opening it applied, which runs up to 20 % ahead of the fingers; in R0 a
    # reading of 0.78 or more after closing meant the fingers had not closed on anything
    gripper=GripperModel(gap=W.finger_gap, closed=0.01, jammed=0.78, leads=True, open_gap_m=0.089),
    step_s=W.SECONDS_PER_ACTION,
    arms=W.ARMS,
)
