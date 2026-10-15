"""Verified model-facing operating notes for the Cartesian ARX X5 surface.

These live beside ``pose`` rather than with the joint cheat-sheet because every
number they quote is one of that module's constants, formatted here instead of
typed a second time. A figure the prose and the tool bounds each spell out on
their own drifts the first time the surface is retuned.

What the tool description, the system message, or the state block already says
is left out rather than repeated in a second wording. The tool owns the action
mechanics and the bounds, the system message owns what the model receives and
what it may not ask for, and these notes own what the spaces cannot express:
frame, geometry, grasp semantics, and gripper polarity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.embodiment_docs import (
    MOUNTING_HEADER,
    format_mounting_notes,
)

from .pose import GRASP_REACH_M, JAW_DEPTH_M, TABLE_SURFACE_Z

#: Lowest grasp point a straight-down jaw can reach before the table stops it.
_TABLETOP_FLOOR_Z = TABLE_SURFACE_Z + JAW_DEPTH_M / 2

#: Grasp point at which the jaws close entirely above the table surface, and so
#: entirely above anything lying flat on it.
_TABLETOP_CLEAR_Z = TABLE_SURFACE_Z + JAW_DEPTH_M


ARX_X5_EEF_DOCS = f"""Two identical 6-DoF ARX X5 arms, prefixed left_ and right_, each with a
parallel-jaw gripper. World frame: +z is up, both bases face world +y (see the
mounting note below), and +x is the lateral axis with the left arm on the -x
side. The table surface is at z = {TABLE_SURFACE_Z:.3f}, and each grasp point reaches about
{GRASP_REACH_M:.2f} m from its own base.
Every position, commanded and reported, is the point between the jaws where an
object is held. To grasp something resting on the table, name that object's
own position: no tool offset is yours to add.
The jaws are {JAW_DEPTH_M * 1000:.0f} mm deep along the tool axis and the commanded point is their
centre, so a straight-down grasp reaches no lower than z = {_TABLETOP_FLOOR_Z:.4f} before the
table stops it. Anything flat lying on the table, such as a tile, a card or a
token, is therefore grasped at a grasp-point z of about {_TABLETOP_FLOOR_Z:.4f} rather than at
an estimated centre height: naming a lower z is safe and lands in the same
place, while a z above {_TABLETOP_CLEAR_Z:.2f} closes the jaws entirely above such an object.
Objects tall enough to stand above the table are grasped at their own height
as usual.
Gripper 0 is fully closed, 1 is fully open, and commanding 0 always closes as
far as the object allows.
Each wrist camera rides its own arm, so which views you have is itself
something you can command. When the head camera is blocked, or the working
arm's own wrist sits too close to show what it is doing, move the idle arm
until its wrist looks at the work: a motion that only buys a view is worth its
env steps when the alternative is aiming blind.
On anything that is not a simple block, where you take hold of it decides how
hard the rest of the task is. Choose the grasp that leaves the part you still
have to insert, pour from, hang or set down clear of the jaws and pointing the
way it will need to go, rather than whichever face is easiest to close on.
Scale every motion to how close it is to touching something. Full-size moves
belong in free space; once the jaws are within about a centimetre of the
object, the table, or wherever a held object is going down, keep corrections
to a few millimetres per call and re-read the cameras between them. A carried
object extends the hand, so measure that centimetre from the edges of whatever
is held, not from the jaws.
When a task is built from several placements, the order they are done in is
part of the task rather than a detail. Work out which piece has to be in place
before which, and do first whatever the rest will stand on, reach over, or
need the room beside.
Most of these tasks also score the arms themselves: the work does not score
until both arms are back within 0.15 m and 20 degrees of the pose they began
the episode in, with the objects left as the task asked. Getting home is the
last step of the task rather than tidying up afterwards, so keep enough env
steps in hand to make it.
The advertised per-dimension bounds only reject gross magnitude errors: a pose
inside them can still be refused as unreachable by the Cartesian planner, and
that refusal spends a turn and gains no env steps, so keep targets near the
current grasp point. A refusal is likelier the more one call asks for, so
travel to the new position first and turn the tool in a later call rather than
doing both at once. Large rotations are more reliable in two or three steps
than in one. Both arms named in one call are planned as a unit and refused as
one: a pose the planner will not take for either arm leaves the other standing
still as well, so move one arm at a time whenever either target is uncertain."""


def build_arx_x5_eef_docs(mount_poses: Mapping[str, Sequence[float]] | None = None) -> str:
    """Cartesian operating notes, with the same live mounting appendix as joint docs."""
    mounting = format_mounting_notes(mount_poses or {})
    if not mounting:
        return ARX_X5_EEF_DOCS
    return ARX_X5_EEF_DOCS + "\n\n" + mounting


def eef_docs_from_joint_docs(joint_docs: str) -> str:
    """Keep the live mounting appendix, replace the joint cheat-sheet."""
    _, found, rest = joint_docs.partition(MOUNTING_HEADER)
    if not found:
        return ARX_X5_EEF_DOCS
    return ARX_X5_EEF_DOCS + "\n\n" + MOUNTING_HEADER + rest
