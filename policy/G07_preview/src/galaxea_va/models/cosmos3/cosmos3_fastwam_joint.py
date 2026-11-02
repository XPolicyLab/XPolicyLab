"""``Cosmos3FastWAMJoint`` — cosmos3 port of ``wan22.fastwam_joint.FastWAMJoint``.

The joint regime lets the action tokens attend **all** video tokens (not just the first
frame), so the closed-loop action policy must denoise the full video jointly with the
action. The class fixes that mask semantics and overrides :meth:`infer_action` to run
the joint video+action denoise loop.
"""

from __future__ import annotations



from .cosmos3_fastwam import Cosmos3FastWAM


class Cosmos3FastWAMJoint(Cosmos3FastWAM):
    """FastWAM joint variant: action attends all video; ``infer_action`` denoises video jointly."""

    supports_fdm = False
    evaluation_action_modes = ("joint",)
