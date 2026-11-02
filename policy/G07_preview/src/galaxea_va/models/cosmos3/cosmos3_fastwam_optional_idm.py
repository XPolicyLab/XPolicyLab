"""``Cosmos3FastWAMOptionalIDM`` — cosmos3 port of ``wan22.fastwam_optional_idm.FastWAMOptionalIDM``.

Same teacher-forcing training as :class:`Cosmos3FastWAMIDM`, but the action's access to the
cond (teacher-forced) video is stochastic per sample: with probability ``action_idm_prob`` the
action reads ALL cond video (IDM), otherwise only the first cond frame (the base first-frame
regime). At inference an ``action_infer_mode`` switch picks between the IDM two-stage path and
the base first-frame action-only path, so a single checkpoint serves both regimes.

As in the parent Cosmos3 IDM path, every clean conditioning frame receives no
timestep embedding. A zero timestep is never used as a substitute for clean
conditioning; only condition frames actually noised during training receive
``E(t)``.
"""

from __future__ import annotations

from typing import Optional


from galaxea_va.utils.logging_config import get_logger

from .cosmos3_fastwam_idm import Cosmos3FastWAMIDM

logger = get_logger(__name__)


class Cosmos3FastWAMOptionalIDM(Cosmos3FastWAMIDM):
    """IDM with per-sample stochastic teacher-forcing + an idm/first_frame inference switch."""

    action_idm_prob: float
    evaluation_action_modes = ("idm", "first_frame")

    @classmethod
    def from_pretrained(cls, *, action_idm_prob: float, **kwargs):
        """Load Optional IDM and initialize its per-sample IDM probability."""
        action_idm_prob = float(action_idm_prob)
        if not 0.0 <= action_idm_prob <= 1.0:
            raise ValueError(
                f"`action_idm_prob` must be in [0, 1], got {action_idm_prob}."
            )
        model = super().from_pretrained(**kwargs)
        model.action_idm_prob = action_idm_prob
        return model
