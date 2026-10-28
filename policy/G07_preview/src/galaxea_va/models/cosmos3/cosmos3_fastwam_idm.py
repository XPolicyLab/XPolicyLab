"""``Cosmos3FastWAMIDM`` — cosmos3 port of ``wan22.fastwam_idm.FastWAMIDM``.

Teacher-forcing (inverse-dynamics) regime. Training runs a three-branch generator sequence
``[noisy_video | cond_video | proprio | action]``: the noisy branch supplies the video
denoise target, the cond branch is a clean (or, with prob ``video_cond_noise_prob``, noised)
teacher-forced video that the action tokens read, and action never sees the noisy branch.
Clean condition frames receive no timestep embedding; ``time_embedder(0)`` is
not used as a clean-condition marker. Only frames actually noised by the
training scheduler receive ``E(t)``.

Inference is two-stage: (1) fully denoise the video, (2) freeze it as a per-layer K/V cache
(``_prefill_action_condition_cache``) and denoise action with
``_forward_action_with_cache``. Supports all four cosmos3 combos (reasoner text/img × action
cogen/split); no new trainable params (the cond branch reuses
``proj_in``/``time_embedder``), so the loader/checkpoint layout is unchanged.
"""

from __future__ import annotations



from galaxea_va.utils.logging_config import get_logger

from .cosmos3_fastwam_joint import Cosmos3FastWAMJoint

logger = get_logger(__name__)


class Cosmos3FastWAMIDM(Cosmos3FastWAMJoint):
    """Teacher-forcing inverse-dynamics variant (two-stage: denoise video -> denoise action vs clean video)."""

    video_cond_noise_prob: float
    supports_fdm = True
    evaluation_action_modes = ("idm",)

    @classmethod
    def from_pretrained(cls, *, video_cond_noise_prob: float, **kwargs):
        """Load an IDM model and initialize its teacher-forcing noise probability."""
        video_cond_noise_prob = float(video_cond_noise_prob)
        if not 0.0 <= video_cond_noise_prob <= 1.0:
            raise ValueError(
                "`video_cond_noise_prob` must be in [0, 1], got "
                f"{video_cond_noise_prob}."
            )
        model = super().from_pretrained(**kwargs)
        model.video_cond_noise_prob = video_cond_noise_prob
        return model

    # -------------------------------------------------- teacher-forcing masks/pos

    # ------------------------------------------------------------------ training

    # ----------------------------------------------------------------- inference
