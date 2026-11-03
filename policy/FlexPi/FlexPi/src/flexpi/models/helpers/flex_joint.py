"""Flex-joint training helpers for FlexPiUnifiedJoint.

Per-sample randomization of:
  * stream presence (video, dino, pointmap)
  * joint flags (joint_video, joint_dino, joint_pointmap)

When enabled, a single training run learns every regime in the 8-way
``(jv, jd, jp)`` × {present/absent} space, so the same checkpoint can
be dispatched at inference into any subset (action-only, video+action,
dino+3d+action, etc.) by passing kwargs to ``infer_action``.

The flex config is a tiny dataclass kept here to avoid bloating
``flexpi/model.py``. The actual sampling, masking, and loss-masking
hooks live in the model files; this module just provides
parameter container + sampling helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FlexJointConfig:
    """Per-stream presence + joint probabilities for flex training.

    Video, DINO, and pointmap presence are independent Bernoulli per sample.
    Joint flags ``(j_v, j_d, j_p)`` are independent Bernoulli per sample,
    forced False when the corresponding stream is absent (unless
    ``cross_modal_predict_X`` is True, in which case the rem_X tokens are
    still denoised conditioned on the other present streams).
    """

    enabled: bool = False
    # Presence (per-sample Bernoulli; 1.0 = always present).
    p_present_video: float = 1.0
    p_present_dino: float = 1.0
    p_present_pointmap: float = 1.0
    # Joint flags (per-sample Bernoulli).
    p_jv: float = 0.5
    p_jd: float = 0.5
    p_jp: float = 0.5
    # When True and the stream is presence-dropped on a sample, keep
    # denoising its rem tokens conditioned on the other present streams.
    cross_modal_predict_video: bool = False
    cross_modal_predict_dino: bool = False
    cross_modal_predict_pointmap: bool = False

    def __post_init__(self):
        for name in ("p_present_video", "p_present_dino", "p_present_pointmap", "p_jv", "p_jd", "p_jp"):
            val = float(getattr(self, name))
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"FlexJointConfig.{name} must be in [0, 1], got {val}")
            setattr(self, name, val)


@dataclass
class FlexBatchFlags:
    """Per-sample boolean tensors of shape ``[B]`` (on the model device).

    Set on ``model._batch_flex`` for the duration of a single
    ``training_loss(...)`` call. Consumed by:

    * token-zeroing hook (zeros ff + rem of absent streams unless cm).
    * mask builder (per-sample edits + B-stack into ``[B, S, S]``).
    * loss-mask hook (multiplies per-sample stream loss by presence mask).
    """

    present_v: torch.Tensor    # [B] bool
    present_d: torch.Tensor    # [B] bool
    present_p: torch.Tensor    # [B] bool
    j_v: torch.Tensor          # [B] bool — already AND-ed with (present_v | cm_v)
    j_d: torch.Tensor          # [B] bool — already AND-ed with (present_d | cm_d)
    j_p: torch.Tensor          # [B] bool — already AND-ed with (present_p | cm_p)
    cm_v: bool                 # cross_modal_predict_video (scalar)
    cm_d: bool                 # cross_modal_predict_dino (scalar)
    cm_p: bool                 # cross_modal_predict_pointmap (scalar)

    @property
    def B(self) -> int:
        return int(self.present_v.shape[0])


def sample_flex_batch_flags(
    cfg: FlexJointConfig,
    batch_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    pointmap_off: bool = False,
) -> FlexBatchFlags:
    """Draw per-sample flex flags for one training step.

    Independent Bernoulli for each of: present_v, present_d, present_p, j_v, j_d, j_p.
    Joint flags are then AND-ed with (presence | cross_modal) so a flag
    pointing at an absent non-cross-modal stream collapses to False.

    Rejection-sample the all-absent corner: when ``(present_v, present_d,
    present_p) == (False, False, False)`` for a sample, flip one stream to
    True (uniform pick among streams whose ``p_present_X > 0``). Guarantees
    every training sample has at least one conditioning anchor — closes the
    train/deploy gap where the deploy path raises if all three are absent.

    ``pointmap_off`` (the model's ``_pointmap_globally_off``) means the run
    carries no pointmap stream at all, so there is nothing for the pointmap
    knobs to randomize: ``present_p``/``j_p`` are forced False and ``p`` is
    barred from the rejection flip, which would otherwise hand a sample an
    anchor that does not exist. No-op when pointmap is off via the legacy
    ``p_present_pointmap=0, p_jp=0`` corner — those already produce all-False.
    """
    def _bern(p: float) -> torch.Tensor:
        if p >= 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if p <= 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        u = torch.rand(batch_size, device=device, generator=generator)
        return u < p

    present_v = _bern(cfg.p_present_video)
    present_d = _bern(cfg.p_present_dino)
    present_p = _bern(0.0 if pointmap_off else cfg.p_present_pointmap)

    # Rejection: pick one allowed stream uniformly when all three came up False.
    # Only consider streams with p_present_X > 0 (respect user intent: a stream
    # explicitly set to 0 is never flipped). If all three are 0 the user
    # configured a pathological case — leave the all-absent state untouched.
    positive_streams: list[str] = []
    if cfg.p_present_video    > 0: positive_streams.append("v")
    if cfg.p_present_dino     > 0: positive_streams.append("d")
    if cfg.p_present_pointmap > 0 and not pointmap_off: positive_streams.append("p")
    if positive_streams:
        all_false = (~present_v) & (~present_d) & (~present_p)
        if all_false.any():
            n = len(positive_streams)
            pick = torch.randint(
                0, n, (batch_size,), device=device, generator=generator,
            )
            for i, name in enumerate(positive_streams):
                flip = all_false & (pick == i)
                if name == "v":   present_v = present_v | flip
                elif name == "d": present_d = present_d | flip
                else:             present_p = present_p | flip

    j_v = _bern(cfg.p_jv)
    j_d = _bern(cfg.p_jd)
    j_p = _bern(0.0 if pointmap_off else cfg.p_jp)

    # Third pointmap knob, same treatment as present_p / j_p: with no pointmap
    # stream there is nothing to predict cross-modally. Observably a no-op today
    # — every reader of cm_p sits behind an `Sp = ptpf = 0` guard that makes the
    # value moot — so this is for consistency, not a fix: it keeps all three
    # knobs neutralized together rather than leaving one live and its safety
    # resting on a guard elsewhere.
    cm_p = bool(cfg.cross_modal_predict_pointmap) and not pointmap_off

    # Force joint=False when stream absent and not cross-modal-predicted.
    if not cfg.cross_modal_predict_video:
        j_v = j_v & present_v
    if not cfg.cross_modal_predict_dino:
        j_d = j_d & present_d
    if not cm_p:
        j_p = j_p & present_p

    return FlexBatchFlags(
        present_v=present_v, present_d=present_d, present_p=present_p,
        j_v=j_v, j_d=j_d, j_p=j_p,
        cm_v=bool(cfg.cross_modal_predict_video),
        cm_d=bool(cfg.cross_modal_predict_dino),
        cm_p=cm_p,
    )
