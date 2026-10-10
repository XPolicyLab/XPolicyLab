"""Stream isolation and loss balancing.

These are the invariants the dual-head design rests on: a platform without
whole-body degrees of freedom must neither be supervised on them nor able to
attend to them, and a 58-dim head must not outweigh a 22-dim one.
"""

from __future__ import annotations

import torch

from mmabc.models.attention import FAR, NEAR, STREAM_ID, build_attention_mask
from mmabc.models.heads import future_alignment_loss, masked_action_loss

CHUNK, NEAR_STEPS, CTX, FUT = 6, 3, 5, 4
HEADS = {"manip": (0, 58), "aux": (58, 80)}


def _layout(include_future: bool = True):
    """Mirror MMABCJointExpert._token_layout."""
    ids, segs = [], []
    for name in ("manip", "aux"):
        ids.append(STREAM_ID[name])
        segs.append(NEAR)
        for step in range(CHUNK):
            ids.append(STREAM_ID[name])
            segs.append(NEAR if step < NEAR_STEPS else FAR)
    if include_future:
        ids += [STREAM_ID["future"]] * (2 * FUT)
        segs += [NEAR] * FUT + [FAR] * FUT
    return torch.tensor(ids), torch.tensor(segs)


def _sees(mask, ids, segs, b, qs, qseg, ks, kseg) -> bool:
    qi = [i for i in range(len(ids)) if ids[i] == STREAM_ID[qs] and segs[i] == qseg]
    ki = [i for i in range(len(ids)) if ids[i] == STREAM_ID[ks] and segs[i] == kseg]
    return bool(mask[b, 0][qi][:, ki].any())


def _mask(aux_active, context_valid=None, include_future=True):
    ids, segs = _layout(include_future)
    b = len(aux_active)
    cv = context_valid if context_valid is not None else torch.ones(b, CTX)
    return (
        build_attention_mask(
            ids, segs, aux_active=torch.tensor(aux_active), context_len=CTX, context_valid=cv
        ),
        ids,
        segs,
    )


def test_near_segment_never_reads_the_far_segment():
    """The executed half of the plan must not depend on the speculative tail."""
    mask, ids, segs = _mask([1.0])
    for qs in ("manip", "aux", "future"):
        for ks in ("manip", "aux", "future"):
            assert not _sees(mask, ids, segs, 0, qs, NEAR, ks, FAR), f"{qs} near read {ks} far"


def test_far_segment_reads_the_near_segment():
    mask, ids, segs = _mask([1.0])
    assert _sees(mask, ids, segs, 0, "manip", FAR, "manip", NEAR)
    assert _sees(mask, ids, segs, 0, "manip", FAR, "aux", NEAR)
    assert _sees(mask, ids, segs, 0, "future", FAR, "future", NEAR)
    # And remains bidirectional inside itself.
    assert _sees(mask, ids, segs, 0, "manip", FAR, "manip", FAR)


def test_action_heads_coordinate_within_a_segment():
    mask, ids, segs = _mask([1.0])
    assert _sees(mask, ids, segs, 0, "manip", NEAR, "aux", NEAR)
    assert _sees(mask, ids, segs, 0, "aux", NEAR, "manip", NEAR)


def test_aux_stream_is_cut_when_inactive():
    mask, ids, segs = _mask([1.0, 0.0])
    assert _sees(mask, ids, segs, 0, "manip", NEAR, "aux", NEAR)
    assert not _sees(mask, ids, segs, 1, "manip", NEAR, "aux", NEAR)
    assert not _sees(mask, ids, segs, 1, "manip", FAR, "aux", NEAR)


def test_future_never_leaks_into_actions():
    """The future frame is unavailable at deployment, so actions must not read it."""
    mask, ids, segs = _mask([1.0])
    for qs in ("manip", "aux"):
        for qseg in (NEAR, FAR):
            for kseg in (NEAR, FAR):
                assert not _sees(mask, ids, segs, 0, qs, qseg, "future", kseg)
    for kseg in (NEAR, FAR):
        assert not _sees(mask, ids, segs, 0, "future", NEAR, "manip", kseg)


def test_padding_context_tokens_are_never_attended():
    valid = torch.ones(1, CTX)
    valid[0, 2] = 0.0
    mask, ids, _ = _mask([1.0], context_valid=valid)
    assert not mask[0, 0][:, len(ids) + 2].any()
    assert mask[0, 0][:, len(ids) + 0].any()


def test_context_is_keys_only_so_mask_is_wider_than_tall():
    mask, ids, _ = _mask([1.0])
    assert mask.shape[-2] == len(ids)
    assert mask.shape[-1] == len(ids) + CTX


def test_every_token_can_attend_to_something():
    """A fully masked row would make softmax produce NaN."""
    mask, _, _ = _mask([0.0], context_valid=torch.zeros(1, CTX))
    assert mask.any(dim=-1).all()


def test_heads_contribute_equally_regardless_of_width():
    """Equal per-dim error in both heads gives equal loss, 14 dims vs 5."""
    target = torch.zeros(1, CHUNK, 80)
    mask = torch.zeros(1, CHUNK, 80)
    mask[:, :, 0:14] = 1
    mask[:, :, 58:63] = 1
    pred = {"manip": torch.zeros(1, CHUNK, 58), "aux": torch.zeros(1, CHUNK, 22)}
    pred["manip"][..., 0:14] = 1.0
    pred["aux"][..., 0:5] = 1.0
    loss, per_head, _ = masked_action_loss(pred, target, mask, HEADS)
    assert abs(per_head["manip"].item() - per_head["aux"].item()) < 1e-6
    assert abs(loss.item() - 1.0) < 1e-6


def test_inactive_head_produces_no_gradient():
    target = torch.zeros(2, CHUNK, 80)
    mask = torch.zeros(2, CHUNK, 80)
    mask[:, :, 0:14] = 1
    mask[0, :, 58:63] = 1  # only sample 0 has whole-body labels
    pm = torch.randn(2, CHUNK, 58, requires_grad=True)
    pa = torch.randn(2, CHUNK, 22, requires_grad=True)
    active = {"manip": torch.tensor([1.0, 1.0]), "aux": torch.tensor([1.0, 0.0])}
    loss, _, counts = masked_action_loss(
        {"manip": pm, "aux": pa}, target, mask, HEADS, head_active=active
    )
    loss.backward()
    assert pa.grad[1].abs().sum().item() == 0.0
    assert pa.grad[0].abs().sum().item() > 0.0
    assert pa.grad[0, :, 5:].abs().sum().item() == 0.0  # unlabelled dims
    assert pm.grad[1].abs().sum().item() > 0.0
    assert counts["aux"].item() == 1.0


def test_inactive_head_cannot_poison_the_loss_with_a_nonfinite_prediction():
    """An unsupervised head's output is unconstrained and may be non-finite.

    Nothing bounds a prediction that no label ever touches, and in an 8-node run
    the whole-body head reached inf on samples that had no whole-body labels.
    Masking by multiplication turned that into `inf * 0 = NaN`, so the head that
    was supposed to be strictly isolated took down the whole batch: every rank
    saw a NaN loss while the manipulation term stayed perfectly finite.
    """
    target = torch.zeros(2, CHUNK, 80)
    mask = torch.zeros(2, CHUNK, 80)
    mask[:, :, 0:14] = 1
    mask[0, :, 58:63] = 1  # only sample 0 has whole-body labels
    pm = torch.randn(2, CHUNK, 58, requires_grad=True)
    pa = torch.randn(2, CHUNK, 22)
    pa[1, :, 7] = float("inf")  # unsupervised slot on the unlabelled sample
    pa[1, :, 9] = float("nan")
    pa.requires_grad_(True)
    active = {"manip": torch.tensor([1.0, 1.0]), "aux": torch.tensor([1.0, 0.0])}

    loss, per_head, _ = masked_action_loss(
        {"manip": pm, "aux": pa}, target, mask, HEADS, head_active=active
    )
    assert torch.isfinite(loss), f"loss was {loss.item()}"
    assert torch.isfinite(per_head["aux"]), f"aux term was {per_head['aux'].item()}"

    # The gradient has to survive too: a finite loss reached through a NaN
    # backward is just as fatal and far harder to see.
    loss.backward()
    assert torch.isfinite(pa.grad).all()
    assert torch.isfinite(pm.grad).all()
    assert pa.grad[1].abs().sum().item() == 0.0


def test_per_sample_normalisation_equalises_dimension_counts():
    """A wide platform must not out-weigh a narrow one for having more joints."""
    target = torch.zeros(2, CHUNK, 80)
    mask = torch.zeros(2, CHUNK, 80)
    mask[0, :, 0:40] = 1  # wide
    mask[1, :, 0:4] = 1  # narrow
    pred = {"manip": torch.ones(2, CHUNK, 58), "aux": torch.zeros(2, CHUNK, 22)}
    _, per_head, _ = masked_action_loss(pred, target, mask, HEADS)
    assert abs(per_head["manip"].item() - 1.0) < 1e-6


def test_future_alignment_is_scale_invariant():
    pred = torch.randn(2, 8, 16)
    assert torch.allclose(
        future_alignment_loss(pred, pred * 7.0),
        torch.zeros(()),
        atol=1e-5,
    )


def test_future_alignment_survives_collapsed_and_nan_targets():
    """A black frame or a broken teacher must not NaN the step."""
    pred = torch.randn(2, 8, 16)
    tgt = torch.zeros(2, 8, 16)
    tgt[0, 0] = float("nan")
    tgt[1, 3] = float("inf")
    loss = future_alignment_loss(pred, tgt)
    assert torch.isfinite(loss)


def test_future_alignment_drops_invalid_horizons():
    pred = torch.ones(2, 4, 8)
    tgt = torch.ones(2, 4, 8)
    valid = torch.tensor([1.0, 0.0])
    loss = future_alignment_loss(pred, tgt, valid)
    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1e-5


def test_split_masks_drop_the_invisible_bank():
    from mmabc.models.attention import split_action_future_masks

    mask, ids, _ = _mask([1.0])
    n_act = int((ids != STREAM_ID["future"]).sum())
    n_fut = int((ids == STREAM_ID["future"]).sum())
    act, fut = split_action_future_masks(mask, n_act, n_fut)
    assert act.shape[-2] == n_act
    assert act.shape[-1] == n_act + CTX
    assert fut.shape[-2] == n_fut
    assert fut.shape[-1] == n_fut + CTX


def test_current_frame_selection_needs_a_timestep_axis():
    """Guards a bug that fed the backbone a 3-pixel strip at deployment.

    Training batches carry images as (B, V, T, H, W, 3) and the model selects
    the current frame with [:, :, 0]. A deployment path that omits the timestep
    axis produces a 5-dim tensor, where the same slice silently takes the first
    row of pixels instead. The shapes still flow through the processor, so
    nothing raises; the policy just stops seeing.
    """
    B, V, H, W = 2, 3, 8, 8
    correct = torch.zeros(B, V, 2, H, W, 3)
    assert correct[:, :, 0].shape == (B, V, H, W, 3)

    missing_axis = torch.zeros(B, V, H, W, 3)
    assert missing_axis[:, :, 0].shape != (B, V, H, W, 3)
    assert missing_axis[:, :, 0].shape == (B, V, W, 3)
