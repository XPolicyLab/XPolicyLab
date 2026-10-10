"""Block-sparse attention masks for manipulation, whole-body and future streams.

Action streams cannot read future tokens. Near-segment queries cannot read
far-segment keys; inactive whole-body tokens are masked from other streams."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# When on, every attention reports a NaN gradient together with the mask and
# input statistics that produced it. Off by default: it costs a hook and a
# finiteness scan per call, which is not worth paying for in a real run.
_DEBUG_SDPA = False


def set_sdpa_debug(enabled: bool) -> None:
    global _DEBUG_SDPA
    _DEBUG_SDPA = bool(enabled)

# Streams that own weights and produce queries. The context is not one of them.
STREAMS = ("manip", "aux", "future")
STREAM_ID = {name: i for i, name in enumerate(STREAMS)}

NEAR, FAR = 0, 1

# Stream visibility is [query][key]; action queries cannot read future tokens.
_STREAM_VISIBILITY = {
    ("manip", "manip"): True,
    ("manip", "aux"): True,
    ("manip", "future"): False,
    ("aux", "manip"): True,
    ("aux", "aux"): True,
    ("aux", "future"): False,
    ("future", "manip"): False,
    ("future", "aux"): False,
    ("future", "future"): True,
}


def stream_visibility(device: torch.device) -> torch.Tensor:
    vis = torch.zeros(len(STREAMS), len(STREAMS), dtype=torch.bool, device=device)
    for (q, k), ok in _STREAM_VISIBILITY.items():
        vis[STREAM_ID[q], STREAM_ID[k]] = ok
    return vis


def build_attention_mask(
    stream_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    aux_active: torch.Tensor,
    context_len: int,
    context_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-sample mask over queries and keys."""
    device = stream_ids.device
    B = aux_active.shape[0]
    Q = stream_ids.shape[0]

    vis = stream_visibility(device)
    self_mask = vis[stream_ids[:, None], stream_ids[None, :]]  # (Q, Q)

    # Segment causality: a near query never reads a far key. Applied across
    # streams as well as within one, so the near actions are also shielded from
    # the far half of the future prediction.
    near_q = (segment_ids == NEAR)[:, None]
    far_k = (segment_ids == FAR)[None, :]
    self_mask = self_mask & ~(near_q & far_k)

    mask = self_mask[None].expand(B, Q, Q).clone()

    # Remove the whole-body tokens for samples that have no whole-body labels:
    # they carry no information and must not be readable.
    is_aux = stream_ids == STREAM_ID["aux"]
    if bool(is_aux.any()):
        dead = (aux_active <= 0.0)[:, None] & is_aux[None, :]  # (B, Q)
        mask &= ~dead[:, :, None]
        mask &= ~dead[:, None, :]

    if context_len:
        # Every query may read the context, except its padding positions.
        ctx = (context_valid > 0.0)[:, None, :].expand(B, Q, context_len)
        mask = torch.cat([mask, ctx], dim=-1)

    # A row with no keys at all would make softmax produce NaN. Keep such a
    # token attending to itself; nothing reads it, so it stays inert.
    empty = ~mask.any(dim=-1)
    if bool(empty.any()):
        eye = torch.eye(Q, dtype=torch.bool, device=device)
        pad = torch.zeros(Q, context_len, dtype=torch.bool, device=device)
        diag = torch.cat([eye, pad], dim=-1)[None].expand(B, Q, Q + context_len)
        mask = mask | (diag & empty[:, :, None])
    return mask[:, None]


def split_action_future_masks(
    mask: torch.Tensor, n_action: int, n_future: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a packed [action | future | context] mask into two SDPA masks.

    Actions never attend to the future stream and the future stream never
    attends to actions, so those keys are dead weight. Dropping them shrinks
    the score matrix on the first noise draw, where the future bank is live.
    """
    action = torch.cat(
        [mask[:, :, :n_action, :n_action], mask[:, :, :n_action, n_action + n_future :]],
        dim=-1,
    )
    future = torch.cat(
        [
            mask[:, :, n_action : n_action + n_future, n_action : n_action + n_future],
            mask[:, :, n_action : n_action + n_future, n_action + n_future :],
        ],
        dim=-1,
    )
    return action, future


def _report_nan_grad(
    which: str,
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    """Describe an attention whose backward produced a NaN from a finite forward.

    Nothing downstream notices such a NaN until the entire gradient is ruined,
    and by then the origin is unrecoverable. The facts that separate the
    plausible causes are how many keys the affected rows were allowed to read
    and how large the inputs were, so both are printed beside the offending
    row indices.
    """
    import os
    import sys

    rank = os.environ.get("RANK", "?")
    bad = (~torch.isfinite(grad)).any(dim=-1)  # (B, H, N)
    b_idx, _, n_idx = bad.nonzero(as_tuple=True)
    n_q, n_k = q.shape[2], k.shape[2]
    lines = [
        f"[sdpa-nan r{rank}] grad_{which}: {int(bad.sum())} of {bad.numel()} slots non-finite; "
        f"q_len={n_q} k_len={n_k} "
        f"mask={None if mask is None else tuple(mask.shape)}",
        f"[sdpa-nan r{rank}] |q|max={float(q.abs().max()):.4g} "
        f"|k|max={float(k.abs().max()):.4g} |v|max={float(v.abs().max()):.4g}",
    ]
    if mask is not None:
        # grad_q is indexed by query, grad_k and grad_v by key, so the two need
        # different tallies: an unread query row and an unattended key column are
        # separate failure modes and only one of them is visible per tensor.
        per_query = mask.sum(dim=-1)
        per_key = mask.sum(dim=-2)
        lines.append(
            f"[sdpa-nan r{rank}] keys/query min={int(per_query.min())}; "
            f"queries/key min={int(per_key.min())}; "
            f"empty_query_rows={int((per_query == 0).sum())}; "
            f"unattended_keys={int((per_key == 0).sum())}"
        )
        if len(n_idx):
            pairs = list(zip(b_idx[:8].tolist(), n_idx[:8].tolist()))
            tally = per_query if which == "q" else per_key
            lines.append(
                f"[sdpa-nan r{rank}] first bad (batch,slot)={pairs}; "
                f"in_context={[bool(n >= n_q) for _, n in pairs]}; "
                f"counts={[int(tally[b, 0, n]) for b, n in pairs]}"
            )
    print("\n".join(lines), file=sys.stderr, flush=True)


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    dropout_p: float = 0.0,
    debug: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention over (B, H, N, D) with a bool mask."""
    dtype_in = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    if mask is not None and mask.dtype == torch.bool:
        # Halved so that adding it to a real score cannot overflow the dtype.
        floor = torch.finfo(q.dtype).min / 2
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(~mask, floor)
    elif mask is not None:
        mask = mask.float()

    if debug or _DEBUG_SDPA:
        def _check(g, name):
            if not torch.isfinite(g).all():
                try:
                    _report_nan_grad(name, g, q, k, v, mask)
                except Exception as exc:  # a diagnostic must never kill a run
                    print(f"[sdpa-nan] reporter failed: {exc!r}", flush=True)

        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.requires_grad:
                tensor.register_hook(lambda g, name=name: _check(g, name))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
    return out.to(dtype_in)
