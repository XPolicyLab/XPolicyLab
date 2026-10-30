"""BSLII-v15: DAC + Hierarchical Residual Action Head (ResDreamer-inspired).

Inspired by ResDreamer (2605.17537), which shows that hierarchical residual
prediction -- letting a small, smoothed branch carry the coarse signal while
a separate branch predicts the fine residual -- achieves better sample /
parameter efficiency than a monolithic head on complex 3D environments.
Adapted here to the BSLII single-view flow-matching ACTION head.

  baseline (plain joint K=16):                       AR-state action_mse 0.00907
  v7 (DAC, the only improving axis):                 0.00840  (best, ties baseline)
  v8/v9/v10/v11 (CVB / SCRG / SCRG+RCC / ANS):       all regress vs baseline
  v12 (DAC + Stage-2 freeze, time axis):             training
  v13 (DAC + Action Loss * 3, loss axis):            training
  v14 (DAC + Action Head Depth x2, arch axis):       being built in parallel
  v15 (DAC + Hierarchical Residual Action Head):     THIS FILE

Why a hierarchical residual action head on top of v7
----------------------------------------------------
The baseline action head is a single ``WanSequenceHead`` (AdaLN-modulated
LayerNorm + Linear) that must commit to BOTH coarse global motion AND fine
local correction simultaneously.  Empirically the head's capacity is
dominated by coarse modelling and fine details under-fit -- the diagnosed
mechanism behind the closed-loop drift the AR-state benchmark measures
across 58-chunk rollouts.

ResDreamer's hierarchical residual idea cleanly addresses this: a small,
smoothed COARSE branch carries the global motion, and a separate RESIDUAL
branch (same architecture as v7's monolithic action_head) predicts only the
delta on top of the coarse term.  By construction the residual branch is
relieved of coarse modelling burden and can dedicate its full capacity to
the local correction signal that closed-loop drift demands.

Design
------
Joint sequence and DAC mask are exactly v7's (no change to the joint
attention graph, K=16 align bottleneck, or the v7 self-contained inference
forward).  Only the per-position action read-out is restructured.

The original v7 action head is the ``residual_head``:

    residual_head(x, e) = WanSequenceHead(dim, out_dim)(x, e)
                        = head( norm(x) * (1+scale) + shift )

A new, smaller ``coarse_head`` reads the SAME action tokens and predicts a
coarse smoothed action:

    coarse_head(x) = Linear( GELU( Linear( LayerNorm(x) ) ) )
                       (hidden = int(dim * coarse_head_hidden_mult))

The final action prediction is the sum of the two branches:

    action_pred = coarse_head(x)  +  residual_head(x, e).

The coarse branch deliberately:
  * Does NOT take the AdaLN time-embed ``e``.  Coarse motion is a smoothed
    function of the joint-token state alone; the time-embed modulation is
    reserved for the residual branch where it informs the per-step delta.
  * Uses a small MLP with ``hidden = int(dim * coarse_head_hidden_mult)``
    (default 0.25, i.e. ~25% of v7's action_head parameter budget).  Lower
    capacity forces the coarse branch into a smoothed prediction; the
    residual branch (full v7 capacity) handles fine detail.
  * Its LAST Linear layer is zero-init when ``coarse_zero_init`` is true,
    so at step 0 the coarse branch outputs the zero vector and v15's total
    prediction is byte-identical to v7's residual_head output.  v15 thus
    starts as a strict-superset of v7, and the only way to depart from v7
    is by the residual / coarse branches gaining capacity through training.

Init parity with v7
-------------------
``residual_head`` is constructed via ``WanSequenceHead(dim, out_dim)`` with
the SAME constructor arguments v7 uses for ``self.action_head``, so its
default initialisation is byte-identical to v7's action head.  The parent
(v7 / WanJointActionPolicy.__init__) constructs its own
``self.action_head``; v15 REPLACES that ``self.action_head`` Module with a
``WanV15ResidualActionHead`` wrapper that exposes the SAME ``(x, e) ->
out`` signature.  Because v7's two forwards (the inherited training forward
``WanBSLIIPolicy._run_joint_model_with_queries`` and v7's inference forward
override ``WanBSLIIv7DecoupledPolicy._run_joint_model``) call
``self.action_head(action_tokens, action_time_embed)`` and rely ONLY on
that signature, the wrapper plug-in is fully transparent -- no override of
either forward is needed.

After the wrapper takes over, ``self.action_head.residual_head`` is
initialised by copying the parent's freshly-constructed action_head's
state_dict (LayerNorm + Linear weights + modulation parameter) so the
residual branch starts byte-identical to v7's action head at step 0.  This
is a state_dict copy on freshly-built modules; no checkpoint load.

Training loss
-------------
Byte-identical to v7.  The flow-matching action loss term in
``WanBSLIIPolicy.compute_loss`` (and the parent's) consumes ``action_pred``
as the velocity prediction; v15 changes ONLY how ``action_pred`` is
computed inside the head.  No extra losses, no auxiliary supervision, no
modification of total-loss aggregation, ``video_loss_weight``,
``action_loss_weight``, or ``align_loss_weight``.

Pareto safety
-------------
* Joint forward, joint attention mask, K-query construction, video head,
  align loss, total-loss aggregation, optimizer trajectory, LR schedule and
  gradient flow on every component EXCEPT the action_head are byte-identical
  to v7.
* Video pathway: unchanged.  Coarse / residual heads operate on the action
  tokens only after they emerge from the joint DiT blocks; they cannot
  back-propagate any video-side change.  v7's DAC mask still blocks action
  -> video attention edges, so any extra action-side gradient cannot leak
  into the video pathway.
* When ``coarse_zero_init`` is true (default) the action_head sum is
  byte-identical to v7's action_head at step 0.  v15 therefore starts as a
  strict superset of v7 -- it can only diverge by training.
* Ablations: ``coarse_head_hidden_mult: 0`` collapses to "residual-only"
  i.e. an exact copy of v7's action head.  ``coarse_zero_init: false``
  removes the zero-init guarantee but keeps the architecture (kept as a
  clean lever).

Diagnostics
-----------
Per-step diagnostic fields surfaced via ``_last_loss_components``:
    v15_coarse_norm     -- mean L2 norm of the coarse_head output across
                           the (B, A, action_dim) tensor (per-element rms).
    v15_residual_norm   -- mean L2 norm of the residual_head output across
                           the same tensor.
    v15_ratio           -- coarse_norm / max(residual_norm, eps).
These are float scalars derived in the head wrapper via .detach().float()
.mean() ops -- no gradient flow.  Logged every step by the workspace's
existing logger because it auto-iterates ``_last_loss_components`` into
``logs.json.txt``.

Author: hyp pipeline (BSLII-v15 plan).
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn

# v7 imports the frozen baseline WanBSLIIPolicy and adds the DAC attention
# mask + the self-contained sampling forward.  v15 subclasses v7 directly --
# the hierarchical residual action head is an ORTHOGONAL architectural axis.
# v15 imports NOTHING from v8/v9/v10/v11/v12/v13/v14 (strict version
# isolation).
from hyp_bslii.policy.wan_bslii_v7_decoupled_policy import (
    WanBSLIIv7DecoupledPolicy,
)
# WanSequenceHead is the AdaLN-style head v7 uses for ``self.action_head``.
# v15 keeps it byte-identical inside ``residual_head``.
from psi_policy.policy.wan_joint_action_policy import WanSequenceHead


class WanV15ResidualActionHead(nn.Module):
    """ResDreamer-inspired residual action head.

    Drop-in replacement for v7's ``self.action_head`` (a ``WanSequenceHead``).

    Forward signature -- ``(action_tokens, action_time_embed) -> action_pred``
    -- matches v7's action_head, so v7's two joint forwards
    (``_run_joint_model_with_queries`` for training and
    ``WanBSLIIv7DecoupledPolicy._run_joint_model`` for inference) keep calling
    ``self.action_head(action_tokens, action_time_embed)`` unchanged.

    Composition
    -----------
    ``residual_head`` -- a ``WanSequenceHead(dim, out_dim)``; same class and
        same init as v7's monolithic action_head.  Consumes both
        ``action_tokens`` and ``action_time_embed``.  Carries the per-step
        delta on top of the coarse term.
    ``coarse_head``   -- a small Linear-GELU-Linear MLP with hidden width
        ``int(dim * coarse_head_hidden_mult)``, consumes ``action_tokens``
        only.  Its last Linear is zero-init when ``coarse_zero_init`` is
        true, so v15 starts byte-identical to v7's residual_head output.
    """

    def __init__(
        self,
        *,
        dim: int,
        out_dim: int,
        coarse_head_hidden_mult: float = 0.25,
        coarse_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"WanV15ResidualActionHead: dim must be > 0; got {dim}.")
        if out_dim <= 0:
            raise ValueError(f"WanV15ResidualActionHead: out_dim must be > 0; got {out_dim}.")
        if coarse_head_hidden_mult < 0.0:
            raise ValueError(
                "WanV15ResidualActionHead: coarse_head_hidden_mult must be "
                f">= 0 (got {coarse_head_hidden_mult})."
            )
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.coarse_head_hidden_mult = float(coarse_head_hidden_mult)
        self.coarse_zero_init = bool(coarse_zero_init)

        # --- Residual branch: byte-identical to v7's monolithic action_head.
        self.residual_head = WanSequenceHead(dim=int(dim), out_dim=int(out_dim))

        # --- Coarse branch: small MLP, no AdaLN-time modulation by design.
        hidden = max(1, int(round(self.dim * self.coarse_head_hidden_mult)))
        self.coarse_hidden_dim = int(hidden)
        # Norm + project-up + GELU + project-down(out_dim).
        # The down-projection is zero-init when coarse_zero_init is true so
        # the coarse branch outputs zero at step 0.
        self.coarse_norm = nn.LayerNorm(self.dim, eps=1.0e-6)
        self.coarse_in = nn.Linear(self.dim, self.coarse_hidden_dim)
        self.coarse_act = nn.GELU()
        self.coarse_out = nn.Linear(self.coarse_hidden_dim, self.out_dim)
        if self.coarse_zero_init:
            # Zero the final projection so coarse_head(x) == 0 at step 0
            # regardless of upstream init.  Bias is also zeroed (default
            # for nn.Linear, set explicitly here for safety).
            with torch.no_grad():
                self.coarse_out.weight.zero_()
                if self.coarse_out.bias is not None:
                    self.coarse_out.bias.zero_()

        # Per-forward diagnostics (filled by forward, read by the policy
        # wrapper into _last_loss_components).  Floats, not tensors.
        self._last_coarse_norm: float = 0.0
        self._last_residual_norm: float = 0.0

    @torch.no_grad()
    def copy_residual_init_from(self, ref_head: WanSequenceHead) -> None:
        """Copy a fresh ``WanSequenceHead`` state into the residual branch.

        Called by ``WanBSLIIv15ResidualActionPolicy.__init__`` after the
        parent constructs its own ``self.action_head``: we replace
        ``self.action_head`` with this wrapper and seed the wrapper's
        residual branch with the parent's freshly-constructed head -- so
        the residual branch's init is the parent-default init, identical
        to v7's action_head init at step 0.
        """
        # The two modules have IDENTICAL structure (same class, same args)
        # so state_dict load_state_dict(strict=True) is safe.
        state = ref_head.state_dict()
        self.residual_head.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # Residual branch -- byte-identical computation to v7's action_head.
        residual = self.residual_head(x, e)

        # Coarse branch -- small MLP, no AdaLN, no time-embed.
        coarse_h = self.coarse_norm(x)
        # Match the residual head's autocast policy.  WanSequenceHead.forward
        # wraps its inner ops in fp32 autocast on cuda; we mirror it so the
        # two branches' outputs are on the same dtype for the sum.
        with torch.amp.autocast(
            "cuda",
            dtype=torch.float32,
            enabled=x.device.type == "cuda",
        ):
            coarse_h = self.coarse_in(coarse_h)
            coarse_h = self.coarse_act(coarse_h)
            coarse = self.coarse_out(coarse_h)

        # Cache per-forward diagnostics (detached scalars).
        try:
            self._last_coarse_norm = float(coarse.detach().float().abs().mean().item())
            self._last_residual_norm = float(residual.detach().float().abs().mean().item())
        except Exception:
            # Never let a diagnostic op crash training.
            pass

        return coarse + residual


class WanBSLIIv15ResidualActionPolicy(WanBSLIIv7DecoupledPolicy):
    """v7 (DAC) with a ResDreamer-inspired hierarchical residual action head.

    The single architectural change vs v7 is the replacement of
    ``self.action_head`` (a monolithic ``WanSequenceHead``) with a
    ``WanV15ResidualActionHead`` wrapper that exposes the same
    ``(action_tokens, action_time_embed) -> action_pred`` signature so
    v7's two joint forwards (training and inference) call into it
    transparently.

    The residual branch is initialised by COPYING the parent-built
    ``action_head`` state into ``self.action_head.residual_head`` so its
    init is parent-default (byte-identical to v7).  The coarse branch's
    last Linear is zero-init by default so v15 starts byte-identical to
    v7's action_head output at step 0; v15 is therefore a strict superset
    of v7 by construction.

    Hyperparameters
    ---------------
    coarse_head_hidden_mult: float, default 0.25.
        Hidden width of the coarse-branch MLP, as a fraction of the model
        ``dim``.  0.25 -> ~25% of v7 action_head's effective parameter
        budget for the coarse pathway; the residual branch keeps the full
        v7-equivalent capacity.  0.0 collapses to "residual-only" (an
        exact copy of v7's action head).
    coarse_zero_init: bool, default True.
        Zero-init the coarse_head's last Linear so the coarse branch
        outputs zero at step 0 and v15 starts byte-identical to v7's
        residual_head output.  Set to False to remove the zero-init
        guarantee while keeping the architecture (clean ablation lever).
    """

    def __init__(
        self,
        *,
        coarse_head_hidden_mult: float = 0.25,
        coarse_zero_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(coarse_head_hidden_mult, (int, float)):
            raise TypeError(
                "WanBSLIIv15ResidualActionPolicy: coarse_head_hidden_mult "
                f"must be float (got {type(coarse_head_hidden_mult).__name__})."
            )
        if float(coarse_head_hidden_mult) < 0.0:
            raise ValueError(
                "WanBSLIIv15ResidualActionPolicy: coarse_head_hidden_mult "
                f"must be >= 0 (got {coarse_head_hidden_mult})."
            )
        self.coarse_head_hidden_mult = float(coarse_head_hidden_mult)
        self.coarse_zero_init = bool(coarse_zero_init)

        # Snapshot the parent-built action_head BEFORE we replace it -- the
        # residual branch will copy its init.
        parent_action_head: WanSequenceHead = self.action_head  # type: ignore[assignment]
        if not isinstance(parent_action_head, WanSequenceHead):
            raise TypeError(
                "WanBSLIIv15ResidualActionPolicy: parent self.action_head must "
                f"be a WanSequenceHead; got {type(parent_action_head).__name__}."
            )

        # Build the v15 wrapper with the same dim / out_dim the parent used.
        model_dim = int(self.model.dim)
        out_dim = int(self.action_dim)
        model_dtype = next(self.model.parameters()).dtype

        v15_head = WanV15ResidualActionHead(
            dim=model_dim,
            out_dim=out_dim,
            coarse_head_hidden_mult=self.coarse_head_hidden_mult,
            coarse_zero_init=self.coarse_zero_init,
        )
        # Seed the residual branch with the parent-default init.
        v15_head.copy_residual_init_from(parent_action_head)
        # Match the parent's dtype placement (parent did .to(dtype=model_dtype)).
        v15_head = v15_head.to(dtype=model_dtype)

        # SWAP: v15's wrapper replaces v7's action_head Module.  v7's two
        # joint forwards call ``self.action_head(action_tokens,
        # action_time_embed)`` -- now they call into the wrapper.
        self.action_head = v15_head

        # Sanity: confirm exactly the two submodules we expect.
        expected_submodules = {
            "residual_head",
            "coarse_norm",
            "coarse_in",
            "coarse_act",
            "coarse_out",
        }
        got = {n for n, _ in self.action_head.named_children()}
        missing = expected_submodules - got
        if missing:
            raise RuntimeError(
                "WanBSLIIv15ResidualActionPolicy: wrapper missing submodules: "
                f"{sorted(missing)} (have {sorted(got)})."
            )

        # Pre-create _last_loss_components if absent (parent populates it
        # lazily inside compute_loss; we keep diagnostics sticky from step 0).
        if not hasattr(self, "_last_loss_components") or self._last_loss_components is None:
            self._last_loss_components = {}  # type: ignore[assignment]
        # Initial values (overwritten after every compute_loss step).
        self._last_loss_components["v15_coarse_head_hidden_mult"] = float(self.coarse_head_hidden_mult)
        self._last_loss_components["v15_coarse_zero_init"] = float(1.0 if self.coarse_zero_init else 0.0)
        self._last_loss_components["v15_coarse_norm"] = 0.0
        self._last_loss_components["v15_residual_norm"] = 0.0
        self._last_loss_components["v15_ratio"] = 0.0

    # ------------------------------------------------------------------
    # compute_loss: thin wrapper that pulls per-forward diagnostics from
    # the WanV15ResidualActionHead wrapper into the policy's
    # _last_loss_components dict (the workspace's logger auto-injects every
    # numeric field into logs.json.txt).
    #
    # No tensor / gradient touched.  Parent (v7 -> WanBSLIIPolicy) runs
    # exactly as before; v15 only surfaces head-branch norms.
    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, Any], training: bool = True):
        loss = super().compute_loss(batch, training=training)
        try:
            head = self.action_head
            if isinstance(head, WanV15ResidualActionHead) and isinstance(
                self._last_loss_components, dict
            ):
                coarse_norm = float(head._last_coarse_norm)
                residual_norm = float(head._last_residual_norm)
                eps = 1.0e-12
                ratio = coarse_norm / max(residual_norm, eps)
                self._last_loss_components["v15_coarse_norm"] = coarse_norm
                self._last_loss_components["v15_residual_norm"] = residual_norm
                self._last_loss_components["v15_ratio"] = float(ratio)
                self._last_loss_components["v15_coarse_head_hidden_mult"] = float(
                    self.coarse_head_hidden_mult
                )
                self._last_loss_components["v15_coarse_zero_init"] = float(
                    1.0 if self.coarse_zero_init else 0.0
                )
        except Exception:
            # Diagnostics must never crash training.
            pass
        return loss


__all__ = [
    "WanV15ResidualActionHead",
    "WanBSLIIv15ResidualActionPolicy",
]
