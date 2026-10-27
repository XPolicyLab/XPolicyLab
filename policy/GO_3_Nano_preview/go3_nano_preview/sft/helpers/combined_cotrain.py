"""Combined cotrain: fold vgen / ae / cotrain (and later vlm_rm) into ONE
optimizer update via gradient accumulation.

The batch-wise scheme (see PhaseScheduler) runs one phase per optimizer step
and toggles requires_grad per phase transition. The combined scheme instead
runs every objective as a *micro-step* inside a single accumulation window
(``gradient_accumulation_steps == number of micro-steps``). All params stay
trainable; which params a micro-step actually updates is decided by the forward
graph (e.g. the cotrain path detaches / no_grads VGen when cotrain_freeze_vgen
is set), never by requires_grad toggling.

Why a list, not a hardcoded sequence — the extension seam (口子): to fold in
vlm_rm later, add ``"vlm_rm"`` to ``order`` and bump
``gradient_accumulation_steps`` to match. The warmup invariant keeps
``len(build_micro_steps(...)) == grad_accum`` by substituting an immature slot
(cotrain / vlm_rm) with a mature objective, so ``gas`` never changes at runtime
(DeepSpeed fixes ``gradient_accumulation_steps`` at init).

This module is a pure helper: it imports nothing from the trainer/model/dataset
modules (so dynamic class loading never double-imports them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Objectives whose judge/reward signal is meaningless until VGen/AE have warmed
# up. During warmup their slot is replaced by ``warmup_substitute``.
_WARMUP_GATED = ("cotrain", "vlm_rm")

KNOWN_OBJECTIVES: Tuple[str, ...] = ("vgen", "ae", "cotrain", "vlm_rm", "vgen_ego")
_INDEPENDENT_BATCH: Tuple[str, ...] = ("vgen_ego",)
_DEFAULT_ORDER: Tuple[str, ...] = ("vgen", "ae", "cotrain")
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "vgen": 1.0,
    "ae": 1.0,
    "cotrain": 1.0,
    "vlm_rm": 1.0,
    "vgen_ego": 1.0,
}


@dataclass(frozen=True)
class MicroStep:
    """One objective inside a combined optimizer update."""

    name: str       # "vgen" | "ae" | "cotrain" | "vgen_ego" (later "vlm_rm")
    weight: float   # loss weight; sets the relative balance between objectives
    # True 时该微步自带数据源，训练循环必须为它单独取一份 batch。
    uses_independent_batch: bool = False


@dataclass(frozen=True)
class CombinedCotrainConfig:
    """Resolved combined-cotrain settings (parsed from ``args.combined_cotrain``)."""

    enabled: bool
    order: Tuple[str, ...]
    weights: Dict[str, float]
    warmup_steps: int
    warmup_substitute: str
    share_batch: bool
    # M3: compute the VLM hidden states + VAE latents once in the vgen micro-step
    # and reuse them in the ae / cotrain micro-steps (instead of recomputing).
    # Precondition: share_batch (all micro-steps must see the same batch for the
    # shared encode/VLM to be valid).
    share_features: bool
    fuse_vgen_ego: bool = False
    pipeline_ae: bool = False

    @property
    def grad_accum(self) -> int:
        """Required ``gradient_accumulation_steps`` == one full micro-step cycle."""
        return len(self.order)

    @property
    def execution_steps(self) -> int:
        """Physical backward calls; grad_accum remains the objective divisor."""
        return self.grad_accum - int(self.fuse_vgen_ego)


def parse_combined_cotrain(args) -> CombinedCotrainConfig:
    """Parse ``args.cotrain.combined_cotrain``. Returns a disabled config when absent."""
    cfg = getattr(args.cotrain, "combined_cotrain", None) or {}
    enabled = bool(cfg.get("enabled", False))

    order = tuple(cfg.get("order", _DEFAULT_ORDER))
    weights = dict(_DEFAULT_WEIGHTS)
    weights.update({k: float(v) for k, v in (cfg.get("weights", {}) or {}).items()})
    warmup_steps = int(cfg.get("warmup_steps", 0))
    warmup_substitute = str(cfg.get("warmup_substitute", "ae"))
    share_batch = bool(cfg.get("share_batch", False))
    share_features = bool(cfg.get("share_features", False))
    fuse_vgen_ego = bool(cfg.get("fuse_vgen_ego", False))
    pipeline_ae = bool(cfg.get("pipeline_ae", False))

    if pipeline_ae:
        if not fuse_vgen_ego:
            raise ValueError("pipeline_ae requires the fused VGen execution path")
        if not args.model.use_vgen or not args.model.use_ae:
            raise ValueError("pipeline_ae requires both VGen and AE objectives enabled")
        if args.cotrain.imf_jvp_impl != "fast":
            raise ValueError("pipeline_ae currently requires the deterministic fast JVP path")
        if args.data.ae_train is not None:
            raise ValueError("pipeline_ae requires the shared Robot train geometry")

    if fuse_vgen_ego:
        if not (enabled and share_batch and share_features):
            raise ValueError("fuse_vgen_ego requires the shared Robot execution path")
        if order != ("vgen", "ae", "cotrain", "vgen_ego") or warmup_steps != 0:
            raise ValueError("fuse_vgen_ego requires the four-objective order without warmup substitution")
        if weights["vgen"] != weights["vgen_ego"]:
            raise ValueError("fuse_vgen_ego currently requires equal Robot/Ego weights")
        if args.data.repeat_in_batch != 1 or args.data.video_repeat_in_batch != 1:
            raise ValueError("fuse_vgen_ego requires repeat_in_batch=video_repeat_in_batch=1")
        if args.model.gradient_vlm2vgen:
            raise ValueError("fuse_vgen_ego requires frozen VLM inputs")
        if not args.model.vae_before_vlm_input_prepare:
            raise ValueError("fuse_vgen_ego requires early VAE encoding for each source")
        if args.model.frame_concat_conditioning:
            raise ValueError("fuse_vgen_ego currently supports token conditioning only")

    if enabled:
        if not order:
            raise ValueError("combined_cotrain.order must be a non-empty list.")
        unknown = [name for name in order if name not in KNOWN_OBJECTIVES]
        if unknown:
            raise ValueError(
                f"combined_cotrain.order contains unknown objectives {unknown}; "
                f"valid objectives are {list(KNOWN_OBJECTIVES)}."
            )
        # 窗口的共享 batch 在第 0 个微步取出，所以首位必须是复用共享 batch 的
        # 目标；否则 share_batch 下后续微步会拿到独立数据源的 batch。
        if order[0] in _INDEPENDENT_BATCH:
            raise ValueError(
                f"combined_cotrain.order[0]={order[0]!r} uses an independent data "
                f"source and cannot lead the window; put a shared-batch objective "
                f"(e.g. 'vgen') first."
            )
        if warmup_steps < 0:
            raise ValueError(f"combined_cotrain.warmup_steps must be >= 0, got {warmup_steps}.")
        if warmup_substitute in _WARMUP_GATED:
            raise ValueError(
                f"combined_cotrain.warmup_substitute={warmup_substitute!r} is itself "
                f"warmup-gated; use a mature objective (e.g. 'vgen' or 'ae')."
            )
        if share_features and not share_batch:
            raise ValueError(
                "combined_cotrain.share_features requires share_batch=true: the ae / "
                "cotrain micro-steps reuse the vgen micro-step's VAE encode + VLM "
                "forward, which is only valid when all micro-steps see the same batch."
            )

    return CombinedCotrainConfig(
        enabled=enabled,
        order=order,
        weights=weights,
        warmup_steps=warmup_steps,
        warmup_substitute=warmup_substitute,
        share_batch=share_batch,
        share_features=share_features,
        fuse_vgen_ego=fuse_vgen_ego,
        pipeline_ae=pipeline_ae,
    )


def build_micro_steps(global_step: int, cfg: CombinedCotrainConfig) -> List[MicroStep]:
    """Return the micro-step sequence for the optimizer update at ``global_step``.

    Length is always ``cfg.grad_accum``. During warmup the cotrain / vlm_rm slot
    is replaced by ``cfg.warmup_substitute`` (so the judge never runs on an
    untrained VGen), but the list length — and therefore ``gas`` — is unchanged.
    ``global_step`` is constant across an accumulation window, so every
    micro-step in the same window agrees on whether it is in warmup.
    """
    in_warmup = global_step < cfg.warmup_steps
    steps: List[MicroStep] = []
    for name in cfg.order:
        effective = cfg.warmup_substitute if (in_warmup and name in _WARMUP_GATED) else name
        steps.append(
            MicroStep(
                name=effective,
                weight=cfg.weights.get(effective, 1.0),
                uses_independent_batch=effective in _INDEPENDENT_BATCH,
            )
        )
    return steps
