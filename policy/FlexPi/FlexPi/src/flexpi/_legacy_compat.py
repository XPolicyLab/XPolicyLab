"""Load FastWAM-lineage checkpoints on this codebase.

FlexPi is a cleaned-up rewrite of the FastWAM research codebase. The model
architecture was kept; what was dropped is ~30 ablation switches plus two
deployment modes. Checkpoints from that lineage therefore carry a `config.yaml`
whose `_target_`s are `fastwam.*` and whose model block contains keys
`create_flexpi` no longer accepts — even though the weights map onto this model
one-to-one.

This module is the compatibility entry point. The `fastwam` alias package (see
`src/fastwam/`) re-exports it so a legacy `_target_` resolves verbatim, with no
edit to the checkpoint and no rewritten config on disk.

Two deployment modes genuinely cannot load and are refused by name:

  pointmap_mode='encoder_cond'   pointmap tokenized by its own frozen ViT
                                 instead of the WAN VAE — different head modules
  disable_video_stream=True      CLWM-style 3-stream — different token layout

`FlexPi.load_checkpoint` already refuses both, but only once weight loading
starts. Checking here fails earlier and names the config key.

Every other removed key is dropped only under a stated condition that makes
dropping provably a no-op. A condition that does not hold, or a key not listed
at all, raises instead of being silently discarded: the failure mode being
guarded against is a checkpoint for which one of these knobs was load-bearing,
which would otherwise build a plausible-looking model with mismatched weights.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _is_off(v: Any) -> bool:
    return v is False or v is None or v == "none" or v == 0


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    """dict / DictConfig -> plain dict, resolving interpolations."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(obj, DictConfig):
            return dict(OmegaConf.to_container(obj, resolve=True))
    except ImportError:
        pass
    return dict(obj)


def FlexJointConfigType():
    """The already-built config class, so an instance is passed through untouched."""
    from .models.helpers.flex_joint import FlexJointConfig

    return FlexJointConfig


# (key, why-it-is-safe-to-drop, predicate over the whole legacy model cfg)
_DROPPABLE_WHEN: List[Tuple[str, str, Callable[[Dict[str, Any]], bool]]] = [
    # Pure ablation switches: dropping is a no-op exactly when they are off.
    ("video_pred_x0",                 "must be False",  lambda c: _is_off(c.get("video_pred_x0"))),
    ("action_pred_x0",                "must be False",  lambda c: _is_off(c.get("action_pred_x0"))),
    ("pointmap_pred_x0",              "must be False",  lambda c: _is_off(c.get("pointmap_pred_x0"))),
    ("use_video_rfg",                 "must be False",  lambda c: _is_off(c.get("use_video_rfg"))),
    ("use_dino_rfg",                  "must be False",  lambda c: _is_off(c.get("use_dino_rfg"))),
    ("use_pointmap_rfg",              "must be False",  lambda c: _is_off(c.get("use_pointmap_rfg"))),
    ("rfg_normalize_prior",           "must be False",  lambda c: _is_off(c.get("rfg_normalize_prior"))),
    ("frame_local_action_attn",       "must be False",  lambda c: _is_off(c.get("frame_local_action_attn"))),
    ("frame_local_cross_modal_attn",  "must be False",  lambda c: _is_off(c.get("frame_local_cross_modal_attn"))),
    ("aux_channel_concat",            "must be 'none'", lambda c: _is_off(c.get("aux_channel_concat"))),
    ("dino_first_frame_only",         "must be False",  lambda c: _is_off(c.get("dino_first_frame_only"))),
    ("pointmap_first_frame_only",     "must be False",  lambda c: _is_off(c.get("pointmap_first_frame_only"))),
    ("pointmap_no_first_frame_anchor","must be False",  lambda c: _is_off(c.get("pointmap_no_first_frame_anchor"))),
    ("dino_input_gate_enabled",       "must be False",  lambda c: _is_off(c.get("dino_input_gate_enabled"))),
    ("pointmap_input_gate_enabled",   "must be False",  lambda c: _is_off(c.get("pointmap_input_gate_enabled"))),
    ("pointmap_cam_patches",          "must be None",   lambda c: c.get("pointmap_cam_patches") is None),
    ("pointmap_cam_regions",          "must be None",   lambda c: c.get("pointmap_cam_regions") is None),
    ("pointmap_dim",                  "must be None",   lambda c: c.get("pointmap_dim") is None),
    ("pointmap_temporal_stride",      "must be 1 (no-op)", lambda c: c.get("pointmap_temporal_stride") in (None, 1)),

    # Consumed above: pointmap_mode is translated into enable_pointmap, and
    # disable_video_stream is rejected outright when set. Listed here so the
    # filter does not then report them as unrecognized.
    ("pointmap_mode",        "translated to enable_pointmap; must be 'vae_parallel' or None",
     lambda c: c.get("pointmap_mode", "vae_parallel") in (None, "vae_parallel")),
    ("disable_video_stream", "FlexPi always carries the video stream; must be False",
     lambda c: _is_off(c.get("disable_video_stream"))),

    # Gate init values are dead once the corresponding gate is off.
    ("dino_input_gate_init",     "requires dino_input_gate_enabled=False",
     lambda c: _is_off(c.get("dino_input_gate_enabled"))),
    ("pointmap_input_gate_init", "requires pointmap_input_gate_enabled=False",
     lambda c: _is_off(c.get("pointmap_input_gate_enabled"))),

    # FlexPi always builds dino_feature_norm as a LayerNorm, and its DINO output
    # head is always a plain Linear (dino_proj_out) — so these two only match
    # FlexPi's fixed behavior, they cannot select it.
    ("dino_feature_norm",     "FlexPi always applies it; must be True",
     lambda c: c.get("dino_feature_norm", True) is True),
    ("dino_unshuffle_head",   "FlexPi's DINO head is always Linear; must be 'linear'",
     lambda c: str(c.get("dino_unshuffle_head", "linear")) == "linear"),
    ("dino_unshuffle_mlp_dim","requires dino_unshuffle_head='linear' (then unused)",
     lambda c: str(c.get("dino_unshuffle_head", "linear")) == "linear"),
    ("dino_input_size",       "derived from dino_cam_patches in FlexPi; must be 224",
     lambda c: int(c.get("dino_input_size", 224)) == 224),

    # The pointmap ViT encoder only exists under pointmap_mode='encoder_cond',
    # which was removed. Under 'vae_parallel' the pointmap goes through the WAN
    # VAE and no separate backbone is constructed, so these are inert.
    ("pointmap_backbone",        "requires pointmap_mode='vae_parallel' (no separate encoder)",
     lambda c: c.get("pointmap_mode", "vae_parallel") == "vae_parallel"),
    ("pointmap_backbone_stage",  "requires pointmap_mode='vae_parallel'",
     lambda c: c.get("pointmap_mode", "vae_parallel") == "vae_parallel"),
    ("freeze_pointmap_encoder",  "requires pointmap_mode='vae_parallel'",
     lambda c: c.get("pointmap_mode", "vae_parallel") == "vae_parallel"),
    ("pointmap_pool_factor",     "requires pointmap_mode='vae_parallel'",
     lambda c: c.get("pointmap_mode", "vae_parallel") == "vae_parallel"),
]

# flex_joint is a 10-field dataclass here; the legacy block carries extra
# research knobs. Same rule: droppable only while provably inert.
_FLEX_JOINT_DROPPABLE_WHEN: List[Tuple[str, str, Callable[[Dict[str, Any]], bool]]] = [
    ("action_loss_weighting",          "must be False", lambda c: _is_off(c.get("action_loss_weighting"))),
    ("action_loss_weight_joint",       "requires action_loss_weighting=False",
     lambda c: _is_off(c.get("action_loss_weighting"))),
    ("action_loss_weight_action_only", "requires action_loss_weighting=False",
     lambda c: _is_off(c.get("action_loss_weighting"))),
    ("correlated_joint",               "must be False", lambda c: _is_off(c.get("correlated_joint"))),
    ("prob_curriculum",                "must be False", lambda c: _is_off(c.get("prob_curriculum"))),
    ("prob_curriculum_warmup_frac",    "requires prob_curriculum=False",
     lambda c: _is_off(c.get("prob_curriculum"))),
    ("regime_film_enabled",            "must be False", lambda c: _is_off(c.get("regime_film_enabled"))),
    ("regime_film_trunk_dim",          "requires regime_film_enabled=False",
     lambda c: _is_off(c.get("regime_film_enabled"))),
    ("regime_token_enabled",           "must be False", lambda c: _is_off(c.get("regime_token_enabled"))),
    ("regime_token_hidden",            "requires regime_token_enabled=False",
     lambda c: _is_off(c.get("regime_token_enabled"))),
]


def _filter(cfg: Dict[str, Any], accepted: set, rules, label: str) -> Dict[str, Any]:
    """Drop keys `accepted` does not contain, but only per an explicit rule."""
    by_key = {k: (why, pred) for k, why, pred in rules}
    kept, dropped, failed, unknown = {}, [], [], []

    for key, value in cfg.items():
        if key in accepted:
            kept[key] = value
            continue
        if key not in by_key:
            unknown.append(key)
            continue
        why, pred = by_key[key]
        if pred(cfg):
            dropped.append(key)
        else:
            failed.append(f"{key}={value!r} ({why})")

    if unknown or failed:
        lines = [f"Cannot load this legacy {label} config on FlexPi."]
        if failed:
            lines.append(
                "  These keys are set to values FlexPi cannot reproduce:\n    "
                + "\n    ".join(failed)
            )
        if unknown:
            lines.append(
                "  These keys are unrecognized — FlexPi may have removed the feature, "
                "or this compat layer needs a rule for them:\n    "
                + ", ".join(sorted(unknown))
            )
        lines.append(
            "  Dropping a key silently is refused on purpose: if it was load-bearing "
            "for this checkpoint, the model would build with mismatched weights and no error."
        )
        raise ValueError("\n".join(lines))

    if dropped:
        logger.info("[FlexPi compat] %s: dropped %d inert legacy key(s): %s",
                    label, len(dropped), ", ".join(sorted(dropped)))
    return kept


def create_fastwam_unified_joint(**kwargs: Any):
    """Compat entry point for `_target_: fastwam.runtime.create_fastwam_unified_joint`.

    Validates the legacy model config, translates it, and calls
    `flexpi.runtime.create_flexpi`. Nothing from the FastWAM codebase is
    imported — this is FlexPi building a FlexPi model.
    """
    from .runtime import create_flexpi

    # Loud, because a `fastwam` name exists on some machines for real: this line
    # is how you confirm which code actually built the model.
    logger.info("[FlexPi compat] legacy FastWAM config -> flexpi.runtime.create_flexpi")

    cfg = dict(kwargs)

    # --- the two removed deployment modes, refused by name -----------------
    mode = cfg.get("pointmap_mode", "vae_parallel")
    if mode is not None and mode != "vae_parallel":
        raise ValueError(
            f"pointmap_mode={mode!r} no longer exists in FlexPi (only 'vae_parallel' "
            "survives). The pointmap head modules differ, so these weights cannot map "
            "onto this model."
        )
    if cfg.get("disable_video_stream"):
        raise ValueError(
            "disable_video_stream=True no longer exists in FlexPi — it carries the video "
            "stream always, and that checkpoint has a different token layout."
        )

    # --- pointmap_mode (string, multi-mode) -> enable_pointmap (bool) -------
    # The one genuine semantic change: FlexPi kept a single pointmap path, so the
    # mode selector collapsed to an on/off switch.
    if "enable_pointmap" not in cfg and "pointmap_mode" in cfg:
        cfg["enable_pointmap"] = cfg["pointmap_mode"] is not None

    # --- nested flex_joint --------------------------------------------------
    # Hydra hands this over as a DictConfig, not a dict, so match on Mapping and
    # convert explicitly — an isinstance(..., dict) test silently skips the
    # filter and the extra keys then reach FlexJointConfig(**...) as a TypeError.
    fj = cfg.get("flex_joint")
    if fj is not None and not isinstance(fj, FlexJointConfigType()):
        from .models.helpers.flex_joint import FlexJointConfig
        import dataclasses

        fj_dict = _to_plain_dict(fj)
        fj_accepted = {f.name for f in dataclasses.fields(FlexJointConfig)}
        cfg["flex_joint"] = _filter(
            fj_dict, fj_accepted, _FLEX_JOINT_DROPPABLE_WHEN, "flex_joint"
        )

    accepted = set(inspect.signature(create_flexpi).parameters)
    return create_flexpi(**_filter(cfg, accepted, _DROPPABLE_WHEN, "model"))
