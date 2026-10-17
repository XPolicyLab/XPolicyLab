"""Attach the test-adapted success/failure distillation objective to FastWAM.

Explicitly TEST-ADAPTED / NOT HELD OUT.  Wraps the ActionDiT block Linears in
LoRA adapters (the frozen `.base` layers remain the released policy) and loads
the oracle-episode lookup tables consumed by
``FastWAM._training_loss_success_distillation``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from fastwam.utils.logging_config import get_logger

from .lora import apply_lora_to_linear_suffixes

logger = get_logger(__name__)

DEFAULT_TARGET_SUFFIXES = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)


def attach_action_success_distillation(model, lora_cfg: dict | None, distill_cfg: dict | None) -> None:
    lora_cfg = dict(lora_cfg or {})
    distill_cfg = dict(distill_cfg or {})

    if lora_cfg.get("enabled", False):
        suffixes = tuple(lora_cfg.get("target_suffixes") or DEFAULT_TARGET_SUFFIXES)
        result = apply_lora_to_linear_suffixes(
            model.action_expert,
            target_suffixes=suffixes,
            rank=int(lora_cfg.get("rank", 16)),
            alpha=float(lora_cfg.get("alpha", lora_cfg.get("rank", 16))),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            init_seed=lora_cfg.get("init_seed", None),
        )
        expected = len(model.action_expert.blocks) * len(suffixes)
        if result.replaced != expected:
            raise ValueError(
                f"LoRA wrapped {result.replaced} Linear layers but expected {expected} "
                f"({len(model.action_expert.blocks)} blocks x {len(suffixes)} suffixes)"
            )
        model.action_lora_rank = int(lora_cfg.get("rank", 16))
        model.action_lora_alpha = float(lora_cfg.get("alpha", lora_cfg.get("rank", 16)))
        model.action_lora_save_merged = bool(lora_cfg.get("save_lora_merged", True))
        # The shared DiTBlock objects are wrapped in place, so the MoT mixture
        # sees the same LoRALinear modules; assert that identity explicitly.
        if model.mot.mixtures["action"] is not model.action_expert:
            raise RuntimeError("MoT action mixture is not the wrapped ActionDiT instance")

    model.action_success_distillation_enabled = bool(distill_cfg.get("enabled", False))
    if not model.action_success_distillation_enabled:
        return
    if not lora_cfg.get("enabled", False):
        raise ValueError("action_success_distillation requires action_lora_config.enabled=true")

    # Golden-noise self-distillation (HELD OUT, no test data): the frozen released
    # policy run through the deployed solver from the deployment's fixed action noise
    # (one CPU float32 tensor per run seed, re-seeded at every replan) on ordinary
    # training frames gives the target a*(s); the LoRA student is flow-matched toward
    # a*(s) under random noise/timesteps.  Nothing here touches RoboTwin test scenes.
    if str(distill_cfg.get("mode", "oracle")) == "golden":
        model.success_distill_mode = "golden"
        model.golden_noise_seed = int(distill_cfg.get("golden_noise_seed", 42))
        model.golden_num_inference_steps = int(distill_cfg.get("num_inference_steps", 10))
        shift = distill_cfg.get("sigma_shift", None)
        model.golden_sigma_shift = None if shift is None else float(shift)
        model.golden_target_weight = float(distill_cfg.get("golden_target_weight", 1.0))
        model.golden_data_weight = float(distill_cfg.get("golden_data_weight", 0.0))
        model.golden_num_noises = int(distill_cfg.get("golden_num_noises", 1))
        model.golden_select = str(distill_cfg.get("golden_select", "single"))
        if model.golden_num_inference_steps <= 0:
            raise ValueError("golden num_inference_steps must be positive")
        if model.golden_sigma_shift is not None and model.golden_sigma_shift <= 0:
            raise ValueError("golden sigma_shift must be positive or null")
        if model.golden_select not in ("single", "mean", "medoid"):
            raise ValueError(f"golden_select must be single|mean|medoid, got {model.golden_select!r}")
        if model.golden_select == "single" and model.golden_num_noises != 1:
            raise ValueError("golden_select=single requires golden_num_noises=1")
        if model.golden_select != "single" and model.golden_num_noises < 2:
            raise ValueError(f"golden_select={model.golden_select} requires golden_num_noises>=2")
        logger.warning(
            "Enabled GOLDEN-NOISE self-distillation (held out, no test data): noise_seed=%d num_noises=%d "
            "select=%s steps=%d shift=%s target_w=%.3f data_w=%.3f LoRA rank=%d",
            model.golden_noise_seed, model.golden_num_noises, model.golden_select,
            model.golden_num_inference_steps, model.golden_sigma_shift, model.golden_target_weight,
            model.golden_data_weight, model.action_lora_rank,
        )
        return
    model.success_distill_mode = "oracle"

    cases_path = Path(str(distill_cfg["oracle_cases_path"])).expanduser()
    rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no oracle cases in {cases_path}")
    indices = [int(row["output_episode_index"]) for row in rows]
    if sorted(indices) != list(range(len(rows))):
        raise ValueError("oracle_cases.jsonl output_episode_index must be exactly 0..N-1")

    failure_counts = Counter(
        (row["entry"]["task"], row["entry"]["phase"]) for row in rows if not row["entry"]["released_baseline_success"]
    )
    if not failure_counts:
        raise ValueError("oracle metadata contains no released-baseline failures")
    mean_count = sum(failure_counts.values()) / float(len(failure_counts))
    failure_lookup = torch.zeros(len(rows), dtype=torch.bool)
    balance_lookup = torch.ones(len(rows), dtype=torch.float32)
    for row in rows:
        index = int(row["output_episode_index"])
        entry = row["entry"]
        if not entry["released_baseline_success"]:
            failure_lookup[index] = True
            balance_lookup[index] = float(mean_count / failure_counts[(entry["task"], entry["phase"])])
    actual_failures = int(failure_lookup.sum().item())
    actual_anchors = len(rows) - actual_failures
    for key, actual in (("expected_failure_episodes", actual_failures), ("expected_anchor_episodes", actual_anchors)):
        expected = distill_cfg.get(key)
        if expected is not None and int(expected) != actual:
            raise ValueError(f"{key}: expected {int(expected)}, got {actual}")

    model._success_distill_failure_lookup = failure_lookup
    model._success_distill_balance_lookup = balance_lookup
    model.success_distill_original_dataset_index = int(distill_cfg.get("original_dataset_index", 0))
    model.success_distill_num_inference_steps = int(distill_cfg.get("num_inference_steps", 10))
    model.success_distill_sigma_shift = float(distill_cfg.get("sigma_shift", 5.0))
    model.success_distill_noise_seed = int(distill_cfg.get("noise_seed", 42))
    model.success_distill_correction_weight = float(distill_cfg.get("correction_weight", 1.0))
    model.success_distill_keep_weight = float(distill_cfg.get("keep_weight", 4.0))
    model.success_distill_failure_proximity_weight = float(distill_cfg.get("failure_proximity_weight", 0.25))
    model.success_distill_correction_sampling = str(distill_cfg.get("correction_sampling", "deployment_grid"))
    if model.success_distill_correction_sampling not in ("deployment_grid", "random"):
        raise ValueError("correction_sampling must be deployment_grid or random")
    for name in ("success_distill_correction_weight", "success_distill_keep_weight", "success_distill_failure_proximity_weight"):
        if getattr(model, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if model.success_distill_num_inference_steps <= 0 or model.success_distill_sigma_shift <= 0:
        raise ValueError("num_inference_steps and sigma_shift must be positive")
    logger.warning(
        "Enabled action success distillation (%s): correction_sampling=%s cases=%s failures=%d anchors=%d groups=%d "
        "LoRA rank=%d steps=%d shift=%.2f seed=%d weights=(fix %.3f, keep %.3f, prox %.3f)",
        "HELD OUT fresh-seed demos" if all(row["entry"].get("scene_split") == "train" for row in rows) else "TEST-ADAPTED",
        model.success_distill_correction_sampling,
        cases_path, actual_failures, actual_anchors, len(failure_counts), model.action_lora_rank,
        model.success_distill_num_inference_steps, model.success_distill_sigma_shift,
        model.success_distill_noise_seed, model.success_distill_correction_weight,
        model.success_distill_keep_weight, model.success_distill_failure_proximity_weight,
    )
