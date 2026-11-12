# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse and enforce XPolicyLab RL capability gates."""

from __future__ import annotations

from typing import Any, Iterable

RL_MODE_EVAL = "eval"
RL_MODE_SFT = "sft"
RL_MODE_PPO = "ppo"
RL_MODE_GRPO = "grpo"
RL_MODE_NFT = "nft"
SUPPORTED_RL_MODES = (RL_MODE_EVAL, RL_MODE_SFT, RL_MODE_PPO, RL_MODE_GRPO, RL_MODE_NFT)
# Modes whose rollout must record log-probs and whose update recomputes them.
RL_TRAIN_MODES = (RL_MODE_PPO, RL_MODE_GRPO)
RL_SAMPLE_MODES = (*RL_TRAIN_MODES, RL_MODE_NFT)


def normalize_rl_modes(modes: Iterable[str] | None) -> tuple[str, ...]:
    """Lower-case, de-duplicate, and validate RL mode names."""
    if not modes:
        return (RL_MODE_EVAL,)
    seen: list[str] = []
    for raw in modes:
        mode = str(raw).strip().lower()
        if mode not in SUPPORTED_RL_MODES:
            raise ValueError(
                f"Unknown RL mode {raw!r}. Supported modes: {SUPPORTED_RL_MODES}."
            )
        if mode not in seen:
            seen.append(mode)
    return tuple(seen) if seen else (RL_MODE_EVAL,)


def get_declared_rl_modes(policy: Any) -> tuple[str, ...]:
    """Read the modes the policy enables (``ModelTemplate.get_rl_modes``)."""
    getter = getattr(policy, "get_rl_modes", None)
    if callable(getter):
        return normalize_rl_modes(getter())
    return (RL_MODE_EVAL,)


def require_rl_mode(policy: Any, required_mode: str) -> tuple[str, ...]:
    """Fail fast if ``required_mode`` is not in the policy's declared modes."""
    required = str(required_mode).strip().lower()
    if required not in SUPPORTED_RL_MODES:
        raise ValueError(
            f"Unknown required RL mode {required_mode!r}. "
            f"Supported modes: {SUPPORTED_RL_MODES}."
        )
    declared = get_declared_rl_modes(policy)
    if required not in declared:
        policy_name = getattr(policy, "model_cfg", {})
        if isinstance(policy_name, dict):
            policy_name = policy_name.get("policy_name", type(policy).__name__)
        else:
            policy_name = type(policy).__name__
        raise RuntimeError(
            f"XPolicyLab policy {policy_name!r} does not allow RL mode {required!r}. "
            f"Enabled modes: {declared}. Add it to actor.model.xpolicylab.rl.modes "
            "if the policy lists it in RL_MODES."
        )
    return declared


def infer_required_rl_mode(model_cfg: Any) -> str:
    """Return ``actor.model.xpolicylab.required_mode`` (default: eval)."""
    xcfg = getattr(model_cfg, "xpolicylab", None)
    explicit = getattr(xcfg, "required_mode", None) if xcfg is not None else None
    if explicit:
        return str(explicit).strip().lower()
    return RL_MODE_EVAL
