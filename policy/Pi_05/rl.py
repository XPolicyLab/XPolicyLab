"""XPolicyLab bridge to the pinned RLinf v0.4 native PyTorch Pi0.5.

Only the RL backend imports this module. Preprocessing remains owned by the
XPolicyLab adapter; sampling and likelihood recomputation belong to Pi0RL.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import torch

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.openpi import (
    _apply_openpi_param_dtypes,
    _freeze_after_load,
    _pi0_dtype_name,
    checkpoint,
)
from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL, Pi0RLConfig


@dataclasses.dataclass(frozen=True)
class Pi05RLConfig(Pi0RLConfig):
    noise_method: str = "flow_sde"
    noise_level: float = 0.5
    num_steps: int = 10
    action_chunk: int = 50
    action_env_dim: int = 14
    torch_compile: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> Pi05RLConfig:
        cfg = dict(cfg or {})
        # Explicit compatibility for the former no-op/default settings.
        if cfg.pop("safe_get_logprob", False):
            raise ValueError(
                "v0.4 requires real Gaussian logprobs; safe_get_logprob=True is unsupported."
            )
        if cfg.pop("value_hidden_sizes", None) is not None:
            raise ValueError(
                "v0.4 owns value head sizes; custom value_hidden_sizes are unsupported."
            )
        if cfg.pop("is_nft", False):
            raise ValueError("Pi_05 declares eval/PPO/GRPO only; NFT is not enabled.")
        unknown = set(cfg) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown Pi_05 rl_head keys: {sorted(unknown)}")
        if "noise_logvar_range" in cfg:
            cfg["noise_logvar_range"] = tuple(cfg["noise_logvar_range"])
        result = cls(**cfg)
        if result.noise_method not in (
            "flow_ode",
            "flow_sde",
            "flow_cps",
            "flow_noise",
        ):
            raise ValueError(f"Unsupported noise_method: {result.noise_method}")
        if result.num_steps < 2 or min(result.action_chunk, result.action_env_dim) <= 0:
            raise ValueError(
                "Pi05 requires num_steps >= 2 and positive action dimensions/horizon."
            )
        if result.value_vlm_mode not in ("mean_token", "first_token", "last_token"):
            raise ValueError(f"Unsupported value_vlm_mode: {result.value_vlm_mode}")
        lo, hi = result.noise_logvar_range
        if not 0 < lo <= hi or not 0 <= result.noise_level <= 1:
            raise ValueError(
                "Expected positive ordered noise std range and noise_level in [0, 1]."
            )
        if (
            result.add_value_head
            and not result.value_after_vlm
            and not result.joint_logprob
        ):
            raise ValueError(
                "v0.4 suffix values average different denoising steps during sampling/recompute; "
                "use value_after_vlm=true or joint_logprob=true with a value head."
            )
        return result

    def native(self) -> Pi0RLConfig:
        fields = {f.name for f in dataclasses.fields(Pi0RLConfig)}
        return Pi0RLConfig(
            **{k: v for k, v in dataclasses.asdict(self).items() if k in fields}
        )


def observation_from_inputs(inputs: dict[str, torch.Tensor]) -> Observation:
    """Translate transformed XPolicyLab tensors and normalize uint8 pixels once."""
    # OpenPI transforms retain uint8 pixels. The native Observation loader maps
    # them to [-1, 1]; replay must store that same normalized representation.
    return Observation.from_dict(
        {
            "image": {k[6:]: v for k, v in inputs.items() if k.startswith("image/")},
            "image_mask": {
                k[11:]: v for k, v in inputs.items() if k.startswith("image_mask/")
            },
            "state": inputs["state"],
            "tokenized_prompt": inputs["tokenized_prompt"],
            "tokenized_prompt_mask": inputs["tokenized_prompt_mask"],
        }
    )


def replay_observation(
    inputs: dict[str, torch.Tensor] | Observation,
) -> dict[str, torch.Tensor]:
    """Store the v0.4 preprocessed replay format, before image preprocessing."""
    obs = inputs if isinstance(inputs, Observation) else observation_from_inputs(inputs)
    return {
        **{f"obs_image__{k}": v.detach().clone() for k, v in obs.images.items()},
        **{
            f"obs_image_mask__{k}": v.detach().clone()
            for k, v in obs.image_masks.items()
        },
        "obs_state": obs.state.detach().clone(),
        "tokenized_prompt": obs.tokenized_prompt.detach().clone(),
        "tokenized_prompt_mask": obs.tokenized_prompt_mask.detach().clone(),
    }


class Pi05RLPolicy(Pi0RL):
    """Use the v0.4 sampler with XPolicyLab's task-specific transforms."""

    @torch.no_grad()
    def sample_preprocessed(
        self,
        inputs,
        *,
        mode="train",
        compute_values=True,
        noise=None,
        rng=None,
        **kwargs,
    ):
        del kwargs  # RLinf can pass model-independent sampling options.
        obs = observation_from_inputs(inputs)
        if mode == "eval":
            return self._predict_eval(obs, noise=noise, rng=rng)
        if mode != "train":
            raise ValueError(f"Expected train or eval, got {mode!r}.")
        if self.rl_cfg.noise_method == "flow_ode" or (
            self.rl_cfg.noise_method != "flow_noise" and self.rl_cfg.noise_level <= 0
        ):
            raise ValueError(
                "PPO/GRPO require a stochastic sampler with nonzero noise."
            )
        replay = replay_observation(obs)
        actions, extras = self._predict_train(
            obs, noise=noise, rng=rng, compute_values=compute_values
        )
        extras["forward_inputs"].update(replay)
        return actions, extras


def load_weights(policy: Pi0RL, path: str | Path) -> None:
    """Import base weights or a full XPolicyLab/RLinf model state (not optimizer state)."""
    path = Path(path)
    full = checkpoint.resolve_full_weights(path)
    safe = checkpoint.resolve_model_safetensors(path)
    if full is not None:
        from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

        state = checkpoint._normalize_state_dict(
            as_state_dict(torch.load(full, map_location="cpu", weights_only=True))
        )
        wrapped = [k.startswith("backbone.") for k in state]
        if any(wrapped):
            if not all(wrapped):
                raise ValueError(
                    "Mixed bare/backbone checkpoint keys are not supported."
                )
            state = {k[len("backbone.") :]: v for k, v in state.items()}
        state = checkpoint._convert_official_openpi_keys(state, full)
        incompatible = policy.load_state_dict(state, strict=False)
        missing = [
            k
            for k in incompatible.missing_keys
            if not k.startswith(("value_head.", "noise_head."))
        ]
        unexpected = [
            k
            for k in incompatible.unexpected_keys
            if not (k.startswith("value_head.") and not hasattr(policy, "value_head"))
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"Pi05 checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
    elif safe is not None:
        checkpoint.load_base_safetensors(policy, safe)
    else:
        raise FileNotFoundError(
            f"No model.safetensors, shard index or full_weights.pt found at {path}."
        )


def load_pi05_rl_policy(
    model_config,
    weight_path: str,
    rl_config: Pi05RLConfig,
    *,
    output_transform: Callable,
    precision=None,
    embed_dtype=None,
) -> Pi05RLPolicy:
    if not model_config.pi05:
        raise ValueError("Pi05RLPolicy requires pi05=True.")
    if (
        rl_config.action_chunk > model_config.action_horizon
        or rl_config.action_env_dim > model_config.action_dim
    ):
        raise ValueError(
            "Environment chunk/dimension cannot exceed the model horizon/dimension."
        )
    aliases = {"float32": "fp32", "bfloat16": "bf16"}
    target_dtype = torch_dtype_from_precision(aliases.get(precision, precision))
    activation_dtype = (
        target_dtype if embed_dtype is None
        else torch_dtype_from_precision(aliases.get(embed_dtype, embed_dtype))
    )
    native = Pi0Config(
        **{
            name: getattr(model_config, name)
            for name in (
                "pi05",
                "paligemma_variant",
                "action_expert_variant",
                "action_dim",
                "action_horizon",
                "max_token_len",
            )
        },
        dtype=_pi0_dtype_name(activation_dtype),
        discrete_state_input=getattr(model_config, "discrete_state_input", None),
    )
    set_torch_compile(rl_config.torch_compile)
    policy = Pi05RLPolicy(
        native,
        num_steps=rl_config.num_steps,
        action_chunk=rl_config.action_chunk,
        action_env_dim=rl_config.action_env_dim,
        rl_cfg=rl_config.native(),
    )
    policy._input_transform_fn = lambda value: (
        value
    )  # replay uses stored preprocessed tensors
    policy._output_transform_fn = output_transform
    load_weights(policy, weight_path)
    _freeze_after_load(policy, "rl")
    _apply_openpi_param_dtypes(policy, target_dtype)
    return policy
