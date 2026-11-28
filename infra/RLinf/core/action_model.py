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

"""RLinf ``BasePolicy`` wrapper around an XPolicyLab ``Model``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .capabilities import (
    RL_MODE_NFT,
    RL_MODE_PPO,
    RL_SAMPLE_MODES,
    RL_TRAIN_MODES,
    require_rl_mode,
)
from .obs import env_obs_to_xpolicy_obs_list, pack_action_steps


def _rl_num_steps(xpolicy: Any) -> int | None:
    """Denoise step count the NFT worker reads from ``model.config.num_steps``."""
    candidates = [xpolicy, getattr(xpolicy, "policy", None), getattr(xpolicy, "model", None)]
    for obj in candidates:
        if obj is None:
            continue
        rl_config = getattr(obj, "rl_config", None)
        num_steps = getattr(rl_config, "num_steps", None)
        if num_steps is None:
            num_steps = getattr(obj, "num_steps", None)
        if num_steps is not None:
            return int(num_steps)
    return None


def _is_overridden_rl_method(instance: Any, name: str) -> bool:
    """True when the policy class overrides ``ModelTemplate``'s stub."""
    impl = getattr(type(instance), name, None)
    if impl is None or not callable(impl):
        return False
    qualname = getattr(impl, "__qualname__", "")
    return not str(qualname).startswith("ModelTemplate.")


class XPolicyLabForRLActionPrediction(nn.Module, BasePolicy):
    """Thin adapter: capability gate + obs/action pack + XPolicyLab RL methods.

    The XPolicyLab policy keeps its trainable ``nn.Module`` in ``policy`` or
    ``model``; it is registered here as ``backbone`` so FSDP, optimizers and
    weight sync see its parameters.
    """

    def __init__(
        self,
        xpolicy: Any,
        *,
        action_dim: int,
        num_action_chunks: int,
        required_mode: str,
    ):
        super().__init__()
        self.xpolicy = xpolicy
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.required_mode = str(required_mode).strip().lower()
        self.declared_modes = require_rl_mode(xpolicy, self.required_mode)
        if self.required_mode == RL_MODE_NFT:
            for name in ("predict_action_batch_rl", "nft_forward"):
                if not _is_overridden_rl_method(xpolicy, name):
                    raise NotImplementedError(f"NFT policy must implement {name}.")
        if self.required_mode in RL_TRAIN_MODES:
            missing = [
                name
                for name in ("predict_action_batch_rl", "compute_logprob_value")
                if not _is_overridden_rl_method(xpolicy, name)
            ]
            if missing:
                raise NotImplementedError(
                    f"XPolicyLab policy enables {self.required_mode!r} but does not "
                    f"implement {missing}."
                )
        self.action_type = getattr(xpolicy, "action_type", None) or getattr(
            xpolicy, "ACTION_TYPE", "joint"
        )
        self.robot_action_dim_info = getattr(xpolicy, "robot_action_dim_info", None)

        backbone = getattr(xpolicy, "policy", None)
        if not isinstance(backbone, nn.Module):
            backbone = getattr(xpolicy, "model", None)
        if isinstance(backbone, nn.Module):
            self.backbone = backbone
            # FSDP wrap policy reads these on the wrapper, not the inner module.
            self._no_split_modules = getattr(backbone, "_no_split_modules", None)
            self._no_split_names = getattr(backbone, "_no_split_names", None)
        else:
            if self.required_mode in RL_SAMPLE_MODES:
                raise TypeError("Training requires a real nn.Module on policy or model.")
            self.backbone = nn.Identity()
            self._no_split_modules = None
            self._no_split_names = None
        # NFT worker reads model.config.num_steps on this wrapper, not the inner policy.
        num_steps = _rl_num_steps(xpolicy)
        self.config = SimpleNamespace(
            num_steps=num_steps,
            action_chunk=self.num_action_chunks,
            action_dim=self.action_dim,
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if not isinstance(forward_type, ForwardType):
            forward_type = ForwardType(forward_type)
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        raise NotImplementedError(
            f"XPolicyLab adapter does not implement forward_type={forward_type}."
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs_list = env_obs_to_xpolicy_obs_list(
            env_obs,
            action_type=self.action_type,
            robot_action_dim_info=self.robot_action_dim_info,
        )
        # Train and in-training eval share the RL sampler. A pure eval run keeps
        # get_action_batch, so an unfinished template can still define the methods.
        if self.required_mode in RL_SAMPLE_MODES:
            actions, extras = self.xpolicy.predict_action_batch_rl(
                obs_list,
                mode=mode,
                compute_values=compute_values,
                **kwargs,
            )
        else:
            self.xpolicy.update_obs_batch(obs_list)
            action_lists = self.xpolicy.get_action_batch(
                env_idx_list=list(range(len(obs_list)))
            )
            actions = pack_action_steps(
                action_lists,
                action_type=self.action_type,
                robot_action_dim_info=self.robot_action_dim_info,
            )
            extras = {}

        actions = torch.as_tensor(np.asarray(actions, dtype=np.float32))
        if actions.ndim == 2:
            actions = actions[:, None, :]
        self._check_action_shape(actions)
        batch_size = actions.shape[0]

        extras = dict(extras or {})
        needs_logprob = mode == "train" and self.required_mode in RL_TRAIN_MODES
        needs_value = (
            needs_logprob and compute_values and self.required_mode == RL_MODE_PPO
        )
        prev_logprobs = extras.get("prev_logprobs")
        if prev_logprobs is None:
            if needs_logprob:
                raise RuntimeError(
                    f"{self.required_mode} rollout needs prev_logprobs from the "
                    "XPolicyLab policy; do not invent a fake logprob."
                )
            # NFT's inherited transport uses this field for batch/version shapes,
            # never for a probability ratio. PPO/GRPO still require real values.
            prev_logprobs = torch.zeros((batch_size, actions.shape[1]))
        prev_values = extras.get("prev_values")
        if prev_values is None:
            if needs_value:
                raise RuntimeError(
                    "PPO rollout needs prev_values; enable the policy value head."
                )
            prev_values = torch.zeros((batch_size, 1))
        extras["prev_logprobs"] = torch.as_tensor(prev_logprobs).float()
        extras["prev_values"] = torch.as_tensor(prev_values).float()
        forward_inputs = dict(extras.get("forward_inputs") or {})
        forward_inputs.setdefault("action", actions.reshape(batch_size, -1))
        extras["forward_inputs"] = forward_inputs
        return actions, extras

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_values: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.required_mode == RL_MODE_NFT:
            raise RuntimeError("NFT samples have no log-probs; use ForwardType.NFT.")
        if not _is_overridden_rl_method(self.xpolicy, "compute_logprob_value"):
            raise NotImplementedError(
                "XPolicyLab policy does not implement compute_logprob_value; "
                "PPO / GRPO updates are not available."
            )
        return self.xpolicy.compute_logprob_value(
            forward_inputs, compute_values=compute_values, **kwargs
        )

    def nft_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, Any]:
        if not _is_overridden_rl_method(self.xpolicy, "nft_forward"):
            raise NotImplementedError(
                "XPolicyLab policy does not implement nft_forward."
            )
        return self.xpolicy.nft_forward(forward_inputs, **kwargs)

    def set_global_step(self, global_step: int) -> None:
        method = getattr(self.backbone, "set_global_step", None)
        if callable(method):
            method(global_step)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        method = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if method is None:
            raise NotImplementedError("Backbone does not support gradient checkpointing.")
        method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def sft_forward(self, data: Any, **kwargs: Any) -> torch.Tensor:
        if not _is_overridden_rl_method(self.xpolicy, "sft_forward"):
            raise NotImplementedError(
                "XPolicyLab policy does not implement sft_forward."
            )
        sft = self.xpolicy.sft_forward
        if isinstance(data, tuple) and len(data) == 2:
            return sft(obs_list=data[0], actions=data[1])
        if isinstance(data, dict):
            if "obs_list" in data:
                return sft(
                    obs_list=data["obs_list"],
                    actions=data.get("actions", data.get("action")),
                )
            return sft(**data)
        return sft(data)

    def _check_action_shape(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: policy returned {actions.shape[-1]}, "
                f"expected actor.model.action_dim={self.action_dim}."
            )
        if actions.shape[1] > self.num_action_chunks:
            raise ValueError(
                f"Action horizon mismatch: policy returned {actions.shape[1]}, "
                f"expected <= actor.model.num_action_chunks={self.num_action_chunks}."
            )
