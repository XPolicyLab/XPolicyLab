from __future__ import annotations

import inspect
from typing import Dict, Optional, Tuple

import torch

from psi_policy.model.common.normalizer import LinearNormalizer
from psi_policy.model.diffusion.transformer_for_action import TransformerForAction
from psi_policy.model.observation.obs_composer import ObsComposer
from psi_policy.policy.base_policy import BaseImagePolicy


class PsiPolicy(BaseImagePolicy):
    """PSI Policy(v2): 与 v1 对齐的 RS-IMLE + Transformer 训练语义。"""

    def __init__(
        self,
        *,
        obs_composer: ObsComposer,
        action_dim: int,
        action_horizon: int,
        n_layer: int = 6,
        n_head: int = 4,
        p_drop_attn: float = 0.1,
        use_attn_mask: bool = False,
        n_samples_per_condition: int = 20,
        rs_imle_epsilon: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__()

        self.obs_composer = obs_composer
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.n_samples_per_condition = int(n_samples_per_condition)
        self.rs_imle_epsilon = float(rs_imle_epsilon)
        self.kwargs = kwargs

        if self.action_dim <= 0:
            raise ValueError("action_dim 必须为正整数")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon 必须为正整数")
        if self.n_samples_per_condition <= 0:
            raise ValueError("n_samples_per_condition 必须为正整数")

        obs_shape, obs_part_length = self.obs_composer.output_shape()
        n_emb = int(obs_shape[-1])
        obs_tokens = int(obs_shape[-2])

        self.model = TransformerForAction(
            input_dim=self.action_dim,
            output_dim=self.action_dim,
            action_horizon=self.action_horizon,
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_emb=n_emb,
            max_cond_tokens=obs_tokens,
            p_drop_attn=float(p_drop_attn),
            obs_part_length=obs_part_length,
            use_attn_mask=bool(use_attn_mask),
        )
        self.normalizer = LinearNormalizer()

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del output_path

        if "past_action" in obs_dict:
            raise NotImplementedError("past_action not implemented")

        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        obs_tokens = self.obs_composer(nobs, training=False)

        noise = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=self.device,
        )
        pred_actions, _ = self.model(noise, cond=obs_tokens, training=False, gen_attn_map=False)
        assert pred_actions.shape == (batch_size, self.action_horizon, self.action_dim), (
            f"pred_actions.shape={pred_actions.shape}, "
            f"expected=({batch_size}, {self.action_horizon}, {self.action_dim})"
        )
        pred_actions = self.normalizer["action"].unnormalize(pred_actions)
        return {"action": pred_actions, "action_pred": pred_actions}

    def bidirection_predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
        prev_traj: Optional[torch.Tensor] = None,
        latency_step: int = 0,
        execute_step: int = 8,
        n_sample: int = 32,
    ) -> Dict[str, torch.Tensor]:
        del output_path

        if prev_traj is None:
            return self.predict_action(obs_dict)

        if "past_action" in obs_dict:
            raise NotImplementedError("past_action not implemented")

        nobs = self.normalizer.normalize(obs_dict)

        obs_tokens = self.obs_composer(nobs, training=False).repeat_interleave(n_sample, dim=0)
        noise = torch.randn(
            n_sample,
            self.action_horizon,
            self.action_dim,
            device=self.device,
        )

        pred_actions, _ = self.model(noise, cond=obs_tokens, training=False, gen_attn_map=False)

        start = int(execute_step) + int(latency_step)
        prev_end = prev_traj[:, start:].reshape(1, -1)
        gen_start = pred_actions[:, : self.action_horizon - start].reshape(n_sample, -1)
        distances = torch.cdist(gen_start, prev_end)
        _, best_idx = distances.min(dim=0)

        chosen = pred_actions[best_idx]
        chosen = self.normalizer["action"].unnormalize(chosen)
        return {"action": chosen, "action_pred": chosen}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        lr: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        betas: Tuple[float, float] = (0.95, 0.999),
    ):
        param_dict = {name: p for name, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        if fused_available:
            return torch.optim.AdamW(optim_groups, lr=float(lr), betas=betas, fused=True)
        return torch.optim.AdamW(optim_groups, lr=float(lr), betas=betas)

    def rs_imle_loss(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        epsilon: float = 0.03,
    ) -> torch.Tensor:
        batch_size = real_samples.shape[0]
        num_samples = fake_samples.shape[1]

        real_flat = real_samples.reshape(batch_size, 1, -1)
        fake_flat = fake_samples.reshape(batch_size, num_samples, -1)

        distances = torch.cdist(real_flat, fake_flat).squeeze(1)
        valid_samples = (distances > float(epsilon)).float()

        max_dist = distances.max()
        min_dist, _ = (distances + (1 - valid_samples) * max_dist).min(dim=1)

        valid_real = (min_dist < max_dist).float()
        numerator = (min_dist * valid_real).sum()
        denominator = valid_real.sum()
        loss = numerator / denominator.clamp_min(1.0)
        return torch.where(denominator > 0, loss, torch.zeros_like(loss))

    def compute_loss(self, batch: dict, training: bool = True) -> torch.Tensor:
        assert "valid_mask" not in batch, "'valid_mask' should not be in batch"

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"].float())

        batch_size = nactions.shape[0]
        obs_tokens = self.obs_composer(nobs, training=training)

        noise = torch.randn(
            batch_size * self.n_samples_per_condition,
            *nactions.shape[1:],
            device=self.device,
        )
        repeated_cond = obs_tokens.repeat_interleave(self.n_samples_per_condition, dim=0)

        pred_actions, _ = self.model(
            noise,
            cond=repeated_cond,
            training=training,
            gen_attn_map=False,
        )
        pred_actions = pred_actions.reshape(batch_size, self.n_samples_per_condition, *nactions.shape[1:])
        return self.rs_imle_loss(nactions, pred_actions, epsilon=self.rs_imle_epsilon)

    def forward(self, batch: dict, training: bool = True) -> torch.Tensor:
        return self.compute_loss(batch=batch, training=training)


__all__ = ["PsiPolicy"]
