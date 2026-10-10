"""Rectified flow with clean-action prediction and velocity-space loss.

The path is x_t = (1 - t) * noise + t * action. sigma_min bounds the
velocity conversion near t=1; explicit velocity prediction remains supported."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowConfig:
    chunk_size: int = 32
    execute_steps: int = 16
    num_inference_steps: int = 5
    repeated_noise_draws: int = 8
    time_beta: tuple[float, float] = (1.5, 1.0)
    sigma_min: float = 0.05
    loss_type: str = "v"
    # x predicts clean actions; v predicts path velocity.
    pred_type: str = "x"
    # Unit-mean weights retain relative time weighting while stabilising the loss scale.
    normalize_weight: bool = True


class RectifiedFlow:
    def __init__(self, cfg: FlowConfig) -> None:
        self.cfg = cfg

    def sample_time(self, batch: int, device: torch.device) -> torch.Tensor:
        a, b = self.cfg.time_beta
        dist = torch.distributions.Beta(
            torch.tensor(a, device=device), torch.tensor(b, device=device)
        )
        # Keep t strictly below 1 so the velocity weight stays finite.
        return dist.sample((batch,)).clamp(max=1.0 - 1e-3)

    def interpolate(
        self, actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        tt = t.view(-1, *([1] * (actions.dim() - 1)))
        return (1.0 - tt) * noise + tt * actions

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Per-sample weight turning a sample-space error into velocity space."""
        if self.cfg.loss_type != "v":
            return torch.ones_like(t)
        w = 1.0 / (1.0 - t).clamp(min=self.cfg.sigma_min) ** 2
        if self.cfg.normalize_weight:
            w = w / w.mean().clamp(min=1e-6)
        return w

    def training_target(
        self, actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The regression target and per-sample weight for one noise draw.

        * ``pred_type == "x"``: the target is the clean action chunk and the
          weight is the v-space reweighting ``loss_weight(t)``.
        * ``pred_type == "v"``: the target is the path velocity
          ``actions - noise`` and the weight is 1 (v-space loss is native).
        """
        if self.cfg.pred_type == "v":
            return actions - noise, torch.ones_like(t)
        return actions, self.loss_weight(t)

    def to_velocity(
        self, prediction: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Instantaneous velocity implied by the network's raw output."""
        if self.cfg.pred_type == "v":
            return prediction
        tt = t.view(-1, *([1] * (x_t.dim() - 1)))
        return (prediction - x_t) / (1.0 - tt).clamp(min=self.cfg.sigma_min)

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler integration from noise to a clean chunk."""
        steps = self.cfg.num_inference_steps
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i * dt, device=device, dtype=torch.float32)
            pred = denoise_fn(x, t)
            x = x + dt * self.to_velocity(pred.float(), x.float(), t).to(x.dtype)
        return x
