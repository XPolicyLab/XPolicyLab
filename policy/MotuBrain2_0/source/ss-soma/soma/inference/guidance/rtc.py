"""SimpleRTC: gradient-free action blending for smooth chunk transitions.

Implements the soft-masking strategy from simple_rtc.md:
  A'[t] = w[t] * a_prev[t] + (1 - w[t]) * a_pred[t]

where w[t] is a decay weight based on inference delay d and fusion window L.

Pure functions, no state, torch.compile safe.
"""

from dataclasses import dataclass
import math

import torch

from soma.inference.config import SimpleRTCConfig


@dataclass(frozen=True)
class RTCState:
    """Prepared action constraint for one diffusion trajectory."""

    mode: str
    action_length: int
    constraint: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    prefix_length: int = 0

    def initialize(self, action: torch.Tensor) -> torch.Tensor:
        """Apply the constraint to newly initialized action noise."""
        return self.after_solver(action)

    def before_prediction(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare constrained actions and action timesteps for prediction."""
        if self.mode != "train" or self.prefix_length <= 0:
            return action, timestep.unsqueeze(0)
        action = self._lock_prefix(action)
        action_timestep = timestep.expand(self.action_length).clone()
        action_timestep[: self.prefix_length] = 0
        return action, action_timestep

    def after_solver(self, action: torch.Tensor) -> torch.Tensor:
        """Reapply the constraint after one solver update."""
        if self.constraint is None:
            return action
        if self.mode == "train":
            return self._lock_prefix(action)
        if self.weights is None:
            raise RuntimeError("simple RTC weights were not prepared")
        return rtc_blend(
            action.t(),
            self.constraint,
            self.weights,
        ).t()

    def _lock_prefix(self, action: torch.Tensor) -> torch.Tensor:
        if self.constraint is None or self.prefix_length <= 0:
            return action
        locked = action.clone()
        locked[:, : self.prefix_length] = self.constraint[
            : self.prefix_length
        ].to(action.dtype).t()
        return locked


class RTCGuidance:
    """Build per-run RTC state from model-space action constraints."""

    def __init__(
        self,
        config: SimpleRTCConfig,
        device: str | torch.device,
    ) -> None:
        self._config = config
        self._device = torch.device(device)

    def prepare(
        self,
        constraint: torch.Tensor | None,
        *,
        action_length: int,
        rtc_delay: int,
    ) -> RTCState:
        """Prepare simple or train-style RTC state for one diffusion run."""
        mode = self._config.type
        if mode not in ("simple", "train"):
            raise ValueError(
                "SimpleRTCConfig.type must be 'simple' or 'train', got "
                f"{mode!r}"
            )
        if not self._config.enabled or constraint is None:
            return RTCState(mode=mode, action_length=action_length)
        if constraint.ndim == 0:
            raise TypeError("RTC action constraint must have a sequence length")
        if len(constraint) == 0:
            return RTCState(mode=mode, action_length=action_length)

        delay = resolve_rtc_delay(self._config, rtc_delay)
        prefix_length = min(delay, len(constraint), action_length)
        if mode == "train":
            return RTCState(
                mode=mode,
                action_length=action_length,
                constraint=constraint,
                prefix_length=prefix_length,
            )

        fusion_window = self._config.fusion_window
        if fusion_window is None:
            fusion_window = len(constraint)
        else:
            fusion_window = min(fusion_window, len(constraint))
        weights = compute_rtc_weights(
            action_length,
            delay,
            fusion_window,
            self._config.decay,
            device=self._device,
            dtype=torch.float32,
        )
        return RTCState(
            mode=mode,
            action_length=action_length,
            constraint=constraint,
            weights=weights,
            prefix_length=prefix_length,
        )


def resolve_rtc_delay(rtc_cfg: SimpleRTCConfig, rtc_d: int) -> int:
    """Resolve the effective RTC delay from config and runtime metadata."""
    auto_delay = getattr(rtc_cfg, "auto_delay", True)
    delay = rtc_d if auto_delay else getattr(rtc_cfg, "delay", 0)
    return max(int(delay), 0)


def compute_rtc_weights(
    chunk_size: int,
    delay: int,
    fusion_window: int,
    decay: str = "linear",
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute per-timestep RTC blending weights.

    Args:
        chunk_size: H, total action steps in the chunk
        delay: d, steps strictly constrained to previous actions (w=1)
        fusion_window: second boundary. Weight decays from delay to this step.
        decay: "linear", "quadratic", or "exponential"
        device: torch device
        dtype: torch dtype

    Returns:
        w: [H] weights in [0, 1]. w[t]=1 means fully use a_prev,
           w[t]=0 means fully use model prediction.
    """
    step_idx = torch.arange(chunk_size, device=device, dtype=dtype)
    delay = max(int(delay), 0)
    decay_end = max(int(fusion_window), delay)

    if decay_end <= delay:
        base_weight = (step_idx < delay).to(dtype)
    else:
        # Before `delay`, weight is clamped to 1. After `decay_end`,
        # weight is clamped to 0. The interval in between decays linearly.
        base_weight = ((decay_end - step_idx) / (decay_end - delay)).clamp(0, 1)

    if decay == "linear":
        return base_weight
    if decay == "quadratic":
        return base_weight**2
    if decay == "exponential":
        # Smooth curve that preserves endpoints: f(0)=0 and f(1)=1.
        e_minus_1 = math.e - 1
        return base_weight * (torch.exp(base_weight) - 1) / e_minus_1
    raise ValueError(f"Unknown RTC decay type: {decay!r}")


def rtc_blend(
    a_pred: torch.Tensor,
    a_prev: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Blend model prediction with previous chunk's residual actions.

    Args:
        a_pred: [N, action_dim] model's current x-prediction (after ODE step)
        a_prev: [R, action_dim] unexecuted remainder from previous chunk (R <= N)
        weights: [N] blending weights from compute_rtc_weights

    Returns:
        blended: [N, action_dim] fused action prediction
    """
    R = min(a_prev.shape[0], a_pred.shape[0])
    if R == 0:
        return a_pred
    out = a_pred.clone()
    w = weights[:R].unsqueeze(-1)  # [R, 1] for broadcasting
    out[:R] = w * a_prev[:R].to(out.dtype) + (1 - w) * a_pred[:R]
    return out
