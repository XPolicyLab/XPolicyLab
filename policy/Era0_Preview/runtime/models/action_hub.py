# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping, Tuple, Dict, Type
import json
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------------------
    # Action representation and transforms
    # ---------------------------------------------------------------------


    def configure_target_mode(self, mode: str) -> None:
        """Optional hook for setting loss target type (e.g., action vs velocity)."""
        return None

    # ---------------------------------------------------------------------
    # Space-level hooks
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
        norm_key: Any | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """Default: return unchanged."""
        return action

    def normalize_action_for_metric(
        self,
        action: torch.Tensor,
        norm_key: Any | None = None,
    ) -> torch.Tensor:
        """Map raw actions into the space used for cross-sample error metrics."""
        return action

    def to_diffusion_state(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """
        Convert model output into the state used by the next denoising step.
        Default is identity (model output already in state space).
        """
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


def _normalize_valid_mask(ref: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(ref, dtype=ref.dtype, device=ref.device)
    mask = valid_mask.to(device=ref.device, dtype=ref.dtype)
    if mask.shape != ref.shape:
        raise ValueError(f"valid_mask shape mismatch: {mask.shape} vs {ref.shape}")
    return (mask > 0).to(dtype=ref.dtype)


# =============================================================================
# Implementations
# =============================================================================
@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    _GRIPPER_LOSS_TYPES = ("bce", "mse")

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self._target_mode = "action"
        # Gripper loss switch (Path A). Default "bce" preserves the released
        # X-VLA training contract and the head's logit-output convention: the
        # post-processed action still passes through torch.sigmoid at inference
        # (postprocess is unchanged). When set to "mse", the training-time loss
        # applies torch.sigmoid to the raw logits inside compute_loss BEFORE
        # taking MSE against a [0,1] target, so the eval-side [0,1] contract
        # and every downstream `> 0.5` threshold keep working without any
        # eval-code change. Configured via configure_gripper_loss().
        self._gripper_loss_type = "bce"
        # Gripper proprio switch. Default False preserves the released X-VLA
        # contract: preprocess zeroes proprio[..., gripper_idx] so the model
        # cannot condition on the current gripper open-ratio. Set True to feed
        # the live gripper open-ratio (already in [0,1]) into proprio, which is
        # required for meaningfully learning a per-step gripper delta (or any
        # gripper policy that depends on the current gripper state). Persisted
        # in the model config as `keep_gripper_proprio` so `from_pretrained`
        # restores it deterministically at eval; the RoboDojo eval server packs
        # the live gripper into proprio regardless -- this flag only decides
        # whether preprocess re-zeroes it before the transformer sees it.
        self._keep_gripper_proprio = False
        # Optional XYZ_SCALE override (default OFF -> class attribute 500.0).
        #
        # XYZ_SCALE=500 is calibrated for ABSOLUTE xyz targets. When the data
        # handler switches to anchor-relative xyz (e.g.
        # XVLA_ROBODOJO_ACTION_DELTA=1), the position MSE shrinks by the target
        # variance ratio and the position term loses loss share, silently
        # changing the loss balance alongside the parameterization. Setting this
        # override keeps the two variables separable.
        #
        # RoboDojo sim reference, from the absolute-mode baselines' tensorboard
        # (ratio of windowed means over steps 3000-10000; single-step values are
        # unusable, CV ~25%): position/rot6d loss ratio ~= 3.86, i.e. position
        # holds ~79.5% of (position + rot6d). Measured target variance ratio
        # abs/delta = 2.18x, so 500 * 2.18 = 1090 preserves that balance IF the
        # error MSE tracks the target variance -- see
        # scripts/train_xvla_robodojo_sim_delta.sh for the full caveat.
        _xyz_scale_raw = os.environ.get("XVLA_EE6D_XYZ_SCALE", "").strip()
        if _xyz_scale_raw:
            try:
                _xyz_scale = float(_xyz_scale_raw)
            except ValueError as exc:
                raise ValueError(
                    f"XVLA_EE6D_XYZ_SCALE must be float-convertible, got {_xyz_scale_raw!r}"
                ) from exc
            if not (_xyz_scale > 0.0):
                raise ValueError(f"XVLA_EE6D_XYZ_SCALE must be > 0, got {_xyz_scale}")
            logger.warning(
                "[ee6d] XYZ_SCALE overridden by XVLA_EE6D_XYZ_SCALE: %.1f -> %.1f",
                type(self).XYZ_SCALE, _xyz_scale,
            )
            self.XYZ_SCALE = _xyz_scale
        # Norm buffers — only active after load_norm_stats() (flow matching)
        self.register_buffer("_norm_mean", torch.zeros(self.dim_action))
        self.register_buffer("_norm_std", torch.ones(self.dim_action))
        self.register_buffer("_norm_enabled", torch.zeros(1))
        self.register_buffer("_norm_mode_id", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_norm_q_low", torch.full((self.dim_action,), -1.0))
        self.register_buffer("_norm_q_high", torch.full((self.dim_action,), 1.0))
        # Regression normalization is intentionally independent from the legacy
        # flow-matching buffers above.  RoboDojo's per-step delta action and
        # absolute proprio have different distributions, so they must not
        # share one set of EE20 statistics.
        self.register_buffer("_regression_action_mean", torch.zeros(self.dim_action))
        self.register_buffer("_regression_action_std", torch.ones(self.dim_action))
        self.register_buffer("_regression_action_q_low", torch.full((self.dim_action,), -1.0))
        self.register_buffer("_regression_action_q_high", torch.full((self.dim_action,), 1.0))
        self.register_buffer("_regression_proprio_mean", torch.zeros(self.dim_action))
        self.register_buffer("_regression_proprio_std", torch.ones(self.dim_action))
        self.register_buffer("_regression_proprio_q_low", torch.full((self.dim_action,), -1.0))
        self.register_buffer("_regression_proprio_q_high", torch.full((self.dim_action,), 1.0))
        self.register_buffer("_regression_norm_requested", torch.zeros(1))
        self.register_buffer("_regression_norm_stats_loaded", torch.zeros(1))
        self.register_buffer("_regression_norm_mode_id", torch.zeros(1, dtype=torch.long))

    def configure_target_mode(self, mode: str) -> None:
        mode_norm = str(mode).strip().lower()
        if mode_norm in {"action", "velocity"}:
            self._target_mode = mode_norm

    def configure_gripper_loss(self, kind: str) -> None:
        """Select the gripper loss type. See ``_gripper_loss_type`` for details.

        Args:
            kind: One of ``"bce"`` (default; released contract) or ``"mse"``
                (sigmoid the logits, then MSE against the [0,1] target).
        """
        kind_norm = str(kind).strip().lower()
        if kind_norm not in self._GRIPPER_LOSS_TYPES:
            raise ValueError(
                f"Unsupported gripper_loss_type={kind}. "
                f"Expected one of {self._GRIPPER_LOSS_TYPES}"
            )
        self._gripper_loss_type = kind_norm

    def configure_keep_gripper_proprio(self, keep: bool) -> None:
        """Toggle whether preprocess zeroes ``proprio[..., gripper_idx]``.

        See ``_keep_gripper_proprio`` for the full contract. Persisted to the
        model config so `from_pretrained` restores it at eval.
        """
        self._keep_gripper_proprio = bool(keep)

    def configure_regression_normalization(self, enabled: bool) -> None:
        """Enable the opt-in EE6D regression normalization contract."""
        self._regression_norm_requested.fill_(int(bool(enabled)))

    @staticmethod
    def _stats_tensor(payload: Mapping[str, Any], key: str, *, path: str) -> torch.Tensor:
        value = payload.get(key)
        if not isinstance(value, list) or len(value) != EE6DActionSpace.dim_action:
            raise ValueError(
                f"Invalid RoboDojo EE6D regression norm stats {path}: "
                f"{key!r} must contain {EE6DActionSpace.dim_action} values"
            )
        return torch.tensor(value, dtype=torch.float32)

    @staticmethod
    def _stats_quantile_tensor(payload: Mapping[str, Any], prefix: str, side: str, *, path: str) -> torch.Tensor:
        aliases = (
            f"{prefix}_q_{side}",
            f"{prefix}_q{'01' if side == 'low' else '99'}",
        )
        for key in aliases:
            if key in payload:
                return EE6DActionSpace._stats_tensor(payload, key, path=path)
        raise ValueError(
            f"Invalid RoboDojo EE6D regression norm stats {path}: missing one of {aliases}"
        )

    @staticmethod
    def _fix_gripper_stats(*values: torch.Tensor) -> None:
        # Gripper remains in raw [0, 1] space because regression uses logits
        # and BCE/MSE-on-sigmoid.  Its statistics are recorded for provenance
        # but never allowed to change that model/eval contract.
        for value, fill in zip(values, (0.0, 1.0, 0.0, 1.0)):
            value[list(EE6DActionSpace.gripper_idx)] = fill

    def _load_regression_norm_stats(self, payload: Mapping[str, Any], path: str) -> bool:
        if payload.get("schema") != "robodojo_ee6d_regression_norm_v1":
            return False
        action_mean = self._stats_tensor(payload, "action_mean", path=path)
        action_std = self._stats_tensor(payload, "action_std", path=path)
        action_low = self._stats_quantile_tensor(payload, "action", "low", path=path)
        action_high = self._stats_quantile_tensor(payload, "action", "high", path=path)
        proprio_mean = self._stats_tensor(payload, "proprio_mean", path=path)
        proprio_std = self._stats_tensor(payload, "proprio_std", path=path)
        proprio_low = self._stats_quantile_tensor(payload, "proprio", "low", path=path)
        proprio_high = self._stats_quantile_tensor(payload, "proprio", "high", path=path)
        self._fix_gripper_stats(action_mean, action_std, action_low, action_high)
        self._fix_gripper_stats(proprio_mean, proprio_std, proprio_low, proprio_high)
        for low, high, label in ((action_low, action_high, "action"), (proprio_low, proprio_high, "proprio")):
            if torch.any(high < low):
                raise ValueError(f"Invalid RoboDojo EE6D regression norm stats {path}: {label} q_high < q_low")
        self._regression_action_mean.copy_(action_mean)
        self._regression_action_std.copy_(action_std.clamp(min=1e-6))
        self._regression_action_q_low.copy_(action_low)
        self._regression_action_q_high.copy_(action_high)
        self._regression_proprio_mean.copy_(proprio_mean)
        self._regression_proprio_std.copy_(proprio_std.clamp(min=1e-6))
        self._regression_proprio_q_low.copy_(proprio_low)
        self._regression_proprio_q_high.copy_(proprio_high)
        self._regression_norm_mode_id.fill_(1)
        self._regression_norm_stats_loaded.fill_(1)
        return True

    def load_norm_stats(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if self._load_regression_norm_stats(payload, path):
            return
        if "mean" not in payload or "std" not in payload:
            raise ValueError(f"Invalid norm stats json (missing mean/std): {path}")
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        for gi in self.gripper_idx:
            if gi < len(mean):
                mean[gi] = 0.0
                std[gi] = 1.0
        self._norm_mean.copy_(mean)
        self._norm_std.copy_(std.clamp(min=1e-6))
        if "q_low" in payload and "q_high" in payload:
            low = torch.tensor(payload["q_low"], dtype=torch.float32)
            high = torch.tensor(payload["q_high"], dtype=torch.float32)
            for gi in self.gripper_idx:
                if gi < len(low):
                    low[gi] = -1.0
                    high[gi] = 1.0
            self._norm_q_low.copy_(low)
            self._norm_q_high.copy_(high)
            self._norm_mode_id.fill_(1)  # quantile
        else:
            self._norm_mode_id.fill_(0)  # zscore
        self._norm_enabled.fill_(1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_enabled.item() == 0:
            return x
        if self._norm_mode_id.item() == 1:  # quantile → [-1, 1]
            low = self._norm_q_low.to(x.device, x.dtype)
            high = self._norm_q_high.to(x.device, x.dtype)
            span = (high - low).clamp(min=1e-6)
            x = torch.max(torch.min(x, high), low)
            return (x - low) * (2.0 / span) - 1.0
        mean = self._norm_mean.to(x.device, x.dtype)
        std = self._norm_std.to(x.device, x.dtype)
        return (x - mean) / std

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_enabled.item() == 0:
            return x
        if self._norm_mode_id.item() == 1:  # quantile
            low = self._norm_q_low.to(x.device, x.dtype)
            high = self._norm_q_high.to(x.device, x.dtype)
            span = (high - low).clamp(min=1e-6)
            return (x + 1.0) * (0.5 * span) + low
        mean = self._norm_mean.to(x.device, x.dtype)
        std = self._norm_std.to(x.device, x.dtype)
        return x * std + mean

    @property
    def _regression_norm_active(self) -> bool:
        return (
            self._target_mode == "action"
            and bool(self._regression_norm_requested.item())
            and bool(self._regression_norm_stats_loaded.item())
        )

    def validate_regression_normalization(self) -> None:
        if not bool(self._regression_norm_requested.item()):
            return
        if self._target_mode != "action":
            raise ValueError("EE6D regression normalization requires action_loss_mode=regression")
        if not bool(self._regression_norm_stats_loaded.item()):
            raise ValueError(
                "EE6D regression normalization requires --action_norm_stats built by "
                "tools/robodojo/build_ee6d_regression_norm_stats.py"
            )

    def _normalize_regression(self, x: torch.Tensor, *, proprio: bool) -> torch.Tensor:
        if not self._regression_norm_active:
            return x
        if proprio:
            low, high = self._regression_proprio_q_low, self._regression_proprio_q_high
        else:
            low, high = self._regression_action_q_low, self._regression_action_q_high
        low = low.to(x.device, x.dtype)
        high = high.to(x.device, x.dtype)
        span = (high - low).clamp(min=1e-6)
        clipped = torch.maximum(torch.minimum(x, high), low)
        out = (clipped - low) * (2.0 / span) - 1.0
        out[..., self.gripper_idx] = x[..., self.gripper_idx]
        return out

    def _denormalize_regression_action(self, x: torch.Tensor) -> torch.Tensor:
        if not self._regression_norm_active:
            return x
        low = self._regression_action_q_low.to(x.device, x.dtype)
        high = self._regression_action_q_high.to(x.device, x.dtype)
        out = (x + 1.0) * (0.5 * (high - low).clamp(min=1e-6)) + low
        out[..., self.gripper_idx] = x[..., self.gripper_idx]
        return out


    @property
    def _fm_norm_active(self) -> bool:
        """True only when both flow-matching mode and norm stats are loaded."""
        return self._target_mode == "velocity" and bool(self._norm_enabled.item())

    def preprocess(self, proprio, action, mode="train", norm_key: Any | None = None):
        """
        action-mode (regression): zero gripper in action; optional q01/q99
        normalization is enabled only by the persisted EE6D regression switch.
        velocity-mode (flow): keep action gripper; normalize when load_norm_stats() was called.

        Gripper proprio zero-out is gated on ``_keep_gripper_proprio``. Default
        False preserves the released X-VLA contract (zero gripper proprio);
        True feeds the live gripper open-ratio to the model.
        """
        proprio_m = proprio.clone()
        action_m = action.clone()
        if not self._keep_gripper_proprio:
            proprio_m[..., self.gripper_idx] = 0.0
        if self._target_mode != "velocity":
            action_m[..., self.gripper_idx] = 0.0
        if self._fm_norm_active:
            proprio_m = self._normalize(proprio_m)
            action_m = self._normalize(action_m)
        elif self._regression_norm_active:
            proprio_m = self._normalize_regression(proprio_m, proprio=True)
            action_m = self._normalize_regression(action_m, proprio=False)
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """Denormalize active normalized modes, then map grippers into [0,1]."""
        if self._fm_norm_active:
            action = self._denormalize(action)
        elif self._regression_norm_active:
            action = self._denormalize_regression_action(action)
        if action.size(-1) > max(self.gripper_idx):
            if self._target_mode == "velocity":
                action[..., self.gripper_idx] = action[..., self.gripper_idx].clamp(0.0, 1.0)
            else:
                action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action

    def to_diffusion_state(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        # The regression model emits normalized non-gripper channels, while the
        # iterative state remains raw. This mirrors CrossEmbActionSpace and
        # prevents repeated quantile transforms from accumulating drift.
        if self._regression_norm_active:
            return self.postprocess(action, norm_key=norm_key)
        return action


@register_action("ee6d_padded")
class EE6DPaddedActionSpace(EE6DActionSpace):
    """EE6D with a wider model-facing tensor and masked zero-padding tail.

    The physical RoboDojo action remains the first 20 EE6D channels.  The
    remaining channels exist only to preserve Hy-VLA's 32-wide public action
    contract; they are never supervised or allowed into diffusion sampling.
    """

    _LOSS_PROFILE_HY_NORMALIZED_MSE = "hy_normalized_mse"
    _LOSS_PROFILE_OFFICIAL_EE6D_REGRESSION = "official_ee6d_regression"
    _OFFICIAL_GRIPPER_SCALE = 1.0
    _OFFICIAL_XYZ_SCALE = 500.0
    _OFFICIAL_ROT_SCALE = 10.0

    def __init__(self, real_dim: int = 20, max_dim: int = 32):
        if int(real_dim) != EE6DActionSpace.dim_action:
            raise ValueError(
                f"ee6d_padded requires real_dim={EE6DActionSpace.dim_action}, got {real_dim}"
            )
        if int(max_dim) < int(real_dim):
            raise ValueError(f"max_dim={max_dim} must be >= real_dim={real_dim}")
        super().__init__()
        self.real_dim = int(real_dim)
        self.dim_action = int(max_dim)
        self.dim_proprio = int(max_dim)
        self._loss_profile = self._LOSS_PROFILE_HY_NORMALIZED_MSE

    def configure_target_mode(self, mode: str) -> None:
        super().configure_target_mode(mode)
        self._validate_loss_profile()

    def configure_loss_profile(self, profile: str) -> None:
        normalized = str(profile or self._LOSS_PROFILE_HY_NORMALIZED_MSE).strip().lower()
        aliases = {
            "": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "default": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "hy": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "hy_normalized_mse": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "official": self._LOSS_PROFILE_OFFICIAL_EE6D_REGRESSION,
            "official_ee6d_regression": self._LOSS_PROFILE_OFFICIAL_EE6D_REGRESSION,
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {
            self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            self._LOSS_PROFILE_OFFICIAL_EE6D_REGRESSION,
        }
        if normalized not in allowed:
            raise ValueError(
                "ee6d_padded action_loss_profile must be one of "
                f"{sorted(allowed)}, got {profile!r}"
            )
        self._loss_profile = normalized
        self._validate_loss_profile()

    def _validate_loss_profile(self) -> None:
        if (
            self._loss_profile == self._LOSS_PROFILE_OFFICIAL_EE6D_REGRESSION
            and self._target_mode != "action"
        ):
            raise ValueError(
                "official_ee6d_regression requires action_loss_mode=regression."
            )

    def _active(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] < self.real_dim:
            raise ValueError(
                f"ee6d_padded expected at least {self.real_dim} channels, got {value.shape[-1]}"
            )
        return value[..., : self.real_dim]

    def preprocess(self, proprio, action, mode="train", norm_key: Any | None = None):
        """Prepare Hy's padded action condition without a second normalization.

        ``ee6d_padded`` must not invoke the legacy 20-D normalizer because the
        Hy adapter owns the 32-D state/action representation.  The official
        raw EE6D regression path nevertheless retains the released action
        condition: gripper logits are predicted, not fed back as noisy action
        inputs to the transformer.
        """
        del mode, norm_key
        if self._loss_profile != self._LOSS_PROFILE_OFFICIAL_EE6D_REGRESSION:
            return proprio, action
        action_m = action.clone()
        action_m[..., self.gripper_idx] = 0.0
        return proprio, action_m

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """Keep Hy normalized outputs intact; restore sigmoid for official regression."""
        del norm_key
        if self._loss_profile != self._LOSS_PROFILE_OFFICIAL_EE6D_REGRESSION:
            return action
        output = action.clone()
        output[..., self.gripper_idx] = torch.sigmoid(output[..., self.gripper_idx])
        return output


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """Joint-space layout with joints + gripper only."""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()


    def preprocess(self, proprio, action, mode="train", norm_key: Any | None = None):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot variant of EE6DActionSpace using MSE for all components."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()


    def preprocess(self, proprio, action, mode="train", norm_key: Any | None = None):
        """No preprocessing applied in AGIBOT variant."""
        return proprio, action

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """AGIBOT does not postprocess."""
        return action


@register_action("ee6d_mobile15")
class EE6DMobile15ActionSpace(BaseActionSpace):
    """RoboCasa PandaOmron mobile-manipulator action space (single-arm + base).

    Layout (dim_action=20 to remain warm-start-compatible with the standard
    EE6D 20D head; the trailing 5 dims are unused / masked to zero):

        [0:3]   ee_pos              ← same slot as EE6D POS_IDX_1
        [3:9]   ee_rot6d            ← same slot as EE6D ROT_IDX_1
        [9]     gripper (logit)     ← same slot as EE6D gripper[0]
        [10:14] base_motion         ← NEW: normalized OSC delta ∈ [-1, 1]
                                      channels 0..2 drive the mobile base;
                                      channel 3 drives the TORSO lift (gym
                                      wrapper maps base_motion[3:4] ->
                                      robot0_torso) and is identically 0 in the
                                      current dataset, kept for schema parity
        [14]    control_mode        ← NEW: {0: arm mode, 1: base mode}
        [15:20] unused (mask=0)

    Handler upstream must emit action of shape [T, 15] (with
    XVLA_ROBOCASA_ACTION_SPACE=mobile15) which then gets padded to [T, 20] by
    `_pad_or_trim(..., dim=20)` in lerobot_generic.py; the corresponding
    valid_dim [T, 15] is padded to [T, 20] with trailing zeros, so dims
    [15..19] are automatically masked out of the loss. All of dims [0..14] are
    supervised on every frame — base_motion carries an exact zero target on
    arm-mode frames rather than being masked out (see
    docs/memory-decisions.md, valid_mask semantics rule).

    Loss composition (five terms):
        position_loss    : MSE on [0:3]   * XYZ_SCALE
        rotate6D_loss    : MSE on [3:9]   * ROT_SCALE
        gripper_loss     : BCE on [9]     * GRIPPER_SCALE
        base_motion_loss : MSE on [10:14] * BASE_SCALE  (new)
        control_mode_loss: MSE on [14]    * MODE_SCALE  (new; 0/1 target)

    Reweighting (opt-in, via --reweight_base_motion in train/train.py):
        If enabled, BASE_SCALE ← BASE_SCALE * base_scale_ratio and
        MODE_SCALE ← MODE_SCALE * mode_scale_ratio at construction time.
        Defaults keep base+mode at the same magnitude as ee_pos + gripper,
        which is safe when composite tasks dominate the mix; reweight is
        needed if the base_motion loss collapses toward zero (model trivially
        predicts base=0 because the frame-share of active base motion is
        low; see docs/robocasa_v1_eval.md §4.6 for rationale).

    Cross-emb slot warning:
        The dims [10..14] overlap the `right_ee` slot of the cross_emb 20D
        bimanual layout (models/action_hub.py::CrossEmbActionSpace). Never
        combine this space with cross_emb bimanual data in a single run; the
        base+mode signal would collide with right-arm ee_pos/rot supervision.
    """

    dim_action = 20  # keep warm-start compatibility with EE6D 20D head

    # Slot definitions (indexes into the padded 20D action vector)
    EE_POS_IDX = (0, 1, 2)              # ee_pos (3)
    EE_ROT_IDX = (3, 4, 5, 6, 7, 8)     # ee_rot6d (6)
    GRIPPER_IDX = (9,)                  # ee_gripper (1) — BCE
    BASE_MOTION_IDX = (10, 11, 12, 13)  # base_motion (4) — MSE
    CONTROL_MODE_IDX = (14,)            # control_mode (1) — MSE (target 0/1)

    # Follow EE6D's magnitude convention for the arm part.
    #
    # ── BASE_SCALE history: 500 → 50 → 5 ──────────────────────────────────
    # base_motion is supervised on every frame but only carries non-zero signal
    # on the ~15% base-mode frames of the mixed atomic+composite data. A
    # base-side scale matching EE_POS (500) dominated total loss (~77%) and
    # starved the arm; 50 brought it to ~37%.
    #
    # 2026-08-01: lowered again 50 → 5. This OVERRULES the earlier
    # "不要动 BASE_SCALE" invariant in docs/memory-decisions.md, on evidence the
    # earlier analysis did not have. That analysis assumed the large base share
    # was benign ("arm 已收敛 + base 稀疏慢收敛"). Both premises are false:
    #
    #   1. Arm did NOT converge. position_loss plateaued at 0.236 (RMSE 21.7mm)
    #      vs 0.0352 (RMSE 8.4mm) for the arm10 config that reached ~35% SR;
    #      atomic_seen SR was 1/180.
    #   2. base_motion learned NOTHING, it is not "slowly converging":
    #      RMSE 0.062 ≈ std_gt 0.059 ⇒ R² ≈ 0, and the loss moved only
    #      0.247→0.192 between 35k and 115k steps.
    #   3. Single-variable controlled comparison (runs 2026-07-29 vs 07-30,
    #      identical train_args, only the base_motion/control_mode loss commit
    #      differs), matched at steps 85-95k:
    #          base share 73.1% → 36.6%
    #          position_loss   0.3453 → 0.2583  (-25.2%)
    #          rotate6D_loss   0.0548 → 0.0470  (-14.2%)
    #          gripper_loss    0.0666 → 0.0525  (-21.2%)
    #      ⇒ base's loss share DOES suppress arm learning. Not benign.
    #
    # Why base_motion is unlearnable here (so its share is irreducible *in this
    # architecture*, making down-weighting the correct call rather than a
    # band-aid): base_motion is a velocity-like command that is in fact highly
    # predictable — measured lag-1 autocorrelation 0.985-0.995 and a
    # "copy previous command" predictor scores R² ≈ 0.98. But the model is fed
    # a SINGLE timestep with no action history and no base velocity (18D proprio
    # adds base_pos/base_quat, i.e. position, not velocity). Under that partial
    # observability the command is genuinely ambiguous, and an L2 objective
    # answers ambiguity with the conditional mean — which for a signal that is 0
    # on 85% of frames collapses to ≈0. That is exactly the observed R²≈0.
    #
    # ⇒ The real fix for composite/base is flow matching (sample from the
    #   conditional distribution instead of averaging it), optionally plus
    #   action history / multi-frame input. NOT proprio, and NOT scale tuning.
    #   Until then, 5 keeps base out of the arm's way: it puts the loss
    #   composition back near the only configuration validated on RoboCasa
    #   (arm10, ~35% SR): position ~67%, rot ~13%, grip ~13%, base ~5%.
    #
    # Tuning note: prefer changing this constant over enabling
    # --reweight_base_motion (that flag MULTIPLIES the scale up and exists only
    # for the obsolete BASE_SCALE=500 semantics). --base_scale_ratio can still
    # be used for experiments, but it requires --reweight_base_motion.
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    GRIPPER_SCALE = 1.0
    BASE_SCALE = 5.0     # 500 → 50 → 5; see the block above before changing
    MODE_SCALE = 1.0     # MSE against 0/1; unit scale is sufficient

    # gripper_idx marks channels that are treated as logits: zeroed in the
    # flow-matching input (preprocess) and sigmoid-decoded at inference
    # (postprocess). ONLY dim 9 (arm gripper, BCE) belongs here.
    #
    # control_mode (dim 14) is deliberately NOT in gripper_idx. It is trained
    # with plain MSE against a 0/1 target and left as a normal regressed dim.
    # Rationale: BCE_with_logits pushes the logit unboundedly (larger logit =
    # smaller loss), and under the 16.5% base-frame class imbalance this
    # saturated sigmoid(dim14) to ~0.977 — the model predicted "base mode" on
    # 98-99% of arm-only eval frames, driving the base during atomic tasks
    # (see docs/memory-decisions.md). MSE on 0/1 keeps the target bounded and
    # symmetric, so it cannot run away to one class. Client-side decode uses a
    # plain `> 0.5` threshold on the raw (un-sigmoided) output.
    gripper_idx = (9,)

    def __init__(
        self,
        reweight_base_motion: bool = False,
        base_scale_ratio: float = 3.0,
        mode_scale_ratio: float = 3.0,
    ):
        """
        Args:
            reweight_base_motion: if True, multiply BASE_SCALE by base_scale_ratio and
                MODE_SCALE by mode_scale_ratio at construction time. Default: False
                (base + mode inherit their raw EE-comparable scale).
            base_scale_ratio: multiplier applied to BASE_SCALE when
                reweight_base_motion=True. Default: 3.0 (roughly makes base+mode
                jointly account for ~65% of total loss instead of ~40% under the
                pretrain_human300 mix; see docs/robocasa_v1_eval.md §4.6).
            mode_scale_ratio: multiplier applied to MODE_SCALE when
                reweight_base_motion=True. Default: 3.0.
        """
        super().__init__()
        self._reweight_base_motion = bool(reweight_base_motion)
        self._base_scale_ratio = float(base_scale_ratio)
        self._mode_scale_ratio = float(mode_scale_ratio)
        # Apply reweighting once at construction; class-level defaults stay pristine.
        self._effective_base_scale = self.BASE_SCALE * (
            self._base_scale_ratio if self._reweight_base_motion else 1.0
        )
        self._effective_mode_scale = self.MODE_SCALE * (
            self._mode_scale_ratio if self._reweight_base_motion else 1.0
        )


    def preprocess(self, proprio, action, mode="train", norm_key: Any | None = None):
        """Zero out gripper logit in proprio/action, like EE6D's action-mode path.

        base_motion + control_mode are NOT modified: they are already in the
        physical space the model should reproduce (normalized OSC delta ∈
        [-1, 1] for base, ±1 for control_mode).
        """
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., list(self.gripper_idx)] = 0.0
        action_m[..., list(self.gripper_idx)] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """Apply sigmoid to gripper logits; leave arm/base/mode dims as-is.

        The eval client (evaluation/robocasa/robocasa_client.py) is responsible
        for splitting the 20D output into arm(10) + base(4) + mode(1) and
        packing them into the robocasa gym env's dict action.
        """
        if action.size(-1) > max(self.gripper_idx):
            action[..., list(self.gripper_idx)] = torch.sigmoid(
                action[..., list(self.gripper_idx)]
            )
        return action


@register_action("auto")
class AutoActionSpace(BaseActionSpace):
    """
    Auto-detecting action space that adapts to any action dimension.

    - Model outputs max_dim for compatibility with pretrained models
    - Loss is computed only on the first real_dim dimensions
    - Postprocess trims output back to real_dim

    Args:
        real_dim: The actual action dimension from the dataset/policy feature
        max_dim: The model's output dimension for pretrained VLA compatibility
    """

    JOINTS_SCALE = 100.0

    def __init__(self, real_dim: int, max_dim: int = 20):
        super().__init__()
        self.real_dim = real_dim
        self.dim_action = max_dim  # Model-facing dimension
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Pad real_dim → max_dim (zeros for the dummy channels)."""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            # If dimension doesn't match either, pad/trim to real_dim first
            if x.size(-1) < self.real_dim:
                pad_shape = list(x.shape[:-1]) + [self.real_dim - x.size(-1)]
                pad = x.new_zeros(pad_shape)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.real_dim]

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Trim model output max_dim → real_dim."""
        return x[..., : self.real_dim]


    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
        norm_key: Any | None = None,
    ):
        """
        Pad action from real_dim to max_dim for the model.
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        """
        Trim model output from max_dim to real_dim for real robot control.
        """
        return self._trim_to_real_dim(action)


@register_action("dm05_native")
class DM05NativeActionSpace(BaseActionSpace):
    """Native DM05 action vector with unscaled masked MSE loss.

    DM05's flow-matching objective is a mean squared velocity error over all
    action channels. Unlike ``AutoActionSpace``, it must not apply the legacy
    joint-space ``JOINTS_SCALE=100`` multiplier or trim/pad the native head.
    """

    def __init__(self, action_dim: int = 32):
        super().__init__()
        self.real_dim = int(action_dim)
        self.dim_action = int(action_dim)
        self.dim_proprio = int(action_dim)


    def preprocess(self, proprio, action, mode="train", norm_key=None):
        del mode, norm_key
        return proprio, action


@register_action("cross_emb")
class CrossEmbActionSpace(BaseActionSpace):
    """
    Cross-embodiment canonical action space.

    Canonical layout (base_dim=60):
      left_ee(10) + left_hand(20) + right_ee(10) + right_hand(20)
    and reserve channels up to max_dim (default 64) for forward compatibility.
    """

    # Fixed semantic slots in the first 60 dims.
    LEFT_EE_SLICE = slice(0, 10)
    LEFT_HAND_SLICE = slice(10, 30)
    RIGHT_EE_SLICE = slice(30, 40)
    RIGHT_HAND_SLICE = slice(40, 60)
    GRIPPER_IDX = (9, 39)
    EE_EQ_POS_IDX_1 = (0, 1, 2)
    EE_EQ_POS_IDX_2 = (30, 31, 32)
    EE_EQ_ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    EE_EQ_ROT_IDX_2 = (33, 34, 35, 36, 37, 38)
    EE_EQ_KEEP_IDX = tuple(
        list(range(LEFT_EE_SLICE.start, LEFT_EE_SLICE.stop))
        + list(range(RIGHT_EE_SLICE.start, RIGHT_EE_SLICE.stop))
    )

    # Paper-aligned scales (match EE6DActionSpace XYZ_SCALE=500 / ROT_SCALE=10 /
    # GRIPPER_SCALE=1). Applied uniformly across ee + joint families so cross_emb
    # loss numbers are directly comparable to ee20d (and to each other across
    # rep × semantics ablations). HAND_SCALE bumped 20→500 so joint dims get the
    # same position-scale weight as EE position dims (joints normalized to ±1 ≈
    # EE xyz normalized magnitude).
    EE_SCALE = EE6DActionSpace.XYZ_SCALE       # 500
    HAND_SCALE = EE6DActionSpace.XYZ_SCALE     # 500 (joints treated as position-class)
    GRIPPER_SCALE = EE6DActionSpace.GRIPPER_SCALE  # 1
    EE_EQ_POS_SCALE = EE6DActionSpace.XYZ_SCALE
    EE_EQ_ROT_SCALE = EE6DActionSpace.ROT_SCALE
    EE_EQ_GRIPPER_SCALE = EE6DActionSpace.GRIPPER_SCALE
    DROID_STATE_CLIPPED_EE_LOSS_WEIGHT = 0.25
    _HAND_LOSS_NORMALIZATION_MODES = {
        "local_mean",
        "global_mean",
        "global_joint",
    }
    _REGRESSION_LOSSES = {"mse", "smooth_l1", "l1"}
    _GRIPPER_LOSS_TYPES = ("bce", "mse")
    _LOSS_PROFILE_HY_NORMALIZED_MSE = "hy_normalized_mse"
    _LOSS_PROFILE_CROSS_EMB_NATIVE = "cross_emb_native"
    _NORM_MODE_TO_ID = {"zscore": 0, "quantile": 1}
    _NORM_ID_TO_MODE = {0: "zscore", 1: "quantile"}
    _BANK_BUFFER_NAMES = (
        "_bank_mean",
        "_bank_std",
        "_bank_low",
        "_bank_high",
        "_bank_count",
        "_bank_loss_dim_weight",
    )

    def __init__(
        self,
        base_dim: int = 60,
        max_dim: int = 64,
        ee_equivalent: bool = False,
        norm_bank_count: int = 1,
    ):
        super().__init__()
        self.base_dim = int(base_dim)
        self.dim_action = int(max_dim)
        self.dim_proprio = int(max_dim)
        self.gripper_idx = tuple(i for i in self.GRIPPER_IDX if i < self.base_dim)
        self.ee_equivalent = bool(ee_equivalent)
        num_banks = max(1, int(norm_bank_count))

        if self.dim_action < self.base_dim:
            raise ValueError(f"max_dim({self.dim_action}) must be >= base_dim({self.base_dim})")

        self.register_buffer("_norm_mean", torch.zeros(self.dim_action))
        self.register_buffer("_norm_std", torch.ones(self.dim_action))
        self.register_buffer("_norm_enabled", torch.zeros(1))
        self.register_buffer("_norm_mode_id", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_norm_low", torch.full((self.dim_action,), -1.0))
        self.register_buffer("_norm_high", torch.full((self.dim_action,), 1.0))
        self.register_buffer("_clip_low", torch.full((self.dim_action,), float("-inf")))
        self.register_buffer("_clip_high", torch.full((self.dim_action,), float("inf")))
        self.register_buffer("_clip_enabled", torch.zeros(1))
        self.register_buffer("_norm_count", torch.zeros(self.dim_action))
        self.register_buffer("_norm_count_enabled", torch.zeros(1))
        self.register_buffer("_loss_dim_weight", torch.ones(self.dim_action))
        self.register_buffer("_bank_mean", torch.zeros(num_banks, self.dim_action))
        self.register_buffer("_bank_std", torch.ones(num_banks, self.dim_action))
        self.register_buffer("_bank_low", torch.full((num_banks, self.dim_action), -1.0))
        self.register_buffer("_bank_high", torch.full((num_banks, self.dim_action), 1.0))
        self.register_buffer("_bank_count", torch.zeros(num_banks, self.dim_action))
        self.register_buffer("_bank_loss_dim_weight", torch.ones(num_banks, self.dim_action))
        self._regression_loss = "mse"
        self._smooth_l1_beta = 1.0
        self._loss_dim_weight_power = 0.0
        self._loss_min_dim_count = 0.0
        self._loss_target_mode = "action"
        # Gripper loss switch (Path A). Default "bce" preserves the released
        # cross-emb training contract and the head's logit-output convention:
        # postprocess still passes gripper logits through torch.sigmoid, so
        # every eval-side [0,1] threshold keeps working. When set to "mse",
        # compute_loss applies torch.sigmoid to gripper logits BEFORE MSE
        # against the [0,1] binarized target, so no eval-side change is needed.
        # Configured via configure_gripper_loss(). Only affects the action-mode
        # gripper branch; velocity/FM mode already regresses gripper.
        self._gripper_loss_type = "bce"
        # Native CrossEmb keeps its paper-aligned physical loss by default.
        # The Hy adapter explicitly selects ``hy_normalized_mse`` when its
        # dataset already supplies z-scored actions and flow targets.
        self._loss_profile = self._LOSS_PROFILE_CROSS_EMB_NATIVE
        self._hand_loss_normalization = "global_joint"
        self._pending_loss_stats: dict[str, tuple[torch.Tensor, torch.Tensor, float]] | None = None
        self._norm_bank_keys: list[str] = ["__global__"]
        self._norm_bank_index: dict[str, int] = {"__global__": 0}
        self._set_norm_mode("zscore")
        self._sync_global_bank()
        self.register_load_state_dict_post_hook(self._compat_state_dict_post_hook)

    def _resize_bank_buffers(self, num_banks: int) -> None:
        """Safely resize all bank buffers to (num_banks, dim_action).

        Uses register_buffer to re-create tensors when in-place resize_
        fails (e.g. after from_pretrained loads non-resizable storage).
        """
        D = self.dim_action
        names_defaults = [
            ("_bank_mean", 0.0), ("_bank_std", 1.0),
            ("_bank_low", -1.0), ("_bank_high", 1.0),
            ("_bank_count", 0.0), ("_bank_loss_dim_weight", 1.0),
        ]
        for name, fill in names_defaults:
            old = getattr(self, name)
            if old.shape == (num_banks, D):
                continue
            new = torch.full((num_banks, D), fill, dtype=old.dtype, device=old.device)
            # Use untyped_storage() (TypedStorage is deprecated since torch 2.0).
            try:
                storage_size = old.untyped_storage().size() // old.element_size()
            except Exception:
                storage_size = old.numel()
            safe_rows = min(storage_size // D, old.shape[0], num_banks)
            if safe_rows > 0:
                new[:safe_rows].copy_(old.contiguous()[:safe_rows])
            self.register_buffer(name, new)

    _BANK_BUFFER_NAMES = (
        "_bank_mean",
        "_bank_std",
        "_bank_low",
        "_bank_high",
        "_bank_count",
        "_bank_loss_dim_weight",
    )

    def _maybe_resize_bank_buffers_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
    ) -> None:
        bank_rows = None
        for name in self._BANK_BUFFER_NAMES:
            value = state_dict.get(f"{prefix}{name}")
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == self.dim_action:
                bank_rows = int(value.shape[0])
                break
        if bank_rows is None or bank_rows <= 0:
            return
        current_rows = getattr(self, self._BANK_BUFFER_NAMES[0]).shape[0]
        if current_rows != bank_rows:
            self._resize_bank_buffers(bank_rows)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # HF from_pretrained may load buffers one-by-one with assign=True. For
        # multi-bank checkpoints, align bank buffer rows before the actual load.
        self._maybe_resize_bank_buffers_from_state_dict(state_dict, prefix)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _needs_legacy_bank_rebuild(self, missing_keys: Iterable[str] | None = None) -> bool:
        if self._bank_mean.shape[0] != 1:
            return False
        if self._norm_bank_keys not in ([], ["__global__"]):
            return False
        if missing_keys is not None:
            for key in missing_keys:
                leaf = str(key).split(".")[-1]
                if leaf.startswith("_bank_"):
                    return True
        if (
            self._norm_enabled.item() <= 0
            and self._clip_enabled.item() <= 0
            and self._norm_count_enabled.item() <= 0
        ):
            return False
        return (
            not torch.allclose(self._bank_mean[0], self._norm_mean)
            or not torch.allclose(self._bank_std[0], self._norm_std)
            or not torch.allclose(self._bank_low[0], self._norm_low)
            or not torch.allclose(self._bank_high[0], self._norm_high)
            or not torch.allclose(self._bank_count[0], self._norm_count)
            or not torch.allclose(self._bank_loss_dim_weight[0], self._loss_dim_weight)
        )

    def _compat_state_dict_post_hook(self, module, incompatible_keys) -> None:
        missing_keys = getattr(incompatible_keys, "missing_keys", None)
        if self._needs_legacy_bank_rebuild(missing_keys):
            # Older cross_emb checkpoints predate per-bank buffers. Rebuild the
            # single global bank from the restored global norm/clip statistics.
            self._sync_global_bank()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) > self.dim_action:
            return x[..., : self.dim_action]
        pad_shape = list(x.shape[:-1]) + [self.dim_action - x.size(-1)]
        return torch.cat([x, x.new_zeros(pad_shape)], dim=-1)


    def _compute_loss_dim_weight(self, counts: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(counts)
        if bool(torch.any(counts > 0.0)):
            counts = counts.clamp(min=0.0)
            keep = counts > 0.0
            if self._loss_min_dim_count > 0:
                keep = keep & (counts >= float(self._loss_min_dim_count))
            weights = keep.to(dtype=weights.dtype)
            if self._loss_dim_weight_power > 0 and bool(keep.any()):
                max_count = counts[keep].max().clamp(min=1.0)
                ratio = torch.where(keep, counts / max_count, torch.zeros_like(counts))
                dim_w = ratio.clamp(min=1e-6).pow(float(self._loss_dim_weight_power))
                weights = torch.where(keep, dim_w, torch.zeros_like(weights))
        if self.gripper_idx:
            weights[list(self.gripper_idx)] = 1.0
        if float(weights.sum().item()) <= 0:
            weights.fill_(1.0)
        positive = weights > 0
        if bool(positive.any()):
            mean_w = weights[positive].mean().clamp(min=1e-6)
            weights = torch.where(positive, weights / mean_w, weights)
        return weights

    def _mask_degenerate_hand_quantiles(
        self,
        weights: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        counts: torch.Tensor,
        *,
        counts_available: bool,
    ) -> torch.Tensor:
        out = weights.clone()
        hand_idx = [
            i
            for i in (
                list(range(self.LEFT_HAND_SLICE.start, self.LEFT_HAND_SLICE.stop))
                + list(range(self.RIGHT_HAND_SLICE.start, self.RIGHT_HAND_SLICE.stop))
            )
            if i < self.base_dim
        ]
        if hand_idx:
            hand_counts = counts[hand_idx].clamp(min=0.0)
            hand_observed = (
                hand_counts > 0
                if counts_available
                else torch.ones_like(hand_counts, dtype=torch.bool)
            )
            if self._norm_mode() == "quantile" and self._clip_enabled.item() > 0:
                hand_dynamic = (high[hand_idx] - low[hand_idx]) > 1e-6
            else:
                hand_dynamic = torch.ones_like(hand_observed)
            hand_keep = hand_observed & hand_dynamic
            hand_weights = out[hand_idx]

            # min_dim_count may suppress genuinely observed long-tail hand
            # dimensions. Retain them with continuous count weighting; only
            # unobserved or quantile-degenerate dimensions are hard-masked.
            if self._loss_dim_weight_power > 0 and bool(hand_keep.any()):
                positive_counts = counts[counts > 0]
                max_count = (
                    positive_counts.max().clamp(min=1.0)
                    if bool(positive_counts.numel())
                    else hand_counts.new_tensor(1.0)
                )
                retained_weight = (
                    hand_counts / max_count
                ).clamp(min=1e-6).pow(float(self._loss_dim_weight_power))
            else:
                retained_weight = torch.ones_like(hand_weights)
            hand_weights = torch.where(
                hand_keep & (hand_weights <= 0),
                retained_weight,
                hand_weights,
            )
            out[hand_idx] = torch.where(
                hand_keep,
                hand_weights,
                torch.zeros_like(hand_weights),
            )
        positive = out > 0
        if bool(positive.any()):
            out = torch.where(
                positive,
                out / out[positive].mean().clamp(min=1e-6),
                out,
            )
        return out

    def _refresh_loss_dim_weight(self) -> None:
        counts = self._norm_count if self._norm_count_enabled.item() > 0 else torch.zeros_like(self._norm_count)
        global_weight = self._compute_loss_dim_weight(counts)
        global_weight = self._mask_degenerate_hand_quantiles(
            global_weight,
            self._norm_low,
            self._norm_high,
            counts,
            counts_available=bool(self._norm_count_enabled.item() > 0),
        )
        self._loss_dim_weight.copy_(global_weight)

        bank_counts = self._bank_count
        bank_weights = torch.stack(
            [
                self._mask_degenerate_hand_quantiles(
                    self._compute_loss_dim_weight(bank_counts[i]),
                    self._bank_low[i],
                    self._bank_high[i],
                    bank_counts[i],
                    counts_available=bool(self._norm_count_enabled.item() > 0),
                )
                for i in range(bank_counts.shape[0])
            ],
            dim=0,
        )
        if self._bank_loss_dim_weight.shape != bank_weights.shape:
            self._resize_bank_buffers(bank_weights.shape[0])
        self._bank_loss_dim_weight.copy_(bank_weights)

    def configure_target_mode(self, mode: str) -> None:
        mode_norm = str(mode or "action").strip().lower().replace("-", "_")
        if mode_norm not in {"action", "velocity"}:
            raise ValueError(f"Unsupported target_mode={mode}. Expected 'action' or 'velocity'.")
        self._loss_target_mode = mode_norm

    def configure_loss_profile(self, profile: str) -> None:
        normalized = str(profile or self._LOSS_PROFILE_HY_NORMALIZED_MSE).strip().lower()
        aliases = {
            "": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "default": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "hy": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "hy_normalized_mse": self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            "native": self._LOSS_PROFILE_CROSS_EMB_NATIVE,
            "cross_emb_native": self._LOSS_PROFILE_CROSS_EMB_NATIVE,
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {
            self._LOSS_PROFILE_HY_NORMALIZED_MSE,
            self._LOSS_PROFILE_CROSS_EMB_NATIVE,
        }
        if normalized not in allowed:
            raise ValueError(
                "cross_emb action_loss_profile must be one of "
                f"{sorted(allowed)}, got {profile!r}"
            )
        self._loss_profile = normalized


    def configure_gripper_loss(self, kind: str) -> None:
        """Select the gripper loss type. See ``_gripper_loss_type`` for details.

        Args:
            kind: One of ``"bce"`` (default; released contract) or ``"mse"``
                (sigmoid the logits, then MSE against the [0,1] target).

        Only affects the action-mode branch of ``compute_loss``. The
        velocity/flow-matching path is unchanged (it already regresses gripper
        as a continuous target).
        """
        kind_norm = str(kind).strip().lower()
        if kind_norm not in self._GRIPPER_LOSS_TYPES:
            raise ValueError(
                f"Unsupported gripper_loss_type={kind}. "
                f"Expected one of {self._GRIPPER_LOSS_TYPES}"
            )
        self._gripper_loss_type = kind_norm


    def _set_norm_mode(self, mode: str) -> None:
        mode_name = str(mode or "zscore").strip().lower().replace("-", "_")
        if mode_name not in self._NORM_MODE_TO_ID:
            raise ValueError(
                f"Unsupported norm_mode={mode}. Expected one of {sorted(self._NORM_MODE_TO_ID.keys())}"
            )
        if mode_name == "quantile" and self._clip_enabled.item() <= 0:
            raise ValueError("norm_mode=quantile requires q_low/q_high clip bounds in norm stats.")
        self._norm_mode_id.fill_(self._NORM_MODE_TO_ID[mode_name])

    def _norm_mode(self) -> str:
        mode_id = int(self._norm_mode_id.item())
        return self._NORM_ID_TO_MODE.get(mode_id, "zscore")

    def norm_bank_keys(self) -> tuple[str, ...]:
        return tuple(self._norm_bank_keys)

    def _sync_global_bank(self) -> None:
        # Keep row 0 as the global bank, but do not collapse multi-bank tables
        # back to a single row after per-family stats have been loaded.
        num_banks = max(1, int(self._bank_mean.shape[0]))
        expected_shape = (num_banks, self.dim_action)
        if self._bank_mean.shape != expected_shape:
            self._resize_bank_buffers(num_banks)
        self._bank_mean[0].copy_(self._norm_mean)
        self._bank_std[0].copy_(self._norm_std)
        self._bank_low[0].copy_(self._norm_low)
        self._bank_high[0].copy_(self._norm_high)
        self._bank_count[0].copy_(self._norm_count)
        self._bank_loss_dim_weight[0].copy_(self._loss_dim_weight)
        if num_banks == 1 or not self._norm_bank_keys:
            self._norm_bank_keys = ["__global__"]
        elif len(self._norm_bank_keys) != num_banks:
            tail = [k for k in self._norm_bank_keys if k != "__global__"]
            tail = tail[: max(0, num_banks - 1)]
            while len(tail) < max(0, num_banks - 1):
                tail.append(f"__bank_{len(tail) + 1}__")
            self._norm_bank_keys = ["__global__", *tail]
        elif self._norm_bank_keys[0] != "__global__":
            self._norm_bank_keys = ["__global__", *[k for k in self._norm_bank_keys if k != "__global__"]]
        self._rebuild_norm_bank_index(self._norm_bank_keys)

    @staticmethod
    def _canonicalize_norm_bank_key(key: str) -> str:
        """Rewrite ``|robot=<label>`` via embodiment aliases (e.g. r1lite→galaxea_r1_lite)."""
        if "|robot=" not in key:
            return key
        # Local import avoids pulling datasets at module import for non-cross-emb paths.
        from datasets.cross_emb_semantics import normalize_cross_emb_robot_group

        prefix, _, robot = key.rpartition("|robot=")
        canon = normalize_cross_emb_robot_group(robot)
        if not canon:
            return key
        return f"{prefix}|robot={canon}"

    def _rebuild_norm_bank_index(self, keys: list[str]) -> None:
        index = {key: idx for idx, key in enumerate(keys)}
        # Register canonical robot aliases so stale stats (robot=r1lite) match
        # train-time keys (robot=galaxea_r1_lite) without requiring an offline remap.
        for key, idx in list(index.items()):
            canon = self._canonicalize_norm_bank_key(key)
            index.setdefault(canon, idx)
        self._norm_bank_index = index

    def _set_norm_banks(self, banks: list[tuple[str, dict]]) -> None:
        num_banks = max(1, len(banks))
        mean = torch.zeros((num_banks, self.dim_action), dtype=self._bank_mean.dtype, device=self._bank_mean.device)
        std = torch.ones((num_banks, self.dim_action), dtype=self._bank_std.dtype, device=self._bank_std.device)
        low = torch.full((num_banks, self.dim_action), -1.0, dtype=self._bank_low.dtype, device=self._bank_low.device)
        high = torch.full((num_banks, self.dim_action), 1.0, dtype=self._bank_high.dtype, device=self._bank_high.device)
        count = torch.zeros((num_banks, self.dim_action), dtype=self._bank_count.dtype, device=self._bank_count.device)
        keys: list[str] = []

        for idx, (key, payload) in enumerate(banks):
            bank_mean = torch.as_tensor(payload["mean"], dtype=mean.dtype, device=mean.device).view(-1)
            bank_std = torch.as_tensor(payload["std"], dtype=std.dtype, device=std.device).view(-1).clamp(min=1e-6)
            if bank_mean.numel() != self.dim_action or bank_std.numel() != self.dim_action:
                raise ValueError(
                    f"norm bank dim mismatch for key={key}: mean={bank_mean.numel()} std={bank_std.numel()} expected={self.dim_action}"
                )
            mean[idx].copy_(bank_mean)
            std[idx].copy_(bank_std)

            bank_low = payload.get("q_low", payload.get("q01"))
            bank_high = payload.get("q_high", payload.get("q99"))
            if bank_low is not None and bank_high is not None:
                low_t = torch.as_tensor(bank_low, dtype=low.dtype, device=low.device).view(-1)
                high_t = torch.as_tensor(bank_high, dtype=high.dtype, device=high.device).view(-1)
                if low_t.numel() != self.dim_action or high_t.numel() != self.dim_action:
                    raise ValueError(
                        f"norm bank clip dim mismatch for key={key}: low={low_t.numel()} high={high_t.numel()} expected={self.dim_action}"
                    )
                high_t = torch.maximum(high_t, low_t)
                low[idx].copy_(low_t)
                high[idx].copy_(high_t)

            bank_count = payload.get("count")
            if bank_count is not None:
                count_t = torch.as_tensor(bank_count, dtype=count.dtype, device=count.device).view(-1)
                if count_t.numel() != self.dim_action:
                    raise ValueError(
                        f"norm bank count dim mismatch for key={key}: count={count_t.numel()} expected={self.dim_action}"
                    )
                count[idx].copy_(count_t.clamp(min=0.0))
            keys.append(str(key))

        self._resize_bank_buffers(num_banks)
        self._bank_mean.copy_(mean)
        self._bank_std.copy_(std)
        self._bank_low.copy_(low)
        self._bank_high.copy_(high)
        self._bank_count.copy_(count)
        self._norm_bank_keys = keys
        self._rebuild_norm_bank_index(keys)

    def _resolve_norm_bank_indices(self, norm_key: Any | None, batch_size: int) -> torch.Tensor | None:
        if norm_key is None or batch_size <= 0:
            return None
        if isinstance(norm_key, torch.Tensor):
            if norm_key.ndim == 0:
                norm_key = [str(norm_key.item())]
            else:
                norm_key = [str(v) for v in norm_key.detach().cpu().tolist()]
        elif isinstance(norm_key, str):
            norm_key = [norm_key]
        elif isinstance(norm_key, tuple):
            norm_key = list(norm_key)
        elif not isinstance(norm_key, list):
            norm_key = [str(norm_key)]

        keys = [str(v) for v in norm_key]
        if len(keys) == 1 and batch_size > 1:
            keys = keys * batch_size
        if len(keys) != batch_size:
            raise ValueError(f"norm_key batch mismatch: got {len(keys)} keys for batch_size={batch_size}")
        if self._norm_bank_keys == ["__global__"]:
            indices = [0 for _ in keys]
        else:
            resolved: list[int] = []
            missing: list[str] = []
            for key in keys:
                idx = self._norm_bank_index.get(key)
                if idx is None:
                    idx = self._norm_bank_index.get(self._canonicalize_norm_bank_key(key))
                if idx is None:
                    missing.append(key)
                else:
                    resolved.append(idx)
            if missing:
                raise KeyError(
                    "unknown norm_key(s) for loaded multi-bank stats: "
                    f"missing={sorted(set(missing))} available={tuple(self._norm_bank_keys)}"
                )
            indices = resolved
        return torch.as_tensor(indices, dtype=torch.long, device=self._bank_mean.device)

    @staticmethod
    def _expand_bank_tensor(
        values: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if ref.dim() <= 1:
            return values[0]
        view_shape = [values.shape[0]] + [1] * max(0, ref.dim() - 2) + [ref.shape[-1]]
        expanded_shape = [ref.shape[0]] + [1] * max(0, ref.dim() - 2) + [ref.shape[-1]]
        return values.view(*view_shape).expand(*expanded_shape)

    def _select_bank_tensor(
        self,
        ref: torch.Tensor,
        bank_tensor: torch.Tensor,
        bank_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if bank_indices is None:
            selected = bank_tensor[:1]
        else:
            if bank_indices.numel() > 0:
                min_idx = int(bank_indices.min().item())
                max_idx = int(bank_indices.max().item())
                if min_idx < 0 or max_idx >= int(bank_tensor.shape[0]):
                    raise IndexError(
                        "norm bank index out of range: "
                        f"min={min_idx} max={max_idx} banks={int(bank_tensor.shape[0])} "
                        f"keys={tuple(self._norm_bank_keys)}"
                    )
            selected = bank_tensor.index_select(0, bank_indices.to(device=bank_tensor.device))
        selected = selected.to(device=ref.device, dtype=ref.dtype)
        return self._expand_bank_tensor(selected, ref)
    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor, *, norm_mode: str | None = None) -> None:
        mean = mean.to(dtype=self._norm_mean.dtype, device=self._norm_mean.device)
        std = std.to(dtype=self._norm_std.dtype, device=self._norm_std.device)
        if mean.numel() != self.dim_action or std.numel() != self.dim_action:
            raise ValueError(
                f"norm dim mismatch: got mean={mean.numel()} std={std.numel()}, expected {self.dim_action}"
            )
        self._norm_mean.copy_(mean.view(-1))
        self._norm_std.copy_(std.view(-1).clamp(min=1e-6))
        self._norm_enabled.fill_(1)
        # Keep gripper channels in logit space.
        if self.gripper_idx:
            self._norm_mean[list(self.gripper_idx)] = 0.0
            self._norm_std[list(self.gripper_idx)] = 1.0
        if norm_mode is not None:
            self._set_norm_mode(norm_mode)
        self._sync_global_bank()

    def set_clip_bounds(self, low: torch.Tensor, high: torch.Tensor) -> None:
        low = low.to(dtype=self._clip_low.dtype, device=self._clip_low.device)
        high = high.to(dtype=self._clip_high.dtype, device=self._clip_high.device)
        if low.numel() != self.dim_action or high.numel() != self.dim_action:
            raise ValueError(
                f"clip dim mismatch: got low={low.numel()} high={high.numel()}, expected {self.dim_action}"
            )
        low = low.view(-1)
        high = high.view(-1)
        high = torch.maximum(high, low)
        self._clip_low.copy_(low)
        self._clip_high.copy_(high)
        self._norm_low.copy_(low)
        self._norm_high.copy_(high)
        self._clip_enabled.fill_(1)
        if self.gripper_idx:
            self._clip_low[list(self.gripper_idx)] = float("-inf")
            self._clip_high[list(self.gripper_idx)] = float("inf")
            # Keep gripper channels identity-mapped for BCE target semantics.
            self._norm_low[list(self.gripper_idx)] = -1.0
            self._norm_high[list(self.gripper_idx)] = 1.0
        self._sync_global_bank()

    def load_norm_stats(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "mean" not in payload or "std" not in payload:
            raise ValueError(f"Invalid norm stats json (missing mean/std): {path}")
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        self.set_norm_stats(mean, std)
        if "count" in payload:
            count = torch.tensor(payload["count"], dtype=torch.float32)
            if count.numel() != self.dim_action:
                raise ValueError(
                    f"Invalid norm stats count dim={count.numel()}, expected {self.dim_action}"
                )
            self._norm_count.copy_(count.view(-1).clamp(min=0.0))
            self._norm_count_enabled.fill_(1)
        else:
            self._norm_count.zero_()
            self._norm_count_enabled.fill_(0)
        if "q_low" in payload and "q_high" in payload:
            low = torch.tensor(payload["q_low"], dtype=torch.float32)
            high = torch.tensor(payload["q_high"], dtype=torch.float32)
            self.set_clip_bounds(low, high)
        elif "q01" in payload and "q99" in payload:
            low = torch.tensor(payload["q01"], dtype=torch.float32)
            high = torch.tensor(payload["q99"], dtype=torch.float32)
            self.set_clip_bounds(low, high)
        else:
            self._clip_enabled.fill_(0)
        norm_mode = payload.get("norm_mode")
        if norm_mode is None:
            norm_mode = "quantile" if self._clip_enabled.item() > 0 else "zscore"
        self._set_norm_mode(norm_mode)
        banks_payload = payload.get("banks")
        if isinstance(banks_payload, dict) and banks_payload:
            order = payload.get("bank_keys")
            if not isinstance(order, list) or not order:
                order = list(banks_payload.keys())
            ordered_banks: list[tuple[str, dict]] = [("__global__", payload)]
            for key in order:
                if key in banks_payload and isinstance(banks_payload[key], dict):
                    ordered_banks.append((str(key), dict(banks_payload[key])))
            self._set_norm_banks(ordered_banks)
        else:
            self._norm_bank_keys = ["__global__"]
            self._norm_bank_index = {"__global__": 0}
            self._sync_global_bank()
        self._refresh_loss_dim_weight()
        self._sync_global_bank()

    def _normalize(self, x: torch.Tensor, *, norm_key: Any | None = None) -> torch.Tensor:
        if self._norm_enabled.item() <= 0:
            return x
        bank_indices = self._resolve_norm_bank_indices(norm_key, int(x.shape[0]) if x.ndim > 0 else 0)
        if self._norm_mode() == "quantile":
            x_raw = x
            low = self._select_bank_tensor(x, self._bank_low, bank_indices)
            high = self._select_bank_tensor(x, self._bank_high, bank_indices)
            span_raw = high - low
            span = span_raw.clamp(min=1e-6)
            active = span_raw > 1e-6
            x = torch.max(torch.min(x, high), low)
            x = (x - low) * (2.0 / span) - 1.0
            # Degenerate quantile bounds (q_low==q_high) carry no scaling information.
            # Keep these channels in raw space to avoid forcing a synthetic -1 target.
            if x.dim() > 0:
                x = torch.where(active, x, x_raw)
            if self.gripper_idx:
                x[..., self.gripper_idx] = x_raw[..., self.gripper_idx]
            return x
        mean = self._select_bank_tensor(x, self._bank_mean, bank_indices)
        std = self._select_bank_tensor(x, self._bank_std, bank_indices)
        return (x - mean) / std

    def _denormalize(self, x: torch.Tensor, *, norm_key: Any | None = None) -> torch.Tensor:
        if self._norm_enabled.item() <= 0:
            return x
        bank_indices = self._resolve_norm_bank_indices(norm_key, int(x.shape[0]) if x.ndim > 0 else 0)
        if self._norm_mode() == "quantile":
            low = self._select_bank_tensor(x, self._bank_low, bank_indices)
            high = self._select_bank_tensor(x, self._bank_high, bank_indices)
            span_raw = high - low
            span = span_raw.clamp(min=1e-6)
            active = span_raw > 1e-6
            x_denorm = (x + 1.0) * (0.5 * span) + low
            if x.dim() > 0:
                x_denorm = torch.where(active, x_denorm, x)
            if self.gripper_idx:
                x_denorm[..., self.gripper_idx] = x[..., self.gripper_idx]
            return x_denorm
        mean = self._select_bank_tensor(x, self._bank_mean, bank_indices)
        std = self._select_bank_tensor(x, self._bank_std, bank_indices)
        return x * std + mean

    def _apply_clip(self, x: torch.Tensor, *, norm_key: Any | None = None) -> torch.Tensor:
        if self._clip_enabled.item() <= 0:
            return x
        bank_indices = self._resolve_norm_bank_indices(norm_key, int(x.shape[0]) if x.ndim > 0 else 0)
        low = self._select_bank_tensor(x, self._bank_low, bank_indices).clone()
        high = self._select_bank_tensor(x, self._bank_high, bank_indices).clone()
        if self.gripper_idx:
            low[..., self.gripper_idx] = float("-inf")
            high[..., self.gripper_idx] = float("inf")
        return torch.max(torch.min(x, high), low)

    def _build_valid_mask(self, ref: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            out = ref.new_zeros(ref.shape)
            out[..., : self.base_dim] = 1.0
            return out
        out = self._pad_to_model_dim(valid_mask.to(dtype=ref.dtype))
        return (out > 0).to(dtype=ref.dtype)

    def _apply_ee_equivalent_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_model_dim(x)
        if self.base_dim < self.dim_action:
            x[..., self.base_dim:] = 0.0
        keep = x.new_zeros(self.dim_action)
        keep[list(self.EE_EQ_KEEP_IDX)] = 1.0
        view_shape = [1] * (x.dim() - 1) + [self.dim_action]
        return x * keep.view(*view_shape)


    def preprocess(self, proprio, action, mode="train", norm_key: Any | None = None):
        if self.ee_equivalent:
            proprio_m = self._apply_ee_equivalent_input_mask(proprio)
            action_m = self._apply_ee_equivalent_input_mask(action)
            if self.gripper_idx:
                proprio_m[..., self.gripper_idx] = 0.0
                action_m[..., self.gripper_idx] = 0.0
            return proprio_m, action_m

        proprio_m = self._pad_to_model_dim(proprio)
        action_m = self._pad_to_model_dim(action)
        if self.base_dim < self.dim_action:
            proprio_m[..., self.base_dim:] = 0.0
            action_m[..., self.base_dim:] = 0.0
        proprio_m = self._apply_clip(proprio_m, norm_key=norm_key)
        action_m = self._apply_clip(action_m, norm_key=norm_key)
        proprio_m = self._normalize(proprio_m, norm_key=norm_key)
        action_m = self._normalize(action_m, norm_key=norm_key)
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        action = self._pad_to_model_dim(action)
        if self.ee_equivalent:
            if self.base_dim < self.dim_action:
                action[..., self.base_dim:] = 0.0
            keep = action.new_zeros(self.dim_action)
            keep[list(self.EE_EQ_KEEP_IDX)] = 1.0
            view_shape = [1] * (action.dim() - 1) + [self.dim_action]
            action = action * keep.view(*view_shape)
            if self.gripper_idx:
                # ee_equivalent always trains gripper as BCE logits.
                action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
            return action
        action = self._denormalize(action, norm_key=norm_key)
        action = self._apply_clip(action, norm_key=norm_key)
        if self.gripper_idx:
            if self._loss_target_mode == "velocity":
                # Flow-matching / velocity training: gripper output is the integrated
                # raw value in [0,1] (identity-mapped channel), NOT a logit.
                # Applying sigmoid would silently squash [0,1] to [0.5, 0.731] and
                # prevent the robot from ever fully closing/opening the gripper.
                action[..., self.gripper_idx] = action[..., self.gripper_idx].clamp(0.0, 1.0)
            else:
                # BCE training: gripper output is a logit, sigmoid recovers probability.
                action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action

    def normalize_action_for_metric(
        self,
        action: torch.Tensor,
        norm_key: Any | None = None,
    ) -> torch.Tensor:
        action = self._pad_to_model_dim(action)
        if self.ee_equivalent:
            return action
        action = self._apply_clip(action, norm_key=norm_key)
        return self._normalize(action, norm_key=norm_key)

    def to_diffusion_state(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        # Keep iterative denoising in raw action space to avoid repeated normalization drift.
        return self.postprocess(action, norm_key=norm_key)


# =============================================================================
# RoboCasa delta12 —— 与 pi0.5 对齐的 flow-matching action space
# =============================================================================
@register_action("delta12")
class Delta12ActionSpace(BaseActionSpace):
    """RoboCasa v1.0 的 12 维 raw OSC delta，按 pi0.5 的方式训练。

    与既有 `ee6d` / `ee6d_mobile15` 的三点根本差异（这也是本类存在的全部理由）：

    1. **target 是 delta，不是绝对位姿。**
       `ee6d_mobile15` 把 OSC delta 合成成绝对目标位姿（`target = pose[t] + delta*0.05`），
       于是「输出≈proprio」成为一个低 loss 的捷径解。本类直接监督 parquet 里的
       raw delta（env 顺序，见 `datasets/domain_handler/simulations.py` 顶部的
       `robocasa_parquet_action_to_env12`），该捷径**不存在**。
    2. **loss 无 per-group scale。**
       `ee6d_mobile15` 有 `XYZ_SCALE=500 / ROT_SCALE=10 / GRIPPER_SCALE / BASE_SCALE /
       MODE_SCALE` 五个手调常量；本类是**单项均匀 masked MSE**，与 pi0.5 的
       `jnp.mean(jnp.square(v_t - u_t))`（`openpi/src/openpi/models/pi0.py:214`）一致。
       量纲差异全部交给 action norm stats。
    3. **z-score 归一化。**
       与 pi0.5 的 robocasa365 配置一致：`use_quantile_norm=False`
       （`openpi/src/openpi/training/config.py:737`，官方 ckpt 的 `norm_stats.json`
       只有 mean/std、没有 q01/q99）。归一化在本类的 `preprocess` 里做、
       反归一化在 `postprocess` 里做，因此统计量随 ckpt 一起走，部署无需外部文件。

    维度布局（`dim_action = 20`，保 warm-start 兼容 20D head）
    -------------------------------------------------------
        [0:3]   ee_pos_delta      归一化 OSC delta ∈ [-1,1]（×0.05 m/unit）
        [3:6]   ee_rot_delta      归一化 OSC axis-angle ∈ [-1,1]（×0.5 rad/unit）
        [6]     gripper           **±1 原始值**（不是 0/1，pi0.5 也是 ±1）
        [7:11]  base_motion       base_x / base_y / base_yaw / torso ∈ [-1,1]
        [11]    control_mode      **±1 原始值**（-1=arm, +1=base）
        [12:20] 未用（handler 侧 `valid_dim` 只给前 12 维，pad 维被 mask 掉）

    ⚠️ 与 pi0.5 的已知结构差异
    --------------------------
    * **padding 维**：pi0.5 把 32 维里的 20 个 padding 维**也算进 loss**
      （`u_t = noise - 0 = noise` 是纯不可预测噪声，给 loss 贴一个 ~0.625 的常数地板）。
      本类用 `valid_mask` 把 dims[12:20] 屏蔽掉。⇒ **loss 数值与 openpi 不可直接比大小**，
      但梯度更干净。
    * Action/state normalization follows OpenPI's `(x-mean)/(std+1e-6)`
      contract. Delta12's strict launcher always enables state normalization.

    ⚠️ flow matching 的接法
    ---------------------
    `models/modeling_xvla.py` 的 `flow_matching` 分支调用的是 `compute_loss`
    （**不是** `compute_flow_matching_loss` —— 后者自 commit `08e90e7` 起已成死代码）。
    本类的 `compute_loss` 本身就是「归一化空间里的均匀 MSE」，对速度场 target 直接可用，
    所以 FM 分支无需改动 `modeling_xvla.py` 即可正确工作。
    `compute_flow_matching_loss` 也提供为同一实现的别名，两条路都对。
    """

    dim_action = 20
    REAL_DIM = 12

    EE_POS_IDX = (0, 1, 2)
    EE_ROT_IDX = (3, 4, 5)
    GRIPPER_IDX = (6,)
    BASE_MOTION_IDX = (7, 8, 9, 10)
    CONTROL_MODE_IDX = (11,)

    # gripper 在本空间里是**普通回归维**（±1），不是 logit ⇒ 不做 sigmoid、
    # 不排除在归一化之外。留空以避免任何依赖 gripper_idx 的通用逻辑误触发。
    gripper_idx: Tuple[int, ...] = ()

    DIM_NAMES = (
        "ee_pos_dx", "ee_pos_dy", "ee_pos_dz",
        "ee_rot_dx", "ee_rot_dy", "ee_rot_dz",
        "gripper",
        "base_x", "base_y", "base_yaw", "torso",
        "control_mode",
    )

    def __init__(self, norm_proprio: bool | None = None):
        super().__init__()
        if norm_proprio is None:
            norm_proprio = str(os.environ.get("XVLA_DELTA12_NORM_PROPRIO", "0")).strip() == "1"
        self._norm_proprio_flag = bool(norm_proprio)
        d = self.dim_action
        self.register_buffer("_norm_mean", torch.zeros(d), persistent=True)
        self.register_buffer("_norm_std", torch.ones(d), persistent=True)
        self.register_buffer("_norm_enabled", torch.zeros(1, dtype=torch.long), persistent=True)
        self.register_buffer("_pro_mean", torch.zeros(d), persistent=True)
        self.register_buffer("_pro_std", torch.ones(d), persistent=True)
        self.register_buffer("_pro_enabled", torch.zeros(1, dtype=torch.long), persistent=True)
        self._target_mode = "action"

    # ------------------------------------------------------------------ hooks
    def configure_target_mode(self, mode: str) -> None:
        mode_norm = str(mode or "action").strip().lower()
        if mode_norm in {"action", "velocity"}:
            self._target_mode = mode_norm

    def load_norm_stats(self, path: str) -> None:
        """读 `tools/build_robocasa_delta12_norm_stats.py` 产出的 JSON（z-score）。

        schema（与 `EE6DActionSpace.load_norm_stats` 兼容的超集）：
            {"mean": [D], "std": [D],
             "proprio_mean": [D], "proprio_std": [D],
             "std_epsilon": 1e-6, "dim_names": [...], ...}
        ⚠️ 只支持 z-score。给了 `q_low`/`q_high` 会直接报错 —— 不静默降级，
           因为 pi0.5 的 robocasa365 明确用 mean/std。
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "mean" not in payload or "std" not in payload:
            raise ValueError(f"delta12 norm stats 缺 mean/std: {path}")
        if "q_low" in payload or "q_high" in payload or "q01" in payload:
            raise ValueError(
                f"delta12 只支持 z-score，但 {path} 带了 quantile 字段。"
                f"pi0.5 的 robocasa365 用 use_quantile_norm=False，请用 "
                f"tools/build_robocasa_delta12_norm_stats.py 重新生成。"
            )
        if payload.get("schema") != "robocasa_delta12_openpi_v2":
            raise ValueError(
                f"delta12 requires OpenPI-aligned v2 norm stats, got schema="
                f"{payload.get('schema')!r}: {path}. Rebuild with "
                "tools/build_robocasa_delta12_norm_stats.py."
            )
        expected_root = os.environ.get("ERA0_DELTA12_DATA_ROOT", "")
        if not expected_root:
            raise ValueError("Optional delta12 normalization requires ERA0_DELTA12_DATA_ROOT")
        if payload.get("data_roots") != [expected_root]:
            raise ValueError(
                f"delta12 norm stats must be built from {expected_root}, got "
                f"{payload.get('data_roots')!r}: {path}"
            )
        epsilon = float(payload.get("std_epsilon", 1e-6))
        if epsilon <= 0.0:
            raise ValueError(f"delta12 std_epsilon must be positive, got {epsilon}: {path}")

        def _load(mean_key: str, std_key: str, mean_buf, std_buf, tag: str) -> bool:
            if mean_key not in payload or std_key not in payload:
                return False
            m = torch.tensor(payload[mean_key], dtype=torch.float32)
            s = torch.tensor(payload[std_key], dtype=torch.float32)
            if m.numel() < self.dim_action:
                m = torch.cat([m, torch.zeros(self.dim_action - m.numel())])
                s = torch.cat([s, torch.ones(self.dim_action - s.numel())])
            m, s = m[: self.dim_action], s[: self.dim_action]
            hit = (s <= 0).nonzero(as_tuple=True)[0].tolist()
            if hit:
                names = [self.DIM_NAMES[i] if i < len(self.DIM_NAMES) else f"pad{i}" for i in hit]
                print(f"[delta12/{tag}] std_epsilon={epsilon:g} applies to {len(hit)} constant dims: "
                      f"{list(zip(hit, names))}")
            mean_buf.copy_(m)
            std_buf.copy_(s + epsilon)
            return True

        _load("mean", "std", self._norm_mean, self._norm_std, "action")
        self._norm_enabled.fill_(1)
        if self._norm_proprio_flag:
            ok = _load("proprio_mean", "proprio_std", self._pro_mean, self._pro_std, "proprio")
            if not ok:
                raise ValueError(
                    f"XVLA_DELTA12_NORM_PROPRIO=1 但 {path} 里没有 proprio_mean/proprio_std"
                )
            self._pro_enabled.fill_(1)
        print(f"[delta12] action norm stats loaded (z-score) from {path}; "
              f"norm_proprio={self._norm_proprio_flag}")

    # -------------------------------------------------------------- normalize
    def _norm(self, x: torch.Tensor, mean, std, enabled) -> torch.Tensor:
        if int(enabled.item()) == 0:
            return x
        return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

    def _denorm(self, x: torch.Tensor, mean, std, enabled) -> torch.Tensor:
        if int(enabled.item()) == 0:
            return x
        return x * std.to(x.device, x.dtype) + mean.to(x.device, x.dtype)

    def preprocess(self, proprio, action, mode: str = "train", norm_key: Any | None = None):
        return (
            self._norm(proprio, self._pro_mean, self._pro_std, self._pro_enabled),
            self._norm(action, self._norm_mean, self._norm_std, self._norm_enabled),
        )

    def postprocess(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        # 反归一化回 raw OSC delta 空间。**不做 sigmoid、不 clip 到 [-1,1]** ——
        # gripper/control_mode 由评测侧按 `> 0` 取符号（与数据里的 ±1 一致），
        # 越界的连续通道由 OSC 控制器自己饱和（评测侧会记 pos_clip_rate）。
        return self._denorm(action, self._norm_mean, self._norm_std, self._norm_enabled)

    def to_diffusion_state(self, action: torch.Tensor, norm_key: Any | None = None) -> torch.Tensor:
        # FM 的 Euler 积分全程在归一化空间进行 ⇒ 中间态不反归一化。
        return action

    # ------------------------------------------------------------------- loss


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "AutoActionSpace",
    "CrossEmbActionSpace",
    "Delta12ActionSpace",
    "ACTION_REGISTRY",
]
