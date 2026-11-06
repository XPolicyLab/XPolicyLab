"""RoboDojo inference settings and checkpoint-compatible embodiment identifiers."""
from __future__ import annotations

import warnings
from enum import IntEnum
import torch

# Checkpoint padding widths; the active RoboDojo state and action remain 16-D.
MAX_ACTION_DIM = 1024
MAX_STATE_DIM = 1024


class EmbodimentId(IntEnum):
    # Current model constructors still reference this default identifier.
    Latent_action = 0
    # Keep the raw training ID: the released checkpoint maps 81 to model slot 0.
    RoboDojo = 81


class Embodiment_DIM:
    action_dims = {EmbodimentId.Latent_action: 32, EmbodimentId.RoboDojo: 16}
    state_dims = {EmbodimentId.Latent_action: 32, EmbodimentId.RoboDojo: 16}


class base_data_config:
    frame_per_latent = 4
    frame_compression = 4
    num_latents = 32
    embodiment = ''
    # 由子类覆盖为 EmbodimentId 成员。-1 表示 pretrain generic,不进 CategorySpecific 路由。
    embodiment_id: int = -1
    # 该 embodiment 需要读取的相机 key 列表(observation.images.xxx)。
    # 子类按视角顺序覆盖,空列表表示不读视频(如 pretrain Latent_action)。
    obs_cam_keys: list = []
    # Target video resolution (set via DataConfig from toml).
    # Must be divisible by vae_stride * patch_size (e.g. 16*2=32).
    video_height: int = 480
    video_width: int = 640
    norm_eps: float = 1e-8  # floor for std/span to avoid division by zero
    # "zscore"          : (x - mean) / std
    # "minmax"          : 2 * (x - min) / (max - min) - 1     → exactly  [-1, 1] on training data
    # "minmax_quantile" : 2 * (x - q01) / (q99 - q01) - 1     → roughly  [-1, 1] (outlier-robust)
    norm_mode: str = "zscore"
    norm_clamp_abs: float = 5  # outlier margin for minmax / minmax_quantile (None → no clamp)
    action_dim: int = 0
    state_dim: int = 0

    # Indices into the raw 44-dim vector to keep (override per embodiment)
    action_indices: list = None  # None means keep all
    state_indices: list = None

    # Relative-action: subtract the current (window-start) state from the
    # action chunk before normalization (LingBot-style `subtract_state`).
    # Master switch is propagated from DataConfig (toml [data] relative_action);
    # default False → absolute actions (unchanged behavior).
    relative_action: bool = False
    # Which ACTIVE action dims (post action_indices) have the state subtracted.
    # None → all dims shared with state (min(action_dim, state_dim)). Override
    # per embodiment to EXCLUDE gripper/effector dims, which stay absolute
    # (binary open/close — a "relative" gripper is meaningless). Action dim d is
    # assumed to correspond to state dim d (same joint) on these dims.
    relative_action_dims: list = None

    @property
    def action_dim(self) -> int:
        if self.action_indices is not None:
            return len(self.action_indices)
        return self.action_dim

    @property
    def state_dim(self) -> int:
        if self.state_indices is not None:
            return len(self.state_indices)
        return self.state_dim

    def combine_views(self, video_dict: dict) -> torch.Tensor:
        """默认实现:从多视角 video dict 里取 obs_cam_keys 声明的第一个 view。
        每个 embodiment 子类应根据自身视角语义重写本方法
        (例如 concat / stack 多个 view、或按 role 路由)。
        """
        warnings.warn(
            f"{self.embodiment}: using default combine_views (only the first "
            f"view in obs_cam_keys is kept). Override combine_views in the "
            f"embodiment subclass to fuse multiple views.",
            stacklevel=2,
        )
        if not self.obs_cam_keys:
            raise ValueError(
                f"{self.embodiment}: obs_cam_keys is empty, no video to return"
            )
        return video_dict[self.obs_cam_keys[0]]

    @staticmethod
    def _resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # Plain (aspect-distorting) bilinear resize, antialiased — same as
        # FastWAM's torchvision F.resize(size=[h, w], antialias=True).
        return torch.nn.functional.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False, antialias=True
        )

    @staticmethod
    def resize(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize frames to (target_h, target_w) preserving aspect ratio,
        then black-pad the shorter side (letterbox). No cropping.

        Intended for use inside `combine_views` when a subclass needs to
        normalize per-view resolution before concatenation.

        frames: [..., C, H, W]  (last 3 dims are C,H,W; everything before is
                                 collapsed into the batch dim by interpolate).
        Returns: [..., C, target_h, target_w]
        """
        H, W = frames.shape[-2], frames.shape[-1]
        if H == target_h and W == target_w:
            return frames
        scale = min(target_w / W, target_h / H)
        new_h = min(round(H * scale), target_h)
        new_w = min(round(W * scale), target_w)
        # interpolate expects 4D input -> flatten leading dims, then restore
        lead_shape = frames.shape[:-3]
        x = frames.reshape(-1, frames.shape[-3], H, W)
        x = torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = torch.nn.functional.pad(
            x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0
        )
        return x.reshape(*lead_shape, frames.shape[-3], target_h, target_w)

    def select_action_dims(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Select active dims from raw action. raw_action: [..., action_dim]"""
        if self.action_indices is None:
            return raw_action
        idx = torch.tensor(self.action_indices, dtype=torch.long,
                           device=raw_action.device)
        return raw_action[..., idx]

    def select_state_dims(self, raw_state: torch.Tensor) -> torch.Tensor:
        """Select active dims from raw state. raw_state: [..., state_dim]"""
        if self.state_indices is None:
            return raw_state
        idx = torch.tensor(self.state_indices, dtype=torch.long,
                           device=raw_state.device)
        return raw_state[..., idx]

    def relative_dims(self) -> list:
        """Active action dims to subtract state from. [] when relative_action
        is off. Defaults to all dims shared with state when not specified."""
        if not self.relative_action:
            return []
        if self.relative_action_dims is not None:
            return list(self.relative_action_dims)
        return list(range(min(self.action_dim, self.state_dim)))

    def _relative_shift(self, x: torch.Tensor, anchor_state: torch.Tensor | None,
                        sign: float) -> torch.Tensor:
        """Add `sign * anchor_state` to `x` on the relative dims (dim 0), the
        core of LingBot-style `subtract_state`. Pure shift — the caller
        (action_normalize / action_denormalize) decides WHETHER to call it (the
        relative_action / anchor-None discrimination lives there), so there are
        no guards here. `anchor_state` broadcasts against `x`, so the caller
        controls the per-latent / per-step mapping by its shape:
          - training: x=[action_dim, nf, apf, 1], anchor=states[:, :nf]
            ([state_dim, nf, 1, 1]) → broadcasts over the apf actions of block i.
          - inference: x=[action_dim], anchor=[state_dim] (the current state).
        Dims not in relative_dims() (e.g. gripper) are left untouched. Action dim
        d corresponds to state dim d on the relative dims.
        """
        dims = self.relative_dims()
        idx = torch.tensor(dims, dtype=torch.long, device=x.device)
        x = x.clone()
        x[idx] = x[idx] + sign * anchor_state.to(x.dtype)[idx]
        return x

    def expand_action_to_raw(self, action: torch.Tensor) -> torch.Tensor:
        """Expand reduced action back to raw 44-dim (zero-fill unused dims).
        action: [..., action_dim] -> [..., action_dim]
        """
        if self.action_indices is None:
            return action
        shape = list(action.shape)
        shape[-1] = self.action_dim
        raw = torch.zeros(shape, dtype=action.dtype, device=action.device)
        idx = torch.tensor(self.action_indices, dtype=torch.long,
                           device=action.device)
        raw[..., idx] = action
        return raw

    def normalize_zscore(self, actions: torch.Tensor,
                         mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Z-score normalize using mean/std. actions: [action_dim, ...], mean/std: [action_dim].

        Outliers are clamped to ±``norm_clamp_abs`` (same bound as the minmax /
        minmax_quantile paths), matching FastWAM's LinearNormalizer which clamps
        every mode (incl. z-score) to [-5, 5] in the forward (normalize)
        direction. The denormalize direction is intentionally NOT clamped.
        """
        shape = [1] * actions.ndim
        shape[0] = -1  # broadcast along action_dim (first dim)
        m = mean.to(actions.dtype).view(shape)
        s = std.to(actions.dtype).view(shape)
        s = s.clamp(min=self.norm_eps)
        normed = (actions - m) / s
        if self.norm_clamp_abs is not None and self.norm_clamp_abs > 0:
            normed = normed.clamp(-self.norm_clamp_abs, self.norm_clamp_abs)
        return normed

    def denormalize_zscore(self, actions: torch.Tensor,
                           mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize actions from normalized space back to raw scale."""
        shape = [1] * actions.ndim
        shape[0] = -1
        m = mean.to(actions.dtype).view(shape)
        s = std.to(actions.dtype).view(shape)
        return actions * s + m

    # --- min-max with quantile ----------------------
    def normalize_minmax_quantile(self, x: torch.Tensor,
                                  q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        """2 * (x - q01) / (q99 - q01) - 1  →  values in q01..q99 land in [-1, 1].

        x: [dim, ...]; q01/q99: [dim].  Outliers are clamped to
        ±``norm_clamp_abs`` (e.g. 5.0) so noise targets stay finite.
        """
        return self._normalize_linear_range(x, q01, q99)

    def denormalize_minmax_quantile(self, x: torch.Tensor,
                                    q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        return self._denormalize_linear_range(x, q01, q99)

    # --- min-max (true global min/max, no quantile) ----------------------
    def normalize_minmax(self, x: torch.Tensor,
                         x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
        """2 * (x - min) / (max - min) - 1  →  training data lands exactly in [-1, 1].

        x: [dim, ...]; x_min/x_max: [dim].  Same outlier clamp behavior as
        ``normalize_minmax_quantile`` via ``norm_clamp_abs``.
        """
        return self._normalize_linear_range(x, x_min, x_max)

    def denormalize_minmax(self, x: torch.Tensor,
                           x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
        return self._denormalize_linear_range(x, x_min, x_max)

    def _normalize_linear_range(self, x: torch.Tensor,
                                lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.ndim
        shape[0] = -1
        lo = lo.to(x.dtype).view(shape)
        hi = hi.to(x.dtype).view(shape)
        span = (hi - lo).clamp(min=self.norm_eps)
        normed = (x - lo) * 2.0 / span - 1.0
        if self.norm_clamp_abs is not None and self.norm_clamp_abs > 0:
            normed = normed.clamp(-self.norm_clamp_abs, self.norm_clamp_abs)
        return normed

    def _denormalize_linear_range(self, x: torch.Tensor,
                                  lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.ndim
        shape[0] = -1
        lo = lo.to(x.dtype).view(shape)
        hi = hi.to(x.dtype).view(shape)
        return (x + 1.0) * 0.5 * (hi - lo) + lo

    # --- Unified dispatch entrypoints  (call these from datasets) ---------

    def normalize(self,
                  x: torch.Tensor,
                  *,
                  mean: torch.Tensor | None = None,
                  std: torch.Tensor | None = None,
                  q01: torch.Tensor | None = None,
                  q99: torch.Tensor | None = None,
                  x_min: torch.Tensor | None = None,
                  x_max: torch.Tensor | None = None) -> torch.Tensor:
        """Pick the right normalize function based on ``self.norm_mode``."""
        if self.norm_mode == "zscore":
            if mean is None or std is None:
                raise ValueError("norm_mode='zscore' requires mean/std stats.")
            return self.normalize_zscore(x, mean, std)
        if self.norm_mode == "minmax_quantile":
            if q01 is None or q99 is None:
                raise ValueError("norm_mode='minmax_quantile' requires q01/q99 stats.")
            return self.normalize_minmax_quantile(x, q01, q99)
        if self.norm_mode == "minmax":
            if x_min is None or x_max is None:
                raise ValueError("norm_mode='minmax' requires min/max stats.")
            return self.normalize_minmax(x, x_min, x_max)
        raise ValueError(f"unknown norm_mode={self.norm_mode!r}")

    def denormalize(self,
                    x: torch.Tensor,
                    *,
                    mean: torch.Tensor | None = None,
                    std: torch.Tensor | None = None,
                    q01: torch.Tensor | None = None,
                    q99: torch.Tensor | None = None,
                    x_min: torch.Tensor | None = None,
                    x_max: torch.Tensor | None = None) -> torch.Tensor:
        """Inverse of ``normalize`` — dispatched by ``self.norm_mode``."""
        if self.norm_mode == "zscore":
            if mean is None or std is None:
                raise ValueError("norm_mode='zscore' requires mean/std stats.")
            return self.denormalize_zscore(x, mean, std)
        if self.norm_mode == "minmax_quantile":
            if q01 is None or q99 is None:
                raise ValueError("norm_mode='minmax_quantile' requires q01/q99 stats.")
            return self.denormalize_minmax_quantile(x, q01, q99)
        if self.norm_mode == "minmax":
            if x_min is None or x_max is None:
                raise ValueError("norm_mode='minmax' requires min/max stats.")
            return self.denormalize_minmax(x, x_min, x_max)
        raise ValueError(f"unknown norm_mode={self.norm_mode!r}")

    # --- per-field entry points (state vs action) — the SINGLE source of truth
    #     used by BOTH the training dataset and the deploy policy. Stats are
    #     passed in (the caller owns them); norm_mode dispatch + relative-action
    #     handling live here so the two paths can never diverge. -------------

    def state_normalize(self, x: torch.Tensor, *, mean=None, std=None,
                        q01=None, q99=None, x_min=None, x_max=None) -> torch.Tensor:
        """Normalize state by norm_mode (state is never made relative)."""
        return self.normalize(x, mean=mean, std=std, q01=q01, q99=q99,
                              x_min=x_min, x_max=x_max)

    def state_denormalize(self, x: torch.Tensor, *, mean=None, std=None,
                          q01=None, q99=None, x_min=None, x_max=None) -> torch.Tensor:
        """Denormalize state by norm_mode."""
        return self.denormalize(x, mean=mean, std=std, q01=q01, q99=q99,
                                x_min=x_min, x_max=x_max)

    def action_normalize(self, x: torch.Tensor, anchor_state=None, *,
                         mean=None, std=None, q01=None, q99=None,
                         x_min=None, x_max=None, step_axis: int = 2,
                         rel_mean=None, rel_std=None, rel_q01=None, rel_q99=None,
                         rel_min=None, rel_max=None) -> torch.Tensor:
        """Single entry point for action normalization — discriminates here:
          - relative_action ON (and an anchor given): subtract the anchor on the
            relative dims (`_relative_shift`) THEN PER-STEP normalize with the
            rel_* stats (`action_normalize_perstep`). Each chunk step is
            normalized by its own distribution.
          - otherwise: the plain absolute normalize with single-vector stats.
        """
        if self.relative_action and anchor_state is not None:
            x = self._relative_shift(x, anchor_state, -1.0)   # action - anchor
            return self.action_normalize_perstep(
                x, step_axis=step_axis,
                rel_mean=rel_mean, rel_std=rel_std, rel_q01=rel_q01,
                rel_q99=rel_q99, rel_min=rel_min, rel_max=rel_max)
        return self.normalize(x, mean=mean, std=std, q01=q01, q99=q99,
                              x_min=x_min, x_max=x_max)

    def action_denormalize(self, x: torch.Tensor, anchor_state=None, *,
                           mean=None, std=None, q01=None, q99=None,
                           x_min=None, x_max=None, step_axis: int = 2,
                           rel_mean=None, rel_std=None, rel_q01=None, rel_q99=None,
                           rel_min=None, rel_max=None) -> torch.Tensor:
        """Exact inverse of action_normalize — same discrimination:
          - relative_action ON (and an anchor given): PER-STEP denormalize with
            the rel_* stats THEN add the anchor back on the relative dims.
          - otherwise: plain absolute denormalize.
        """
        if self.relative_action and anchor_state is not None:
            x = self.action_denormalize_perstep(
                x, step_axis=step_axis,
                rel_mean=rel_mean, rel_std=rel_std, rel_q01=rel_q01,
                rel_q99=rel_q99, rel_min=rel_min, rel_max=rel_max)
            return self._relative_shift(x, anchor_state, +1.0)   # action + anchor
        return self.denormalize(x, mean=mean, std=std, q01=q01, q99=q99,
                                x_min=x_min, x_max=x_max)

    # --- per-step (relative) normalization --------------------------------
    # Used when relative_action is on AND per-step relative stats are available.
    # Unlike the single-vector path above (one [action_dim] stat shared by every
    # chunk step), these index per-step stats [H, action_dim] by the within-block
    # offset so chunk step j is normalized by rel_stat[j]. Lives here (the config
    # base) so BOTH the training dataset and the deploy policy use identical math.

    def _perstep_stat_view(self, stat: torch.Tensor, ndim: int, step_axis: int,
                           n_steps: int, device, dtype) -> torch.Tensor:
        """Slice a per-step stat [H, action_dim] to `n_steps` along the chunk
        offset and reshape so action_dim lands on axis 0 and the step varies on
        `step_axis` (all other axes broadcast). Offsets >= H clamp to the last
        available step."""
        H, A = stat.shape
        idx = torch.arange(n_steps, device=device).clamp(max=H - 1)
        s = stat.to(device=device, dtype=dtype)[idx]      # [n_steps, A]
        shape = [1] * ndim
        shape[0] = A
        shape[step_axis] = n_steps
        return s.transpose(0, 1).reshape(shape)

    def action_normalize_perstep(self, x: torch.Tensor, *, step_axis: int = 2,
                                 rel_mean=None, rel_std=None,
                                 rel_q01=None, rel_q99=None,
                                 rel_min=None, rel_max=None) -> torch.Tensor:
        """PER-STEP normalize an (already relative-shifted) action chunk.

        `x` has action_dim on axis 0 and the chunk offset on `step_axis`
        (training: [action_dim, nf, apf, 1] → step_axis=2, the apf within-block
        offset). Non-relative dims must already carry the absolute stats inside
        the rel_* arrays (done at aggregation time) so they normalize
        consistently across steps. The relative subtract itself is done by the
        caller (action_normalize). norm_mode dispatch / norm_eps / norm_clamp_abs
        match normalize()."""
        n = x.shape[step_axis]
        def V(stat):
            return self._perstep_stat_view(stat, x.ndim, step_axis, n, x.device, x.dtype)
        if self.norm_mode == "zscore":
            if rel_mean is None or rel_std is None:
                raise ValueError("per-step zscore requires rel_mean/rel_std.")
            normed = (x - V(rel_mean)) / V(rel_std).clamp(min=self.norm_eps)
        elif self.norm_mode == "minmax_quantile":
            if rel_q01 is None or rel_q99 is None:
                raise ValueError("per-step minmax_quantile requires rel_q01/rel_q99.")
            lo, hi = V(rel_q01), V(rel_q99)
            normed = (x - lo) * 2.0 / (hi - lo).clamp(min=self.norm_eps) - 1.0
        elif self.norm_mode == "minmax":
            if rel_min is None or rel_max is None:
                raise ValueError("per-step minmax requires rel_min/rel_max.")
            lo, hi = V(rel_min), V(rel_max)
            normed = (x - lo) * 2.0 / (hi - lo).clamp(min=self.norm_eps) - 1.0
        else:
            raise ValueError(f"unknown norm_mode={self.norm_mode!r}")
        if self.norm_clamp_abs is not None and self.norm_clamp_abs > 0:
            normed = normed.clamp(-self.norm_clamp_abs, self.norm_clamp_abs)
        return normed

    def action_denormalize_perstep(self, x: torch.Tensor, *, step_axis: int = 2,
                                   rel_mean=None, rel_std=None,
                                   rel_q01=None, rel_q99=None,
                                   rel_min=None, rel_max=None) -> torch.Tensor:
        """PER-STEP denormalize — inverse of action_normalize_perstep. The
        relative add-back is done by the caller (action_denormalize)."""
        n = x.shape[step_axis]
        def V(stat):
            return self._perstep_stat_view(stat, x.ndim, step_axis, n,
                                           x.device, x.dtype)
        if self.norm_mode == "zscore":
            return x * V(rel_std) + V(rel_mean)
        if self.norm_mode == "minmax_quantile":
            lo, hi = V(rel_q01), V(rel_q99)
            return (x + 1.0) * 0.5 * (hi - lo) + lo
        if self.norm_mode == "minmax":
            lo, hi = V(rel_min), V(rel_max)
            return (x + 1.0) * 0.5 * (hi - lo) + lo
        raise ValueError(f"unknown norm_mode={self.norm_mode!r}")
