"""Non-autoregressive Motubrain model based on the GVA backbone."""

import copy
import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from falcon.config import TrainerConfig
from falcon.models.motubrain.conditioning import condition_key_layout
from falcon.models.motubrain.layers import (
    MotubrainFinalLayer,
    MotubrainLayer,
)
from falcon.models.motubrain.model_config import MotubrainConfig
from falcon.models.motubrain.modules import (
    MotubrainDecoupledAttention,
    MotubrainJointAttention,
    MotubrainV2AAttention,
)
from falcon.models.motu_3_2_gva.model import MotuGVAModel
from falcon.modules import TimestepEmbed
from soma.inference.models.operators import (
    build_rope_cache,
    ensure_bf16,
    ensure_fp32,
)

logger = logging.getLogger(__name__)


def _cumulative_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Build device-local int32 cumulative sequence lengths."""
    lengths = lengths.to(dtype=torch.int32)
    return torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0, dtype=torch.int32)])


def build_video_key_layout(
    video_key_mask: torch.Tensor,
    video_lengths: torch.Tensor,
    *following_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Build indices and lengths after removing invalid packed video keys."""
    if video_key_mask.ndim != 1 or video_key_mask.dtype != torch.bool:
        raise ValueError("video_key_mask must be a one-dimensional bool tensor")
    sample_count = video_lengths.numel()
    if any(lengths.numel() != sample_count for lengths in following_lengths):
        raise ValueError("video and following key lengths must have equal size")
    if int(video_lengths.sum().item()) != video_key_mask.numel():
        raise ValueError(
            "video_key_mask length must equal the total video token count"
        )

    video_counts = video_lengths.tolist()
    following_counts = [lengths.tolist() for lengths in following_lengths]
    indices = []
    key_lengths = []
    video_offset = 0
    packed_offset = 0
    for sample_index, video_count in enumerate(video_counts):
        sample_video_mask = video_key_mask[
            video_offset : video_offset + video_count
        ]
        valid_video = sample_video_mask.nonzero(as_tuple=False).flatten()
        indices.append(valid_video + packed_offset)
        packed_offset += video_count
        video_offset += video_count
        key_length = valid_video.numel()
        for counts in following_counts:
            count = counts[sample_index]
            indices.append(
                torch.arange(
                    packed_offset,
                    packed_offset + count,
                    device=video_key_mask.device,
                )
            )
            packed_offset += count
            key_length += count
        key_lengths.append(key_length)

    lengths = torch.tensor(
        key_lengths,
        dtype=torch.int32,
        device=video_key_mask.device,
    )
    return (
        torch.cat(indices).to(dtype=torch.long),
        _cumulative_lengths(lengths),
        int(lengths.max().item()),
        lengths,
    )


def _pair_timesteps(
    video_timestep: torch.Tensor,
    action_timestep: torch.Tensor | None,
) -> torch.Tensor:
    """Pair independently sampled video/action timesteps by sample."""
    if video_timestep.ndim != 1:
        raise ValueError("video timestep t must be one-dimensional")
    if action_timestep is None:
        action_timestep = video_timestep
    if action_timestep.shape != video_timestep.shape:
        raise ValueError("t_a must match video timestep shape " f"{video_timestep.shape}, got {action_timestep.shape}")
    return torch.stack((video_timestep, action_timestep), dim=1)


class _MotubrainTimestepEmbed(TimestepEmbed):
    """Embed paired video/action timesteps with the existing timestep MLP."""

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 2 or timesteps.shape[1] != 2:
            raise ValueError("Motubrain timesteps must have shape [batch, video/action]")
        batch_size = timesteps.shape[0]
        embedded = super().forward(timesteps.reshape(-1))
        return embedded.reshape(batch_size, 2, -1)


def build_multiview_hw_offsets(
    view_counts: torch.Tensor,
    view_heights: torch.Tensor,
    view_widths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cumulative H/W RoPE offsets, restarting at each sample."""
    if view_counts.ndim != 1 or view_heights.ndim != 1 or view_widths.ndim != 1:
        raise ValueError("view_counts, view_heights, and view_widths must all be one-dimensional")
    num_views = int(view_counts.sum().item())
    if num_views != view_heights.numel() or num_views != view_widths.numel():
        raise ValueError(
            "sum(view_counts) must equal the number of view heights and widths, "
            f"got {num_views}, {view_heights.numel()}, and {view_widths.numel()}"
        )
    if torch.any(view_counts <= 0):
        raise ValueError("view_counts must contain only positive values")
    if torch.any(view_heights <= 0) or torch.any(view_widths <= 0):
        raise ValueError("view heights and widths must contain only positive values")

    height_values = view_heights.tolist()
    width_values = view_widths.tolist()
    height_offsets: list[int] = []
    width_offsets: list[int] = []
    view_index = 0
    for view_count in view_counts.tolist():
        cumulative_height = 0
        cumulative_width = 0
        for _ in range(view_count):
            height_offsets.append(cumulative_height)
            width_offsets.append(cumulative_width)
            cumulative_height += height_values[view_index]
            cumulative_width += width_values[view_index]
            view_index += 1
    return (
        torch.tensor(height_offsets, dtype=torch.int32, device=view_counts.device),
        torch.tensor(width_offsets, dtype=torch.int32, device=view_counts.device),
    )


def resolve_attention_class(
    *,
    layer_number: int,
    num_layers: int,
    bridge_layer_range: Sequence[int] | None,
    v2a: bool,
) -> type[nn.Module]:
    """Select one layer's attention implementation at the model boundary.

    ``bridge_layer_range=(start, end)`` uses zero-indexed half-open boundaries,
    while ``layer_number`` is one-indexed. For example, ``(6, 18)`` enables
    bridging in layers 7 through 18.
    """
    if not 1 <= layer_number <= num_layers:
        raise ValueError(f"layer_number must be in [1, {num_layers}], got {layer_number}")
    if bridge_layer_range is None:
        return MotubrainV2AAttention if v2a else MotubrainJointAttention
    if len(bridge_layer_range) != 2:
        raise ValueError(
            "bridge_layer_range must contain exactly two zero-indexed half-open boundaries, "
            f"got {tuple(bridge_layer_range)!r}"
        )
    start, end = bridge_layer_range
    if not 0 <= start < end <= num_layers:
        raise ValueError(f"bridge_layer_range must satisfy 0 <= start < end <= {num_layers}, got {(start, end)!r}")
    if layer_number <= start or layer_number > end:
        return MotubrainDecoupledAttention
    return MotubrainV2AAttention if v2a else MotubrainJointAttention


class MotubrainModel(MotuGVAModel):
    """Video-action model for the first non-AR Motubrain release."""

    def __init__(self, cfg: TrainerConfig):
        base_cfg = copy.deepcopy(cfg)
        action_dim = int(base_cfg.model.get("action_dim", 55))
        action_mask_concat = base_cfg.model.get(
            "action_mask_concat",
            False,
        )
        if not isinstance(action_mask_concat, bool):
            raise ValueError(
                "action_mask_concat must be a boolean, got "
                f"{action_mask_concat!r}"
            )
        base_cfg.model["a_in_channels"] = action_dim
        base_cfg.model["a_embed_in_channels"] = action_dim * (
            2 if action_mask_concat else 1
        )
        super().__init__(base_cfg)

        self.motu_cfg = MotubrainConfig(base_cfg)
        self.action_dim = self.motu_cfg.action_dim
        self.a_in_channels = self.motu_cfg.a_in_channels
        self.a_embed_in_channels = self.motu_cfg.a_embed_in_channels
        self.timestep_embed = _MotubrainTimestepEmbed(self.motu_cfg.hidden_size)

        attention_classes = [
            resolve_attention_class(
                layer_number=layer_number,
                num_layers=self.motu_cfg.num_layers,
                bridge_layer_range=self.motu_cfg.bridge_layer_range,
                v2a=self.motu_cfg.v2a,
            )
            for layer_number in range(1, self.motu_cfg.num_layers + 1)
        ]
        blocks = []
        for layer_number, attention_class in enumerate(attention_classes, start=1):
            blocks.append(
                MotubrainLayer(
                    layer_number,
                    self.motu_cfg,
                    attention_class,
                )
            )
        self.vidu_blocks = nn.ModuleList(blocks)
        self.final_layer = MotubrainFinalLayer(
            hidden_size=self.motu_cfg.hidden_size,
            out_channels=self.v_in_channels * self.patch_size**2,
            audio_in_channels=self.action_dim,
        )
        self.timestep_embed.init_weights()
        for block in self.vidu_blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.loss_weight_action = self.motu_cfg.action_loss_weight

        self.view_embed = None
        if self.motu_cfg.view_embed_mode == "additive":
            # Supply zeros directly so creating the table consumes no RNG.
            self.view_embed = nn.Embedding.from_pretrained(
                torch.zeros(len(self.motu_cfg.view_embed_names), self.motu_cfg.hidden_size),
                freeze=False,
            )

        self.register_load_state_dict_pre_hook(self._adapt_gva_state_dict)

    def apply_view_embedding(self, hidden_states: torch.Tensor, mb_meta: dict) -> torch.Tensor:
        """Add camera identity after patch embedding, using the video CP layout."""
        if self.view_embed is None:
            return hidden_states
        if "view_ids_view" not in mb_meta:
            raise ValueError("additive view embedding requires view_ids_view")
        view_ids = mb_meta["view_ids_view"].to(device=hidden_states.device, dtype=torch.long)
        token_ids = torch.repeat_interleave(
            view_ids, mb_meta["Vs_view"].to(device=hidden_states.device, dtype=torch.long)
        )
        token_ids = self.preprocess_context_parallel(token_ids, mb_meta["cuVs"], mb_meta["maxV"])
        return hidden_states + self.view_embed(token_ids)

    def _adapt_gva_state_dict(
        self,
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Keep Motubrain action I/O when loading a shape-incompatible GVA head."""
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        current_state = module.state_dict()
        adaptable = (
            "a_embed.0.weight",
            "final_layer.a_linear.weight",
            "final_layer.a_linear.bias",
        )
        replaced = []
        for name in adaptable:
            source_name = f"{prefix}{name}"
            if source_name not in state_dict:
                continue
            target = current_state[name]
            if state_dict[source_name].shape != target.shape:
                state_dict[source_name] = target.detach().clone()
                replaced.append(name)
        if replaced:
            logger.warning(
                "Loaded GVA backbone while retaining Motubrain action I/O: %s",
                ", ".join(replaced),
            )

    def prepare_mb_meta(self, **kwargs: Any) -> dict:
        """Extend GVA audio metadata without changing its CP layout."""
        videos = kwargs["Vs"]
        num_samples = videos.numel()
        modal_batch_sizes = {name: kwargs[name].numel() for name in ("As", "Cs", "As_padded", "Cs_padded")}
        inconsistent = {name: size for name, size in modal_batch_sizes.items() if size != num_samples}
        if inconsistent:
            raise ValueError(f"all modal metadata must contain {num_samples} samples, got {inconsistent}")

        mb_meta = super().prepare_mb_meta(**kwargs)
        videos = mb_meta["Vs"]
        captions = mb_meta["Cs"]
        vc_lengths = (videos + captions).to(dtype=torch.int32)
        cumulative_vc = _cumulative_lengths(vc_lengths)
        mb_meta.update(
            VCs=vc_lengths,
            cuVCs=cumulative_vc,
            maxVC=vc_lengths.max().item(),
            totalVC=cumulative_vc[-1].item(),
        )

        video_key_mask = mb_meta.get("video_key_mask")
        if video_key_mask is not None:
            vc_layout = build_video_key_layout(
                video_key_mask,
                videos,
                captions,
            )
            vac_layout = build_video_key_layout(
                video_key_mask,
                videos,
                mb_meta["As"],
                captions,
            )
            mb_meta.update(
                VC_key_indices=vc_layout[0],
                cuVCs_key=vc_layout[1],
                maxVC_key=vc_layout[2],
                VCs_key=vc_layout[3],
                VAC_key_indices=vac_layout[0],
                cuVACs_key=vac_layout[1],
                maxVAC_key=vac_layout[2],
                VACs_key=vac_layout[3],
            )

        if "NVs" in mb_meta:
            required = ("Fs_view", "Hs_view", "Ws_view", "cuVs_rope")
            missing = [name for name in required if name not in mb_meta]
            if missing:
                raise ValueError(f"multiview metadata is missing required keys: {missing}")
            num_views = int(mb_meta["NVs"].sum().item())
            if mb_meta["NVs"].numel() != num_samples:
                raise ValueError(
                    f"NVs must contain one view count per sample ({num_samples}), got {mb_meta['NVs'].numel()}"
                )
            view_meta_sizes = {
                "Fs_view": mb_meta["Fs_view"].numel(),
                "Hs_view": mb_meta["Hs_view"].numel(),
                "Ws_view": mb_meta["Ws_view"].numel(),
            }
            if any(size != num_views for size in view_meta_sizes.values()):
                raise ValueError(f"per-view metadata must contain {num_views} views, got {view_meta_sizes}")
            if int(mb_meta["cuVs_rope"][-1].item()) != int(mb_meta["cuVs"][-1].item()):
                raise ValueError("cuVs_rope and cuVs must describe the same total number of video tokens")
            view_frames = mb_meta["Fs_view"]
            view_heights = mb_meta["Hs_view"]
            view_widths = mb_meta["Ws_view"]
            view_lengths = view_frames * view_heights * view_widths
            expected_cumulative = _cumulative_lengths(view_lengths)
            if not torch.equal(mb_meta["cuVs_rope"], expected_cumulative):
                raise ValueError("cuVs_rope must match Fs_view * Hs_view * Ws_view")
            if self.motu_cfg.multiview_rope_mode == "hw_offset":
                h_offsets, w_offsets = build_multiview_hw_offsets(
                    mb_meta["NVs"],
                    view_heights,
                    view_widths,
                )
            else:
                h_offsets = torch.zeros_like(view_heights)
                w_offsets = torch.zeros_like(view_widths)
            mb_meta["h_seqlen_offsets"] = h_offsets
            mb_meta["w_seqlen_offsets"] = w_offsets
        return mb_meta

    def prepare_rope_caches(self, mb_meta: dict) -> None:
        """Build the Falcon RoPE caches shared by full and PP inference."""
        video_local = mb_meta["cuVs"][-1].item() // self.cp_size
        if "NVs" in mb_meta:
            video_cu_tokens = mb_meta["cuVs_rope"]
            video_axes_shapes = [
                mb_meta["Fs_view"],
                mb_meta["Hs_view"],
                mb_meta["Ws_view"],
            ]
            video_axes_offsets = [
                mb_meta.get("f_seqlen_offsets"),
                mb_meta["h_seqlen_offsets"],
                mb_meta["w_seqlen_offsets"],
            ]
        else:
            video_cu_tokens = mb_meta["cuVs"]
            video_axes_shapes = [
                mb_meta["Fs"],
                mb_meta["Hs"],
                mb_meta["Ws"],
            ]
            video_axes_offsets = [
                mb_meta.get("f_seqlen_offsets"),
                mb_meta.get("h_seqlen_offsets"),
                mb_meta.get("w_seqlen_offsets"),
            ]
        mb_meta["video_rope_cache"] = build_rope_cache(
            video_local,
            self.motu_cfg.hidden_size_per_attention_head,
            video_cu_tokens,
            video_axes_shapes,
            self.motu_cfg.video_rope_axes,
            axes_seqlen_offsets=video_axes_offsets,
            cp_rank=self.cp_rank,
            total_local=video_local,
        )

        action_local = mb_meta["cuAs_padded"][-1].item() // self.cp_size
        mb_meta["audio_rope_cache"] = build_rope_cache(
            action_local,
            self.motu_cfg.hidden_size_per_attention_head,
            mb_meta["cuAs_padded"],
            [mb_meta["As_padded"]],
            self.motu_cfg.audio_rope_axes,
            axes_seqlen_offsets=[mb_meta.get("a_seqlen_offsets")],
            cp_rank=self.cp_rank,
            total_local=action_local,
        )

    def prepare_action_conditions(
        self,
        mb_meta: dict,
        action_conditions: dict[str, dict] | None,
        device: torch.device,
    ) -> None:
        """Validate named inputs, shard features, and prepare sample-local keys.

        Source names in canonical sorted order determine key order. Feature
        keys are target Motubrain layers; encoder layer selection is external.
        """
        configured = self.motu_cfg.action_conditions
        supplied = action_conditions or {}
        if set(supplied) != set(configured):
            raise ValueError("action_conditions must contain exactly the configured inputs")
        if not configured:
            return
        prepared = {}
        layouts = {}
        layout_cache = {}
        for name, spec in configured.items():
            condition = supplied[name]
            features = condition["features"]
            if set(features) != set(spec["layers"]):
                raise ValueError(f"action condition {name!r} must contain every configured target layer")
            lengths = condition["lengths"].to(device=device, dtype=torch.int32)
            padded = condition["padded_lengths"].to(device=device, dtype=torch.int32)
            if torch.any(padded % self.cp_size):
                raise ValueError("padded condition lengths must be divisible by CP size")
            total = int(padded.sum())
            local = total // self.cp_size
            local_features = {}
            for layer, value in features.items():
                if value.shape != (total, spec["feature_dim"]):
                    raise ValueError(f"action condition {name!r} shape differs from lengths or configured width")
                local_features[layer] = value[
                    self.cp_rank * local : (self.cp_rank + 1) * local
                ].to(device=device, dtype=torch.bfloat16)
            prepared[name] = {"features": local_features, "lengths": lengths}
            for layer in spec["layers"]:
                layer_layouts = layouts.setdefault(layer, {})
                sources = (*layer_layouts, name)
                if sources not in layout_cache:
                    bases = {
                        "A_padded": mb_meta["As_padded"],
                        "VAC": mb_meta.get("VACs_key", mb_meta["VACs"]),
                    }
                    for previous in layer_layouts:
                        bases = {base: counts + prepared[previous]["lengths"] for base, counts in bases.items()}
                    layout_cache[sources] = {
                        base: condition_key_layout(counts, lengths, padded) for base, counts in bases.items()
                    }
                layer_layouts[name] = layout_cache[sources]
        mb_meta["action_conditions"] = prepared
        mb_meta["condition_key_layouts"] = layouts

    def forward(
        self,
        v: torch.Tensor,
        a: torch.Tensor,
        caption: torch.Tensor,
        t: torch.Tensor,
        t_a: torch.Tensor | None = None,
        force_drop_ids: bool | None = None,
        action_conditions: dict[str, dict] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GVA with independently sampled video and action timesteps."""
        mb_meta = self.prepare_mb_meta(**kwargs)
        self.prepare_action_conditions(mb_meta, action_conditions, v.device)
        paired_timesteps = _pair_timesteps(t, t_a)
        self.prepare_rope_caches(mb_meta)

        x = self.preprocess_context_parallel(
            v,
            mb_meta["cuVs"],
            mb_meta["maxV"],
        )
        action = self.preprocess_context_parallel(
            a,
            mb_meta["cuAs_padded"],
            mb_meta["maxA_padded"],
        )
        caption = self.preprocess_context_parallel(
            caption,
            mb_meta["cuCs_padded"],
            mb_meta["maxC_padded"],
        )
        x, action, caption, paired_timesteps = ensure_bf16((x, action, caption, paired_timesteps))

        adaln = self.timestep_embed(paired_timesteps)
        caption = self.c_embed(
            caption,
            mb_meta["cuCs_padded"],
            mb_meta["Cs"],
            force_drop_ids,
        )
        hidden_states = torch.utils.checkpoint.checkpoint(
            self.patch_embed,
            x,
            use_reentrant=False,
        )
        hidden_states = self.apply_view_embedding(hidden_states, mb_meta)
        action_hidden_states = torch.utils.checkpoint.checkpoint(
            self.a_embed,
            action,
            use_reentrant=False,
        )
        for block in self.vidu_blocks:
            hidden_states, action_hidden_states, caption = block(
                hidden_states,
                action_hidden_states,
                caption,
                adaln,
                mb_meta,
            )

        hidden_states, action_hidden_states = torch.utils.checkpoint.checkpoint(
            self.final_layer,
            hidden_states,
            action_hidden_states,
            adaln,
            mb_meta,
            use_reentrant=False,
        )
        if self.cp_size > 1:
            hidden_states = self.postprocess_context_parallel(
                hidden_states,
                mb_meta["cuVs"],
                mb_meta["maxV"],
            )
            action_hidden_states = self.postprocess_context_parallel(
                action_hidden_states,
                mb_meta["cuAs_padded"],
                mb_meta["maxA_padded"],
            )
        return ensure_fp32(hidden_states), ensure_fp32(action_hidden_states)
