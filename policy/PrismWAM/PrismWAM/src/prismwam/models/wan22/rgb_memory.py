"""PrismWAM Stage 2 with a sparse RGB memory (RoboDojo); Stage 1 is unchanged.

The RGB history is three observed frames (the episode's first frame, the frame
32 steps back and the current frame), each encoded on its own. They build a
fixed per-plan context. RGB goal prediction stays supervised, but neither
depth nor the stored memory can read that target. Current proprio is used by
depth/action only, so the cached RGB keys/values do not depend on the state.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from omegaconf import OmegaConf


from .rgb_depth_action_depth_parent import (
    PrismWAMSharedDepthParent,
    create_prismwam_rgb_depth_action as create_parent,
)
from .sparse_memory import (
    RGBMemoryConfig,
    append_rgb_kv,
    memory_mask,
    memory_suffix_mask,
    pack_rgb_memory,
    training_rgb_mask,
)


class RGBMemoryPrismWAM(PrismWAMSharedDepthParent):
    def setup_rgb_memory(self, config: dict[str, Any]) -> None:
        self.rgb_memory_config = RGBMemoryConfig(**config)
        self.rgb_history_num_frames = 3
        self.rgb_condition_latent_frames = 3
        if not self.video_use_text_embedding:
            raise ValueError("The RGB memory model requires the text embedding.")
        if self.rgb_expert.blocks is not self.video_expert.blocks:
            raise ValueError(
                "The RGB memory model must retain the shared RGB/Depth backbone"
            )
        if (
            self.rgb_history_frame_stride != self.rgb_memory_config.raw_frame_stride
            or self.rgb_memory_config.vae_temporal_stride != 4
        ):
            raise ValueError("The episode cache requires Wan temporal stride 4")
        self._memory_lengths = None
        self._rgb_context_has_proprio = False
        self._validating = False
        self.reset_rgb_memory()

    @contextmanager
    def loss_validation_context(self):
        """Loss-only validation runs in training mode; keep it noise-free."""
        self._validating = True
        try:
            yield
        finally:
            self._validating = False

    def reset_rgb_memory(self) -> None:
        self._rgb_memory_state = None
        self._rgb_memory_context = None
        self._rgb_memory_context_mask = None
        self._rgb_memory_latents = None

    def _append_proprio_to_context(self, context, context_mask, proprio):
        self._rgb_context_has_proprio = (
            self.proprio_encoder is not None and proprio is not None
        )
        return super()._append_proprio_to_context(context, context_mask, proprio)

    def _rgb_context(self, context, mask):
        if self._rgb_context_has_proprio:
            return (context[:, :-1], mask[:, :-1])
        return (context, mask)

    @torch.no_grad()
    def _encode_memory_video(self, video: torch.Tensor) -> torch.Tensor:
        video = self._resize_rgb_spatial(
            video.to(device=self.device, dtype=self.torch_dtype)
        )
        encoded = self.rgb_vae.encode(video, device=self.device, tiled=False)
        if isinstance(encoded, list):
            encoded = torch.stack(encoded)
        if encoded.ndim != 5 or encoded.shape[2] != (video.shape[2] - 1) // 4 + 1:
            raise ValueError("RGB memory VAE geometry mismatch")
        return encoded.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_independent_rgb_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Encode each raw observation in a separate one-frame VAE call."""
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 1:
            raise ValueError("Independent RGB frames require [B,3,T,H,W]")
        return torch.cat(
            [
                self._encode_memory_video(video[:, :, index : index + 1])
                for index in range(video.shape[2])
            ],
            dim=2,
        )

    def _encode_rgb_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 4:
            raise ValueError("Sparse training expects RGB [anchor,past,current,F8]")
        prefix = self._encode_independent_rgb_frames(video[:, :, :3])
        target = self._encode_memory_video(video[:, :, 3:])
        return torch.cat((prefix, target), dim=2)

    def build_inputs(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> dict[str, Any]:
        if "rgb_memory" in sample:
            raise ValueError(
                "Independent sparse RGB must not receive a full causal prefix"
            )
        inputs = super().build_inputs(sample, tiled=tiled)
        self._memory_lengths = torch.full(
            (inputs["rgb_latents"].shape[0],), 3, device=self.device, dtype=torch.long
        )
        return inputs

    def training_loss_with_components(self, sample, tiled=False):
        try:
            return super().training_loss_with_components(sample, tiled=tiled)
        finally:
            self._memory_lengths = None

    def _mix_clean_conditioning_latents(
        self, latents: torch.Tensor, *, stream: str
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Apply clean-video conditioning regularization.

        Each selected sample/frame gets an independent alpha ~ U[0,1] and
        ``(1-alpha) * clean + alpha * N(0,1)``.  Targets remain based on the
        original clean latent, and action latents are never corrupted.
        Validation is kept deterministic: the noise applies only while the MoT
        is in training mode and outside ``loss_validation_context``.
        """
        config = self.rgb_memory_config
        probability = config.conditioning_noise_probability(stream)
        streams = str(config.conditioning_latent_noise_stream)
        enabled_streams = {
            "rgb": {"rgb"},
            "depth": {"depth"},
            "rgb_depth": {"rgb", "depth"},
            "none": set(),
        }[streams]
        if (
            probability == 0.0
            or stream not in enabled_streams
            or (not self.mot.training)
            or self._validating
        ):
            return (latents, {"fraction": 0.0, "mean_ratio": 0.0})
        if latents.ndim != 5:
            raise ValueError(
                f"Conditioning latent must be [B,C,T,H,W], got {tuple(latents.shape)}"
            )
        (batch, frames) = (int(latents.shape[0]), int(latents.shape[2]))
        history = latents[:, :, :frames]
        selected = torch.rand(batch, device=latents.device) < probability
        ratio = torch.rand(
            (batch, 1, frames, 1, 1), device=latents.device, dtype=latents.dtype
        )
        mixed = history * (1.0 - ratio) + torch.randn_like(history) * ratio
        selector = selected[:, None, None, None, None]
        mixed = torch.where(selector, mixed, history)
        if frames < latents.shape[2]:
            mixed = torch.cat((mixed, latents[:, :, frames:]), dim=2)
        mean_ratio = (
            float(ratio[selected].mean().detach()) if bool(selected.any()) else 0.0
        )
        return (
            mixed,
            {
                "fraction": float(selected.float().mean().detach()),
                "mean_ratio": mean_ratio,
            },
        )

    def _rgb_pre(
        self,
        latents,
        timestep,
        context,
        context_mask,
        *,
        condition_frames,
        start_frame=0,
    ):
        (context, context_mask) = self._rgb_context(context, context_mask)
        return self.rgb_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            temporal_rope_positions=list(
                range(start_frame, start_frame + latents.shape[2])
            ),
            num_condition_frames=condition_frames,
        )

    def _suffix_pre(
        self,
        depth_latents,
        action_latents,
        timestep_depth,
        timestep_action,
        context,
        context_mask,
    ):
        depth = self.video_expert.pre_dit(
            x=depth_latents,
            timestep=timestep_depth,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            num_condition_frames=self.depth_condition_latent_frames,
        )
        action = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        depth_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=depth["tokens"].shape[1],
            video_tokens_per_frame=depth["meta"]["tokens_per_frame"],
            device=depth_latents.device,
            num_condition_frames=self.depth_condition_latent_frames,
        )
        return (depth, action, depth_mask)

    def _predict_three_noise(
        self,
        *,
        rgb_latents,
        depth_latents,
        action_latents,
        timestep_rgb,
        timestep_depth,
        timestep_action,
        context,
        context_mask,
    ):
        if self._memory_lengths is None:
            raise RuntimeError("Missing RGB memory prefix lengths")
        pre = self._rgb_pre(
            rgb_latents,
            timestep_rgb,
            context,
            context_mask,
            condition_frames=self.rgb_condition_latent_frames,
        )
        packed = pack_rgb_memory(
            pre,
            condition_frames=self.rgb_condition_latent_frames,
            lengths=self._memory_lengths,
            expert=self.rgb_expert,
        )
        (rgb_mask, latest) = training_rgb_mask(
            packed, self._memory_lengths, self.rgb_memory_config
        )
        (depth, action, depth_mask) = self._suffix_pre(
            depth_latents,
            action_latents,
            timestep_depth,
            timestep_action,
            context,
            context_mask,
        )
        mask = memory_mask(rgb_mask, latest, depth_mask, action["tokens"].shape[1])
        states = {"rgb": packed, "video": depth, "action": action}
        outputs = self.mot(
            embeds_all={k: v["tokens"] for (k, v) in states.items()},
            attention_mask=mask,
            freqs_all={k: v["freqs"] for (k, v) in states.items()},
            context_all={
                "rgb": self._video_context_payload(packed),
                "video": self._video_context_payload(depth),
                "action": self._text_context_payload(
                    action["context"], action["context_mask"]
                ),
            },
            t_mod_all={k: v["t_mod"] for (k, v) in states.items()},
        )
        return (
            self.rgb_expert.post_dit(outputs["rgb"][:, packed["visual_indices"]], pre),
            self.video_expert.post_dit(outputs["video"], depth),
            self.action_expert.post_dit(outputs["action"], action),
        )

    @torch.no_grad()
    def _encode_rgb_history_images_latents_tensor(self, rgb_history_images, **kwargs):
        del kwargs
        if (
            rgb_history_images.ndim != 5
            or rgb_history_images.shape[0] != 1
            or rgb_history_images.shape[2] != 3
        ):
            raise ValueError("Sparse inference expects [1,3,3,H,W] raw observations")
        return self._encode_independent_rgb_frames(rgb_history_images)

    def _infer_action_latents(self, **kwargs):
        self._rgb_context_has_proprio = False
        return super()._infer_action_latents(**kwargs)

    @torch.no_grad()
    def _prefill_clean_rgb_cache(self, *, condition_rgb, context, context_mask):
        if condition_rgb.shape[0] != 1:
            raise ValueError("Reset/keep one RGB memory per evaluation episode")
        (rgb_context, rgb_mask) = self._rgb_context(context, context_mask)
        if condition_rgb.shape[2] != 3:
            raise ValueError(
                "Sparse prefill requires exactly three observation latents"
            )
        selected = condition_rgb
        lengths = torch.tensor([3], device=condition_rgb.device)
        pre = self._rgb_pre(
            selected,
            selected.new_zeros(1),
            context,
            context_mask,
            condition_frames=3,
            start_frame=0,
        )
        packed = pack_rgb_memory(
            pre, condition_frames=3, lengths=lengths, expert=self.rgb_expert
        )
        self._rgb_memory_state = append_rgb_kv(
            self.mot, packed, self.rgb_memory_config, previous=None
        )
        self._rgb_memory_context = rgb_context.clone()
        self._rgb_memory_context_mask = rgb_mask.clone()
        self._rgb_memory_latents = condition_rgb.clone()
        cache = self._rgb_memory_state["layers"]
        return ({"memory_prefix_len": cache[0]["prefix_seq_len"]}, cache)

    @torch.no_grad()
    def _predict_depth_action_noise_with_rgb_cache(
        self,
        *,
        rgb_pre,
        rgb_kv_cache,
        depth_latents,
        action_latents,
        timestep_depth,
        timestep_action,
        context,
        context_mask,
    ):
        (depth, action, depth_mask) = self._suffix_pre(
            depth_latents,
            action_latents,
            timestep_depth,
            timestep_action,
            context,
            context_mask,
        )
        prefix = rgb_pre["memory_prefix_len"]
        mask = memory_suffix_mask(
            rgb_tokens=int(prefix),
            depth_mask=depth_mask,
            action_tokens=int(action["tokens"].shape[1]),
        )
        states = {"video": depth, "action": action}
        outputs = self.mot.forward_with_prefix_expert_cache(
            prefix_expert_name="rgb",
            prefix_kv_cache=rgb_kv_cache,
            embeds_all={k: v["tokens"] for (k, v) in states.items()},
            attention_mask=mask,
            freqs_all={k: v["freqs"] for (k, v) in states.items()},
            context_all={
                "video": self._video_context_payload(depth),
                "action": self._text_context_payload(
                    action["context"], action["context_mask"]
                ),
            },
            t_mod_all={k: v["t_mod"] for (k, v) in states.items()},
        )
        return (
            self.video_expert.post_dit(outputs["video"], depth),
            self.action_expert.post_dit(outputs["action"], action),
        )

    def _extra_checkpoint_payload(self):
        payload = super()._extra_checkpoint_payload()
        payload["rgb_memory_config"] = self.rgb_memory_config.to_dict()
        return payload

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if payload.get("rgb_memory_config") != self.rgb_memory_config.to_dict():
            raise ValueError("Checkpoint RGB memory settings are different")
        del payload
        result = super().load_checkpoint(path, optimizer=optimizer)
        self.reset_rgb_memory()
        return result


def create_prismwam_rgb_depth_action(
    *, rgb_memory, model_dtype, device, **kwargs
) -> RGBMemoryPrismWAM:
    config = (
        OmegaConf.to_container(rgb_memory, resolve=True)
        if OmegaConf.is_config(rgb_memory)
        else rgb_memory
    )
    RGBMemoryConfig(**dict(config))  # validate before loading any weights
    if kwargs.get("rgb_history_num_frames") not in (1, 3, 5):
        raise ValueError("Sparse RGB configuration has invalid history declaration")
    kwargs["rgb_history_num_frames"] = 1
    model = create_parent(
        model_dtype=model_dtype, device=device, _model_class=RGBMemoryPrismWAM, **kwargs
    )
    model.setup_rgb_memory(dict(config))
    return model
