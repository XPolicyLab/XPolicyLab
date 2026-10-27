from __future__ import annotations

import logging
import random
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange


from policy_core.utils.WAM_policy.data_utils import apply_color_jitter_to_video
from policy_core.utils.WAM_policy.video_sparse_utils import _encode_2dvae_frames, normalize_sparse_downsample_factor

logger = logging.getLogger(__name__)


def _apply_rotary_emb(x: torch.Tensor, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _random_temporal_token_keep(
    x: torch.Tensor,
    keep_ratio: float,
    mask_cur_obs: bool = False,
):
    batch_size, num_frames, num_tokens, channels = x.shape
    assert 0 < keep_ratio <= 1.0, "keep_ratio must be in (0, 1]"
    kept_len = int(num_tokens * keep_ratio)

    if not mask_cur_obs:
        x_first = x[:, 0]
        x_rest = x[:, 1:]
        scores = torch.rand(
            batch_size, num_frames - 1, num_tokens, device=x.device
        )
        indices = scores.topk(kept_len, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
        x_rest_selected = torch.gather(x_rest, dim=2, index=indices)
        x_rest_selected = x_rest_selected.reshape(
            batch_size, (num_frames - 1) * kept_len, channels
        )
    else:
        scores = torch.rand(
            batch_size, num_frames, num_tokens, device=x.device
        )
        indices = scores.topk(kept_len, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
        x_selected = torch.gather(x, dim=2, index=indices)
        x_first = x_selected[:, 0]
        x_rest_selected = x_selected[:, 1:].reshape(
            batch_size, (num_frames - 1) * kept_len, channels
        )
    return x_first, x_rest_selected


class WAM_policyTaskModel(nn.Module):
    def __init__(
        self,
        *,
        vgen: nn.Module | None = None,
        vgen_high: nn.Module | None = None,
        action_expert: nn.Module | None = None,
        dsrl_policy: nn.Module | None = None,
        vlm_reward_head: nn.Module | None = None,
        tokenizer=None,
        text_encoder=None,
        uncond_prompt_embeds=None,
    ) -> None:
        peers = (vgen, vgen_high, action_expert, dsrl_policy, vlm_reward_head)
        if all(module is None for module in peers):
            raise ValueError(
                "WAM_policyTaskModel requires at least one of: vgen, vgen_high, "
                "action_expert, dsrl_policy, vlm_reward_head"
            )
        super().__init__()
        if vgen is not None:
            self.vgen = vgen
        if vgen_high is not None:
            self.vgen_high = vgen_high
        if action_expert is not None:
            self.ae = action_expert
        if dsrl_policy is not None:
            self.dsrl_policy = dsrl_policy
        if vlm_reward_head is not None:
            self.vlm_reward_head = vlm_reward_head
        self.tokenizer = tokenizer
        object.__setattr__(self, "text_encoder", text_encoder)
        self.uncond_prompt_embeds = uncond_prompt_embeds

    @property
    def action_expert(self) -> nn.Module | None:
        return getattr(self, "ae", None)


    def forward(self, inputs):
        if self.training:
            raise RuntimeError("WAM_policy runtime is inference-only; call eval() first")
        return self.infer_forward(inputs)

    def infer_forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        request = dict(inputs)
        objective = request.pop("objective", None)
        if objective == "ae":
            return self.infer_ae_forward(**request)
        raise ValueError(f"Unsupported WAM_policy inference objective: {objective!r}")

    @staticmethod
    def _profile_start(profiler, name: str) -> None:
        if profiler is not None:
            profiler.start_region(name)

    @staticmethod
    def _profile_end(profiler, name: str) -> None:
        if profiler is not None:
            profiler.end_region(name)

    @staticmethod
    def rescale_for_ae(
        latents,
        *,
        cfg: Any,
        stats_tracker,
        vae,
        ae_video_scaling_factor: float,
    ):
        if latents is None or stats_tracker is not None:
            return latents
        if not cfg.validation.ae_rescale:
            return latents
        ratio = cfg.validation.ae_rescale_ratio
        if ratio is None:
            ratio = (
                float(ae_video_scaling_factor)
                / float(vae.config.scaling_factor)
            )
        return latents * float(ratio)

    def _prepare_ae_latents(
        self,
        *,
        video,
        video_latents_for_ae,
        sparse_length: int,
        n_view: int,
        cfg: Any,
        vae,
        stats_tracker,
        ae_video_scaling_factor: float,
        apply_color_jitter: bool = False,
        profiler=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_latents_for_ae is not None:
            if isinstance(video_latents_for_ae, tuple):
                dense_latents, sparse_latents = video_latents_for_ae
            else:
                n_frame = video_latents_for_ae.shape[2]
                if sparse_length <= 0 or sparse_length >= n_frame:
                    raise ValueError(
                        f"Invalid sparse_length={sparse_length} for AE "
                        f"latent frames={n_frame}."
                    )
                dense_latents = video_latents_for_ae[:, :, :-sparse_length]
                sparse_latents = video_latents_for_ae[:, :, -sparse_length:]
            dense_latents = self.rescale_for_ae(
                dense_latents, cfg=cfg,
                stats_tracker=stats_tracker,
                vae=vae,
                ae_video_scaling_factor=ae_video_scaling_factor,
            )
            sparse_latents = self.rescale_for_ae(
                sparse_latents, cfg=cfg,
                stats_tracker=stats_tracker,
                vae=vae,
                ae_video_scaling_factor=ae_video_scaling_factor,
            )
            return dense_latents, sparse_latents

        if video is None:
            raise ValueError("AE requires either video or video_latents_for_ae.")
        n_frame = video.shape[3]
        if sparse_length <= 0 or sparse_length >= n_frame:
            raise ValueError(
                f"Invalid sparse_length={sparse_length} for AE "
                f"video frames={n_frame}."
            )
        raw_video = video
        if apply_color_jitter and cfg.data.use_color_jitter:
            raw_video = rearrange(
                raw_video, "b c v t h w -> (b v) c t h w"
            )
            raw_video = apply_color_jitter_to_video(raw_video)
            raw_video = rearrange(
                raw_video,
                "(b v) c t h w -> b c v t h w",
                v=n_view,
            )

        n_frame_dense = n_frame - sparse_length
        dense_video = raw_video[:, :, :, :-sparse_length]
        sparse_video = raw_video[:, :, :, -sparse_length:]
        dense_frames = rearrange(
            dense_video, "b c v t h w -> (b v t) c h w"
        )
        sparse_frames = rearrange(
            sparse_video, "b c v t h w -> (b v t) c h w"
        )
        with torch.no_grad():
            self._profile_start(profiler, "vae_encode")
            dense_latents = _encode_2dvae_frames(
                vae, dense_frames, cfg.model.batch_size_2dvae, stats_tracker
            )
            sparse_latents = _encode_2dvae_frames(
                vae, sparse_frames, cfg.model.batch_size_2dvae, stats_tracker
            )
            self._profile_end(profiler, "vae_encode")
        dense_latents = rearrange(
            dense_latents,
            "(bv t) c h w -> bv c t h w",
            t=n_frame_dense,
        )
        sparse_latents = rearrange(
            sparse_latents,
            "(bv t) c h w -> bv c t h w",
            t=sparse_length,
        )
        return dense_latents, sparse_latents

    @staticmethod
    def ae_sparse_rope_args(cfg: Any) -> dict[str, Any]:
        factor = int(
            normalize_sparse_downsample_factor(
                cfg.cotrain.video_sparse_downsample_factor
            )
        )
        return {"is_sparse": factor > 1, "factor": factor}

    def _build_ae_vision_condition(
        self,
        *,
        ae_latents_dense: torch.Tensor,
        ae_latents_sparse: torch.Tensor,
        n_view: int,
        cfg: Any,
        augment_vision: bool = False,
        repeat_in_batch: int = 1,
    ) -> tuple[torch.Tensor, list[int] | None]:
        n_frame_dense = ae_latents_dense.shape[2]
        sparse_length = ae_latents_sparse.shape[2]
        _, _, _, l_h_dense, l_w_dense = ae_latents_dense.shape
        _, _, _, l_h_sparse, l_w_sparse = ae_latents_sparse.shape
        dense_tokens = rearrange(
            ae_latents_dense, "bv c t h w -> bv (t h w) c"
        )
        sparse_tokens = rearrange(
            ae_latents_sparse, "bv c t h w -> bv (t h w) c"
        )
        if augment_vision:
            dense_tokens = _augment_vision_latents(
                dense_tokens,
                n_frame=n_frame_dense,
                h=l_h_dense,
                w=l_w_dense,
                noise_std=random.uniform(0.01, 0.1),
            )
            sparse_tokens = _augment_vision_latents(
                sparse_tokens,
                n_frame=sparse_length,
                h=l_h_sparse,
                w=l_w_sparse,
                noise_std=random.uniform(0.01, 0.15),
            )

        dense_rope = self.action_expert.get_vision_rope(
            dense_tokens,
            n_frame_dense,
            l_h_dense,
            l_w_dense,
            is_sparse=False,
            factor=1,
        )
        dense_tokens = _apply_rotary_emb(dense_tokens, dense_rope)
        dense_tokens = rearrange(
            dense_tokens, "(b v) l c -> b v l c", v=n_view
        )

        sparse_offset = torch.zeros(
            (
                sparse_tokens.shape[0],
                n_frame_dense * l_h_sparse * l_w_sparse,
                sparse_tokens.shape[-1],
            ),
            device=sparse_tokens.device,
            dtype=sparse_tokens.dtype,
        )
        sparse_with_offset = torch.cat(
            [sparse_offset, sparse_tokens], dim=1
        )
        sparse_rope = self.action_expert.get_vision_rope(
            sparse_with_offset,
            n_frame_dense + sparse_length,
            l_h_sparse,
            l_w_sparse,
            **self.ae_sparse_rope_args(cfg),
        )
        sparse_tokens = _apply_rotary_emb(
            sparse_with_offset, sparse_rope
        )[:, sparse_offset.shape[1]:]
        sparse_tokens = rearrange(
            sparse_tokens, "(b v) l c -> b v l c", v=n_view
        )
        permutation = None
        if repeat_in_batch > 1:
            dense_tokens = dense_tokens.repeat(repeat_in_batch, 1, 1, 1)
            sparse_tokens = sparse_tokens.repeat(
                repeat_in_batch, 1, 1, 1
            )
            permutation = torch.randperm(dense_tokens.shape[0]).tolist()
            dense_tokens = dense_tokens[permutation]
            sparse_tokens = sparse_tokens[permutation]

        dense_tokens = rearrange(
            dense_tokens,
            "b v (t h w) c -> (b v) t (h w) c",
            t=n_frame_dense,
            h=l_h_dense,
            w=l_w_dense,
        )
        sparse_tokens = rearrange(
            sparse_tokens,
            "b v (t h w) c -> (b v) t (h w) c",
            t=sparse_length,
            h=l_h_sparse,
            w=l_w_sparse,
        )
        dense_current, dense_history = _random_temporal_token_keep(
            dense_tokens,
            keep_ratio=1 / float(cfg.data.mask_ratio),
            mask_cur_obs=False,
        )
        sparse_current, sparse_history = _random_temporal_token_keep(
            sparse_tokens,
            keep_ratio=1 / float(cfg.data.mask_ratio),
            mask_cur_obs=True,
        )
        dense_condition = (
            dense_history
            if cfg.data.drop_cur_obs
            else torch.cat((dense_current, dense_history), dim=1)
        )
        sparse_condition = torch.cat(
            (sparse_current, sparse_history), dim=1
        )
        return (
            rearrange(
                torch.cat((dense_condition, sparse_condition), dim=1),
                "(b v) l c -> b v l c",
                v=n_view,
            ),
            permutation,
        )


    @torch.no_grad()
    def infer_ae_forward(
        self,
        *,
        video,
        video_latents_for_ae,
        n_view: int,
        cfg: Any,
        data_config,
        vae,
        stats_tracker,
        ae_video_scaling_factor: float,
        pipe_ae,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        view_ids: torch.Tensor,
        action_chunk: int,
        action_dim: int,
        num_inference_steps: int,
        body_emb_id: torch.Tensor | None = None,
    ):
        ae_latents_dense, ae_latents_sparse = self._prepare_ae_latents(
            video=video,
            video_latents_for_ae=video_latents_for_ae,
            sparse_length=int(data_config["sparse_length"]),
            n_view=n_view, cfg=cfg,
            vae=vae,
            stats_tracker=stats_tracker,
            ae_video_scaling_factor=ae_video_scaling_factor,
        )
        vision_cond, _ = self._build_ae_vision_condition(
            ae_latents_dense=ae_latents_dense,
            ae_latents_sparse=ae_latents_sparse,
            n_view=n_view, cfg=cfg,
        )
        # Remove the two gripper values from the padded Robotwin state.
        joint_indices = [index for index in range(32) if index not in (6, 13)]
        if body_emb_id is not None and not torch.all(body_emb_id == 11):
            raise ValueError('Only robotwin body embedding ID 11 is supported')
        state_cond = state[..., joint_indices]
        return pipe_ae.infer(
            vision_feature=vision_cond,
            num_inference_steps=num_inference_steps,
            action_chunk=action_chunk,
            state_cond=state_cond,
            action_dim=action_dim,
            action_mask=action_mask,
            view_ids=view_ids,
            body_emb_id=body_emb_id,
        )[0]
