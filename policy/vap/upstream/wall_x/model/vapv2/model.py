"""Isolated VAPv2 forward: text is updated and carried through every joint block.

Data/loss shape contracts and Video/Action embeddings follow VAP. No changes to
legacy VAP forward are required. Tensor inputs are floating [B,S,D]; positions
[B,S,4]; validity bool [B,S]. Only Text/Memory/I are bidirectional.
"""

import torch
from torch.utils.checkpoint import checkpoint
from wall_x.model.vap.model import VAPTokenModel, VAPTokenOutput
from wall_x.model.vap.layout import (
    AttentionLayout,
    InterleaveLayout,
    ParallelTeacherForcingLayout,
)
from wall_x.model.vap.memory import FirstFrameMemoryTokens
from wall_x.model.vap.rope import action_position_ids as build_action_position_ids
from .block import VAPv2Block
from .layout import JointConditionLayout
from .rope import text_position_ids as build_text_position_ids
from .forward_rope import ForwardRoPECache, rope_forward_cache_enabled


class VAPv2TokenModel(VAPTokenModel):
    # Used by the shared sampler; legacy VAP retains its history-safe fallback.
    supports_memory_prefix_kv_cache = True

    @classmethod
    def from_existing(cls, core):
        if core.first_frame_memory is not None:
            raise ValueError("VAPv2 supports prefix Memory only")
        core.__class__ = cls
        core.blocks = torch.nn.ModuleList(VAPv2Block(block) for block in core.blocks)
        # These legacy text modules must not remain unused trainable parameters.
        core.text_embedding = None
        core.global_time_anchor = None
        return core

    def forward_action_stage_with_prefix_cache(self, *args, **kwargs):
        from .kv_cache import prefill

        return prefill(self, *args, **kwargs)

    def forward_active_action_with_prefix_cache(self, *args, **kwargs):
        from .kv_cache import decode

        return decode(self, *args, **kwargs)

    def forward(
        self,
        video_tokens: torch.Tensor,
        noisy_actions: torch.Tensor,
        *,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        video_position_ids: torch.Tensor,
        condition_video_tokens: int,
        text_context: torch.Tensor | None = None,
        text_valid_mask: torch.Tensor | None = None,
        text_position_ids_override: torch.Tensor | None = None,
        action_position_ids: torch.Tensor | None = None,
        video_camera_ids: torch.Tensor | None = None,
        layout: AttentionLayout | None = None,
        capture_action_prefix_rank: int | None = None,
        action_prefix_embedding_dropout_scale: torch.Tensor | None = None,
        state_condition: torch.Tensor | None = None,
        first_frame_memory: FirstFrameMemoryTokens | None = None,
        memory_prefix_valid: torch.Tensor | None = None,
    ) -> VAPTokenOutput:
        if capture_action_prefix_rank is not None:
            from wall_x.model.vap.layout import ActionStageLayout

            if self.training or torch.is_grad_enabled():
                raise RuntimeError(
                    "prefix cache capture is inference-only and requires no_grad"
                )
            if (
                not isinstance(layout, (ActionStageLayout, InterleaveLayout))
                or capture_action_prefix_rank not in range(1, 2 * self.num_chunks, 2)
                or (isinstance(layout, ActionStageLayout)
                    and capture_action_prefix_rank != layout.active_rank)
            ):
                raise ValueError(
                    "VAPv2 capture requires a causal IVAVA layout and active action rank"
                )
            if self.state_film is not None:
                raise NotImplementedError(
                    "VAPv2 cache does not support State-FiLM"
                )
        if first_frame_memory is not None:
            raise ValueError("VAPv2 supports prefix Memory only, not an Anchor Bank")
        if video_tokens.ndim != 3 or video_tokens.shape[-1] != self.video_dim:
            raise ValueError(
                f"video_tokens must be [B,S,{self.video_dim}], got {video_tokens.shape}"
            )
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_input_dim:
            raise ValueError(
                f"noisy_actions must be [B,S,{self.action_input_dim}], "
                f"got {noisy_actions.shape}"
            )
        batch_size = video_tokens.shape[0]
        if noisy_actions.shape[0] != batch_size:
            raise ValueError("video/action batch sizes must match")
        if video_camera_ids is None:
            raise ValueError(
                "video_camera_ids are required for categorical view identity"
            )
        if tuple(video_camera_ids.shape) != tuple(video_tokens.shape[:2]):
            raise ValueError("video_camera_ids must be [B,Sv]")
        if (
            video_camera_ids.min().item() < 0
            or video_camera_ids.max().item() >= self.max_views
        ):
            raise ValueError(f"video_camera_ids must be in [0,{self.max_views})")
        valid_video_timestep_shapes = {
            (batch_size,),
            (batch_size, video_tokens.shape[1]),
        }
        if tuple(video_timestep.shape) not in valid_video_timestep_shapes:
            raise ValueError("video_timestep must be [B] or per-token [B,Sv]")
        valid_action_timestep_shapes = {
            (batch_size,),
            (batch_size, noisy_actions.shape[1]),
        }
        if tuple(action_timestep.shape) not in valid_action_timestep_shapes:
            raise ValueError("action_timestep must be [B] or per-token [B,Sa]")
        if condition_video_tokens <= 0:
            raise ValueError("condition_video_tokens must be positive")
        if layout is None:
            future_video_tokens = video_tokens.shape[1] - condition_video_tokens
            if future_video_tokens <= 0 or future_video_tokens % self.num_chunks:
                raise ValueError(
                    "future video token length after I must be positive and divisible "
                    "by num_chunks"
                )
            if noisy_actions.shape[1] % self.num_chunks:
                raise ValueError(
                    "action token length must be divisible by configured num_chunks"
                )
            layout = InterleaveLayout(
                num_chunks=self.num_chunks,
                condition_video_tokens=condition_video_tokens,
                video_tokens_per_chunk=future_video_tokens // self.num_chunks,
                action_tokens_per_chunk=noisy_actions.shape[1] // self.num_chunks,
            )
        else:
            if layout.num_chunks != self.num_chunks:
                raise ValueError("attention layout chunk count does not match model")
            if layout.condition_video_tokens != condition_video_tokens:
                raise ValueError(
                    "attention layout condition length does not match input"
                )
            if video_tokens.shape[1] != layout.num_video_tokens:
                raise ValueError("video token length does not match attention layout")
            if noisy_actions.shape[1] != layout.num_action_tokens:
                raise ValueError("action token length does not match attention layout")
        if action_position_ids is None:
            base_action_positions = build_action_position_ids(
                batch_size,
                self.num_chunks,
                layout.action_tokens_per_chunk,
                chunk_stride=layout.action_tokens_per_chunk,
                device=noisy_actions.device,
            )
            action_positions = (
                layout.expand_action(base_action_positions, base_action_positions)
                if isinstance(layout, ParallelTeacherForcingLayout)
                else base_action_positions
            )
        else:
            action_positions = action_position_ids
        # Embed sampled Video times and one clean Text time in ONE invocation,
        # matching the single-use DMuon hook contract for these shared modules.
        video_times = (
            video_timestep[:, None] if video_timestep.ndim == 1 else video_timestep
        )
        combined_times = torch.cat(
            (video_times, torch.zeros_like(video_times[:, :1])), dim=1
        )
        time_embedding = self.video_time.embed(combined_times, dtype=video_tokens.dtype)
        combined_mod = self.video_time.project(time_embedding, dtype=video_tokens.dtype)
        text_mod = combined_mod[:, -1]
        if video_timestep.ndim == 1:
            video_time_embedding, video_mod = time_embedding[:, 0], combined_mod[:, 0]
        else:
            video_time_embedding, video_mod = (
                time_embedding[:, :-1],
                combined_mod[:, :-1],
            )
        action = self.action_identity(self.action_encoder(noisy_actions))
        if action_prefix_embedding_dropout_scale is not None:
            if not self.training:
                raise RuntimeError("action prefix embedding dropout is training-only")
            if not isinstance(layout, ParallelTeacherForcingLayout):
                raise TypeError(
                    "action prefix embedding dropout requires a parallel "
                    "teacher-forcing layout"
                )
            expected_scale_shape = (batch_size, 1, self.action_dim)
            if (
                tuple(action_prefix_embedding_dropout_scale.shape)
                != expected_scale_shape
            ):
                raise ValueError(
                    "action prefix embedding dropout scale must be "
                    f"{expected_scale_shape}, got "
                    f"{tuple(action_prefix_embedding_dropout_scale.shape)}"
                )
            prefix_tokens = int(layout.action_tokens_per_chunk)
            if not layout.robust_a0:
                # Preserve the historical tensor expression exactly when the
                # feature is disabled.
                clean_a0 = action[:, :prefix_tokens] * (
                    action_prefix_embedding_dropout_scale.to(action)
                )
                action = torch.cat((clean_a0, action[:, prefix_tokens:]), dim=1)
            else:
                # Physical Action storage is [A0_clean,A0_robust,A0_noisy,
                # A1_clean,A1_noisy], so only the robust copy is weakened.
                prefix_start = prefix_tokens
                prefix_end = prefix_start + prefix_tokens
                dropped_prefix = action[:, prefix_start:prefix_end] * (
                    action_prefix_embedding_dropout_scale.to(action)
                )
                action = torch.cat(
                    (
                        action[:, :prefix_start],
                        dropped_prefix,
                        action[:, prefix_end:],
                    ),
                    dim=1,
                )
        if self.state_film is None:
            if state_condition is not None:
                raise ValueError(
                    "state_condition was provided but State FiLM is disabled"
                )
        else:
            if state_condition is None:
                raise ValueError(
                    "State FiLM is enabled but state_condition was not provided"
                )
            action = self.state_film(action, state_condition, layout=layout)
        action_time_embedding = self.action_time.embed(
            action_timestep, dtype=action.dtype
        )
        action_mod = self.action_time.project(action_time_embedding, dtype=action.dtype)
        if text_context is None or self.text_input_projector is None:
            raise ValueError(
                "VAPv2 requires Qwen context and a trainable text projector"
            )
        if text_context.shape[0] != batch_size:
            raise ValueError("Text and V/A batch sizes must match")
        valid = text_valid_mask
        if valid is None:
            valid = getattr(self, "_v2_text_valid_mask", None)
        if valid is None:
            raise ValueError("VAPv2 requires explicit Qwen token validity")
        if valid.dtype != torch.bool or valid.shape != text_context.shape[:2]:
            raise ValueError("Text validity must match raw Qwen context [B,L]")
        if valid.device != text_context.device or not bool(valid.any(dim=1).all()):
            raise ValueError(
                "Text validity must share context device and include at least EOS"
            )
        if text_position_ids_override is not None:
            raise ValueError(
                "VAPv2 text positions are derived from actual text validity"
            )
        context = self.text_input_projector(text_context, valid)
        text_positions = build_text_position_ids(valid)
        video = self.video_identity(video_tokens, video_camera_ids)
        if memory_prefix_valid is not None:
            m = memory_prefix_valid.shape[1]
            video = torch.cat(
                (
                    video[:, :m].masked_fill(~memory_prefix_valid[..., None], 0),
                    video[:, m:],
                ),
                dim=1,
            )
        joint_layout = JointConditionLayout(layout, valid, memory_prefix_valid)
        if self.attention_backend == "flex_attention":
            joint_layout.flex_block_mask(video.device)
        rope_cache = None
        if rope_forward_cache_enabled():
            rope_cache = ForwardRoPECache(
                text=text_positions, video=video_position_ids, action=action_positions,
                head_dim=self.head_dim, axes_dim=self.rope_axes_dim,
            )
        checkpoint_layers = self.gradient_checkpoint_num_layers
        if checkpoint_layers is None:
            checkpoint_layers = len(self.blocks)
        layer_caches = []
        for index, block in enumerate(self.blocks):

            def block_forward(v, a, t, module=block):
                return module(
                    v,
                    a,
                    t,
                    video_time_mod=video_mod,
                    action_time_mod=action_mod,
                    text_time_mod=text_mod,
                    video_position_ids=video_position_ids,
                    action_position_ids=action_positions,
                    text_position_ids=text_positions,
                    joint_layout=joint_layout,
                    capture_prefix_rank=capture_action_prefix_rank,
                    rope_cache=rope_cache,
                )

            if (
                self.training
                and self.gradient_checkpointing_enabled
                and index < checkpoint_layers
            ):
                # Text is an explicit input/output; no mutable module field is
                # used for per-layer state during GC backward recomputation.
                video, action, context = checkpoint(
                    block_forward, video, action, context, use_reentrant=False
                )
            else:
                result = block_forward(video, action, context)
                if capture_action_prefix_rank is None:
                    video, action, context = result
                else:
                    video, action, context, cache = result
                    layer_caches.append(cache)
        output = VAPTokenOutput(
            video_hidden=video,
            action_hidden=action,
            video_velocity=self.video_head(video, video_time_embedding),
            action_velocity=self.action_head(action, action_time_embedding),
            layout=layout,
        )
        if capture_action_prefix_rank is None:
            return output
        from .kv_cache import JointActionPrefixCache
        active_chunk = (capture_action_prefix_rank - 1) // 2
        start = active_chunk * layout.action_tokens_per_chunk
        end = start + layout.action_tokens_per_chunk
        visible_end = (joint_layout.text_len + layout.condition_video_tokens
                       + active_chunk * layout.video_tokens_per_chunk + end)
        return output, JointActionPrefixCache(
            tuple(layer_caches),
            action_positions[:, start:end].detach().clone(),
            joint_layout.valid[:, :visible_end].detach().clone(),
            id(self),
            action_positions.detach().clone(),
            start,
            joint_layout,
        )
