"""Chunk-autoregressive Cosmos3 FastWAM with persistent observed-video memory.

The variant keeps Cosmos3-Nano's published UND/generator parameters, but owns
its AR sequence layout, masks, positions, teacher-forcing forward, and online
KV-cache lifecycle.  Only ``reasoner_input='first_frame_text'`` and the shared
``cogen`` action path are supported.

Observed-video memory and every other clean video condition are encoded without
a timestep embedding. ``E(0)`` is not a clean-condition marker; timestep
embeddings are added only to explicitly noisy future-video frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from galaxea_va.models.inference_rng import inference_generator
from galaxea_va.utils.logging_config import get_logger

from ..fdm import action_video_group_mask
from ..action_padding import zero_padded_action_dims
from ..inference_warmup import warmup_ar_inference
from .cosmos3_fastwam import _MODALITY_MARGIN, _compile_inference
from .cosmos3_fastwam_optional_idm import Cosmos3FastWAMOptionalIDM
from .mrope import build_vae_position_ids

logger = get_logger(__name__)


@dataclass
class Cosmos3ARObservedChunk:
    """One real observation update kept for soft-reset replay.

    ``frames[1,3,T,H,W]`` were observed while executing the action chunk whose UND was
    built from ``boundary_frame[1,3,1,H,W]``; ``und_k``/``und_v``/``und_keep_mask`` are that UND.
    """

    boundary_frame: torch.Tensor
    frames: torch.Tensor
    und_k: list[torch.Tensor]
    und_v: list[torch.Tensor]
    und_keep_mask: torch.Tensor


@dataclass
class Cosmos3ARCacheState:
    """Persistent batch-one inference state for real observed-video memory.

    ``video_k``/``video_v`` contain only real observations.  The pending UND
    cache belongs to the action chunk whose real observations will arrive on
    the next call; generated future video is never stored here.
    """

    video_k: Optional[list[torch.Tensor]] = None
    video_v: Optional[list[torch.Tensor]] = None
    latent_positions: Optional[torch.Tensor] = None
    tokens_per_latent: Optional[int] = None
    patch_h: Optional[int] = None
    patch_w: Optional[int] = None
    latents_per_chunk: Optional[int] = None
    video_temporal_offset: Optional[int] = None
    pending_und_k: Optional[list[torch.Tensor]] = None
    pending_und_v: Optional[list[torch.Tensor]] = None
    pending_und_keep_mask: Optional[torch.Tensor] = None
    observed_chunks: int = 0
    # Whether the streaming decoder includes every real observation in this cache epoch.
    decoder_stream_synced: bool = False
    # Latest real frame [1,3,1,H,W]: the boundary of the next observation update.
    last_frame: Optional[torch.Tensor] = None
    # Latest infer_reset_keep_chunks observation updates, oldest first; empty when the keep is 0.
    observed_history: list[Cosmos3ARObservedChunk] = field(default_factory=list)


class Cosmos3FastWAMAR(Cosmos3FastWAMOptionalIDM):
    """Optional-IDM Cosmos3 variant with chunk-causal training and online memory."""

    @classmethod
    def from_pretrained(
        cls,
        *,
        action_idm_prob: float,
        video_cond_noise_prob: float,
        train_num_chunks: int,
        infer_sliding_window: int,
        action_video_freq_ratio: int = 4,
        chunk_offset_range: int = 0,
        infer_reset_every: int = 32,
        infer_reset_keep_chunks: int = 0,
        max_attn_window_size: Optional[int] = None,
        random_attn_window_size: bool = False,
        compile_training_denoise: bool = False,
        **kwargs,
    ) -> "Cosmos3FastWAMAR":
        """Load Cosmos3-Nano with the fixed AR topology.

        This variant owns ``first_frame_text`` UND, the shared ``cogen`` action
        path, integer latent positions, and the grouped training core.
        These are class behavior rather than configurable modes.

        Args:
            action_idm_prob: Probability that action uses generated future-video conditioning.
            video_cond_noise_prob: Probability of noising the teacher-forced future-video branch.
            train_num_chunks: Maximum action/video chunks in a teacher-forcing sample.
            action_video_freq_ratio: Action steps per input video frame.
            infer_sliding_window: Maximum observed chunks retained by the sliding KV cache.
                Zero retains only the latest boundary latent.
            chunk_offset_range: Inclusive training range for the random generator
                mRoPE chunk offset. Zero keeps every training sequence at its base position.
            infer_reset_every: Number of observation chunks processed before
                rebuilding the Transformer and streaming VAE caches from the latest frame.
            infer_reset_keep_chunks: Observation chunks replayed from raw frames at each
                rebuild (soft reset), so memory spans [keep, reset_every] chunks instead of
                restarting from one frame. Zero keeps the hard reset. Must be smaller than
                ``infer_reset_every`` and ``infer_sliding_window``; checked at each rebuild
                because evaluation may switch the inference context per episode.
            max_attn_window_size: Maximum number of historical observed-video chunks
                visible during training. ``None`` keeps the full training history.
            random_attn_window_size: Whether each sample draws its training attention window
                equally between ``max_attn_window_size`` and half of it (floored, at least 1).
            compile_training_denoise: Compile the AR teacher-forcing denoise core.
            **kwargs: Remaining Cosmos3 pretrained-construction arguments.
        """
        train_num_chunks = int(train_num_chunks)
        infer_sliding_window = int(infer_sliding_window)
        chunk_offset_range = int(chunk_offset_range)
        infer_reset_every = int(infer_reset_every)
        infer_reset_keep_chunks = int(infer_reset_keep_chunks)
        max_attn_window_size = None if max_attn_window_size is None else int(max_attn_window_size)
        random_attn_window_size = bool(random_attn_window_size)
        if train_num_chunks < 1:
            raise ValueError(f"`train_num_chunks` must be >= 1, got {train_num_chunks}.")
        if infer_sliding_window < 0:
            raise ValueError(f"`infer_sliding_window` must be >= 0, got {infer_sliding_window}.")
        if chunk_offset_range < 0:
            raise ValueError(f"`chunk_offset_range` must be >= 0, got {chunk_offset_range}.")
        if infer_reset_every < 1:
            raise ValueError(
                "`infer_reset_every` must be >= 1, "
                f"got {infer_reset_every}."
            )
        if infer_reset_keep_chunks < 0:
            raise ValueError(f"`infer_reset_keep_chunks` must be >= 0, got {infer_reset_keep_chunks}.")
        if max_attn_window_size is not None and not (1 <= max_attn_window_size <= train_num_chunks):
            raise ValueError(
                "`max_attn_window_size` must be within "
                f"[1, train_num_chunks={train_num_chunks}], "
                f"got {max_attn_window_size}."
            )
        if random_attn_window_size and max_attn_window_size is None:
            raise ValueError(
                "`random_attn_window_size=True` requires a non-null `max_attn_window_size`."
            )
        model = super().from_pretrained(
            action_idm_prob=action_idm_prob,
            video_cond_noise_prob=video_cond_noise_prob,
            reasoner_input="first_frame_text",
            action_path="cogen",
            enable_fps_modulation=False,
            compile_training_denoise=compile_training_denoise,
            **kwargs,
        )
        # Action inference encodes observations as a stream and never decodes video.
        required_vae_stream_methods = ("encode_stream", "reset_encode_stream", "reset_decode_stream")
        missing_vae_stream_methods = [
            name for name in required_vae_stream_methods if not hasattr(model.vae.model, name)
        ]
        if missing_vae_stream_methods:
            raise ValueError(
                "Cosmos3FastWAMAR requires VAE streaming encode support; "
                f"missing {missing_vae_stream_methods}."
            )
        if model.video_fdm_prob and model.video_cond_noise_prob:
            raise ValueError("AR FDM and noisy-future IDM cannot be enabled together.")
        model.train_num_chunks = train_num_chunks
        model.action_video_freq_ratio = int(action_video_freq_ratio)
        model.infer_sliding_window = infer_sliding_window
        model.chunk_offset_range = chunk_offset_range
        model.infer_reset_every = infer_reset_every
        model.infer_reset_keep_chunks = infer_reset_keep_chunks
        model.max_attn_window_size = max_attn_window_size
        model.random_attn_window_size = random_attn_window_size
        model.reset_ar_cache()
        return model

    def reset_ar_cache(self) -> None:
        """Clear observed-video memory and restart the causal VAE streams."""
        self._ar_cache = Cosmos3ARCacheState()
        self.vae.model.reset_encode_stream()
        self.vae.model.reset_decode_stream()

    @torch.no_grad()
    def _prefill_ar_und_cache(
        self,
        *,
        und_tokens: torch.Tensor,
        und_pad_mask: torch.Tensor,
        und_position_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run one current-boundary UND group and cache its per-layer K/V.

        Args:
            und_tokens: Embedded image/text condition ``[1,U,D]``.
            und_pad_mask: Padded UND positions ``[1,U]``.
            und_position_ids: UND mRoPE coordinates ``[1,U,3]``.
        """
        backbone = self.backbone
        und = und_tokens
        keep_und_key = ~und_pad_mask
        und_seq_len = und.shape[1]
        causal = torch.arange(und_seq_len, device=self.device)
        und_mask = (causal[None, :] <= causal[:, None])[None] & keep_und_key[:, None, :]
        und_mask = (
            und_mask
            | torch.eye(
                und_seq_len,
                dtype=torch.bool,
                device=self.device,
            )[None]
        )
        cos_und, sin_und = backbone.rotary_emb(
            und_position_ids,
            device=und.device,
            dtype=und.dtype,
        )
        cos_und = cos_und.squeeze(0)
        sin_und = sin_und.squeeze(0)

        cache_k, cache_v = [], []
        for layer in backbone.layers:
            attn = layer.self_attn
            und_norm = layer.input_layernorm(und)
            q_und, k_und, v_und = attn._project(
                und_norm,
                attn.to_q,
                attn.to_k,
                attn.to_v,
                attn.norm_q,
                attn.norm_k,
                cos_und,
                sin_und,
            )
            und = und + attn.to_out(attn._attend(q_und, k_und, v_und, und_mask))
            und = und + layer.mlp(layer.post_attention_layernorm(und))
            cache_k.append(k_und)
            cache_v.append(v_und)
        return cache_k, cache_v

    @torch.no_grad()
    def _build_current_und_cache(
        self,
        *,
        prompt: str,
        current_image: torch.Tensor,
        compile_action_infer: bool,
        reference_image: Optional[torch.Tensor] = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        int,
    ]:
        """Encode current image/text and return UND K/V, key mask, and video offset."""
        und_tokens, und_pad_mask, und_position_ids, next_und_position = self._prepare_ar_und(
            images=current_image, prompts=prompt, reference_image=reference_image,
        )
        if compile_action_infer:
            if not hasattr(self, "_prefill_ar_und_cache_compiled"):
                self._prefill_ar_und_cache_compiled = _compile_inference(self._prefill_ar_und_cache)
            prefill_ar_und_cache = self._prefill_ar_und_cache_compiled
        else:
            prefill_ar_und_cache = self._prefill_ar_und_cache
        cache_k, cache_v = prefill_ar_und_cache(
            und_tokens=und_tokens,
            und_pad_mask=und_pad_mask,
            und_position_ids=und_position_ids,
        )
        if compile_action_infer:
            cache_k = [cache.clone() for cache in cache_k]
            cache_v = [cache.clone() for cache in cache_v]
        return (
            cache_k,
            cache_v,
            ~und_pad_mask,
            int(next_und_position) + _MODALITY_MARGIN,
        )

    def _video_segment_position_ids(
        self,
        *,
        start_latent_position: int,
        latent_t: int,
        patch_h: int,
        patch_w: int,
        video_temporal_offset: int,
    ) -> torch.Tensor:
        """Build absolute mRoPE IDs for one observed or generated video segment."""
        position_ids, _ = build_vae_position_ids(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=(int(video_temporal_offset) + int(start_latent_position)),
            reset_spatial_indices=True,
        )
        return position_ids.to(self.device)

    def _generator_layer_with_cache(
        self,
        *,
        layer,
        x: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one generator-expert layer against fixed cached context.

        Args:
            layer: Current Cosmos3 transformer layer.
            x: Projected generator tokens shaped ``[B,S,D]``.
            rotary_cos/rotary_sin: RoPE tensors for ``x``.
            context_k/context_v: Fixed UND/video K/V prepended before the
                current tokens, shaped ``[B,S_cache,H_kv,D_head]``.
            attention_mask: Visibility mask for queries in
                ``x`` over ``[context | current]`` keys.

        Returns:
            Updated generator tokens plus the current layer's unappended K/V.
        """
        attn = layer.self_attn
        normalized_tokens = layer.input_layernorm_moe_gen(x)
        query, current_k, current_v = attn._project(
            normalized_tokens,
            attn.add_q_proj,
            attn.add_k_proj,
            attn.add_v_proj,
            attn.norm_added_q,
            attn.norm_added_k,
            rotary_cos,
            rotary_sin,
        )
        x = x + attn.to_add_out(
            attn._attend(
                query,
                torch.cat([context_k, current_k], dim=1),
                torch.cat([context_v, current_v], dim=1),
                attention_mask,
            )
        )
        x = x + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(x))
        return x, current_k, current_v

    @torch.no_grad()
    def _extend_video_cache(
        self,
        *,
        video_latents: torch.Tensor,
        und_k: list[torch.Tensor],
        und_v: list[torch.Tensor],
        und_keep_mask: torch.Tensor,
        video_k: Optional[list[torch.Tensor]],
        video_v: Optional[list[torch.Tensor]],
        start_latent_position: int,
        video_temporal_offset: int,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        int,
        int,
        int,
    ]:
        """Return K/V with one clean video segment appended, without mutating AR state.

        New video queries attend the supplied UND group, the supplied video
        cache, and the complete new block. This primitive serves both persistent
        real-observation memory and a temporary IDM future-video extension.
        The appended segment is clean and therefore receives no timestep
        embedding; in particular it is not encoded with ``E(0)``.
        """
        backbone = self.backbone
        if int(start_latent_position) == 0:
            if video_latents.shape[2] != 1:
                raise ValueError(
                    "The initial observed-video cache segment must contain "
                    f"exactly one latent, got {video_latents.shape[2]}."
                )
        (
            video_tokens,
            latent_t,
            patch_h,
            patch_w,
            _tokens_per_latent,
        ) = backbone._clean_video_tokens(video_latents)
        position_ids = self._video_segment_position_ids(
            start_latent_position=start_latent_position,
            latent_t=latent_t,
            patch_h=patch_h,
            patch_w=patch_w,
            video_temporal_offset=video_temporal_offset,
        )
        cos_video, sin_video = backbone.rotary_emb(
            position_ids,
            device=video_tokens.device,
            dtype=video_tokens.dtype,
        )
        cos_video = cos_video.squeeze(0)
        sin_video = sin_video.squeeze(0)
        new_video_k, new_video_v = [], []

        und_seq_len = und_k[0].shape[1]
        cached_video_seq_len = 0 if video_k is None else video_k[0].shape[1]
        video_mask = torch.ones(
            video_tokens.shape[0],
            video_tokens.shape[1],
            und_seq_len + cached_video_seq_len + video_tokens.shape[1],
            dtype=torch.bool,
            device=video_tokens.device,
        )
        video_mask[:, :, :und_seq_len] &= und_keep_mask[:, None, :]

        for layer_index, layer in enumerate(backbone.layers):
            context_k = und_k[layer_index]
            context_v = und_v[layer_index]
            if video_k is not None:
                context_k = torch.cat(
                    [context_k, video_k[layer_index]],
                    dim=1,
                )
                context_v = torch.cat(
                    [context_v, video_v[layer_index]],
                    dim=1,
                )
            video_tokens, current_k, current_v = self._generator_layer_with_cache(
                layer=layer,
                x=video_tokens,
                rotary_cos=cos_video,
                rotary_sin=sin_video,
                context_k=context_k,
                context_v=context_v,
                attention_mask=video_mask,
            )
            new_video_k.append(current_k if video_k is None else torch.cat([video_k[layer_index], current_k], dim=1))
            new_video_v.append(current_v if video_v is None else torch.cat([video_v[layer_index], current_v], dim=1))
        return (
            new_video_k,
            new_video_v,
            latent_t,
            patch_h,
            patch_w,
        )

    def _pop_old_cache_tokens(self, latents_per_chunk: int) -> None:
        """Remove oldest K/V after an append, retaining at most the AR window.

        Args:
            latents_per_chunk: New latent frames produced by one normal
                ``observed_video`` update.
        """
        state = self._ar_cache
        capacity = max(
            1,
            int(self.infer_sliding_window) * int(latents_per_chunk),
        )
        retained_latents = int(state.latent_positions.numel())
        if retained_latents <= capacity:
            return
        remove_latents = retained_latents - capacity
        remove_tokens = remove_latents * int(state.tokens_per_latent)
        # With B=1, contiguous() can retain the old batch stride after slicing.
        # view restores canonical singleton strides without copying the cache.
        for cache in (state.video_k, state.video_v):
            for layer, tensor in enumerate(cache):
                trimmed = tensor[:, remove_tokens:].contiguous()
                cache[layer] = trimmed.view(trimmed.shape)
        state.latent_positions = state.latent_positions[remove_latents:].contiguous()

    @torch.no_grad()
    def _append_clean_latents_to_memory(
        self,
        *,
        latents: torch.Tensor,
        und_k: list[torch.Tensor],
        und_v: list[torch.Tensor],
        und_keep_mask: torch.Tensor,
        start_position: int,
        video_temporal_offset: int,
    ) -> None:
        """Encode clean latents and append their per-layer K/V to AR memory.

        Args:
            latents: Clean observed latents ``[1,C,T,H,W]``.
            und_k/und_v: Per-layer UND K/V for the action chunk that produced
                these observations.
            und_keep_mask: Valid UND keys ``[1,U]``.
            start_position: Absolute latent index of this segment's first frame.
            video_temporal_offset: mRoPE origin immediately after the UND range.
        """
        state = self._ar_cache
        (
            video_k,
            video_v,
            new_latents,
            patch_h,
            patch_w,
        ) = self._extend_video_cache(
            video_latents=latents,
            und_k=und_k,
            und_v=und_v,
            und_keep_mask=und_keep_mask,
            video_k=state.video_k,
            video_v=state.video_v,
            start_latent_position=start_position,
            video_temporal_offset=video_temporal_offset,
        )
        tokens_per_latent = patch_h * patch_w
        if state.tokens_per_latent is not None and (
            state.tokens_per_latent != tokens_per_latent or state.patch_h != patch_h or state.patch_w != patch_w
        ):
            raise ValueError("Observed-video latent grid changed within one AR episode.")
        state.video_k = video_k
        state.video_v = video_v
        state.tokens_per_latent = tokens_per_latent
        state.patch_h = patch_h
        state.patch_w = patch_w
        new_positions = torch.arange(
            start_position,
            start_position + new_latents,
            dtype=torch.long,
            device=self.device,
        )
        state.latent_positions = (
            new_positions if state.latent_positions is None else torch.cat([state.latent_positions, new_positions])
        )

    @torch.no_grad()
    def _update_ar_memory(
        self,
        *,
        observed_video: torch.Tensor,
        current_und_k: list[torch.Tensor],
        current_und_v: list[torch.Tensor],
        current_und_keep_mask: torch.Tensor,
        current_video_temporal_offset: int,
        latents_per_chunk: int,
    ) -> tuple[torch.Tensor, bool, bool]:
        """Stream-encode newly arrived real observations and update AR memory.

        A cold start encodes the latest single frame with the current UND.
        Normal updates encode all new frames with the UND saved by the preceding
        inference call, because those observations were produced while executing
        that preceding action chunk.

        Args:
            observed_video: Cold-start ``[1,3,1,H,W]`` or one complete update
                ``[1,3,T_new,H,W]`` containing no repeated old boundary.
            current_und_k/current_und_v: Current boundary UND cache. The caller
                saves it for observations that will arrive on the next call.
            current_und_keep_mask: Valid keys in the current UND cache.
            current_video_temporal_offset: mRoPE video origin for this UND layout.
            latents_per_chunk: Expected new latent frames per normal update.

        Returns:
            The clean VAE latents appended during this call and whether the
            streaming VAE caches were rebuilt, followed by whether the decoder
            history can consume this update without a gap. RGB decoding belongs
            to :meth:`infer_joint`, not this action path.
        """
        state = self._ar_cache
        cache_initialized = state.latent_positions is not None
        rebuild_caches = not cache_initialized or state.observed_chunks >= self.infer_reset_every
        decoder_history_ready = rebuild_caches or state.decoder_stream_synced
        keep_chunks = int(self.infer_reset_keep_chunks)
        if cache_initialized:
            if state.pending_und_k is None or state.pending_und_v is None or state.pending_und_keep_mask is None:
                raise RuntimeError("Initialized Cosmos3 AR cache is missing pending UND state.")
            if state.latents_per_chunk != int(latents_per_chunk):
                raise ValueError(
                    "AR latents_per_chunk changed within one episode: "
                    f"{state.latents_per_chunk} -> {latents_per_chunk}."
                )
            if state.video_temporal_offset != int(current_video_temporal_offset):
                raise ValueError("AR UND layout changed the video temporal offset within " "one episode.")
            if keep_chunks:
                # This update was observed while executing the pending UND's chunk.
                state.observed_history.append(Cosmos3ARObservedChunk(
                    boundary_frame=state.last_frame, frames=observed_video,
                    und_k=state.pending_und_k, und_v=state.pending_und_v,
                    und_keep_mask=state.pending_und_keep_mask,
                ))
                del state.observed_history[:-keep_chunks]
        # Each segment: (frames, UND K/V and keep mask, whether it starts the causal VAE stream).
        if not rebuild_caches:
            segments = [(observed_video, state.pending_und_k, state.pending_und_v, state.pending_und_keep_mask, False)]
        elif cache_initialized and keep_chunks:
            # Soft reset: replay the latest observed chunks as one fresh training-like crop that
            # starts from a single boundary latent, so memory never drops below keep_chunks.
            if not keep_chunks < min(int(self.infer_reset_every), int(self.infer_sliding_window)):
                raise ValueError(
                    "infer_reset_keep_chunks must be smaller than infer_reset_every and "
                    f"infer_sliding_window, got {keep_chunks}, {self.infer_reset_every}, "
                    f"{self.infer_sliding_window}."
                )
            history = state.observed_history
            logger.info(
                "Soft-resetting Cosmos3 AR caches after %d observation chunks; replaying %d.",
                state.observed_chunks, len(history),
            )
            self.reset_ar_cache()
            state = self._ar_cache
            state.observed_history = history
            oldest = history[0]
            segments = [(oldest.boundary_frame, oldest.und_k, oldest.und_v, oldest.und_keep_mask, True)] + [
                (chunk.frames, chunk.und_k, chunk.und_v, chunk.und_keep_mask, False) for chunk in history
            ]
        else:
            if cache_initialized:
                logger.info(
                    "Resetting Cosmos3 AR caches after %d observation chunks.",
                    state.observed_chunks,
                )
                self.reset_ar_cache()
                state = self._ar_cache
            segments = [(observed_video[:, :, -1:], current_und_k, current_und_v, current_und_keep_mask, True)]
        observed_latents = []
        for frames, und_k, und_v, und_keep_mask, first_chunk in segments:
            latents = self.vae.model.encode_stream(frames, self.vae.scale, first_chunk=first_chunk)
            expected_latents = 1 if first_chunk else latents_per_chunk
            if latents.shape[2] != expected_latents:
                raise ValueError(
                    f"Streaming VAE produced {latents.shape[2]} latents, expected {expected_latents} "
                    f"(first_chunk={first_chunk})."
                )
            self._append_clean_latents_to_memory(
                latents=latents,
                und_k=und_k,
                und_v=und_v,
                und_keep_mask=und_keep_mask,
                start_position=0 if first_chunk else int(state.latent_positions[-1].item()) + 1,
                video_temporal_offset=current_video_temporal_offset,
            )
            observed_latents.append(latents)
        observed_latents = torch.cat(observed_latents, dim=2)
        state.latents_per_chunk = int(latents_per_chunk)
        state.video_temporal_offset = int(current_video_temporal_offset)
        state.last_frame = observed_video[:, :, -1:]
        self._pop_old_cache_tokens(latents_per_chunk)
        # A replayed cache already spans its boundary latent plus the replayed chunks.
        state.observed_chunks = len(segments) if rebuild_caches else state.observed_chunks + 1
        state.decoder_stream_synced = False
        return observed_latents, rebuild_caches, decoder_history_ready

    @torch.no_grad()
    def _denoise_future_video_with_cache(
        self,
        *,
        future_video_latents: torch.Tensor,
        video_timesteps: torch.Tensor,
        und_k: list[torch.Tensor],
        und_v: list[torch.Tensor],
        und_keep_mask: torch.Tensor,
        video_k: list[torch.Tensor],
        video_v: list[torch.Tensor],
        future_position_ids: torch.Tensor,
        action_condition: Optional[torch.Tensor] = None,
        action_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future-video velocity from current UND and real memory."""
        backbone = self.backbone
        (
            future_tokens,
            latent_t,
            patch_h,
            patch_w,
            _tokens_per_latent,
        ) = backbone._video_tokens_with_time(
            future_video_latents,
            video_timesteps,
            torch.ones(
                (future_video_latents.shape[0], future_video_latents.shape[2]),
                dtype=torch.bool,
                device=future_video_latents.device,
            ),
        )
        video_token_count = future_tokens.shape[1]
        if action_condition is not None:
            future_tokens = torch.cat([backbone.fdm_action_encoder(action_condition), future_tokens], dim=1)
            future_position_ids = torch.cat([action_position_ids, future_position_ids], dim=1)
        cos_future, sin_future = backbone.rotary_emb(
            future_position_ids,
            device=future_tokens.device,
            dtype=future_tokens.dtype,
        )
        cos_future = cos_future.squeeze(0)
        sin_future = sin_future.squeeze(0)

        und_seq_len = und_k[0].shape[1]
        cached_video_seq_len = video_k[0].shape[1]
        future_mask = torch.ones(
            future_tokens.shape[0],
            future_tokens.shape[1],
            und_seq_len + cached_video_seq_len + future_tokens.shape[1],
            dtype=torch.bool,
            device=future_tokens.device,
        )
        future_mask[:, :, :und_seq_len] &= und_keep_mask[:, None, :]

        if action_condition is not None:
            action_len = action_condition.shape[1]
            context_len = und_seq_len + cached_video_seq_len
            group_mask = action_video_group_mask(action_len, latent_t, patch_h * patch_w, future_tokens.device)
            # G cannot read observations or noisy video. N reads same-interval G.
            future_mask[:, :action_len, und_seq_len:] = False
            action_groups = torch.arange(action_len, device=future_tokens.device) // (action_len // latent_t)
            future_mask[:, :action_len, context_len:context_len + action_len] = action_groups[:, None] == action_groups[None, :]
            future_mask[:, action_len:, context_len:context_len + action_len] = group_mask

        for layer_index, layer in enumerate(backbone.layers):
            future_tokens, _, _ = self._generator_layer_with_cache(
                layer=layer,
                x=future_tokens,
                rotary_cos=cos_future,
                rotary_sin=sin_future,
                context_k=torch.cat(
                    [und_k[layer_index], video_k[layer_index]],
                    dim=1,
                ),
                context_v=torch.cat(
                    [und_v[layer_index], video_v[layer_index]],
                    dim=1,
                ),
                attention_mask=future_mask,
            )
        future_tokens = backbone.norm_moe_gen(future_tokens[:, -video_token_count:])
        return backbone._unpatchify(
            backbone.proj_out(future_tokens),
            latent_t,
            patch_h,
            patch_w,
        )

    @torch.no_grad()
    def _denoise_ar_action_with_cache(
        self,
        *,
        action_latents: torch.Tensor,
        action_timesteps: torch.Tensor,
        proprio: torch.Tensor,
        und_k: list[torch.Tensor],
        und_v: list[torch.Tensor],
        und_keep_mask: torch.Tensor,
        video_k: list[torch.Tensor],
        video_v: list[torch.Tensor],
        future_video_seq_len: int,
        proprio_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict action velocity from cached video and current ``[P | A]``.

        The supplied video cache is ``[observed video | generated future]`` in
        IDM mode and contains only observed video in first-frame mode. Action
        reads the complete cache, while proprio reads observed video only; this
        matches the AR teacher-forcing mask, where only action receives the
        optional current-future connection.

        Args:
            action_latents: Noisy action tokens shaped ``[B,A,Da]``.
            action_timesteps: Action denoise timestep shaped ``[B]``.
            proprio: Current boundary state shaped ``[B,Dp]``.
            und_k/und_v: Current UND K/V, one tensor per transformer layer.
            und_keep_mask: Valid UND keys shaped ``[B,U]``.
            video_k/video_v: Video K/V ordered as
                ``[observed video | optional generated future]``.
            future_video_seq_len: Fixed generated-future suffix length (zero in
                first-frame mode). Derive the observed prefix from dynamic K/V
                shapes to avoid specializing on a changing Python integer.
            proprio_position_ids: Proprio mRoPE IDs shaped ``[B,1,3]``.
            action_position_ids: Action mRoPE IDs shaped ``[B,A,3]``.

        Returns:
            Predicted action velocity shaped ``[B,A,Da]``.
        """
        observed_video_seq_len = video_k[0].shape[1] - future_video_seq_len
        backbone = self.backbone
        dtype = backbone.proj_in.weight.dtype
        proprio_token = backbone.proprio_encoder(proprio.to(dtype)).unsqueeze(1)
        action_tokens = backbone._action_tokens_cogen(
            action_latents,
            action_timesteps,
        )
        gen = torch.cat([proprio_token, action_tokens], dim=1)
        generator_position_ids = torch.cat(
            [proprio_position_ids, action_position_ids],
            dim=1,
        )
        cos_gen, sin_gen = backbone.rotary_emb(
            generator_position_ids,
            device=gen.device,
            dtype=gen.dtype,
        )
        cos_gen = cos_gen.squeeze(0)
        sin_gen = sin_gen.squeeze(0)
        action_horizon = action_latents.shape[1]

        und_seq_len = und_k[0].shape[1]
        video_seq_len = video_k[0].shape[1]
        action_mask = torch.ones(
            gen.shape[0],
            1 + action_horizon,
            und_seq_len + video_seq_len + 1 + action_horizon,
            dtype=torch.bool,
            device=gen.device,
        )
        action_mask[:, :, :und_seq_len] &= und_keep_mask[:, None, :]
        # Query/key layout is `[P | A] -> [UND | observed video | generated future | P | A]`.
        # P represents the source boundary, so it cannot read the speculative
        # future or action. A intentionally reads both in IDM mode.
        action_mask[:, :1, und_seq_len + observed_video_seq_len : und_seq_len + video_seq_len] = False
        action_mask[:, :1, und_seq_len + video_seq_len + 1 :] = False

        for layer_index, layer in enumerate(backbone.layers):
            gen, _, _ = self._generator_layer_with_cache(
                layer=layer,
                x=gen,
                rotary_cos=cos_gen,
                rotary_sin=sin_gen,
                context_k=torch.cat(
                    [und_k[layer_index], video_k[layer_index]],
                    dim=1,
                ),
                context_v=torch.cat(
                    [und_v[layer_index], video_v[layer_index]],
                    dim=1,
                ),
                attention_mask=action_mask,
            )
        gen = backbone.norm_moe_gen(gen)
        return backbone._decode_action(gen[:, 1:])

    @torch.no_grad()
    def infer_action(
        self,
        *,
        prompt: str,
        observed_video: torch.Tensor,
        num_video_frames: int,
        num_action_denoise_steps: int,
        num_video_denoise_steps: int,
        proprio: Optional[torch.Tensor] = None,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        sigma_shift: Optional[float] = None,
        rand_device: str = "cpu",
        compile_action_infer: bool = False,
        action_infer_mode: str = "first_frame",
        reset_cache: bool = False,
        generate_video: bool = False,
        reference_image: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Generate one action chunk from persistent real-observation memory.

        Args:
            prompt: Current language instruction.
            observed_video: Cold-start image ``[1,3,1,H,W]`` or exactly the
                newly observed frames ``[1,3,num_video_frames-1,H,W]`` since
                the preceding action chunk.
            num_video_frames: Boundary frame plus future/raw observation frames
                represented by one action chunk.
            num_action_denoise_steps: Action flow-matching solver steps.
            num_video_denoise_steps: Future-video solver steps when video is
                requested or required by IDM.
            proprio: Current source-boundary state ``[Dp]`` or ``[1,Dp]``.
            negative_prompt: Unused shared simulator input.
            sigma_shift: Optional scheduler shift override.
            rand_device: Device on which initial float32 noise is sampled.
            compile_action_infer: Compile the UND prefill and repeated video/action
                denoise cores without CUDA Graphs, sharing compiled code across
                cache lengths. Call warmup_inference before serving
                latency-sensitive requests.
            action_infer_mode: ``first_frame`` uses only real memory; ``idm``
                additionally conditions action on generated future video.
            reset_cache: Start a new episode before consuming this input.
            generate_video: Return generated future latent video in
                ``first_frame`` mode; IDM always generates it.

        Returns:
            ``action[action_horizon, action_dim]`` on CPU. When video is
            generated, ``video_latents`` contains the initial boundary only on an
            episode cold start and otherwise contains only the new future latents.
            ``observed_video_latents`` and ``decoder_stream_reset`` carry the
            real-history update needed by :meth:`infer_joint`;
            ``decoder_history_ready`` reports whether no preceding action-only call
            was skipped. This method never runs the VAE decoder.
        """
        self.eval()
        if prompt is None:
            raise ValueError("Cosmos3FastWAMAR requires a text `prompt`.")
        if action_infer_mode not in {"first_frame", "idm"}:
            raise ValueError("`action_infer_mode` must be `first_frame` or `idm`.")
        generate_video = bool(generate_video) or action_infer_mode == "idm"
        action_horizon = self.action_horizon
        num_video_frames = int(num_video_frames)
        if int(num_action_denoise_steps) <= 0:
            raise ValueError("`num_action_denoise_steps` must be positive, got " f"{num_action_denoise_steps}.")
        if generate_video and int(num_video_denoise_steps) <= 0:
            raise ValueError(
                "`num_video_denoise_steps` must be positive when generating " f"video, got {num_video_denoise_steps}."
            )
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if num_video_frames <= 1 or (num_video_frames - 1) % temporal_factor != 0:
            raise ValueError(
                "`num_video_frames-1` must be a positive multiple of the VAE "
                f"temporal factor {temporal_factor}, got {num_video_frames}."
            )
        future_latents = (num_video_frames - 1) // temporal_factor

        if observed_video.ndim == 4:
            observed_video = observed_video.unsqueeze(0)
        if observed_video.ndim != 5 or observed_video.shape[0] != 1 or observed_video.shape[1] != 3:
            raise ValueError(
                "`observed_video` must be [1,3,T,H,W] or [3,T,H,W], " f"got {tuple(observed_video.shape)}."
            )
        observed_frames = int(observed_video.shape[2])
        expected_update_frames = num_video_frames - 1
        if observed_frames not in {1, expected_update_frames}:
            raise ValueError(
                "`observed_video` must contain one cold-start frame or one "
                "complete new chunk: "
                f"expected T=1 or T={expected_update_frames}, got {observed_frames}."
            )
        if reset_cache:
            self.reset_ar_cache()
        cache_initialized = self._ar_cache.latent_positions is not None
        if not cache_initialized and observed_frames != 1:
            raise ValueError(
                "An empty Cosmos3 AR cache must start from exactly one frame; " f"got T={observed_frames}."
            )
        if cache_initialized and observed_frames == 1:
            raise ValueError(
                "A one-frame cold start was received while AR memory exists. "
                "Pass `reset_cache=True` at the episode boundary."
            )

        observed_video = observed_video.to(
            self.device,
            self.torch_dtype,
        )
        current_image = observed_video[:, :, -1]
        (
            current_und_k,
            current_und_v,
            current_und_keep_mask,
            video_temporal_offset,
        ) = self._build_current_und_cache(
            prompt=prompt,
            current_image=current_image,
            compile_action_infer=compile_action_infer,
            reference_image=reference_image,
        )
        memory_update_latents, decoder_stream_reset, decoder_history_ready = self._update_ar_memory(
            observed_video=observed_video,
            current_und_k=current_und_k,
            current_und_v=current_und_v,
            current_und_keep_mask=current_und_keep_mask,
            current_video_temporal_offset=video_temporal_offset,
            latents_per_chunk=future_latents,
        )
        state = self._ar_cache
        state.pending_und_k = current_und_k
        state.pending_und_v = current_und_v
        state.pending_und_keep_mask = current_und_keep_mask

        if proprio is None:
            proprio_condition = torch.zeros(
                1,
                self.proprio_dim,
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            proprio_condition = proprio.reshape(1, -1).to(
                self.device,
                self.torch_dtype,
            )
        if proprio_condition.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` width must be {self.proprio_dim}, " f"got {proprio_condition.shape[1]}.")

        random_generator = inference_generator(self, rand_device)
        action_latents = torch.randn(
            1,
            action_horizon,
            self.action_dim,
            generator=random_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        action_latents = zero_padded_action_dims(action_latents, action_dim_is_pad)
        action_video_k = state.video_k
        action_video_v = state.video_v
        generated_future_latents = None
        boundary_latent = None
        cold_start = observed_frames == 1
        last_observed_position = int(state.latent_positions[-1].item())

        if compile_action_infer:
            for tensor in state.video_k + state.video_v:
                torch._dynamo.mark_dynamic(tensor, 1)
            if not hasattr(self, "_denoise_future_video_with_cache_compiled"):
                self._denoise_future_video_with_cache_compiled = _compile_inference(
                    self._denoise_future_video_with_cache,
                )
            if not hasattr(self, "_denoise_ar_action_with_cache_compiled"):
                self._denoise_ar_action_with_cache_compiled = _compile_inference(
                    self._denoise_ar_action_with_cache,
                )
            denoise_future_video = self._denoise_future_video_with_cache_compiled
            denoise_ar_action = self._denoise_ar_action_with_cache_compiled
        else:
            denoise_future_video = self._denoise_future_video_with_cache
            denoise_ar_action = self._denoise_ar_action_with_cache

        if generate_video:
            latent_height = int(state.patch_h) * self.backbone.cfg.latent_patch_size
            latent_width = int(state.patch_w) * self.backbone.cfg.latent_patch_size
            future_video_latents = torch.randn(
                1,
                self.backbone.cfg.latent_channel,
                future_latents,
                latent_height,
                latent_width,
                generator=random_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(self.device, self.torch_dtype)
            future_position_ids = self._video_segment_position_ids(
                start_latent_position=last_observed_position + 1,
                latent_t=future_latents,
                patch_h=int(state.patch_h),
                patch_w=int(state.patch_w),
                video_temporal_offset=int(state.video_temporal_offset),
            )
            video_timesteps, video_deltas = self.infer_video_scheduler.build_inference_schedule(
                int(num_video_denoise_steps),
                self.device,
                self.torch_dtype,
                shift_override=sigma_shift,
            )
            for video_timestep, video_delta in zip(
                video_timesteps,
                video_deltas,
            ):
                video_velocity = denoise_future_video(
                    future_video_latents=future_video_latents,
                    video_timesteps=video_timestep.reshape(1),
                    und_k=current_und_k,
                    und_v=current_und_v,
                    und_keep_mask=current_und_keep_mask,
                    video_k=state.video_k,
                    video_v=state.video_v,
                    future_position_ids=future_position_ids,
                )
                future_video_latents = self.infer_video_scheduler.step(
                    video_velocity,
                    video_delta,
                    future_video_latents,
                )
            generated_future_latents = future_video_latents
            boundary_latent = memory_update_latents[:, :, -1:]

            if action_infer_mode == "idm":
                (
                    action_video_k,
                    action_video_v,
                    _,
                    _,
                    _,
                ) = self._extend_video_cache(
                    video_latents=future_video_latents,
                    und_k=current_und_k,
                    und_v=current_und_v,
                    und_keep_mask=current_und_keep_mask,
                    video_k=state.video_k,
                    video_v=state.video_v,
                    start_latent_position=last_observed_position + 1,
                    video_temporal_offset=int(state.video_temporal_offset),
                )
                if compile_action_infer:
                    for tensor in action_video_k + action_video_v:
                        torch._dynamo.mark_dynamic(tensor, 1)

        proprio_position_ids = self._video_segment_position_ids(
            start_latent_position=last_observed_position,
            latent_t=1,
            patch_h=int(state.patch_h),
            patch_w=int(state.patch_w),
            video_temporal_offset=int(state.video_temporal_offset),
        )[:, :1]
        action_position_ids, _ = build_vae_position_ids(
            grid_t=action_horizon,
            grid_h=1,
            grid_w=1,
            temporal_offset=(int(state.video_temporal_offset) + last_observed_position),
            reset_spatial_indices=True,
            start_frame_offset=1,
        )
        action_position_ids = action_position_ids.to(self.device)
        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            int(num_action_denoise_steps),
            self.device,
            self.torch_dtype,
            shift_override=sigma_shift,
        )
        for action_timestep, action_delta in zip(
            action_timesteps,
            action_deltas,
        ):
            action_velocity = denoise_ar_action(
                action_latents=action_latents,
                action_timesteps=action_timestep.reshape(1),
                proprio=proprio_condition,
                und_k=current_und_k,
                und_v=current_und_v,
                und_keep_mask=current_und_keep_mask,
                video_k=action_video_k,
                video_v=action_video_v,
                future_video_seq_len=action_video_k[0].shape[1] - state.video_k[0].shape[1],
                proprio_position_ids=proprio_position_ids,
                action_position_ids=action_position_ids,
            )
            action_latents = self.infer_action_scheduler.step(
                action_velocity,
                action_delta,
                action_latents,
            )
            action_latents = zero_padded_action_dims(action_latents, action_dim_is_pad)

        result: dict[str, Any] = {
            "action": action_latents[0].detach().cpu().float(),
        }
        if generated_future_latents is not None:
            result["video_latents"] = (
                torch.cat(
                    [boundary_latent, generated_future_latents],
                    dim=2,
                )
                if cold_start
                else generated_future_latents
            )
            result["observed_video_latents"] = memory_update_latents
            result["decoder_stream_reset"] = decoder_stream_reset
            result["decoder_history_ready"] = decoder_history_ready
        return result

    warmup_inference = warmup_ar_inference
