from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from typing_extensions import override

from .ahawam_chunk_base import AHAWAMChunkBase


class AHAWAM(AHAWAMChunkBase):
    ACTION_VIDEO_READ_MODES = ("current_only", "history_current")

    def configure_chunk_history(
        self,
        *,
        num_history_frames: int = 0,
        action_video_read_mode: str = "current_only",
        video_rope_frame_stride: int = 1,
        detach_history_kv_during_training: bool = True,
        prepend_episode_first_frame: bool = False,
    ) -> None:
        self.num_history_frames = int(num_history_frames)
        if self.num_history_frames < 0:
            raise ValueError(
                f"`num_history_frames` must be >= 0, got {num_history_frames}"
            )
        self.action_video_read_mode = self._normalize_action_video_read_mode(
            action_video_read_mode
        )
        self.video_rope_frame_stride = int(video_rope_frame_stride)
        if self.video_rope_frame_stride <= 0:
            raise ValueError(
                "`video_rope_frame_stride` must be positive, "
                f"got {video_rope_frame_stride}."
            )
        self.detach_history_kv_during_training = bool(
            detach_history_kv_during_training
        )
        self.prepend_episode_first_frame = bool(prepend_episode_first_frame)

    def _configured_num_history_frames(self) -> int:
        return int(getattr(self, "num_history_frames", 0))

    def _normalize_action_video_read_mode(self, mode: str) -> str:
        normalized = str(mode)
        if normalized not in self.ACTION_VIDEO_READ_MODES:
            raise ValueError(
                "`action_video_read_mode` must be one of "
                f"{self.ACTION_VIDEO_READ_MODES}, got {mode!r}."
            )
        return normalized

    def _configured_action_video_read_mode(self) -> str:
        return self._normalize_action_video_read_mode(
            getattr(self, "action_video_read_mode", "current_only")
        )

    def _configured_video_rope_frame_stride(self) -> int:
        stride = int(getattr(self, "video_rope_frame_stride", 1))
        if stride <= 0:
            raise ValueError(f"`video_rope_frame_stride` must be positive, got {stride}.")
        return stride

    def _configured_detach_history_kv_during_training(self) -> bool:
        return bool(getattr(self, "detach_history_kv_during_training", True))

    def _configured_prepend_episode_first_frame(self) -> bool:
        return bool(getattr(self, "prepend_episode_first_frame", False))

    def _get_video_history_valid_len(
        self,
        sample: dict[str, Any],
        *,
        batch_size: int,
        num_history_frames: int,
    ) -> torch.Tensor:
        if num_history_frames <= 0:
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)
        raw_valid_len = sample.get("video_history_valid_len")
        if raw_valid_len is None:
            raise ValueError(
                "`sample['video_history_valid_len']` is required when "
                f"`num_history_frames` is {num_history_frames}."
            )
        valid_len = torch.as_tensor(raw_valid_len, device=self.device, dtype=torch.long)
        if valid_len.ndim == 0:
            valid_len = valid_len.expand(batch_size)
        if valid_len.ndim != 1 or int(valid_len.shape[0]) != batch_size:
            raise ValueError(
                "`video_history_valid_len` must be scalar or [B], "
                f"got shape {tuple(valid_len.shape)} for batch_size={batch_size}."
            )
        if bool(((valid_len < 0) | (valid_len > num_history_frames)).any().item()):
            raise ValueError(
                "`video_history_valid_len` must be in "
                f"[0, {num_history_frames}], got {valid_len.detach().cpu().tolist()}."
            )
        return valid_len

    def _get_video_current_frame_index(
        self,
        sample: dict[str, Any],
        *,
        batch_size: int,
    ) -> torch.Tensor:
        raw_frame_index = sample.get("video_current_frame_index")
        if raw_frame_index is None:
            raise ValueError(
                "`sample['video_current_frame_index']` is required for "
                "sample-global video RoPE positions."
            )
        frame_index = torch.as_tensor(
            raw_frame_index, device=self.device, dtype=torch.long
        )
        if frame_index.ndim == 0:
            frame_index = frame_index.expand(batch_size)
        if frame_index.ndim != 1 or int(frame_index.shape[0]) != batch_size:
            raise ValueError(
                "`video_current_frame_index` must be scalar or [B], "
                f"got shape {tuple(frame_index.shape)} for batch_size={batch_size}."
            )
        return frame_index

    def _get_video_history_frame_indices(
        self,
        sample: dict[str, Any],
        *,
        batch_size: int,
        num_history_frames: int,
    ) -> torch.Tensor:
        if num_history_frames <= 0:
            return torch.empty(
                (batch_size, 0), dtype=torch.long, device=self.device
            )
        raw_frame_indices = sample.get("video_history_frame_indices")
        if raw_frame_indices is None:
            raise ValueError(
                "`sample['video_history_frame_indices']` is required when "
                f"`num_history_frames` is {num_history_frames}."
            )
        frame_indices = torch.as_tensor(
            raw_frame_indices, device=self.device, dtype=torch.long
        )
        if frame_indices.ndim == 1:
            frame_indices = frame_indices.unsqueeze(0)
        if frame_indices.shape != (batch_size, num_history_frames):
            raise ValueError(
                "`video_history_frame_indices` must be [B,N], "
                f"got shape {tuple(frame_indices.shape)} for "
                f"batch_size={batch_size}, N={num_history_frames}."
            )
        return frame_indices

    def _get_main_video_temporal_position_ids(
        self,
        sample: dict[str, Any],
        *,
        batch_size: int,
        current_clip_latent_frames: int,
    ) -> Optional[torch.Tensor]:
        if self._configured_num_history_frames() <= 0:
            return None
        raw_position_ids = sample.get("video_temporal_position_ids")
        if raw_position_ids is None:
            current_frame_index = self._get_video_current_frame_index(
                sample, batch_size=batch_size
            )
            offsets = torch.arange(
                current_clip_latent_frames, device=self.device, dtype=torch.long
            ).unsqueeze(0)
            return current_frame_index.unsqueeze(1) + offsets
        position_ids = torch.as_tensor(
            raw_position_ids, device=self.device, dtype=torch.long
        )
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape != (batch_size, current_clip_latent_frames):
            raise ValueError(
                "`video_temporal_position_ids` must be [B,F], "
                f"got shape {tuple(position_ids.shape)} for "
                f"batch_size={batch_size}, F={current_clip_latent_frames}."
            )
        return position_ids

    def _encode_training_history_latents(
        self,
        sample: dict[str, Any],
        *,
        batch_size: int,
        num_history_frames: int,
        tiled: bool,
    ) -> torch.Tensor:
        if "video_history" not in sample or sample["video_history"] is None:
            raise ValueError(
                "`sample['video_history']` is required when "
                f"`num_history_frames` is {num_history_frames}."
            )
        history = sample["video_history"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if history.ndim != 5:
            raise ValueError(
                "`video_history` must be [B,C,N,H,W], "
                f"got shape {tuple(history.shape)}."
            )
        if (
            int(history.shape[0]) != batch_size
            or int(history.shape[2]) < num_history_frames
        ):
            raise ValueError(
                "`video_history` shape mismatch: "
                f"got {tuple(history.shape)}, expected batch={batch_size}, "
                f"history>={num_history_frames}."
            )
        history = history[:, :, -num_history_frames:]
        flat_history = history.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_history_frames,
            int(history.shape[1]),
            int(history.shape[3]),
            int(history.shape[4]),
        )
        flat_latents = self._encode_input_image_latents_tensor(
            input_image=flat_history,
            tiled=tiled,
        )
        if flat_latents.ndim != 5 or int(flat_latents.shape[2]) != 1:
            raise ValueError(
                "Encoded history latents must be [B*N,C,1,H,W], "
                f"got {tuple(flat_latents.shape)}."
            )
        latent_channels = int(flat_latents.shape[1])
        latent_height = int(flat_latents.shape[3])
        latent_width = int(flat_latents.shape[4])
        return (
            flat_latents.reshape(
                batch_size,
                num_history_frames,
                latent_channels,
                1,
                latent_height,
                latent_width,
            )
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(
                batch_size,
                latent_channels,
                num_history_frames,
                latent_height,
                latent_width,
            )
        )

    def _build_history_video_frame_mask(
        self,
        *,
        num_history_frames: int,
        valid_history_len: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build causal frame mask for history-only frames [B, N, N]."""
        frame_ids = torch.arange(num_history_frames, device=device)
        causal = frame_ids.unsqueeze(1) >= frame_ids.unsqueeze(0)
        batch_size = int(valid_history_len.shape[0])
        valid_frames = torch.ones(
            (batch_size, num_history_frames), dtype=torch.bool, device=device
        )
        history_positions = torch.arange(num_history_frames, device=device).unsqueeze(0)
        valid_history_len = valid_history_len.to(device=device)
        first_valid_history = (num_history_frames - valid_history_len).unsqueeze(1)
        valid_frames = history_positions >= first_valid_history
        return (
            causal.unsqueeze(0)
            & valid_frames.unsqueeze(1)
            & valid_frames.unsqueeze(2)
        )

    def _expand_frame_mask_to_token_mask(
        self,
        frame_mask: torch.Tensor,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        """Expand [B, Fq, Fk] frame mask to [B, Fq*T, Fk*T] token mask."""
        return frame_mask.repeat_interleave(
            tokens_per_frame, dim=1
        ).repeat_interleave(tokens_per_frame, dim=2)
    @override
    def _should_update_action_history(self) -> bool:
        return False

    @override
    def _prepare_inference_cross_attn_kv_cache(
        self,
        *,
        inference_state: dict[str, Any],
        context: torch.Tensor,
        obs_context: torch.Tensor | None,
    ) -> list[dict[str, torch.Tensor]] | None:
        del inference_state, context, obs_context
        return None

    @override
    def _prepare_inference_chunk_conditioning(
        self,
        *,
        chunk_obs_image: torch.Tensor,
        chunk_proprio: torch.Tensor | None,
        chunk_index: int,
        inference_state: dict[str, Any],
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare single-chunk local obs/proprio conditioning for inference."""
        if self.proprio_encoder is None:
            raise ValueError(
                f"{type(self).__name__} requires `proprio_encoder` for chunk inference conditioning."
            )
        if chunk_proprio is None:
            raise ValueError(
                "`chunk_proprio` is required for chunk inference conditioning."
            )
        chunk_obs_image = chunk_obs_image.to(device=self.device, dtype=self.torch_dtype)
        if chunk_obs_image.ndim == 3:
            chunk_obs_image = chunk_obs_image.unsqueeze(0)
        self.proprio_encoder = self.proprio_encoder.to(
            device=chunk_obs_image.device,
            dtype=self.torch_dtype,
        )
        single_chunk_images = chunk_obs_image.unsqueeze(1)
        obs_context, obs_context_mask = (
            self._build_chunk_aligned_obs_context_from_images(
                chunk_obs_images=single_chunk_images,
                tiled=tiled,
            )
        )
        proprio_for_chunk = chunk_proprio.unsqueeze(1)
        obs_context, obs_context_mask = self._append_proprio_to_obs_context(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            chunk_start_proprio=proprio_for_chunk,
        )
        (
            visual_obs_context,
            visual_obs_mask,
            proprio_context,
            proprio_mask,
        ) = self._split_visual_obs_and_proprio_context(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
        )
        chunk_queries = self._build_chunk_kv_queries(
            obs_context=visual_obs_context,
            obs_context_mask=visual_obs_mask,
        )
        inference_state["_chunk_video_kv_cache"] = (
            self.mot.build_chunk_updated_video_kv_cache(
                video_kv_cache=inference_state["video_kv_cache"],
                chunk_queries=chunk_queries,
                video_tokens_per_frame=int(
                    inference_state["video_tokens_per_frame"]
                ),
                chunk_index=0,
            )
        )
        return proprio_context, proprio_mask, chunk_index

    def _concat_history_kv_entries(
        self,
        entries: list[list[dict[str, torch.Tensor]]],
    ) -> list[dict[str, torch.Tensor]]:
        """Concatenate multiple per-layer KV cache entries into one."""
        num_layers = len(entries[0])
        result: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_parts = [e[layer_idx]["k"] for e in entries]
            v_parts = [e[layer_idx]["v"] for e in entries]
            result.append({
                "k": torch.cat(k_parts, dim=1),
                "v": torch.cat(v_parts, dim=1),
            })
        return result

    @override
    def prefill_video(
        self,
        *,
        prompt: Optional[str] = None,
        input_image: torch.Tensor,
        action_horizon: int,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        video_frame_index: Optional[int] = None,
    ) -> dict[str, Any]:
        self.eval()
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by "
                f"`action_chunk_size` ({self.action_chunk_size})."
            )
        video_mask_mode = str(
            getattr(self.video_expert, "video_attention_mask_mode", "")
        )
        if video_mask_mode not in {"first_frame_causal", "per_frame_causal"}:
            raise ValueError(
                "Two-phase inference requires `video_attention_mask_mode` to be "
                "'first_frame_causal' or 'per_frame_causal'."
            )

        latents_action, batch_size = self._prepare_action_start_latents(
            input_image=input_image,
            action_horizon=action_horizon,
            start_latents=None,
            seed=seed,
            rand_device=rand_device,
        )
        context, context_mask = self._prepare_action_context(
            prompt=prompt,
            batch_size=batch_size,
            proprio=None,
            context=context,
            context_mask=context_mask,
        )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )

        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        if not fuse_flag:
            raise ValueError(
                "AHAWAMChunkBase requires `fuse_vae_embedding_in_latents=True`."
            )

        video_rope_frame_stride = self._configured_video_rope_frame_stride()
        current_frame_index = (
            int(video_frame_index)
            if video_frame_index is not None
            else getattr(self, "_observed_frame_index", 0)
        )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            temporal_position_ids=torch.full(
                (1,),
                int(current_frame_index),
                dtype=torch.long,
                device=first_frame_latents.device,
            ),
            clean_prefix_frames=1,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        num_history = self._configured_num_history_frames()
        prior_entries = (
            getattr(self, "_history_kv_entries", None) or []
        ) if num_history > 0 else []

        if prior_entries:
            prefix_cache = self._concat_history_kv_entries(prior_entries)
            prefix_seq_len = int(prefix_cache[0]["k"].shape[1])
            video_attention_mask = torch.ones(
                (video_seq_len, prefix_seq_len + video_seq_len),
                dtype=torch.bool,
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache_with_prefix(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                prefix_video_kv_cache=prefix_cache,
                prefix_video_seq_len=prefix_seq_len,
                video_attention_mask=video_attention_mask,
            )
        else:
            video_kv_cache = self._prefill_action_video_cache(
                video_pre=video_pre,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
            )

        self._observed_frame_index = (
            current_frame_index + action_horizon // video_rope_frame_stride
        )

        if num_history > 0:
            prior_entries.append(video_kv_cache)
            if self._configured_prepend_episode_first_frame() and len(prior_entries) > 1:
                pinned = prior_entries[0]
                max_sliding = num_history - 1
                sliding = prior_entries[1:]
                if len(sliding) > max_sliding:
                    sliding = sliding[-max_sliding:]
                prior_entries = [pinned] + sliding
            elif len(prior_entries) > num_history:
                prior_entries = prior_entries[-num_history:]
            self._history_kv_entries = prior_entries

        state = {
            "start_latents": latents_action,
            "context": context,
            "context_mask": context_mask,
            "batch_size": batch_size,
            "video_pre": video_pre,
            "video_seq_len": video_seq_len,
            "video_tokens_per_frame": video_tokens_per_frame,
            "video_kv_cache": video_kv_cache,
            "action_history_kv_cache": None,
            "action_history_seq_len": 0,
        }
        return state

    def reset_history(self) -> None:
        """Clear accumulated history KV cache entries and frame counter."""
        self._history_kv_entries = []
        self._observed_frame_index = 0

    @override
    def _build_training_obs_context(
        self,
        sample: dict[str, Any],
        action_horizon: int,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._build_chunk_aligned_obs_context_from_video(
            video=sample["video"],
            action_horizon=action_horizon,
            tiled=tiled,
        )

    def _build_block_diagonal_action_attention_mask(
        self,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError(f"`seq_len` must be positive, got {seq_len}")
        if seq_len % self.action_chunk_size != 0:
            raise ValueError(
                f"`seq_len` ({seq_len}) must be divisible by `action_chunk_size` ({self.action_chunk_size})."
            )
        chunk_ids = torch.arange(
            seq_len // self.action_chunk_size, device=device
        ).repeat_interleave(self.action_chunk_size)
        return chunk_ids.unsqueeze(1) == chunk_ids.unsqueeze(0)

    @override
    def _validate_runtime_action_horizon(self, action_horizon: int) -> None:
        max_offset = int(getattr(self, "max_action_offset", 0))
        if max_offset > 0:
            if int(action_horizon) < int(self.action_horizon):
                raise ValueError(
                    f"With max_action_offset={max_offset}, runtime action_horizon must be "
                    f">= configured causal horizon {self.action_horizon}, got {action_horizon}."
                )
            return
        super()._validate_runtime_action_horizon(action_horizon)

    def _has_action_offset(self, sample: dict[str, Any]) -> bool:
        return "action_offset" in sample

    def _normalize_action_offsets(
        self,
        sample: dict[str, Any],
        *,
        batch_size: int,
    ) -> torch.Tensor:
        raw_offset = sample.get("action_offset", 0)
        offsets = torch.as_tensor(raw_offset, device=self.device, dtype=torch.long)
        if offsets.ndim == 0:
            offsets = offsets.expand(batch_size)
        if offsets.ndim != 1 or int(offsets.shape[0]) != batch_size:
            raise ValueError(
                "`action_offset` must be scalar or [B], "
                f"got shape {tuple(offsets.shape)} for batch_size={batch_size}."
            )
        max_action_offset = int(getattr(self, "max_action_offset", 0))
        if bool((offsets < 0).any().item()):
            raise ValueError(
                f"`action_offset` must be nonnegative, got {offsets.detach().cpu().tolist()}."
            )
        if max_action_offset > 0 and bool((offsets > max_action_offset).any().item()):
            raise ValueError(
                "`action_offset` exceeds configured max_action_offset: "
                f"offsets={offsets.detach().cpu().tolist()} max={max_action_offset}."
            )
        return offsets

    def _slice_offset_sequence(
        self,
        sequence: torch.Tensor,
        *,
        offsets: torch.Tensor,
        action_horizon: int,
    ) -> torch.Tensor:
        if sequence.ndim < 2:
            raise ValueError(
                f"Offset sequence must be at least 2D [B,T,...], got {tuple(sequence.shape)}."
            )
        batch_size = int(sequence.shape[0])
        if offsets.shape != (batch_size,):
            raise ValueError(
                f"`offsets` must have shape [{batch_size}], got {tuple(offsets.shape)}."
            )
        positions = offsets.unsqueeze(1) + torch.arange(
            int(action_horizon), device=sequence.device, dtype=torch.long
        ).unsqueeze(0)
        if int(positions.max().item()) >= int(sequence.shape[1]):
            raise ValueError(
                "Offset slice exceeds sequence length: "
                f"max_index={int(positions.max().item())}, steps={int(sequence.shape[1])}."
            )
        gather_index = positions.reshape(batch_size, int(action_horizon), *([1] * (sequence.ndim - 2)))
        gather_index = gather_index.expand(-1, -1, *sequence.shape[2:])
        return torch.gather(sequence, dim=1, index=gather_index)

    def _extract_offset_chunk_start_proprio(
        self,
        proprio: torch.Tensor,
        *,
        offsets: torch.Tensor,
        action_horizon: int,
    ) -> torch.Tensor:
        if proprio.ndim != 3:
            raise ValueError(
                f"`proprio` must be 3D [B,T,D], got shape {tuple(proprio.shape)}."
            )
        if self.proprio_dim is None or int(proprio.shape[2]) != int(self.proprio_dim):
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}."
            )
        if int(action_horizon) % int(self.action_chunk_size) != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by action_chunk_size ({self.action_chunk_size})."
            )
        num_chunks = int(action_horizon) // int(self.action_chunk_size)
        chunk_starts = (
            torch.arange(num_chunks, device=proprio.device, dtype=torch.long)
            * int(self.action_chunk_size)
        )
        positions = offsets.to(device=proprio.device).unsqueeze(1) + chunk_starts.unsqueeze(0)
        if int(positions.max().item()) >= int(proprio.shape[1]):
            raise ValueError(
                "Shifted chunk-start proprio index exceeds available sequence length: "
                f"max_index={int(positions.max().item())}, proprio_steps={int(proprio.shape[1])}."
            )
        gather_index = positions.unsqueeze(-1).expand(-1, -1, int(proprio.shape[2]))
        return torch.gather(proprio, dim=1, index=gather_index)

    def _build_offset_obs_context(
        self,
        sample: dict[str, Any],
        *,
        offsets: torch.Tensor,
        action_horizon: int,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_obs_images = sample.get("chunk_obs_images")
        if chunk_obs_images is None:
            raise ValueError(
                "`sample['chunk_obs_images']` is required when action-offset mode is enabled."
            )
        chunk_obs_images = chunk_obs_images.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        obs_context, obs_context_mask = self._build_chunk_aligned_obs_context_from_images(
            chunk_obs_images=chunk_obs_images,
            tiled=tiled,
        )
        if self.proprio_encoder is not None:
            if "proprio" not in sample or sample["proprio"] is None:
                raise ValueError(
                    "`sample['proprio']` is required when `proprio_dim` is enabled."
                )
            chunk_start_proprio = self._extract_offset_chunk_start_proprio(
                sample["proprio"].to(
                    device=self.device,
                    dtype=self.torch_dtype,
                    non_blocking=True,
                ),
                offsets=offsets,
                action_horizon=action_horizon,
            )
            obs_context, obs_context_mask = self._append_proprio_to_obs_context(
                obs_context=obs_context,
                obs_context_mask=obs_context_mask,
                chunk_start_proprio=chunk_start_proprio,
            )
        return obs_context, obs_context_mask

    @override
    def training_loss(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> tuple[torch.Tensor, dict[str, float]]:
        inputs = self.build_inputs(sample, tiled=tiled)
        if self._has_action_offset(sample):
            return self._training_loss_action_offset(sample, inputs=inputs, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        obs_context = inputs["obs_context"]
        obs_context_mask = inputs["obs_context_mask"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        (
            visual_obs_context,
            visual_obs_mask,
            proprio_context,
            proprio_mask,
        ) = self._split_visual_obs_and_proprio_context(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
        )
        chunk_queries = self._build_chunk_kv_queries(
            obs_context=visual_obs_context,
            obs_context_mask=visual_obs_mask,
        )

        # --- video noise ---
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        latents[:, :, 0:1] = inputs["first_frame_latents"]

        # --- action noise (no teacher forcing) ---
        noise_action = torch.randn_like(action)
        timestep_action = self._sample_action_training_timestep(
            batch_size=batch_size,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        # --- video pre_dit ---
        num_history_frames = self._configured_num_history_frames()
        temporal_position_ids = self._get_main_video_temporal_position_ids(
            sample,
            batch_size=batch_size,
            current_clip_latent_frames=int(input_latents.shape[2]),
        ) if num_history_frames > 0 else None
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            temporal_position_ids=temporal_position_ids,
        )
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        # --- history encoding (if configured) ---
        history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None
        valid_history_len: Optional[torch.Tensor] = None
        effective_history_frames = 0
        _history_len_metric = 0.0
        if num_history_frames > 0:
            current_valid_history_len = self._get_video_history_valid_len(
                sample, batch_size=batch_size, num_history_frames=num_history_frames
            )
            valid_history_len = current_valid_history_len
            _history_len_metric = float(current_valid_history_len.float().mean().item())
            effective_history_frames = int(current_valid_history_len.max().item())
            if effective_history_frames > 0:
                history_frame_indices = self._get_video_history_frame_indices(
                    sample, batch_size=batch_size, num_history_frames=num_history_frames
                )
                history_frame_indices = history_frame_indices[:, -effective_history_frames:]
                with torch.no_grad():
                    history_latents = self._encode_training_history_latents(
                        sample,
                        batch_size=batch_size,
                        num_history_frames=effective_history_frames,
                        tiled=tiled,
                    )
                    history_temporal_ids = history_frame_indices
                    history_pre = self.video_expert.pre_dit(
                        x=history_latents,
                        timestep=torch.zeros_like(timestep_video),
                        context=context,
                        context_mask=context_mask,
                        action=None,
                        temporal_position_ids=history_temporal_ids,
                        fuse_vae_embedding_in_latents=True,
                        clean_prefix_frames=effective_history_frames,
                    )
                    history_frame_mask = self._build_history_video_frame_mask(
                        num_history_frames=effective_history_frames,
                        valid_history_len=valid_history_len,
                        device=history_pre["tokens"].device,
                    )
                    history_attn_mask = self._expand_frame_mask_to_token_mask(
                        frame_mask=history_frame_mask,
                        tokens_per_frame=video_tokens_per_frame,
                    )
                    history_video_kv_cache = self.mot.prefill_video_cache(
                        video_tokens=history_pre["tokens"],
                        video_freqs=history_pre["freqs"],
                        video_t_mod=history_pre["t_mod"],
                        video_context_payload={
                            "context": history_pre["context"],
                            "mask": history_pre["context_mask"],
                        },
                        video_attention_mask=history_attn_mask,
                    )
                history_video_kv_cache = [
                    {"k": layer["k"].detach(), "v": layer["v"].detach()}
                    for layer in history_video_kv_cache
                ]

        # --- action pre_dit (local-only, no teacher forcing; image obs excluded) ---
        obs_proprio_tokens_per_chunk = self._require_obs_proprio_tokens_per_chunk()
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            obs_context=proprio_context,
            obs_context_mask=proprio_mask,
            clean_action_tokens=None,
            clean_timestep=None,
            chunk_size=self.action_chunk_size,
            noisy_position_offset=0,
            single_branch_chunk_causal=True,
            obs_chunk_offset=0,
            obs_context_causal=True,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
        )

        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        if history_video_kv_cache is not None:
            if valid_history_len is None:
                raise ValueError("`valid_history_len` is required with history KV cache.")
            history_seq_len = int(history_video_kv_cache[0]["k"].shape[1])
            video_seq_len_q = int(video_pre["tokens"].shape[1])
            if int(valid_history_len.min().item()) < effective_history_frames:
                positions = torch.arange(
                    effective_history_frames, device=valid_history_len.device
                )
                first_valid = (effective_history_frames - valid_history_len).clamp(min=0).unsqueeze(1)
                valid_frame_mask = positions.unsqueeze(0) >= first_valid  # [B, F]
                valid_token_mask = valid_frame_mask.repeat_interleave(
                    video_tokens_per_frame, dim=1
                )  # [B, history_seq_len]
                history_prefix_mask = valid_token_mask.unsqueeze(1).expand(
                    -1, video_seq_len_q, -1
                )
            else:
                history_prefix_mask = torch.ones(
                    (batch_size, video_seq_len_q, history_seq_len),
                    dtype=torch.bool,
                    device=video_pre["tokens"].device,
                )
            video_attention_mask = video_attention_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            video_attention_mask = torch.cat(
                [history_prefix_mask, video_attention_mask], dim=2
            )

        tokens_out = self.mot.forward_prior_action_with_chunk_updated_kv(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            chunk_queries=chunk_queries,
            video_tokens_per_frame=video_tokens_per_frame,
            action_chunk_size=self.action_chunk_size,
            history_video_kv_cache=history_video_kv_cache,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action_prior = self.action_expert.post_dit(
            tokens_out["action_prior"], action_pre
        )
        pred_video = pred_video[:, :, 1:]
        target_video_no_first = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video_no_first,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        loss_action_prior = self._compute_weighted_action_loss(
            pred_action=pred_action_prior,
            target_action=target_action,
            timestep_action=timestep_action,
            action_is_pad=action_is_pad,
        )

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action_prior * loss_action_prior
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action_prior
            * float(loss_action_prior.detach().item()),
            "history_len": _history_len_metric,
        }

    def _training_loss_action_offset(
        self,
        sample: dict[str, Any],
        *,
        inputs: dict[str, Any],
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        offsets = self._normalize_action_offsets(sample, batch_size=batch_size)
        action_horizon = int(getattr(self, "action_horizon", 0))
        if action_horizon <= 0:
            action_horizon = int(inputs["action"].shape[1]) - int(offsets.max().item())
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by `action_chunk_size` ({self.action_chunk_size})."
            )

        action = self._slice_offset_sequence(
            inputs["action"], offsets=offsets, action_horizon=action_horizon
        )
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is not None:
            action_is_pad = self._slice_offset_sequence(
                action_is_pad, offsets=offsets, action_horizon=action_horizon
            )
        image_is_pad = inputs["image_is_pad"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        obs_context, obs_context_mask = self._build_offset_obs_context(
            sample,
            offsets=offsets,
            action_horizon=action_horizon,
            tiled=tiled,
        )
        (
            visual_obs_context,
            visual_obs_mask,
            proprio_context,
            proprio_mask,
        ) = self._split_visual_obs_and_proprio_context(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
        )
        chunk_queries = self._build_chunk_kv_queries(
            obs_context=visual_obs_context,
            obs_context_mask=visual_obs_mask,
        )

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        if str(self.action_train_timestep_mode) == "per_chunk":
            num_chunks = action_horizon // self.action_chunk_size
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size * num_chunks,
                device=self.device,
                dtype=action.dtype,
            ).view(batch_size, num_chunks)
            timestep_action = timestep_action.repeat_interleave(
                self.action_chunk_size, dim=1
            )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        num_history_frames = self._configured_num_history_frames()
        temporal_position_ids = self._get_main_video_temporal_position_ids(
            sample,
            batch_size=batch_size,
            current_clip_latent_frames=int(input_latents.shape[2]),
        ) if num_history_frames > 0 else None
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            temporal_position_ids=temporal_position_ids,
        )
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None
        valid_history_len: Optional[torch.Tensor] = None
        effective_history_frames = 0
        _history_len_metric = 0.0
        if num_history_frames > 0:
            current_valid_history_len = self._get_video_history_valid_len(
                sample, batch_size=batch_size, num_history_frames=num_history_frames
            )
            valid_history_len = current_valid_history_len
            _history_len_metric = float(current_valid_history_len.float().mean().item())
            effective_history_frames = int(current_valid_history_len.max().item())
            if effective_history_frames > 0:
                history_frame_indices = self._get_video_history_frame_indices(
                    sample, batch_size=batch_size, num_history_frames=num_history_frames
                )
                history_frame_indices = history_frame_indices[:, -effective_history_frames:]
                with torch.no_grad():
                    history_latents = self._encode_training_history_latents(
                        sample,
                        batch_size=batch_size,
                        num_history_frames=effective_history_frames,
                        tiled=tiled,
                    )
                    history_pre = self.video_expert.pre_dit(
                        x=history_latents,
                        timestep=torch.zeros_like(timestep_video),
                        context=context,
                        context_mask=context_mask,
                        action=None,
                        temporal_position_ids=history_frame_indices,
                        fuse_vae_embedding_in_latents=True,
                        clean_prefix_frames=effective_history_frames,
                    )
                    history_frame_mask = self._build_history_video_frame_mask(
                        num_history_frames=effective_history_frames,
                        valid_history_len=valid_history_len,
                        device=history_pre["tokens"].device,
                    )
                    history_attn_mask = self._expand_frame_mask_to_token_mask(
                        frame_mask=history_frame_mask,
                        tokens_per_frame=video_tokens_per_frame,
                    )
                    history_video_kv_cache = self.mot.prefill_video_cache(
                        video_tokens=history_pre["tokens"],
                        video_freqs=history_pre["freqs"],
                        video_t_mod=history_pre["t_mod"],
                        video_context_payload={
                            "context": history_pre["context"],
                            "mask": history_pre["context_mask"],
                        },
                        video_attention_mask=history_attn_mask,
                    )
                history_video_kv_cache = [
                    {"k": layer["k"].detach(), "v": layer["v"].detach()}
                    for layer in history_video_kv_cache
                ]

        obs_proprio_tokens_per_chunk = self._require_obs_proprio_tokens_per_chunk()
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            obs_context=proprio_context,
            obs_context_mask=proprio_mask,
            clean_action_tokens=None,
            clean_timestep=None,
            chunk_size=self.action_chunk_size,
            noisy_position_offset=0,
            single_branch_chunk_causal=True,
            obs_chunk_offset=0,
            obs_context_causal=True,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
        )

        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        if history_video_kv_cache is not None:
            if valid_history_len is None:
                raise ValueError("`valid_history_len` is required with history KV cache.")
            history_seq_len = int(history_video_kv_cache[0]["k"].shape[1])
            video_seq_len_q = int(video_pre["tokens"].shape[1])
            if int(valid_history_len.min().item()) < effective_history_frames:
                positions = torch.arange(
                    effective_history_frames, device=valid_history_len.device
                )
                first_valid = (effective_history_frames - valid_history_len).clamp(min=0).unsqueeze(1)
                valid_frame_mask = positions.unsqueeze(0) >= first_valid
                valid_token_mask = valid_frame_mask.repeat_interleave(
                    video_tokens_per_frame, dim=1
                )
                history_prefix_mask = valid_token_mask.unsqueeze(1).expand(
                    -1, video_seq_len_q, -1
                )
            else:
                history_prefix_mask = torch.ones(
                    (batch_size, video_seq_len_q, history_seq_len),
                    dtype=torch.bool,
                    device=video_pre["tokens"].device,
                )
            video_attention_mask = video_attention_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            video_attention_mask = torch.cat(
                [history_prefix_mask, video_attention_mask], dim=2
            )

        tokens_out = self.mot.forward_prior_action_with_chunk_updated_kv(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            chunk_queries=chunk_queries,
            video_tokens_per_frame=video_tokens_per_frame,
            action_chunk_size=self.action_chunk_size,
            history_video_kv_cache=history_video_kv_cache,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action_prior = self.action_expert.post_dit(
            tokens_out["action_prior"], action_pre
        )
        pred_video = pred_video[:, :, 1:]
        target_video_no_first = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video_no_first,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        loss_action_prior = self._compute_weighted_action_loss(
            pred_action=pred_action_prior,
            target_action=target_action,
            timestep_action=timestep_action,
            action_is_pad=action_is_pad,
        )

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action_prior * loss_action_prior
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action_prior
            * float(loss_action_prior.detach().item()),
            "history_len": _history_len_metric,
            "mean_offset": float(offsets.float().mean().item()),
        }

    def _compute_weighted_action_loss(
        self,
        *,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_token.device,
            dtype=action_loss_token.dtype,
        )
        if action_weight.ndim == 1:
            if action_is_pad is not None:
                valid = (~action_is_pad).to(
                    device=action_loss_token.device,
                    dtype=action_loss_token.dtype,
                )
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(
                    dim=1
                ) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)
            return (action_loss_per_sample * action_weight).mean()
        if action_weight.shape != action_loss_token.shape:
            raise ValueError(
                "`action_weight` shape mismatch: "
                f"got {tuple(action_weight.shape)} vs expected {tuple(action_loss_token.shape)}"
            )
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device,
                dtype=action_loss_token.dtype,
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * action_weight * valid).sum(dim=1).div(
                valid_sum
            ).mean()
        return (action_loss_token * action_weight).mean()

    @override
    def _build_prefilled_action_attention_mask(
        self,
        *,
        current_action_seq_len: int,
        action_history_seq_len: int,
        chunk_start: int,
        device: torch.device,
    ) -> torch.Tensor:
        del action_history_seq_len, chunk_start
        return self.action_expert.build_single_branch_chunk_causal_mask(
            seq_len=current_action_seq_len,
            chunk_size=self.action_chunk_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Evaluation with history
    # ------------------------------------------------------------------

    def _mean_history_len_metric(self, sample: dict[str, Any]) -> float:
        raw_valid_len = sample.get("video_history_valid_len")
        if raw_valid_len is None:
            return 0.0
        valid_len = torch.as_tensor(raw_valid_len, dtype=torch.float32)
        if valid_len.numel() == 0:
            return 0.0
        return float(valid_len.float().mean().item())

    def _normalize_eval_video_history(
        self,
        *,
        video_history: Optional[torch.Tensor],
        video_history_valid_len: Optional[torch.Tensor | int],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if self._configured_num_history_frames() <= 0:
            return None
        if video_history is None:
            return None
        history = video_history.to(device=self.device, dtype=self.torch_dtype)
        if history.ndim == 4:
            history = history.unsqueeze(0)
        if history.ndim != 5:
            raise ValueError(
                "`video_history` must be [C,N,H,W] or [B,C,N,H,W], "
                f"got shape {tuple(history.shape)}."
            )
        if int(history.shape[0]) != batch_size:
            raise ValueError(
                f"`video_history` batch mismatch: {history.shape[0]} vs {batch_size}."
            )
        if batch_size != 1:
            raise ValueError(
                "AHAWAM evaluation history prefill currently expects batch_size=1."
            )
        if video_history_valid_len is None:
            raise ValueError(
                "`video_history_valid_len` is required when `video_history` is passed."
            )
        valid_len = torch.as_tensor(video_history_valid_len, dtype=torch.long)
        if valid_len.ndim == 0:
            valid_len_value = int(valid_len.item())
        elif valid_len.ndim == 1 and int(valid_len.shape[0]) == batch_size:
            valid_len_value = int(valid_len[0].item())
        else:
            raise ValueError(
                "`video_history_valid_len` must be scalar or [B], "
                f"got shape {tuple(valid_len.shape)}."
            )
        if valid_len_value < 0 or valid_len_value > int(history.shape[2]):
            raise ValueError(
                "`video_history_valid_len` out of range: "
                f"{valid_len_value} for history shape {tuple(history.shape)}."
            )
        if valid_len_value == 0:
            return None
        return history[:, :, -valid_len_value:]

    def _normalize_eval_history_frame_indices(
        self,
        *,
        video_history_frame_indices: Optional[torch.Tensor],
        video_history_valid_len: Optional[torch.Tensor | int],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if self._configured_num_history_frames() <= 0:
            return None
        if video_history_frame_indices is None:
            return None
        frame_indices = torch.as_tensor(video_history_frame_indices, dtype=torch.long)
        if frame_indices.ndim == 1:
            frame_indices = frame_indices.unsqueeze(0)
        if frame_indices.ndim != 2 or int(frame_indices.shape[0]) != batch_size:
            raise ValueError(
                "`video_history_frame_indices` must be [N] or [B,N], "
                f"got shape {tuple(frame_indices.shape)}."
            )
        if batch_size != 1:
            raise ValueError(
                "AHAWAM evaluation history prefill currently expects batch_size=1."
            )
        if video_history_valid_len is None:
            raise ValueError(
                "`video_history_valid_len` is required with `video_history_frame_indices`."
            )
        valid_len = torch.as_tensor(video_history_valid_len, dtype=torch.long)
        if valid_len.ndim == 0:
            valid_len_value = int(valid_len.item())
        elif valid_len.ndim == 1 and int(valid_len.shape[0]) == batch_size:
            valid_len_value = int(valid_len[0].item())
        else:
            raise ValueError(
                "`video_history_valid_len` must be scalar or [B], "
                f"got shape {tuple(valid_len.shape)}."
            )
        if valid_len_value == 0:
            return None
        if valid_len_value < 0 or valid_len_value > int(frame_indices.shape[1]):
            raise ValueError(
                "`video_history_valid_len` out of range for "
                f"`video_history_frame_indices`: {valid_len_value}."
            )
        return frame_indices[:, -valid_len_value:]

    def _normalize_eval_current_frame_index(
        self,
        *,
        video_current_frame_index: Optional[torch.Tensor | int],
        batch_size: int,
    ) -> Optional[int]:
        if video_current_frame_index is None:
            return None
        frame_index = torch.as_tensor(video_current_frame_index, dtype=torch.long)
        if frame_index.ndim == 0:
            return int(frame_index.item())
        if frame_index.ndim == 1 and int(frame_index.shape[0]) == batch_size:
            if batch_size != 1:
                raise ValueError(
                    "AHAWAM evaluation currently expects batch_size=1 "
                    "for explicit video frame indices."
                )
            return int(frame_index[0].item())
        raise ValueError(
            "`video_current_frame_index` must be scalar or [B], "
            f"got shape {tuple(frame_index.shape)}."
        )

    @torch.no_grad()
    def _evaluate_action_metrics(
        self,
        *,
        sample: dict[str, Any],
        pred_action: Optional[torch.Tensor],
    ) -> dict[str, float]:
        if not self._has_action_offset(sample):
            return super()._evaluate_action_metrics(sample=sample, pred_action=pred_action)
        action = sample.get("action")
        if action is None or pred_action is None:
            return {}
        offsets = self._normalize_action_offsets(sample, batch_size=int(action.shape[0]))
        action_horizon = int(getattr(self, "action_horizon", 0))
        if action_horizon <= 0:
            action_horizon = int(pred_action.shape[-2])
        eval_sample = dict(sample)
        eval_sample["action"] = self._slice_offset_sequence(
            action.to(device=self.device),
            offsets=offsets,
            action_horizon=action_horizon,
        ).detach().cpu()
        proprio = sample.get("proprio")
        if proprio is not None:
            eval_sample["proprio"] = self._slice_offset_sequence(
                proprio.to(device=self.device),
                offsets=offsets,
                action_horizon=action_horizon,
            ).detach().cpu()
        return super()._evaluate_action_metrics(
            sample=eval_sample, pred_action=pred_action
        )

    @override
    def evaluate_validation(
        self,
        sample: dict[str, Any],
        *,
        eval_num_inference_steps: int,
        eval_dir: Optional[str] = None,
        global_step: int = 0,
        process_index: int = 0,
    ) -> dict[str, float | str]:
        result = super().evaluate_validation(
            sample,
            eval_num_inference_steps=eval_num_inference_steps,
            eval_dir=eval_dir,
            global_step=global_step,
            process_index=process_index,
        )
        result["history_len"] = self._mean_history_len_metric(sample)
        return result

    @override
    def _build_eval_infer_kwargs(
        self,
        *,
        sample: dict[str, Any],
        eval_num_inference_steps: int,
    ) -> dict[str, Any]:
        kwargs = super()._build_eval_infer_kwargs(
            sample=sample,
            eval_num_inference_steps=eval_num_inference_steps,
        )
        if self._has_action_offset(sample):
            kwargs["action_horizon"] = int(self.action_horizon)
            if self.proprio_encoder is not None:
                if "proprio" not in sample or sample["proprio"] is None:
                    raise ValueError(
                        "`sample['proprio']` is required for offset-aware AHAWAM evaluation."
                    )
                offsets = self._normalize_action_offsets(
                    sample, batch_size=int(sample["proprio"].shape[0])
                )
                shifted_proprio = self._extract_offset_chunk_start_proprio(
                    sample["proprio"].to(device=self.device, dtype=self.torch_dtype),
                    offsets=offsets,
                    action_horizon=int(self.action_horizon),
                )
                kwargs["proprio"] = shifted_proprio[0].detach().cpu()
        if self._configured_num_history_frames() <= 0:
            return kwargs
        video_history = sample.get("video_history")
        if video_history is None:
            return kwargs
        if video_history.ndim == 5:
            video_history = video_history[0]
        if video_history.ndim != 4:
            raise ValueError(
                "`sample['video_history']` must be [B,C,N,H,W] or [C,N,H,W], "
                f"got shape {tuple(video_history.shape)}."
            )
        kwargs["video_history"] = video_history
        kwargs["video_history_valid_len"] = sample.get("video_history_valid_len")
        kwargs["video_history_frame_indices"] = sample.get("video_history_frame_indices")
        kwargs["video_current_frame_index"] = sample.get("video_current_frame_index")
        return kwargs

    @override
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        chunk_obs_images: Optional[torch.Tensor] = None,
        video_history: Optional[torch.Tensor] = None,
        video_history_valid_len: Optional[torch.Tensor | int] = None,
        video_history_frame_indices: Optional[torch.Tensor] = None,
        video_current_frame_index: Optional[torch.Tensor | int] = None,
    ):
        del num_frames, action, action_cfg_scale, negative_prompt, text_cfg_scale
        if action_horizon is None:
            action_horizon = self.action_horizon
        if chunk_obs_images is None:
            raise ValueError(
                "`chunk_obs_images` is required for AHAWAM evaluation. "
                "Expected shape [num_chunks,3,H,W] or [B,num_chunks,3,H,W]."
            )

        self._inference_state = None
        self.reset_history()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4:
            raise ValueError(
                "`input_image` must be [3,H,W] or [B,3,H,W], "
                f"got shape {tuple(input_image.shape)}."
            )
        batch_size = int(input_image.shape[0])
        history = self._normalize_eval_video_history(
            video_history=video_history,
            video_history_valid_len=video_history_valid_len,
            batch_size=batch_size,
        )
        history_frame_indices = self._normalize_eval_history_frame_indices(
            video_history_frame_indices=video_history_frame_indices,
            video_history_valid_len=video_history_valid_len,
            batch_size=batch_size,
        )
        current_frame_index = self._normalize_eval_current_frame_index(
            video_current_frame_index=video_current_frame_index,
            batch_size=batch_size,
        )

        if history is not None:
            if history_frame_indices is None:
                raise ValueError(
                    "`video_history_frame_indices` is required when `video_history` "
                    "is passed for sample-global RoPE validation."
                )
            for frame_idx in range(int(history.shape[2])):
                self.infer_action(
                    prompt=prompt,
                    input_image=history[:, :, frame_idx],
                    action_horizon=action_horizon,
                    context=context,
                    context_mask=context_mask,
                    seed=seed,
                    rand_device=rand_device,
                    tiled=tiled,
                    video_frame_index=int(history_frame_indices[0, frame_idx].item()),
                    phase="video",
                )

        self.infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            video_frame_index=current_frame_index,
            phase="video",
        )

        batch_size = int(self._inference_state["batch_size"])  # type: ignore[index]
        normalized = self._normalize_chunk_obs_images(
            chunk_obs_images=chunk_obs_images,
            batch_size=batch_size,
            action_horizon=action_horizon,
        )
        if normalized is None:
            raise ValueError(
                "`chunk_obs_images` normalization failed. "
                "Expected shape [num_chunks,3,H,W] or [B,num_chunks,3,H,W]."
            )

        num_chunks = action_horizon // self.action_chunk_size
        chunk_proprios = None
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`proprio` is required for evaluation when `proprio_dim` is enabled."
                )
            if proprio.ndim == 2:
                chunk_proprios = proprio.unsqueeze(0)
            elif proprio.ndim == 3:
                chunk_proprios = proprio
            else:
                raise ValueError(
                    f"`proprio` must be [num_chunks,D] or [B,num_chunks,D], "
                    f"got shape {tuple(proprio.shape)}"
                )

        chunk_outputs = []
        for chunk_index in range(num_chunks):
            obs_image = normalized[:, chunk_index]
            result = self.infer_action(
                chunk_obs_image=obs_image,
                chunk_proprio=(
                    chunk_proprios[:, chunk_index, :]
                    if chunk_proprios is not None
                    else None
                ),
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                tiled=tiled,
                phase="action",
            )
            chunk_outputs.append(result["action_chunk"])

        action = torch.cat(chunk_outputs, dim=0)
        return {"action": action}

    @torch.no_grad()
    def _prepare_distill_video_state(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> dict[str, Any]:
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        action = inputs["action"]
        obs_context = inputs["obs_context"]
        obs_context_mask = inputs["obs_context_mask"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        if self._has_action_offset(sample):
            offsets = self._normalize_action_offsets(sample, batch_size=batch_size)
            action_horizon = int(getattr(self, "action_horizon", int(action.shape[1])))
            action = self._slice_offset_sequence(
                action,
                offsets=offsets,
                action_horizon=action_horizon,
            )
            obs_context, obs_context_mask = self._build_offset_obs_context(
                sample,
                offsets=offsets,
                action_horizon=action_horizon,
                tiled=tiled,
            )

        (
            visual_obs_context,
            visual_obs_mask,
            proprio_context,
            proprio_mask,
        ) = self._split_visual_obs_and_proprio_context(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
        )
        chunk_queries = self._build_chunk_kv_queries(
            obs_context=visual_obs_context,
            obs_context_mask=visual_obs_mask,
        )

        num_history_frames = self._configured_num_history_frames()
        temporal_position_ids = (
            self._get_main_video_temporal_position_ids(
                sample,
                batch_size=batch_size,
                current_clip_latent_frames=int(input_latents.shape[2]),
            )
            if num_history_frames > 0
            else None
        )
        timestep_video = torch.zeros(
            (batch_size,), dtype=input_latents.dtype, device=self.device
        )
        video_pre = self.video_expert.pre_dit(
            x=input_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            temporal_position_ids=temporal_position_ids,
            clean_prefix_frames=int(input_latents.shape[2]),
        )
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None
        valid_history_len = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        effective_history_frames = 0
        if num_history_frames > 0:
            valid_history_len = self._get_video_history_valid_len(
                sample, batch_size=batch_size, num_history_frames=num_history_frames
            )
            effective_history_frames = int(valid_history_len.max().item())
            if effective_history_frames > 0:
                history_frame_indices = self._get_video_history_frame_indices(
                    sample, batch_size=batch_size, num_history_frames=num_history_frames
                )
                history_frame_indices = history_frame_indices[:, -effective_history_frames:]
                with torch.no_grad():
                    history_latents = self._encode_training_history_latents(
                        sample,
                        batch_size=batch_size,
                        num_history_frames=effective_history_frames,
                        tiled=tiled,
                    )
                    history_pre = self.video_expert.pre_dit(
                        x=history_latents,
                        timestep=torch.zeros_like(timestep_video),
                        context=context,
                        context_mask=context_mask,
                        action=None,
                        temporal_position_ids=history_frame_indices,
                        fuse_vae_embedding_in_latents=True,
                        clean_prefix_frames=effective_history_frames,
                    )
                    history_frame_mask = self._build_history_video_frame_mask(
                        num_history_frames=effective_history_frames,
                        valid_history_len=valid_history_len,
                        device=history_pre["tokens"].device,
                    )
                    history_attn_mask = self._expand_frame_mask_to_token_mask(
                        frame_mask=history_frame_mask,
                        tokens_per_frame=video_tokens_per_frame,
                    )
                    history_video_kv_cache = self.mot.prefill_video_cache(
                        video_tokens=history_pre["tokens"],
                        video_freqs=history_pre["freqs"],
                        video_t_mod=history_pre["t_mod"],
                        video_context_payload={
                            "context": history_pre["context"],
                            "mask": history_pre["context_mask"],
                        },
                        video_attention_mask=history_attn_mask,
                    )
                history_video_kv_cache = [
                    {"k": layer["k"].detach(), "v": layer["v"].detach()}
                    for layer in history_video_kv_cache
                ]

        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        if history_video_kv_cache is not None:
            history_seq_len = int(history_video_kv_cache[0]["k"].shape[1])
            video_seq_len_q = int(video_pre["tokens"].shape[1])
            if int(valid_history_len.min().item()) < effective_history_frames:
                positions = torch.arange(
                    effective_history_frames, device=valid_history_len.device
                )
                first_valid = (effective_history_frames - valid_history_len).clamp(
                    min=0
                ).unsqueeze(1)
                valid_frame_mask = positions.unsqueeze(0) >= first_valid
                valid_token_mask = valid_frame_mask.repeat_interleave(
                    video_tokens_per_frame, dim=1
                )
                history_prefix_mask = valid_token_mask.unsqueeze(1).expand(
                    -1, video_seq_len_q, -1
                )
            else:
                history_prefix_mask = torch.ones(
                    (batch_size, video_seq_len_q, history_seq_len),
                    dtype=torch.bool,
                    device=video_pre["tokens"].device,
                )
            video_attention_mask = video_attention_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            video_attention_mask = torch.cat(
                [history_prefix_mask, video_attention_mask], dim=2
            )

        editor_cache = self.mot.prefill_video_and_editor_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            chunk_queries=chunk_queries,
            video_tokens_per_frame=video_tokens_per_frame,
            history_video_kv_cache=history_video_kv_cache,
        )

        return {
            "editor_cache": editor_cache,
            "context": context,
            "context_mask": context_mask,
            "proprio_context": proprio_context,
            "proprio_mask": proprio_mask,
            "action": action,
            "batch_size": batch_size,
        }

    def _predict_action_flow_with_video_state(
        self,
        noisy_action: torch.Tensor,
        timestep_action: torch.Tensor,
        video_state: dict[str, Any],
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=video_state["context"],
            context_mask=video_state["context_mask"],
            obs_context=video_state["proprio_context"],
            obs_context_mask=video_state["proprio_mask"],
            clean_action_tokens=None,
            clean_timestep=None,
            chunk_size=self.action_chunk_size,
            noisy_position_offset=0,
            single_branch_chunk_causal=True,
            obs_chunk_offset=0,
            obs_context_causal=True,
            obs_proprio_tokens_per_chunk=self._require_obs_proprio_tokens_per_chunk(),
        )
        prior_embed = self.mot.action_branch_embedding[1].to(
            device=action_pre["tokens"].device, dtype=action_pre["tokens"].dtype
        )
        action_tokens = action_pre["tokens"] + prior_embed.view(1, 1, -1)
        action_tokens = self.mot.forward_action_prior_only_with_editor_cache(
            action_tokens=action_tokens,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            editor_cache=video_state["editor_cache"],
            action_chunk_size=self.action_chunk_size,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def rollout_action_prior_only(
        self,
        sample: dict[str, Any],
        num_steps: int = 16,
        capture_indices: tuple[int, ...] = (0, 1, 2, 4, 8, 12, 16),
        sigma_shift: Optional[float] = None,
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        video_state = self._prepare_distill_video_state(sample, tiled=tiled)
        action = video_state["action"]
        batch_size = video_state["batch_size"]
        latents = torch.randn_like(action)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        capture_indices = tuple(sorted({int(step) for step in capture_indices}))
        captured_states: dict[int, torch.Tensor] = {}

        for step_idx, (timestep, delta) in enumerate(zip(timesteps, deltas)):
            if step_idx in capture_indices:
                captured_states[step_idx] = latents.detach().clone()
            timestep_action = timestep.expand(batch_size).to(
                device=self.device, dtype=self.torch_dtype
            )
            flow_pred = self._predict_action_flow_with_video_state(
                noisy_action=latents,
                timestep_action=timestep_action,
                video_state=video_state,
            )
            latents = self.infer_action_scheduler.step(flow_pred, delta, latents)

        final_step = int(num_steps)
        if final_step in capture_indices:
            captured_states[final_step] = latents.detach().clone()

        return {
            "initial_latents": captured_states.get(0),
            "captured_states": captured_states,
            "final_latents": latents.detach().clone(),
            "timesteps": timesteps,
            "capture_step_indices": capture_indices,
            "video_state": video_state,
        }
