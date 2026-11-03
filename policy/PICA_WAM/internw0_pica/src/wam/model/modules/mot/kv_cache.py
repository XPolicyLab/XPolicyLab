from typing import Callable, Optional

import os

import torch


class MoTCacheMixin:
    """Per-replan video-prefix KV cache used by online action denoising."""

    @staticmethod
    def _apply_key_mask(
        attention_mask: torch.Tensor,
        key_mask: Optional[torch.Tensor],
        *,
        allow_self_when_all_masked: bool = False,
    ) -> torch.Tensor:
        if key_mask is None:
            return attention_mask
        if attention_mask.ndim not in (2, 4):
            raise ValueError(
                f"`attention_mask` must be 2D or 4D when applying key_mask, got {tuple(attention_mask.shape)}"
            )
        if key_mask.ndim != 2:
            raise ValueError(f"`key_mask` must be [B,S], got {tuple(key_mask.shape)}")
        if int(attention_mask.shape[-1]) != int(key_mask.shape[1]):
            raise ValueError(
                "`key_mask` seq length mismatch: "
                f"mask={key_mask.shape[1]} vs attention keys={attention_mask.shape[-1]}"
            )
        effective = attention_mask.to(device=key_mask.device, dtype=torch.bool)
        if effective.ndim == 2:
            effective = effective.unsqueeze(0).unsqueeze(1)
        else:
            if int(effective.shape[0]) not in (1, int(key_mask.shape[0])):
                raise ValueError(
                    "`attention_mask` batch dimension must be 1 or match key_mask batch: "
                    f"{effective.shape[0]} vs {key_mask.shape[0]}"
                )
            if int(effective.shape[1]) != 1:
                raise ValueError(
                    f"`attention_mask` 4D shape must be [B,1,Q,K], got {tuple(effective.shape)}"
                )
        effective = effective & key_mask[:, None, None, :].to(dtype=torch.bool)
        if allow_self_when_all_masked:
            empty_batch = ~key_mask.any(dim=1)
            if attention_mask.shape[-2] != attention_mask.shape[-1]:
                raise ValueError(
                    "Cannot add self keys for empty key_mask with non-square attention_mask."
                )
            eye = torch.eye(
                int(attention_mask.shape[-2]),
                dtype=torch.bool,
                device=effective.device,
            ).view(1, 1, int(attention_mask.shape[-2]), int(attention_mask.shape[-1]))
            effective = torch.where(empty_batch[:, None, None, None], eye, effective)
        return effective

    def prefill_expert_cache(
        self,
        expert_name: str,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        context_payload: Optional[dict],
        attention_mask: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        cache_slice: Optional[slice] = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill a single expert and return per-layer K/V for all or selected tokens."""
        if expert_name not in self.mixtures:
            raise ValueError(f"MoT has no expert named {expert_name!r}.")
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S,S], got {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got {tuple(attention_mask.shape)}"
            )
        if int(attention_mask.shape[0]) != int(tokens.shape[1]):
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs tokens={tokens.shape[1]}"
            )
        if key_mask is not None:
            if key_mask.ndim != 2:
                raise ValueError(
                    f"`key_mask` must be [B,S], got {tuple(key_mask.shape)}"
                )
            if tuple(key_mask.shape) != tuple(tokens.shape[:2]):
                raise ValueError(
                    f"`key_mask` shape mismatch: {tuple(key_mask.shape)} vs {tuple(tokens.shape[:2])}"
                )
            key_mask = key_mask.to(device=tokens.device, dtype=torch.bool)

        flex_key_mask = key_mask
        force_sdpa = False
        if self.attention_backend == "flex" and key_mask is not None:
            force_sdpa = not bool(key_mask.all().item())
            if not force_sdpa:
                flex_key_mask = None
        flex_block_mask = None
        if not force_sdpa:
            flex_block_mask = self._build_flex_block_mask(
                attention_mask=attention_mask,
                batch_size=int(tokens.shape[0]),
                query_len=int(tokens.shape[1]),
                key_len=int(tokens.shape[1]),
                device=tokens.device,
            )
        expert = self.mixtures[expert_name]
        x = tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=freqs,
                t_mod=t_mod,
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=attention_mask,
                flex_block_mask=flex_block_mask,
                flex_key_mask=flex_key_mask,
                allow_self_when_all_masked=True,
                force_sdpa=force_sdpa,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=context_payload,
            )
            layer_cache = {
                "k": k[:, cache_slice] if cache_slice is not None else k,
                "v": v[:, cache_slice] if cache_slice is not None else v,
            }
            if key_mask is not None:
                layer_cache["mask"] = (
                    key_mask[:, cache_slice] if cache_slice is not None else key_mask
                )
            kv_cache.append(layer_cache)
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        video_grouped_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
                - `kv_cache`: per-layer projected context K/V
                - `kv_cache_stacked`: stacked projected context K/V
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.
            video_grouped_kv_cache: Optional stacked video K/V with shape
                `[num_layers, B, Sv, H]`, used to reduce compiled group inputs.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for `forward_action_with_video_cache`."
            )
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[
            video_seq_len:total_seq_len, :total_seq_len
        ]

        action_context_kv_cache = None
        action_context_kv_cache_stacked = None
        if action_context_payload is not None:
            action_context_kv_cache = action_context_payload.get("kv_cache")
            action_context_kv_cache_stacked = action_context_payload.get(
                "kv_cache_stacked"
            )
        if action_context_kv_cache is not None:
            if len(action_context_kv_cache) != self.num_layers:
                raise ValueError(
                    "`action_context_payload['kv_cache']` must contain "
                    f"{self.num_layers} layers, got {len(action_context_kv_cache)}."
                )
            for layer_idx, layer_context_kv in enumerate(action_context_kv_cache):
                if (
                    not isinstance(layer_context_kv, (tuple, list))
                    or len(layer_context_kv) != 2
                ):
                    raise ValueError(
                        "`action_context_payload['kv_cache'][%d]` must be a (K, V) pair."
                        % layer_idx
                    )
        if action_context_kv_cache_stacked is not None:
            if (
                not isinstance(action_context_kv_cache_stacked, (tuple, list))
                or len(action_context_kv_cache_stacked) != 2
                or not all(
                    isinstance(value, torch.Tensor)
                    for value in action_context_kv_cache_stacked
                )
                or any(
                    int(value.shape[0]) != self.num_layers
                    for value in action_context_kv_cache_stacked
                )
            ):
                raise ValueError(
                    "`action_context_payload['kv_cache_stacked']` must be a "
                    "(num_layers, B, L, H) K/V pair."
                )

        # Each layer carries video-key validity alongside its cached tensors.
        # Combine it with the current action-key validity for attention.
        cache_mask_presence = ["mask" in layer for layer in video_kv_cache]
        if any(cache_mask_presence) and not all(cache_mask_presence):
            raise ValueError(
                "`video_kv_cache` must provide a key mask for every layer or "
                "for no layers."
            )

        joint_key_mask = None
        if cache_mask_presence and all(cache_mask_presence):
            expected_mask_shape = (
                int(action_tokens.shape[0]),
                int(video_seq_len),
            )
            for layer_idx, layer_cache in enumerate(video_kv_cache):
                layer_mask = layer_cache["mask"]
                if not isinstance(layer_mask, torch.Tensor):
                    raise TypeError(
                        f"`video_kv_cache[{layer_idx}]['mask']` must be a tensor."
                    )
                if tuple(layer_mask.shape) != expected_mask_shape:
                    raise ValueError(
                        f"`video_kv_cache[{layer_idx}]['mask']` shape mismatch: "
                        f"{tuple(layer_mask.shape)} vs {expected_mask_shape}."
                    )

            # `prefill_expert_cache` stores the same video key mask for every
            # layer, so one canonical copy is sufficient for all denoising
            # layers.  Action time steps are all real in online chunk inference.
            video_key_mask = video_kv_cache[0]["mask"].to(
                device=action_tokens.device,
                dtype=torch.bool,
            )
            action_key_mask = torch.ones(
                (int(action_tokens.shape[0]), action_seq_len),
                dtype=torch.bool,
                device=action_tokens.device,
            )
            joint_key_mask = torch.cat([video_key_mask, action_key_mask], dim=1)

        # Match full-MoT training: per-sample invalid keys use the exact SDPA
        # mask path, while all-valid FlexAttention calls keep their fast path.
        flex_key_mask = joint_key_mask
        force_sdpa = False
        if self.attention_backend == "flex" and joint_key_mask is not None:
            force_sdpa = not bool(joint_key_mask.all().item())
            if not force_sdpa:
                flex_key_mask = None

        compile_mode = str(getattr(self, "compile_mode", "off") or "off")
        use_compiled_action_cache = (
            compile_mode != "off"
            and action_context_payload is not None
            and action_context_payload.get("context") is not None
            and action_context_payload.get("mask") is not None
            and hasattr(self, "_get_compiled_action_cache_layer")
        )
        compiled_action_context = None
        compiled_action_context_mask = None
        # Projected context K/V is already backend-stable. Padding it here would
        # inflate every cross-attention lookup; retain padding only for wrappers
        # that still project raw context inside the layer.
        if use_compiled_action_cache and action_context_kv_cache is None:
            (
                compiled_action_context,
                compiled_action_context_mask,
            ) = self._pad_compiled_action_context(
                action_context_payload["context"],
                action_context_payload["mask"],
            )
        elif use_compiled_action_cache:
            compiled_action_context = action_context_payload["context"]
            compiled_action_context_mask = action_context_payload["mask"]

        flex_block_mask = None
        if not use_compiled_action_cache and not force_sdpa:
            flex_block_mask = self._build_flex_block_mask(
                attention_mask=action_attention_mask,
                batch_size=int(action_tokens.shape[0]),
                query_len=action_seq_len,
                key_len=total_seq_len,
                device=action_tokens.device,
            )

        expert = self.mixtures["action"]
        x = action_tokens

        # Per-layer compilation keeps graphs small but still pays Python
        # dispatch for every MoT layer. Group contiguous layers when requested.
        action_cache_group_size = max(
            1, int(os.environ.get("WAM_MOT_ACTION_CACHE_GROUP_SIZE", "1"))
        )
        if (
            use_compiled_action_cache
            and action_context_kv_cache is not None
            and action_context_kv_cache_stacked is not None
            and video_grouped_kv_cache is not None
            and action_cache_group_size > 1
        ):
            if (
                not isinstance(video_grouped_kv_cache, (tuple, list))
                or len(video_grouped_kv_cache) != 2
                or not all(
                    isinstance(value, torch.Tensor)
                    for value in video_grouped_kv_cache
                )
                or any(
                    int(value.shape[0]) != self.num_layers
                    for value in video_grouped_kv_cache
                )
            ):
                raise ValueError(
                    "`video_grouped_kv_cache` must be a stacked "
                    "(num_layers, B, L, H) K/V pair."
                )
            for start_layer in range(0, self.num_layers, action_cache_group_size):
                end_layer = min(
                    self.num_layers, start_layer + action_cache_group_size
                )
                video_keys = video_grouped_kv_cache[0][start_layer:end_layer]
                video_values = video_grouped_kv_cache[1][start_layer:end_layer]
                context_keys = action_context_kv_cache_stacked[0][
                    start_layer:end_layer
                ]
                context_values = action_context_kv_cache_stacked[1][
                    start_layer:end_layer
                ]
                x = self._get_compiled_action_cache_group(start_layer, end_layer)(
                    x,
                    action_freqs,
                    action_t_mod,
                    compiled_action_context,
                    compiled_action_context_mask,
                    context_keys,
                    context_values,
                    video_keys,
                    video_values,
                    action_attention_mask,
                    flex_key_mask,
                )
                self._compiled_action_cache_calls += end_layer - start_layer
            return x

        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            if use_compiled_action_cache:
                layer_cache = video_kv_cache[layer_idx]
                if "k" not in layer_cache or "v" not in layer_cache:
                    raise ValueError(
                        f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                    )
                x = self._get_compiled_action_cache_layer(layer_idx)(
                    x,
                    action_freqs,
                    action_t_mod,
                    compiled_action_context,
                    compiled_action_context_mask,
                    action_context_kv_cache[layer_idx][0]
                    if action_context_kv_cache is not None
                    else None,
                    action_context_kv_cache[layer_idx][1]
                    if action_context_kv_cache is not None
                    else None,
                    layer_cache["k"],
                    layer_cache["v"],
                    action_attention_mask,
                    flex_key_mask,
                )
                self._compiled_action_cache_calls += 1
                continue

            # Action query/key/value remain step-dependent and are recomputed.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
                flex_block_mask=flex_block_mask,
                flex_key_mask=flex_key_mask,
                force_sdpa=force_sdpa,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
                context_kv=(
                    action_context_kv_cache[layer_idx]
                    if action_context_kv_cache is not None
                    else None
                ),
            )
        return x
