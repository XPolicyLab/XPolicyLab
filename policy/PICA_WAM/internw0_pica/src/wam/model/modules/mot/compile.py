"""Optional layerwise Inductor compilation for video/action training."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Dict, Optional
import os

import torch
import torch.nn.functional as F
from wam.utils.logging_config import get_logger

logger = get_logger(__name__)
_MOT_COMPILE_GROUP_SIZE = 1
_MOT_CHECKPOINT_GROUP_SIZE = 3


def _profile_section(profiler, name):
    return profiler.section(name) if profiler is not None else nullcontext()


def _env_flag(name):
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


class MoTCompileMixin:
    def _init_compile(self, mode, checkpointing, pad_to):
        self.compile_mode = str(mode or "off").lower()
        if self.compile_mode not in {"off", "default", "reduce-overhead"}:
            raise ValueError(f"Unsupported MoT compile_mode: {mode!r}")
        if self.compile_mode != "off" and self.mot_checkpoint_mixed_attn:
            raise ValueError(
                "MoT compilation requires mot_checkpoint_mixed_attn=false."
            )
        self.compile_gradient_checkpointing = bool(checkpointing)
        if self.compile_gradient_checkpointing and self.compile_mode == "off":
            raise ValueError(
                "Compile gradient checkpointing requires compile_mode != 'off'."
            )
        self.compile_action_context_pad_to = int(pad_to)
        if self.compile_action_context_pad_to < 0:
            raise ValueError("compile_action_context_pad_to must be non-negative.")
        self._compile_fallback_reasons = set()
        self._compile_context_padding_logged = False
        object.__setattr__(self, "_compiled_groups", {})
        object.__setattr__(self, "_compiled_action_cache_layers", {})
        object.__setattr__(self, "_compiled_action_cache_groups", {})
        object.__setattr__(self, "_compiled_action_cache_calls", 0)

    def _compile_group_ranges(
        self,
        requested_captures: Dict[str, tuple[int, ...]],
    ) -> tuple[tuple[int, int], ...]:
        """Return one independent compiler boundary for every MoT layer."""
        boundaries = set(range(0, self.num_layers, _MOT_COMPILE_GROUP_SIZE))
        boundaries.add(self.num_layers)
        ordered = sorted(boundaries)
        return tuple(zip(ordered, ordered[1:]))

    def _checkpoint_group_ranges(
        self,
        requested_captures: Dict[str, tuple[int, ...]],
    ) -> tuple[tuple[int, int], ...]:
        """Return three-layer checkpoint ranges split at capture boundaries."""
        boundaries = set(range(0, self.num_layers, _MOT_CHECKPOINT_GROUP_SIZE))
        boundaries.add(self.num_layers)
        for layers in requested_captures.values():
            boundaries.update(
                layer_idx + 1
                for layer_idx in layers
                if 0 <= layer_idx < self.num_layers
            )
        ordered = sorted(boundaries)
        return tuple(zip(ordered, ordered[1:]))

    def _get_compiled_group(self, start_layer: int, end_layer: int):
        start_layer = int(start_layer)
        end_layer = int(end_layer)
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise IndexError(
                "MoT compile group range out of bounds: "
                f"[{start_layer}, {end_layer}) for {self.num_layers} layers."
            )
        if end_layer != start_layer + _MOT_COMPILE_GROUP_SIZE:
            raise ValueError(
                "MoT layerwise compile requires exactly one layer per boundary; "
                f"got [{start_layer}, {end_layer})."
            )
        group_range = (start_layer, end_layer)
        compiled_groups = self._compiled_groups
        if group_range not in compiled_groups:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "MoT compile_mode requires a PyTorch build with torch.compile."
                )
            logger.info(
                "Compiling MoT training layer %d/%d lazily: mode=%s "
                "granularity=layer "
                "forward=inductor backward=inductor "
                "dynamic=false fullgraph=false cudagraphs=%s "
                "mix_order_reduction=default",
                start_layer + 1,
                self.num_layers,
                self.compile_mode,
                self.compile_mode == "reduce-overhead"
                and not self.compile_gradient_checkpointing,
            )
            compile_kwargs = {
                "backend": "inductor",
                "dynamic": False,
                "fullgraph": False,
            }
            if self.compile_gradient_checkpointing:
                # Saved activations must survive until checkpoint recomputation
                # finishes; keep Inductor but disable CUDA graph buffer reuse.
                compile_kwargs["options"] = {
                    "triton.cudagraphs": False,
                }
            else:
                compile_kwargs["mode"] = self.compile_mode
            compiled_group = torch.compile(
                self._make_compilable_group_callable(start_layer, end_layer),
                **compile_kwargs,
            )
            compiled_groups[group_range] = compiled_group
        return compiled_groups[group_range]

    def _get_compiled_action_cache_layer(self, layer_idx: int):
        """Return the lazily compiled boundary used by online action denoising."""

        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                "MoT action-cache layer out of bounds: "
                f"{layer_idx} for {self.num_layers} layers."
            )
        compiled_layers = self._compiled_action_cache_layers
        if layer_idx not in compiled_layers:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "MoT compile_mode requires a PyTorch build with torch.compile."
                )
            compile_kwargs = {
                "backend": "inductor",
                "dynamic": False,
                "fullgraph": _env_flag("WAM_MOT_COMPILE_FULLGRAPH"),
                "mode": self.compile_mode,
            }
            logger.info(
                "Compiling MoT online action-cache layer %d/%d: mode=%s "
                "forward=inductor dynamic=false fullgraph=%s",
                layer_idx + 1,
                self.num_layers,
                self.compile_mode,
                compile_kwargs["fullgraph"],
            )
            compiled_layers[layer_idx] = torch.compile(
                self._make_compilable_action_cache_layer_callable(layer_idx),
                **compile_kwargs,
            )
        return compiled_layers[layer_idx]

    def _get_compiled_action_cache_group(self, start_layer: int, end_layer: int):
        """Return one compiled callable for a contiguous action-layer group."""

        start_layer = int(start_layer)
        end_layer = int(end_layer)
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise IndexError(
                "MoT action-cache group out of bounds: "
                f"[{start_layer}, {end_layer}) for {self.num_layers} layers."
            )
        group_range = (start_layer, end_layer)
        compiled_groups = self._compiled_action_cache_groups
        if group_range not in compiled_groups:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "MoT compile_mode requires a PyTorch build with torch.compile."
                )
            compile_kwargs = {
                "backend": "inductor",
                "dynamic": False,
                "fullgraph": _env_flag("WAM_MOT_COMPILE_FULLGRAPH"),
                "mode": self.compile_mode,
            }
            logger.info(
                "Compiling MoT online action-cache group [%d,%d)/%d: mode=%s",
                start_layer + 1,
                end_layer,
                self.num_layers,
                self.compile_mode,
            )
            compiled_groups[group_range] = torch.compile(
                self._make_compilable_action_cache_group_callable(
                    start_layer, end_layer
                ),
                **compile_kwargs,
            )
        return compiled_groups[group_range]

    @torch.no_grad()
    def prefill_action_context_kv_cache(
        self, context: Optional[torch.Tensor]
    ) -> Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Project Action cross-attention context once for the current chunk."""

        if context is None:
            return None
        action_expert = self.mixtures["action"]
        if not bool(getattr(action_expert, "uses_vlm_conditioning", False)):
            # Match ActionDiT.pre_dit: non-VLM conditioning is embedded before it
            # reaches the block cross-attention projections.
            context = action_expert.text_embedding(context)
        cache = []
        for block in action_expert.blocks:
            cross_attn = block.cross_attn
            context_k = cross_attn.norm_k(cross_attn.k(context))
            context_v = cross_attn.v(context)
            cache.append((context_k, context_v))
        return tuple(cache)

    def _make_compilable_action_cache_group_callable(
        self,
        start_layer: int,
        end_layer: int,
    ):
        """Create a fixed multi-layer action-cache compiler boundary."""

        start_layer = int(start_layer)
        end_layer = int(end_layer)

        def action_cache_group(
            action_embed: torch.Tensor,
            action_freqs: torch.Tensor,
            action_t_mod: torch.Tensor,
            action_context: Optional[torch.Tensor],
            action_context_mask: Optional[torch.Tensor],
            action_context_keys: torch.Tensor,
            action_context_values: torch.Tensor,
            video_keys: torch.Tensor,
            video_values: torch.Tensor,
            action_attention_mask: torch.Tensor,
            action_flex_key_mask: Optional[torch.Tensor],
        ) -> torch.Tensor:
            x = action_embed
            for offset, layer_idx in enumerate(range(start_layer, end_layer)):
                x = self._forward_compilable_action_cache_layer(
                    layer_idx,
                    x,
                    action_freqs,
                    action_t_mod,
                    action_context,
                    action_context_mask,
                    action_context_keys[offset],
                    action_context_values[offset],
                    video_keys[offset],
                    video_values[offset],
                    action_attention_mask,
                    action_flex_key_mask,
                )
            return x

        code_name = f"_forward_compilable_action_cache_group_{start_layer}_{end_layer}"
        action_cache_group.__code__ = action_cache_group.__code__.replace(
            co_name=code_name
        )
        action_cache_group.__name__ = code_name
        action_cache_group.__qualname__ = f"{type(self).__qualname__}.{code_name}"
        return action_cache_group

    def _make_compilable_action_cache_layer_callable(self, layer_idx: int):
        """Create a distinct Dynamo frame for one cached-video action layer."""

        layer_idx = int(layer_idx)

        def action_cache_layer(
            action_embed: torch.Tensor,
            action_freqs: torch.Tensor,
            action_t_mod: torch.Tensor,
            action_context: Optional[torch.Tensor],
            action_context_mask: Optional[torch.Tensor],
            action_context_k: Optional[torch.Tensor],
            action_context_v: Optional[torch.Tensor],
            video_k: torch.Tensor,
            video_v: torch.Tensor,
            action_attention_mask: torch.Tensor,
            action_flex_key_mask: Optional[torch.Tensor],
        ) -> torch.Tensor:
            return self._forward_compilable_action_cache_layer(
                layer_idx,
                action_embed,
                action_freqs,
                action_t_mod,
                action_context,
                action_context_mask,
                action_context_k,
                action_context_v,
                video_k,
                video_v,
                action_attention_mask,
                action_flex_key_mask,
            )

        code_name = f"_forward_compilable_action_cache_layer_{layer_idx}"
        action_cache_layer.__code__ = action_cache_layer.__code__.replace(
            co_name=code_name
        )
        action_cache_layer.__name__ = code_name
        action_cache_layer.__qualname__ = f"{type(self).__qualname__}.{code_name}"
        action_cache_layer._mot_layer_idx = layer_idx
        return action_cache_layer

    def _forward_compilable_action_cache_layer(
        self,
        layer_idx: int,
        action_embed: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        action_context_k: Optional[torch.Tensor],
        action_context_v: Optional[torch.Tensor],
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        action_attention_mask: torch.Tensor,
        action_flex_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one action layer against a precomputed video KV cache."""

        expert = self.mixtures["action"]
        block = expert.blocks[int(layer_idx)]
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
            x=action_embed,
            freqs=action_freqs,
            t_mod=action_t_mod,
        )
        mixed = self._mixed_attention(
            q_cat=q_action,
            k_cat=torch.cat([video_k, k_action], dim=1),
            v_cat=torch.cat([video_v, v_action], dim=1),
            attention_mask=action_attention_mask,
            flex_key_mask=action_flex_key_mask,
            force_sdpa=True,
        )
        return self._apply_post_with_optional_checkpoint(
            block=block,
            residual_x=residual_x,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed,
            context_payload={
                "context": action_context,
                "mask": action_context_mask,
            },
            context_kv=(action_context_k, action_context_v)
            if action_context_k is not None and action_context_v is not None
            else None,
        )

    def _make_compilable_group_callable(
        self,
        start_layer: int,
        end_layer: int,
    ):
        """Create a Dynamo frame dedicated to one fixed MoT layer.

        ``functools.partial`` objects are wrapped by Dynamo with the shared
        ``external_utils.inner`` code object. With more than eight bound ranges
        that frame hits Dynamo's default recompile limit and the remaining
        ranges fall back to eager. Cloning this ordinary function's code object
        makes the intended one-graph-per-layer boundary explicit without
        changing process-global Dynamo limits. Compile checkpointing is applied
        by the eager orchestrator outside this callable so checkpoint operators
        and RNG-state helpers never enter the persistent compiler graph.
        """
        start_layer = int(start_layer)
        end_layer = int(end_layer)

        def group_forward(
            video_embed: torch.Tensor,
            action_embed: torch.Tensor,
            attention_mask: torch.Tensor,
            video_freqs: torch.Tensor,
            action_freqs: torch.Tensor,
            video_context: Optional[torch.Tensor],
            video_context_mask: Optional[torch.Tensor],
            action_context: Optional[torch.Tensor],
            action_context_mask: Optional[torch.Tensor],
            video_t_mod: torch.Tensor,
            action_t_mod: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._forward_compilable_group(
                start_layer,
                end_layer,
                video_embed,
                action_embed,
                attention_mask,
                video_freqs,
                action_freqs,
                video_context,
                video_context_mask,
                action_context,
                action_context_mask,
                video_t_mod,
                action_t_mod,
            )

        code_name = f"_forward_compilable_group_{start_layer}_{end_layer}"
        group_forward.__code__ = group_forward.__code__.replace(co_name=code_name)
        group_forward.__name__ = code_name
        group_forward.__qualname__ = f"{type(self).__qualname__}.{code_name}"
        group_forward._mot_layer_range = (start_layer, end_layer)
        return group_forward

    def _pad_compiled_action_context(
        self,
        context: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pad Action cross-attention inputs to one compiler-stable length."""
        target = self.compile_action_context_pad_to
        if context is None or target == 0:
            return context, mask
        if context.ndim != 3:
            raise ValueError(
                "Compiled Action context must be [B,L,D], got "
                f"shape={tuple(context.shape)}."
            )
        length = int(context.shape[1])
        if length > target:
            raise ValueError(
                "Compiled Action context exceeds "
                f"model.mot_compile_action_context_pad_to: {length} > {target}. "
                "Increase the configured limit; the context will not be truncated."
            )
        if mask is None:
            mask = torch.ones(
                (context.shape[0], length), dtype=torch.bool, device=context.device
            )
        elif (
            mask.ndim not in (2, 3)
            or int(mask.shape[0]) != int(context.shape[0])
            or int(mask.shape[-1]) != length
        ):
            raise ValueError(
                "Compiled Action context mask must be [B,L] or [B,Q,L], got "
                f"mask={tuple(mask.shape)} context={tuple(context.shape)}."
            )
        if length == target:
            return context, mask
        pad = target - length
        if not self._compile_context_padding_logged:
            logger.info(
                "Padding MoT Action context to %d tokens for a stable compiled graph.",
                target,
            )
            self._compile_context_padding_logged = True
        return F.pad(context, (0, 0, 0, pad)), F.pad(mask, (0, pad), value=False)

    def _compile_fallback(self, reason: str) -> None:
        if reason in self._compile_fallback_reasons:
            return
        self._compile_fallback_reasons.add(reason)
        logger.warning("MoT compile path disabled for this call: %s", reason)

    def _forward_compilable_layer(
        self,
        layer_idx: int,
        video_embed: torch.Tensor,
        action_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one fixed video/action MoT layer inside one compiler boundary."""
        tokens_all = {"video": video_embed, "action": action_embed}
        freqs_all = {"video": video_freqs, "action": action_freqs}
        t_mod_all = {"video": video_t_mod, "action": action_t_mod}
        context_all = {
            "video": {"context": video_context, "mask": video_context_mask},
            "action": {"context": action_context, "mask": action_context_mask},
        }
        q_chunks = []
        k_chunks = []
        v_chunks = []
        cached = {}
        seq_lens = []

        for name in ("video", "action"):
            expert = self.mixtures[name]
            block = expert.blocks[layer_idx]
            x = tokens_all[name]
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
                freqs=freqs_all[name],
                t_mod=t_mod_all[name],
            )
            q_chunks.append(q)
            k_chunks.append(k)
            v_chunks.append(v)
            seq_lens.append(x.shape[1])
            cached[name] = {
                "block": block,
                "residual_x": residual_x,
                "gate_msa": gate_msa,
                "shift_mlp": shift_mlp,
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp,
                "use_gradient_checkpointing": use_gradient_checkpointing,
            }

        q_cat = torch.cat(q_chunks, dim=1)
        k_cat = torch.cat(k_chunks, dim=1)
        v_cat = torch.cat(v_chunks, dim=1)
        if attention_mask.shape[-2] != q_cat.shape[1]:
            raise ValueError(
                "Attention mask seq length mismatch: "
                f"mask={attention_mask.shape[-2]} vs tokens={q_cat.shape[1]}"
            )
        mixed = self._mixed_attention(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=attention_mask,
            force_sdpa=True,
        )

        start = 0
        for name, seq_len in zip(("video", "action"), seq_lens):
            end = start + seq_len
            cached_expert = cached[name]
            tokens_all[name] = self._apply_post_with_optional_checkpoint(
                block=cached_expert["block"],
                residual_x=cached_expert["residual_x"],
                gate_msa=cached_expert["gate_msa"],
                shift_mlp=cached_expert["shift_mlp"],
                scale_mlp=cached_expert["scale_mlp"],
                gate_mlp=cached_expert["gate_mlp"],
                use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                mixed_slice=mixed[:, start:end, :],
                context_payload=context_all[name],
            )
            start = end

        return tokens_all["video"], tokens_all["action"]

    def _forward_compilable_group(
        self,
        start_layer: int,
        end_layer: int,
        video_embed: torch.Tensor,
        action_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MoT layer inside its own compiler/autograd boundary."""
        video_out = video_embed
        action_out = action_embed
        for layer_idx in range(start_layer, end_layer):
            video_out, action_out = self._forward_compilable_layer(
                layer_idx,
                video_out,
                action_out,
                attention_mask,
                video_freqs,
                action_freqs,
                video_context,
                video_context_mask,
                action_context,
                action_context_mask,
                video_t_mod,
                action_t_mod,
            )
        return video_out, action_out

    def _run_compiled_layer_group(
        self,
        compiled_layers: tuple,
        start_layer: int,
        video_embed: torch.Tensor,
        action_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        *,
        profiler=None,
        profile_mot_layers: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one checkpoint group while retaining per-layer compile frames."""
        video_out = video_embed
        action_out = action_embed
        for offset, compiled_layer in enumerate(compiled_layers):
            layer_idx = int(start_layer) + offset
            layer_prefix = f"layer_{layer_idx:02d}" if profile_mot_layers else "layer"
            try:
                with _profile_section(profiler, f"{layer_prefix}/compiled"):
                    video_out, action_out = compiled_layer(
                        video_out,
                        action_out,
                        attention_mask,
                        video_freqs,
                        action_freqs,
                        video_context,
                        video_context_mask,
                        action_context,
                        action_context_mask,
                        video_t_mod,
                        action_t_mod,
                    )
            except Exception as exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "MoT layerwise torch.compile failed at "
                        f"layer={layer_idx}, mode={self.compile_mode}."
                    )
                raise
        return video_out, action_out

    def _forward_compiled_groups(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        key_masks_all: Dict[str, torch.Tensor],
        requested_captures: Dict[str, tuple[int, ...]],
        profiler=None,
    ):
        with _profile_section(profiler, "key_mask_build"):
            key_mask_all = self._build_key_mask_all(embeds_all, key_masks_all)
            effective_attention_mask = self._apply_key_mask(
                attention_mask, key_mask_all
            )

        video_context = context_all.get("video") or {}
        action_context = context_all.get("action") or {}
        video_out = embeds_all["video"]
        action_out = embeds_all["action"]
        captured: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in requested_captures
        }
        requested_capture_sets = {
            name: frozenset(layers) for name, layers in requested_captures.items()
        }
        profile_mot_layers = profiler is not None and _env_flag(
            "WAM_PROFILE_MOT_LAYERS"
        )

        for start_layer, end_layer in self._checkpoint_group_ranges(requested_captures):
            compiled_layers = tuple(
                self._get_compiled_group(layer_idx, layer_idx + 1)
                for layer_idx in range(start_layer, end_layer)
            )
            group_forward = partial(
                self._run_compiled_layer_group,
                compiled_layers,
                start_layer,
                profiler=profiler,
                profile_mot_layers=profile_mot_layers,
            )
            group_args = (
                video_out,
                action_out,
                effective_attention_mask,
                freqs_all["video"],
                freqs_all["action"],
                video_context.get("context"),
                video_context.get("mask"),
                action_context.get("context"),
                action_context.get("mask"),
                t_mod_all["video"],
                t_mod_all["action"],
            )
            if self.compile_gradient_checkpointing and self.training:
                video_out, action_out = torch.utils.checkpoint.checkpoint(
                    group_forward,
                    *group_args,
                    use_reentrant=False,
                )
            else:
                video_out, action_out = group_forward(*group_args)

            last_layer = end_layer - 1
            group_outputs = {"video": video_out, "action": action_out}
            for name, layers in requested_capture_sets.items():
                if last_layer in layers:
                    captured[name][last_layer] = group_outputs[name]

        tokens = {"video": video_out, "action": action_out}
        if not requested_captures:
            return tokens
        return tokens, captured

    def _compiled_forward_or_eager(
        self,
        embeds_all,
        attention_mask,
        freqs_all,
        context_all,
        t_mod_all,
        key_masks_all=None,
        query_masks_all=None,
        capture_layers=None,
        capture_batch_masks=None,
        profiler=None,
    ):
        reason = None
        if not self.training:
            reason = "evaluation uses the eager attention path"
        elif self.expert_order != ["video", "action"]:
            reason = "layerwise compilation requires two experts"
        elif query_masks_all is not None or capture_batch_masks is not None:
            reason = "query masking and subset captures use the eager path"
        elif attention_mask is None:
            reason = "attention_mask is required"
        elif any(
            name not in embeds_all or name not in freqs_all or name not in t_mod_all
            for name in ("video", "action")
        ):
            reason = "video/action inputs are required"
        elif embeds_all["video"].device.type != "cuda":
            reason = "CUDA tensors are required"
        captures = {
            name: tuple(sorted(int(i) for i in layers))
            for name, layers in (capture_layers or {}).items()
        }
        if set(captures) - {"video", "action"}:
            reason = "additional expert captures use the eager path"
        if reason is not None:
            self._compile_fallback(reason)
            return self._forward_eager(
                embeds_all,
                attention_mask,
                freqs_all,
                context_all,
                t_mod_all,
                key_masks_all,
                query_masks_all,
                capture_layers,
                capture_batch_masks,
                profiler,
            )
        action = context_all.get("action") or {}
        context, mask = self._pad_compiled_action_context(
            action.get("context"), action.get("mask")
        )
        contexts = {**context_all, "action": {"context": context, "mask": mask}}
        return self._forward_compiled_groups(
            embeds_all,
            attention_mask,
            freqs_all,
            contexts,
            t_mod_all,
            key_masks_all or {},
            captures,
            profiler,
        )
