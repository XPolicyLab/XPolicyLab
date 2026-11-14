from __future__ import annotations
from typing import Callable, ClassVar, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .wan_video_dit import flash_attention, modulate, rope_apply
from cloudrobo_wm.utils.logging_config import get_logger
logger = get_logger(__name__)
_compiled_flex_attn: Callable | None = None

def _get_compiled_flex_attn() -> Callable:
    global _compiled_flex_attn
    if _compiled_flex_attn is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex_attn = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex_attn

class MoT(nn.Module):
    attention_mask: ClassVar[Optional['BlockMask']] = None
    attention_mask_seq_len: ClassVar[Optional[int]] = None

    def __init__(self, mixtures: Dict[str, nn.Module], mot_checkpoint_mixed_attn: bool=True, attn_mode: str='sdpa', hbridge_enabled: bool=False, hbridge_bottom_ratio: float=0.25, hbridge_top_ratio: float=0.25):
        super().__init__()
        if attn_mode not in ('sdpa', 'flex'):
            raise ValueError(f"`attn_mode` must be 'sdpa' or 'flex', got {attn_mode!r}")
        if not mixtures:
            raise ValueError('`mixtures` cannot be empty.')
        if 'video' not in mixtures or 'action' not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")
        if not 0.0 <= hbridge_bottom_ratio <= 1.0:
            raise ValueError(f'`hbridge_bottom_ratio` must be in [0,1], got {hbridge_bottom_ratio}')
        if not 0.0 <= hbridge_top_ratio <= 1.0:
            raise ValueError(f'`hbridge_top_ratio` must be in [0,1], got {hbridge_top_ratio}')
        if hbridge_bottom_ratio + hbridge_top_ratio > 1.0:
            raise ValueError(f'`hbridge_bottom_ratio + hbridge_top_ratio` must be <= 1.0, got {hbridge_bottom_ratio} + {hbridge_top_ratio}')
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.checkpoint_full_layers = False
        self.ffn_chunk_size = 0
        self.attn_mode = attn_mode
        self.hbridge_enabled = hbridge_enabled
        self.hbridge_bottom_ratio = hbridge_bottom_ratio
        self.hbridge_top_ratio = hbridge_top_ratio
        if mot_checkpoint_mixed_attn:
            logger.info('Using gradient checkpointing for mixture attention. This will save memory but use more computation.')
        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim
        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(f'All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}')
            if expert.num_heads != self.num_heads:
                raise ValueError(f'All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}')
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(f'All experts must have same attn_head_dim; got {self.attn_head_dim} and {expert.attn_head_dim}')
        logger.info(f'Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}, attn_mode={self.attn_mode}')
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum((p.numel() for p in expert.parameters())) / 1000000000.0:.2f} B")
        if self.hbridge_enabled:
            n_bottom = int(self.num_layers * self.hbridge_bottom_ratio)
            n_top = int(self.num_layers * self.hbridge_top_ratio)
            n_middle = self.num_layers - n_bottom - n_top
            logger.info(f'HBridge enabled: bottom={n_bottom}, middle={n_middle}, top={n_top} layers (ratios: bottom={self.hbridge_bottom_ratio}, top={self.hbridge_top_ratio})')

    def _is_outer_layer(self, layer_idx: int) -> bool:
        if not self.hbridge_enabled:
            return False
        n_bottom = int(self.num_layers * self.hbridge_bottom_ratio)
        n_top = int(self.num_layers * self.hbridge_top_ratio)
        return layer_idx < n_bottom or layer_idx >= self.num_layers - n_top

    def set_attn_mode(self, mode: str):
        if mode not in ('sdpa', 'flex'):
            raise ValueError(f"`mode` must be 'sdpa' or 'flex', got {mode!r}")
        self.attn_mode = mode

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2), shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2))
        return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def _mixed_attention(self, q_cat: torch.Tensor, k_cat: torch.Tensor, v_cat: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if self.attn_mode == 'flex' and MoT.attention_mask is not None and (attention_mask is None):
            actual_seq = q_cat.shape[1]
            mask_total = MoT.attention_mask.shape[-1]
            pad_len = mask_total - actual_seq

            def _forward_flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                if pad_len > 0:
                    q = F.pad(q, (0, 0, 0, pad_len))
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                q4d = rearrange(q, 'b s (n d) -> b n s d', n=self.num_heads)
                k4d = rearrange(k, 'b s (n d) -> b n s d', n=self.num_heads)
                v4d = rearrange(v, 'b s (n d) -> b n s d', n=self.num_heads)
                half_dtypes = (torch.float16, torch.bfloat16)
                if q4d.dtype not in half_dtypes:
                    q4d, k4d, v4d = (q4d.to(torch.bfloat16), k4d.to(torch.bfloat16), v4d.to(torch.bfloat16))
                out = _get_compiled_flex_attn()(q4d, k4d, v4d, block_mask=MoT.attention_mask, kernel_options={'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_M1': 32, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 32})
                out = rearrange(out, 'b n s d -> b s (n d)')
                if pad_len > 0:
                    out = out[:, :actual_seq, :]
                return out
            if self.mot_checkpoint_mixed_attn and self.training and (not self.checkpoint_full_layers):
                return torch.utils.checkpoint.checkpoint(_forward_flex, q_cat, k_cat, v_cat, use_reentrant=False)
            return _forward_flex(q_cat, k_cat, v_cat)
        attn_mask = attention_mask.to(device=q_cat.device) if attention_mask is not None else None

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)
        if self.mot_checkpoint_mixed_attn and self.training and (not self.checkpoint_full_layers):
            return torch.utils.checkpoint.checkpoint(_forward, q_cat, k_cat, v_cat, use_reentrant=False)
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(block, residual_x: torch.Tensor, mixed_attn_out: torch.Tensor, gate_msa: torch.Tensor, shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor, context_payload: Optional[dict], ffn_chunk_size: int=0) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get('context')
            if context is not None:
                context_mask = context_payload.get('mask')
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        if ffn_chunk_size > 0 and mlp_input.shape[1] > ffn_chunk_size:
            mlp_output = torch.cat([block.ffn(part) for part in mlp_input.split(ffn_chunk_size, dim=1)], dim=1)
        else:
            mlp_output = block.ffn(mlp_input)
        x = block.gate(x, gate_mlp, mlp_output)
        return x

    def _build_expert_attention_io(self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        use_gradient_checkpointing = bool(getattr(expert, 'use_gradient_checkpointing', False))
        return (q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing)

    def _apply_post_with_optional_checkpoint(self, block, residual_x: torch.Tensor, gate_msa: torch.Tensor, shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor, use_gradient_checkpointing: bool, mixed_slice: torch.Tensor, context_payload: Optional[dict]) -> torch.Tensor:

        def _post_fn(_mixed_slice: torch.Tensor, _x: torch.Tensor, _gate_msa: torch.Tensor, _shift_mlp: torch.Tensor, _scale_mlp: torch.Tensor, _gate_mlp: torch.Tensor, _block=block, _context_payload=context_payload) -> torch.Tensor:
            return self._apply_expert_post_block(block=_block, residual_x=_x, mixed_attn_out=_mixed_slice, gate_msa=_gate_msa, shift_mlp=_shift_mlp, scale_mlp=_scale_mlp, gate_mlp=_gate_mlp, context_payload=_context_payload, ffn_chunk_size=self.ffn_chunk_size if self.training else 0)
        if use_gradient_checkpointing and self.training and (not self.checkpoint_full_layers):
            return torch.utils.checkpoint.checkpoint(_post_fn, mixed_slice, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_reentrant=False)
        return _post_fn(mixed_slice, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def _forward_joint_inner(self, video_tokens: torch.Tensor, action_tokens: torch.Tensor, video_freqs: torch.Tensor, action_freqs: torch.Tensor, video_t_mod: torch.Tensor, action_t_mod: torch.Tensor, video_context_payload: Optional[dict], action_context_payload: Optional[dict], attention_mask: Optional[torch.Tensor], sub_stream_lens: Optional[list[int]]=None, sub_stream_self_masks: Optional[list[torch.Tensor]]=None) -> tuple[torch.Tensor, torch.Tensor]:
        video_expert = self.mixtures['video']
        action_expert = self.mixtures['action']
        hbridge_active = self.hbridge_enabled and sub_stream_lens is not None and (sub_stream_self_masks is not None)
        Sv = video_tokens.shape[1]
        Sa = action_tokens.shape[1]
        x_video = video_tokens
        x_action = action_tokens
        for layer_idx in range(self.num_layers):
            v_block = video_expert.blocks[layer_idx]
            a_block = action_expert.blocks[layer_idx]
            qv, kv, vv, residual_v, gate_msa_v, shift_mlp_v, scale_mlp_v, gate_mlp_v, _ckpt_v = self._build_expert_attention_io(expert=video_expert, block=v_block, x=x_video, freqs=video_freqs, t_mod=video_t_mod)
            qa, ka, va, residual_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a, _ckpt_a = self._build_expert_attention_io(expert=action_expert, block=a_block, x=x_action, freqs=action_freqs, t_mod=action_t_mod)
            q_cat = torch.cat([qv, qa], dim=1)
            k_cat = torch.cat([kv, ka], dim=1)
            v_cat = torch.cat([vv, va], dim=1)
            if hbridge_active and self._is_outer_layer(layer_idx):
                mixed_chunks = []
                offset = 0
                for sub_len, sub_mask in zip(sub_stream_lens, sub_stream_self_masks):
                    q_sub = q_cat[:, offset:offset + sub_len, :]
                    k_sub = k_cat[:, offset:offset + sub_len, :]
                    v_sub = v_cat[:, offset:offset + sub_len, :]
                    mixed_chunks.append(self._mixed_attention(q_cat=q_sub, k_cat=k_sub, v_cat=v_sub, attention_mask=sub_mask))
                    offset += sub_len
                mixed = torch.cat(mixed_chunks, dim=1)
            else:
                mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)
            mixed_v = mixed[:, :Sv, :]
            mixed_a = mixed[:, Sv:Sv + Sa, :]
            x_video = self._apply_expert_post_block(block=v_block, residual_x=residual_v, mixed_attn_out=mixed_v, gate_msa=gate_msa_v, shift_mlp=shift_mlp_v, scale_mlp=scale_mlp_v, gate_mlp=gate_mlp_v, context_payload=video_context_payload)
            x_action = self._apply_expert_post_block(block=a_block, residual_x=residual_a, mixed_attn_out=mixed_a, gate_msa=gate_msa_a, shift_mlp=shift_mlp_a, scale_mlp=scale_mlp_a, gate_mlp=gate_mlp_a, context_payload=action_context_payload)
        return (x_video, x_action)

    def forward(self, embeds_all: Dict[str, torch.Tensor], attention_mask: Optional[torch.Tensor], freqs_all: Dict[str, torch.Tensor], context_all: Dict[str, Optional[dict]], t_mod_all: Dict[str, torch.Tensor], sub_stream_lens: Optional[list[int]]=None, sub_stream_self_masks: Optional[list[torch.Tensor]]=None):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f'Missing expert tokens for {missing}')
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f'Missing expert freqs for {missing}')
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f'Missing expert t_mod for {missing}')
        if attention_mask is not None:
            if attention_mask.ndim not in (2, 3):
                raise ValueError(f'`attention_mask` must be 2D [S, S] or 3D [B, S, S], got shape {tuple(attention_mask.shape)}')
            if attention_mask.shape[-1] != attention_mask.shape[-2]:
                raise ValueError(f'`attention_mask` must be square in trailing dims, got shape {tuple(attention_mask.shape)}')
        elif self.attn_mode != 'flex' or MoT.attention_mask is None:
            raise ValueError("`attention_mask` is None but attn_mode is not 'flex' or MoT.attention_mask is not set. Call init_flex_mask() or pass a 2D bool mask.")
        hbridge_active = self.hbridge_enabled and sub_stream_lens is not None and (sub_stream_self_masks is not None)
        if hbridge_active:
            if len(sub_stream_lens) != len(sub_stream_self_masks):
                raise ValueError(f'`sub_stream_lens` ({len(sub_stream_lens)}) and `sub_stream_self_masks` ({len(sub_stream_self_masks)}) must have same length.')
            for i, (slen, smask) in enumerate(zip(sub_stream_lens, sub_stream_self_masks)):
                if smask.ndim not in (2, 3) or smask.shape[-1] != slen or smask.shape[-2] != slen:
                    raise ValueError(f'`sub_stream_self_masks[{i}]` must be 2D [S,S] or 3D [B,S,S] of size {slen}, got shape {tuple(smask.shape)}')
        tokens_all = {k: v for k, v in embeds_all.items()}
        for layer_idx in range(self.num_layers):
            args = (layer_idx, tokens_all, attention_mask, freqs_all, context_all, t_mod_all, sub_stream_lens, sub_stream_self_masks, hbridge_active)
            if self.checkpoint_full_layers and self.training and torch.is_grad_enabled():
                tokens_all = torch.utils.checkpoint.checkpoint(self._forward_layer, *args, use_reentrant=False)
            else:
                tokens_all = self._forward_layer(*args)
        return tokens_all

    def _forward_layer(self, layer_idx, tokens_all, attention_mask, freqs_all, context_all, t_mod_all, sub_stream_lens, sub_stream_self_masks, hbridge_active):
        tokens_all = dict(tokens_all)
        q_chunks = []
        k_chunks = []
        v_chunks = []
        cached = {}
        seq_lens = []
        for name in self.expert_order:
            expert = self.mixtures[name]
            block = expert.blocks[layer_idx]
            x = tokens_all[name]
            freqs = freqs_all[name]
            t_mod = t_mod_all[name]
            q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing = self._build_expert_attention_io(expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod)
            q_chunks.append(q)
            k_chunks.append(k)
            v_chunks.append(v)
            seq_lens.append(x.shape[1])
            cached[name] = {'block': block, 'residual_x': residual_x, 'gate_msa': gate_msa, 'shift_mlp': shift_mlp, 'scale_mlp': scale_mlp, 'gate_mlp': gate_mlp, 'use_gradient_checkpointing': use_gradient_checkpointing}
        q_cat = torch.cat(q_chunks, dim=1)
        k_cat = torch.cat(k_chunks, dim=1)
        v_cat = torch.cat(v_chunks, dim=1)
        total_seq = q_cat.shape[1]
        if attention_mask is not None and attention_mask.shape[-1] != total_seq:
            raise ValueError(f'Attention mask seq length mismatch: mask={attention_mask.shape[-1]} vs tokens={total_seq}')
        if hbridge_active and sum(sub_stream_lens) != total_seq:
            raise ValueError(f'`sub_stream_lens` sum must equal total token length: sum={sum(sub_stream_lens)} vs total={total_seq}')
        if hbridge_active and self._is_outer_layer(layer_idx):
            mixed_chunks = []
            offset = 0
            for sub_len, sub_mask in zip(sub_stream_lens, sub_stream_self_masks):
                q_sub = q_cat[:, offset:offset + sub_len, :]
                k_sub = k_cat[:, offset:offset + sub_len, :]
                v_sub = v_cat[:, offset:offset + sub_len, :]
                mixed_chunks.append(self._mixed_attention(q_cat=q_sub, k_cat=k_sub, v_cat=v_sub, attention_mask=sub_mask))
                offset += sub_len
            mixed = torch.cat(mixed_chunks, dim=1)
        else:
            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)
        start = 0
        for name, seq_len in zip(self.expert_order, seq_lens):
            end = start + seq_len
            mixed_slice = mixed[:, start:end, :]
            cached_expert = cached[name]
            block = cached_expert['block']
            context_payload = context_all.get(name)
            updated_tokens = self._apply_post_with_optional_checkpoint(block=block, residual_x=cached_expert['residual_x'], gate_msa=cached_expert['gate_msa'], shift_mlp=cached_expert['shift_mlp'], scale_mlp=cached_expert['scale_mlp'], gate_mlp=cached_expert['gate_mlp'], use_gradient_checkpointing=cached_expert['use_gradient_checkpointing'], mixed_slice=mixed_slice, context_payload=context_payload)
            tokens_all[name] = updated_tokens
            start = end
        return tokens_all
