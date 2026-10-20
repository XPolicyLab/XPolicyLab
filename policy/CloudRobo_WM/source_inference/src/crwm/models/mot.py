import torch
import torch.nn as nn
from typing import Dict, Optional
from .wan_video_dit import flash_attention, modulate, rope_apply
from crwm.utils.logging_config import get_logger
logger = get_logger(__name__)

class MoT(nn.Module):

    def __init__(self, mixtures: Dict[str, nn.Module], hbridge_enabled: bool=False, hbridge_bottom_ratio: float=0.25, hbridge_top_ratio: float=0.25):
        super().__init__()
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
        self.hbridge_enabled = hbridge_enabled
        self.hbridge_bottom_ratio = hbridge_bottom_ratio
        self.hbridge_top_ratio = hbridge_top_ratio
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

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2), shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2))
        return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    @staticmethod
    def _apply_expert_post_block(block, residual_x: torch.Tensor, mixed_attn_out: torch.Tensor, gate_msa: torch.Tensor, shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor, context_payload: Optional[dict]) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get('context')
            if context is not None:
                context_mask = context_payload.get('mask')
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        mlp_output = block.ffn(mlp_input)
        x = block.gate(x, gate_mlp, mlp_output)
        return x

    def _build_expert_attention_io(self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        return (q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

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
            qv, kv, vv, residual_v, gate_msa_v, shift_mlp_v, scale_mlp_v, gate_mlp_v = self._build_expert_attention_io(expert=video_expert, block=v_block, x=x_video, freqs=video_freqs, t_mod=video_t_mod)
            qa, ka, va, residual_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self._build_expert_attention_io(expert=action_expert, block=a_block, x=x_action, freqs=action_freqs, t_mod=action_t_mod)
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

    def _mixed_attention(self, q_cat, k_cat, v_cat, attention_mask=None):
        return flash_attention(q_cat, k_cat, v_cat, self.num_heads, ctx_mask=attention_mask)
