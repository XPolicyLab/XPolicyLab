# Copyright (C) 2026 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import torch
import torch.nn as nn

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from src.models.VLM import qwen3vl_expert
from src.models.VLM.qwen3vl_expert import Qwen3VLForConditionalGeneration
from src.models.VLM.qwen3vl_lce import lce_forward as qwen3_vl_lce_forward
from src.models.policy_head.DiT import DiT, TimestepEmbedder
from src.models.projector.linear_projector import MLPProjector
from src.scheduler.flow import RectifiedFlow
from src.utils.model_utils import auto_cast


qwen3vl_expert.apply_rotary_pos_emb = liger_rotary_pos_emb
qwen3vl_expert.Qwen3VLTextRMSNorm = LigerRMSNorm
qwen3vl_expert.Qwen3VLForConditionalGeneration.forward = qwen3_vl_lce_forward


class XR1(nn.Module):
    def __init__(
        self,
        state_shape=(1, 16),
        action_shape=(30, 29),
        n_choices=5,
        dit_num_layers=36,
        dit_hidden_size=1024,
        dit_is_causal=True,
        num_steps=5,
        freeze_vlm=False,
        enable_dct=False,
        knowledge_insulation=True,
        ffn_gradient_checkpointing=False,
        pretrained_model="Qwen/Qwen3-VL-4B-Instruct",
    ):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.n_choices = n_choices
        self.dit_num_layers = dit_num_layers
        self.dit_hidden_size = dit_hidden_size
        self.dit_is_causal = dit_is_causal
        self.num_steps = num_steps
        self.freeze_vlm = freeze_vlm
        self.dct_coefficient = 1.0 if enable_dct else 0.0
        self.knowledge_insulation = knowledge_insulation
        self.ffn_gradient_checkpointing = ffn_gradient_checkpointing
        self.pretrained_model = pretrained_model

        self._build_model()
        self.scheduler = RectifiedFlow()

    def _build_model(self):
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.pretrained_model,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        ).eval()
        self.vlm.requires_grad_(False)
        self.vlm.model.get_input_embeddings().requires_grad_(False)
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size

        self.state_projector_choice = MLPProjector(
            input_dim=self.state_shape[-1],
            inter_dim=self.vlm_hidden_size,
            output_dim=self.vlm_hidden_size,
            num_layers=2,
        )
        self.action_projector_choice = nn.Sequential(
            MLPProjector(
                input_dim=self.vlm_hidden_size,
                inter_dim=self.vlm_hidden_size,
                output_dim=self.vlm_hidden_size,
                num_layers=4,
            ),
            MLPProjector(
                input_dim=self.vlm_hidden_size,
                output_dim=self.action_shape[-1] * self.n_choices,
                num_layers=1,
            ),
        )
        self.score_projector_choice = nn.Sequential(
            MLPProjector(
                input_dim=self.vlm_hidden_size,
                inter_dim=self.vlm_hidden_size,
                output_dim=self.vlm_hidden_size,
                num_layers=4,
            ),
            MLPProjector(
                input_dim=self.vlm_hidden_size,
                output_dim=self.n_choices,
                num_layers=1,
            ),
        )

        self.dit = DiT(
            hidden_size=self.dit_hidden_size,
            kv_heads=8,
            layer_num=self.dit_num_layers,
            is_causal=self.dit_is_causal,
        )
        self.state_projector = MLPProjector(
            input_dim=self.state_shape[-1],
            inter_dim=self.dit_hidden_size,
            output_dim=self.dit_hidden_size,
            num_layers=2,
        )
        self.action_projector = MLPProjector(
            input_dim=self.action_shape[-1],
            inter_dim=self.dit_hidden_size,
            output_dim=self.dit_hidden_size,
            num_layers=2,
        )
        self.action_output_layer = MLPProjector(
            input_dim=self.dit_hidden_size,
            inter_dim=self.dit_hidden_size,
            output_dim=self.action_shape[-1],
            num_layers=2,
        )

        self.t_embedder = TimestepEmbedder(self.dit_hidden_size)
        self.t_projector = MLPProjector(
            input_dim=self.dit_hidden_size,
            output_dim=6 * self.dit_hidden_size,
            bias=True,
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(Qwen3VLTextConfig.from_pretrained(self.pretrained_model))
        self.sink = nn.Embedding(1, self.dit_hidden_size)
        self.to(torch.bfloat16)

    def get_action_vlm_condition_segments(self, batch):
        if self.knowledge_insulation:
            return batch.pop("action_vlm_condition_segments", None)
        return batch.pop("action_segments", None)

    def get_action_input(self, batch):
        device = batch["input_ids"].device
        if "action" in batch:
            action = batch.pop("action")
            action_mask = batch.pop("action_mask", None)
            state = batch.pop("state")
        else:
            action = torch.zeros((1, *self.action_shape), device=device, dtype=torch.bfloat16)
            action_mask = torch.zeros_like(action, dtype=torch.int32)
            state = torch.zeros((1, *self.state_shape), device=device, dtype=torch.bfloat16)
        return action, action_mask, state

    def unpad(self, past_key_values, position_ids, attention_mask, action_bs, action_segments):
        if action_segments is None:
            return past_key_values, position_ids, attention_mask

        all_keys = torch.stack([layer[0] for layer in past_key_values], dim=0)
        all_values = torch.stack([layer[1] for layer in past_key_values], dim=0)
        seq_len = action_segments[:, 1] - action_segments[:, 0]
        if len(seq_len) != action_bs:
            raise ValueError(f"Expected {action_bs} action segment(s), got {len(seq_len)}")
        max_length = int(seq_len.max().item())
        num_layers = all_keys.size(0)
        num_heads = all_keys.size(2)
        head_dim = all_keys.size(4)
        new_keys = torch.zeros(
            (num_layers, action_bs, num_heads, max_length, head_dim),
            device=all_keys.device,
            dtype=all_keys.dtype,
        )
        new_values = torch.zeros_like(new_keys)
        new_position_ids = torch.zeros((3, action_bs, max_length), device=position_ids.device, dtype=position_ids.dtype)
        new_attention_mask = torch.zeros((action_bs, max_length), device=all_keys.device, dtype=torch.int)
        for idx in range(action_bs):
            start = int(action_segments[idx, 0].item())
            length = int(seq_len[idx].item())
            end = start + length
            new_keys[:, idx, :, :length] = all_keys[:, 0, :, start:end]
            new_values[:, idx, :, :length] = all_values[:, 0, :, start:end]
            new_position_ids[:, idx, :length] = position_ids[:, 0, start:end]
            new_attention_mask[idx, :length] = 1

        return list(zip(new_keys, new_values)), new_position_ids, new_attention_mask

    def dit_forward(self, noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, perf_attn_mask):
        t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        noisy_action = self.action_projector(noisy_action * action_mask)
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()
        hidden_states = self.dit.forward_perf(
            hidden_states,
            past_key_values,
            perf_attn_mask,
            position_embeds,
            t_embeds,
            repeat=1,
        )
        return self.action_output_layer(hidden_states[:, -noisy_action.shape[1] :, :])

    @torch.no_grad()
    def generate(self, batch):
        return self.forward(batch, return_loss=False)

    @auto_cast
    def forward(self, batch, return_loss=False):
        if return_loss:
            raise RuntimeError("This challenge branch only supports evaluation-time inference.")
        if self.training:
            raise RuntimeError("XR1 must be in eval() mode for challenge inference.")

        action_vlm_condition_segments = self.get_action_vlm_condition_segments(batch)
        batch.pop("vlm_action_actual_length", None)
        action, action_mask, state = self.get_action_input(batch)
        batch.pop("vlm_action_target", None)
        batch.pop("vlm_action_mask", None)

        vlm_state = state.flatten(0, 1)
        vlm_state = vlm_state.to(dtype=next(self.state_projector_choice.parameters()).dtype)
        batch["state_embeds"] = self.state_projector_choice(vlm_state)
        vlm_outputs = self.vlm(**batch, use_cache=True)

        action_bs, action_length, _ = action.shape
        _, state_length, _ = state.shape
        q_len = action_length + state_length + 1

        vlm_outputs.past_key_values, vlm_outputs.position_ids, vlm_outputs.attention_mask = self.unpad(
            vlm_outputs.past_key_values,
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            action_bs,
            action_vlm_condition_segments,
        )

        position_ids = (
            torch.arange(0, q_len, device=action.device).view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        position_embeds = self.rotary_emb(action, position_ids)

        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, q_len, -1)
        if self.dit_is_causal:
            causal_mask = torch.tril(torch.ones(action_bs, q_len, q_len, device=action.device), diagonal=0)
        else:
            causal_mask = torch.full((action_bs, q_len, q_len), True, device=action.device)
        perf_attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()

        state_embed = self.state_projector(state)
        noise = torch.randn_like(action)
        return self.scheduler.generate(
            noise,
            num_steps=self.num_steps,
            forward_func=self.dit_forward,
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=position_embeds,
            past_key_values=vlm_outputs.past_key_values,
            perf_attn_mask=perf_attn_mask,
        )
