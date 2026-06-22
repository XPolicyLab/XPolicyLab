"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import re
import math
from typing import List, Tuple
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
# from a1.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

from a1.vla.dit.model import DiT
# from transformers import GemmaForCausalLM, GemmaConfig
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers.cache_utils import DynamicCache
import contextlib
from a1.vla.util import make_att_2d_masks, prepare_attention_bias_4d



class FlowMatchingActionHead(nn.Module):
    """
    Flow Matching action head with cross-attention to prefix memory.

    - prefix_hidden: (B, P, llm_dim) provided by main LLM (Molmo)
    - expert runs in a smaller dimension and attends to projected prefix memory
    - training: used with main LLM to build prefix memory; inference: expert only per step
    """

    def __init__(
        self,
        llm_dim: int = 4096,
        action_dim: int = 7,
        proprio_dim: int = 8,
        horizon: int = 8,
        qwen2_hidden_size: int = 896,
        qwen2_num_layers: int = 18,
        qwen2_num_heads: int = 8,
        qwen2_intermediate_size: int = 4096,
        qwen2_num_kv_heads: int | None = None,
        pvf_func: str = "2d_attn_mask",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.horizon = horizon
        self.qwen2_hidden = qwen2_hidden_size
        self.qwen2_num_layers = qwen2_num_layers
        self.qwen2_num_kv_heads = qwen2_num_kv_heads or qwen2_num_heads
        self.pvf_func = pvf_func

        self.time_encoder = SinusoidalPositionalEncoding(dim=self.qwen2_hidden)

        # Project inputs to Qwen2 hidden size
        self.state_proj = nn.Linear(proprio_dim, self.qwen2_hidden)
        self.action_in_proj = nn.Linear(action_dim, self.qwen2_hidden)
        self.action_time_in = nn.Linear(2 * self.qwen2_hidden, self.qwen2_hidden)
        self.action_time_out = nn.Linear(self.qwen2_hidden, self.qwen2_hidden)

        qwen2_cfg = Qwen2Config(
            # vocab_size=151936,
            hidden_size=qwen2_hidden_size,
            num_hidden_layers=qwen2_num_layers,
            num_attention_heads=qwen2_num_heads,
            intermediate_size=qwen2_intermediate_size,
            # KV heads: prioritize explicitly passed qwen2_num_kv_heads,
            # if not specified, keep consistent with attention heads
            num_key_value_heads=self.qwen2_num_kv_heads,
        )
        # self.gemma = GemmaForCausalLM(config=gemma_cfg)
        self.qwen2 = Qwen2ForCausalLM(config=qwen2_cfg)
        # remove token embeddings; we use inputs_embeds directly
        if hasattr(self.qwen2.model, "embed_tokens"):
            self.qwen2.model.embed_tokens = None

        # Output vector field to action space
        self.action_out = MLPResNet(num_blocks=2, input_dim=self.qwen2_hidden, hidden_dim=self.qwen2_hidden, output_dim=action_dim)
        # self.action_out = MLPResNet(num_blocks=1, input_dim=self.qwen2_hidden, hidden_dim=self.qwen2_hidden, output_dim=action_dim)

    def build_suffix_tokens(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # state: (B, A) or (B, 1, A), x_t: (B, T, A), t: (B,)
        B, T, _ = x_t.shape
        # Unify dtype to that used by linear layer weights to avoid matmul dtype conflicts (e.g., bf16 input x f32 weights)
        head_dtype = self.state_proj.weight.dtype
        device = x_t.device
        # project state to one token
        # print('state',state.shape)
        if state.dim() == 2:
            state_tok = self.state_proj(state.to(head_dtype)).unsqueeze(1)  # (B,1,H)
        elif state.dim() == 3:
            if state.shape[1] == 1:
                state_tok = self.state_proj(state[:, 0, :].to(head_dtype)).unsqueeze(1)  # (B,1,H)
            else:
                state_tok = self.state_proj(state[:, 0, :].to(head_dtype)).unsqueeze(1)  # fallback first token
        elif state.dim() == 4:
            # Accept (B,1,1,A) by squeezing the middle singleton dims
            if state.shape[1] == 1 and state.shape[2] == 1:
                state_tok = self.state_proj(state[:, 0, 0, :].to(head_dtype)).unsqueeze(1)  # (B,1,H)
            else:
                raise ValueError(f"Unsupported state dim: {state.shape}")
        else:
            raise ValueError(f"Unsupported state dim: {state.shape}")
        # print('t',t.shape)
        t_emb = self.time_encoder(t.to(device)).to(head_dtype)  # (B,D)
        # print('t_emb',t_emb.shape)
        t_tok = t_emb.unsqueeze(1).expand(-1, T, -1)  # (B,T,D)
        # print('t_tok',t_tok.shape)
        a_tok = self.action_in_proj(x_t.to(head_dtype))  # (B,T,D)
        # print('a_tok',a_tok.shape)
        at = torch.cat([a_tok, t_tok.to(head_dtype)], dim=-1)
        # print('at',at.shape)
        at = self.action_time_in(at.to(head_dtype))
        # print('at',at.shape)
        at = torch.nn.functional.silu(at)
        # print('at',at.shape)
        at = self.action_time_out(at.to(head_dtype))  # (B,T,D)
        # print('at',at.shape)
        suffix = torch.cat([state_tok.to(head_dtype), at], dim=1)  # (B,1+T,D)
        return suffix

    def predict_vector_field(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], state: torch.Tensor, 
                           x_t: torch.Tensor, t: torch.Tensor, 
                           pos_offset: torch.Tensor | None = None) -> torch.Tensor:
        if self.pvf_func == "2d_attn_mask":
            return self.predict_vector_field_2d_attn_mask(past_key_values, state, x_t, t, pos_offset)
        elif self.pvf_func == "4d_attn_mask":
            return self.predict_vector_field_4d_attn_mask(past_key_values, state, x_t, t, pos_offset)
        else:
            raise ValueError(f"Unsupported pvf function: {self.pvf_func}")

    def predict_vector_field_2d_attn_mask(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], state: torch.Tensor, 
                           x_t: torch.Tensor, t: torch.Tensor, 
                           pos_offset: torch.Tensor | None = None) -> torch.Tensor:

        # Unify dtype to qwen2 parameter dtype
        qwen2_dtype =  next(self.qwen2.parameters()).dtype
        tgt = self.build_suffix_tokens(state, x_t, t).to(qwen2_dtype)  # (B,1+T,Hg)

        batch_size, tgt_len = tgt.shape[:2]
 
        # Compute standard 2D attention_mask (1 for valid tokens, 0 for padding)
        B, L = tgt.shape[:2]  # L = 1 + horizon
        past_len_for_mask = 0
        if past_key_values is not None and len(past_key_values) > 0:
            pk = past_key_values[0][0]
            if pk.dim() >= 3:
                past_len_for_mask = int(pk.shape[-2])

        if past_len_for_mask <= 0:
            raise ValueError("past_key_values is None or empty")

        # Remove padding inside prefix based on pos_offset: valid prefix segment is [0, valid_length), set remaining padding to 0; suffix all 1
        # prefix mask: shape (B, past_len_for_mask)
        prefix_positions = torch.arange(past_len_for_mask, device=tgt.device).unsqueeze(0).expand(B, -1)  # (B, past_len)
        # - (B,): valid prefix length for each sample
        # - (B, past_len): 0/1 mask generated from input_ids, need to sum along dim=1 first to get valid length
        if pos_offset.dim() == 1:
                valid_prefix_lengths = pos_offset.to(device=tgt.device, dtype=torch.long)
        # elif pos_offset.dim() == 2:
        #     valid_prefix_lengths = pos_offset.to(device=tgt.device, dtype=torch.long).sum(dim=1)
        else:
            raise ValueError(f"Unsupported pos_offset shape for mask: {tuple(pos_offset.shape)}")
        valid_prefix_lengths = valid_prefix_lengths.clamp_min(0).clamp_max(past_len_for_mask).view(B, 1)
        prefix_mask = (prefix_positions < valid_prefix_lengths).to(torch.int64)  # 1 for valid, 0 for padding
        # suffix mask: 后缀均为有效
        suffix_mask = torch.ones((B, L), dtype=torch.int64, device=tgt.device)
        attention_mask_2d = torch.cat([prefix_mask, suffix_mask], dim=1)  # (B, past_len_for_mask + L)
        
        # Compatible with new HF cache: Gemma expects Cache object (with get_seq_length)
        # If legacy list[(k,v), ...] is passed, convert to DynamicCache
        pkv_for_qwen2 = past_key_values
        if past_key_values is not None and not hasattr(past_key_values, "get_seq_length"):
            pkv_for_qwen2 = DynamicCache.from_legacy_cache(past_key_values)
        
        # Dynamically set qwen2 model inference layers for early exit training
        self.qwen2.model.config.num_hidden_layers = len(past_key_values)

        outputs = self.qwen2.model(
            inputs_embeds=tgt,
            attention_mask=attention_mask_2d,
            # position_ids=position_ids,
            use_cache=False,
            past_key_values=pkv_for_qwen2,
        )
        # Restore qwen2 model inference layers
        self.qwen2.model.config.num_hidden_layers = self.qwen2_num_layers
        
        h = outputs.last_hidden_state[:, -self.horizon:, :]
        v_pred = self.action_out(h)  # (B,T,A)
        return v_pred.to(x_t.dtype)

    def predict_vector_field_4d_attn_mask(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], state: torch.Tensor, 
                           x_t: torch.Tensor, t: torch.Tensor, 
                           pos_offset: torch.Tensor | None = None) -> torch.Tensor:

        # Unify dtype to qwen2 parameter dtype 
        qwen2_dtype = next(self.qwen2.parameters()).dtype
        # memory = self.memory_proj(prefix_hidden.to(gemma_dtype))  # (B,P,Hg)
        tgt = self.build_suffix_tokens(state, x_t, t).to(qwen2_dtype)  # (B,1+T,Hg)
            
        # seq = torch.cat([memory, tgt], dim=1)  # (B,P+1+T,Hg)
        # 正确设置 position_ids：需在 prefix 长度基础上递增，匹配缓存的 KV 长度
        batch_size, tgt_len = tgt.shape[:2]
        # 优先使用 caller 提供的有效长度偏移（忽略 padding 的有效 token 数），否则回退到 KV 长度
        if pos_offset is not None:
            # pos_offset: (B,)
            base = torch.arange(tgt_len, device=tgt.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            position_ids = base + pos_offset.view(batch_size, 1)
        else:
            raise ValueError("pos_offset is None")
        # Construct openpi-style block causal mask for suffix (state+actions):
        # Block definition: state token as single block (length 1), action sequence as new whole block (length horizon).
        # Fully connected within block; due to past_key_values, suffix can naturally only see prefix (past) and its own block, satisfying "action block can see previous blocks".
        B, L = tgt.shape[:2]  # L = 1 + horizon
        pad_masks = torch.ones((B, L), dtype=torch.bool, device=tgt.device)
        att_masks = torch.zeros((B, L), dtype=torch.int32, device=tgt.device)
        # State block start
        att_masks[:, 0] = 1
        # Action block start (first action position)
        if L > 1:
            att_masks[:, 1] = 1
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        attn_bias_suffix = prepare_attention_bias_4d(att_2d)  # (B,1,L,L)

        # If prefix KV cache exists, need to append zero bias in column dimension (allow attention), forming (B,1,L,past_len+L)
        past_len_for_mask = 0
        # try:
        if past_key_values is not None and len(past_key_values) > 0:
            pk = past_key_values[0][0]
            if pk.dim() >= 3:
                past_len_for_mask = int(pk.shape[-2])

        if past_len_for_mask > 0:
            zero_bias_prefix = torch.zeros(
                (B, 1, L, past_len_for_mask), dtype=attn_bias_suffix.dtype, device=attn_bias_suffix.device
            )
            attn_mask_4d = torch.cat([zero_bias_prefix, attn_bias_suffix], dim=-1)
        else:
            # attn_mask_4d = attn_bias_suffix
            raise ValueError("past_key_values is None or empty")

        # attention_mask_2d = torch.ones((B, past_len_for_mask + L), dtype=torch.int64, device=tgt.device)
        
        # Compatible with new HF cache: Gemma expects Cache object (with get_seq_length)
        # If legacy list[(k,v), ...] is passed, convert to DynamicCache
        pkv_for_qwen2 = past_key_values
        if past_key_values is not None and not hasattr(past_key_values, "get_seq_length"):
            pkv_for_qwen2 = DynamicCache.from_legacy_cache(past_key_values)
        # Dynamically set qwen2 model inference layers for early exit training
        self.qwen2.model.config.num_hidden_layers = len(past_key_values)

        outputs = self.qwen2.model(
            inputs_embeds=tgt,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            use_cache=False,
            past_key_values=pkv_for_qwen2,
            # attention_bias=attn_bias_suffix,
        )
        # Restore qwen2 model inference layers
        self.qwen2.model.config.num_hidden_layers = self.qwen2_num_layers
        
        h = outputs.last_hidden_state[:, -self.horizon:, :]
        v_pred = self.action_out(h)  # (B,T,A)
        return v_pred.to(x_t.dtype)

    @torch.no_grad()
    def sample_noisy_actions(self, ground_truth_actions: torch.Tensor):
        B = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        dtype = ground_truth_actions.dtype
        noise = torch.randn_like(ground_truth_actions,device=device,dtype=dtype)
        # Beta(1.5,1)
        # t = torch.distributions.Beta(torch.tensor(1.5, device=device, dtype=dtype),
        #                              torch.tensor(1.0, device=device, dtype=dtype)).sample((B,))
        # Beta(1.5,1) — 在混合精度下强制使用 float32 参数，避免 bf16/fp16 的数值不稳定
        t = torch.distributions.Beta(
            torch.tensor(1.5, device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32)
        ).sample((B,))
        # avoid value error of bf1f and amp
        t = t.to(torch.float32)
        t = (t * 0.999 + 0.001)
        t_exp = t.view(B, 1, 1)
        x_t_fp32 = t_exp * noise.to(torch.float32) + (1.0 - t_exp) * ground_truth_actions.to(torch.float32)
        x_t = x_t_fp32.to(ground_truth_actions.dtype)
        return {"noise": noise, "x_t": x_t, "t": t}


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        # print('MLPResNet forward()','*'*50,x.shape)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        # print("after layernorm1 x shape",x.shape)
        # print('layer_norm1 weight.shape',self.layer_norm1.weight.shape)
        # print('fc1 weight.shape',self.fc1.weight.shape)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        # print("fc1",x.shape)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_actions_chunk=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*action_dim, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        # print('*'*50,actions_hidden_states.shape)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        # print("L1RegressionActionHead predict_action() actions_hidden_states.shape","*"*50,actions_hidden_states.shape)
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, self.num_actions_chunk, -1) # why?
        # print("***** rearranged_actions_hidden_states.shape",rearranged_actions_hidden_states.shape)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=10, # 2 to 4
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output

class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_actions_chunk=8,
        num_diffusion_steps_train=100,
        num_diffusion_steps_inference=30,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*action_dim, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.num_diffusion_steps_inference = num_diffusion_steps_inference
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, self.num_actions_chunk, self.action_dim), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, self.num_actions_chunk, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred
    

# modified from openvla-oft
# class DiffusionActionHead(nn.Module):
#     """
#     Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

#     Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
#     """

#     def __init__(
#         self,
#         input_dim=4096,
#         hidden_dim=4096,
#         action_dim=7,
#         num_diffusion_steps=100,
#         noise_pred_model ='film',
#         film_num_layers=8,
#     ):
#         super().__init__()
#         self.action_dim = action_dim
#         if noise_pred_model == 'film':
#             self.noise_predictor = NoisePredictionModelWithFiLM(
#                 transformer_hidden_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim,num_layers=film_num_layers)
#         else:
#             self.noise_predictor = NoisePredictionModel(
#                 transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim)
#         self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps, beta_schedule="squaredcos_cap_v2")
#         self.num_diffusion_steps = num_diffusion_steps
#         self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

#     def sample_noisy_actions(self, ground_truth_actions):
#         """
#         Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
#         used as input in the noise prediction network. Returns noise, noisy actions, and the
#         corresponding diffusion timestep embeddings.
#         """
#         # ground_truth_actions: ground-truth actions
#         # - shape: (batch_size, chunk_len, action_dim)
#         batch_size = ground_truth_actions.shape[0]
#         device = ground_truth_actions.device
#         # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
#         noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
#         # Sample random diffusion timesteps (one for each action in batch).
#         timesteps = torch.randint(
#             low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
#         )
#         # Add noise to clean actions according to the magnitude at each diffusion timestep via
#         # closed-form forward diffusion.
#         noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

#         # Get diffusion timestep embeddings as well
#         diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
#         diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

#         return_dict = dict(
#             noise=noise,
#             noisy_actions=noisy_actions,
#             diffusion_timestep_embeddings=diffusion_timestep_embeddings,
#         )

#         return return_dict

#     def predict_noise(self, actions_hidden_states, noisy_actions, diffusion_timestep_embeddings):
#         """
#         Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
#         noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
#         """
#         # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
#         # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
#         batch_size = actions_hidden_states.shape[0]
#         device = actions_hidden_states.device
#         rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
#         # Get diffusion model's noise prediction.
#         noise_pred = self.noise_predictor(rearranged_actions_hidden_states,noisy_actions,diffusion_timestep_embeddings)
#         return noise_pred
 

# class FiLMBlock(nn.Module):
#     """Feature-wise Linear Modulation block"""
#     def __init__(self, dim, condition_dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim)
#         )
        
#         # FiLM parameters
#         self.film = nn.Sequential(
#             nn.Linear(condition_dim, dim * 2),
#             nn.SiLU()
#         )
        
#     def forward(self, x, condition):
#         # x: (B, seq_len, dim)
#         # condition: (B, condition_dim)
        
#         # Get FiLM parameters
#         film_params = self.film(condition)  # (B, dim * 2)
#         gamma, beta = film_params.chunk(2, dim=-1)  # (B, dim), (B, dim)
        
#         # Apply FiLM modulation
#         residual = x
#         x = self.norm(x)
#         x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)  # FiLM modulation
#         x = self.mlp(x) + residual
        
#         return x

# class NoisePredictionModelWithFiLM(nn.Module):
#     def __init__(self, transformer_hidden_dim, hidden_dim, action_dim=7, num_layers=4):
#         super().__init__()
#         # - 起始值: 4层 （平衡表达能力和计算效率）
#         # - 最小值: 2层 （保证基本表达能力）
#         # - 最大值: 8层 （避免过拟合和计算开销）
        
#         # 只对noisy_actions做输入投影
#         self.input_proj = nn.Linear(action_dim, hidden_dim)

#         self.state_proj = nn.Linear(NUM_ACTIONS_CHUNK * action_dim * hidden_dim, hidden_dim)
        
#         # 改进的condition处理：更好地融合时间步和观察信息
#         self.condition_proj = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),  # actions + timestep
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
        
#         # FiLM blocks保持不变
#         self.blocks = nn.ModuleList([
#             FiLMBlock(hidden_dim, hidden_dim)
#             for _ in range(num_layers)
#         ])
        
#         self.output_proj = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, action_dim)
#         )
        
#     def forward(self, noisy_actions, timestep_embeddings, actions_hidden_states):
#         B, chunk_len, _ = noisy_actions.shape
        
#         # B, chunk_len, transformer_hidden_dim = actions_hidden_states.shape
#         flattened = actions_hidden_states.reshape(B,-1)
#         obs_condition = self.state_proj(flattened)

#         # 改进的条件融合
#         # obs_condition = actions_hidden_states.mean(dim=1)  # (B,hidden_dim)
#         time_condition = timestep_embeddings.squeeze(1)   # (B, hidden_dim)
        
#         # 拼接而不是相加，让模型学习如何组合这些信息
#         fused_condition = torch.cat([obs_condition, time_condition], dim=-1)
#         condition = self.condition_proj(fused_condition)  # (B, hidden_dim)
        
#         # 输入处理：只使用noisy_actions
#         x = self.input_proj(noisy_actions)  # (B, chunk_len, hidden_dim)
        
#         # FiLM调制
#         for block in self.blocks:
#             x = block(x, condition)
            
#         return self.output_proj(x)


class DiffusionTransformerActionHead(nn.Module):
    def __init__(
        self,
        # input_dim=4096,
        hidden_dim=2048,
        depth=14,
        num_heads=16,
        cond_len=50,
        cond_dim=3584,  
        action_dim=7,
        action_horizon = 8,
        num_diffusion_steps=100,
        num_diffusion_inference_steps=5,
        pred_type="sample", # epsilon
        ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        self.num_diffusion_steps = num_diffusion_steps
        self.num_diffusion_inference_steps = num_diffusion_inference_steps
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=pred_type,
            clip_sample=False,
        )

        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=pred_type,
        )

        self.model = DiT(output_dim=action_dim,horizon=action_horizon,hidden_size=hidden_dim,depth=depth,
                            num_heads=num_heads,llm_state_cond_len=cond_len,llm_state_cond_dim=cond_dim,)
        self.action_adaptor = self.build_condition_adapter(
            "mlp2x_gelu", # mlp3x_gelu 
            in_features=action_dim ,out_features=hidden_dim)
        
        if cond_dim != hidden_dim:
            self.condition_adaptor = self.build_condition_adapter("mlp2x_gelu", in_features=cond_dim ,out_features=hidden_dim)
        else:
            self.condition_adaptor = nn.Identity()
            
        assert cond_dim != hidden_dim, "cond_dim must be equal to hidden_dim when do not use condition_adaptor"

        # 添加参数初始化检查
        self._verify_parameters()

    def _verify_parameters(self):
        """验证参数是否正确初始化"""
        for name, param in self.named_parameters():
            if param.numel() == 0:
                raise ValueError(f"Parameter {name} is empty: {param.shape}")
            if torch.isnan(param).any():
                raise ValueError(f"Parameter {name} contains NaN values")
        

    def sample_noisy_actions(self, target_actions):
        batch_size = target_actions.shape[0]
        device = target_actions.device
        # Sample noise that we'll add to the actions
        noise = torch.randn(target_actions.shape, dtype=target_actions.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(target_actions, noise, timesteps)
        return noise, noisy_action, timesteps


    def predict_noise_or_sample(self, noise_action,timestep,actions_hidden_states):
        target_dtype = actions_hidden_states.dtype
        # print("******* predict_noise_or_sample() target_dtype",target_dtype)
        # print("******* predict_noise_or_sample() noise_action.dtype",noise_action.dtype)

        noise_action_adapted = self.action_adaptor(noise_action)  # (B, T, D)
        llm_state_c = self.condition_adaptor( actions_hidden_states)
        # model_dtype = next(self.model.parameters()).dtype
        # print("******* predict_noise_or_sample() model_dtype",model_dtype)
        pred = self.model(noise_action_adapted, timestep,llm_state_c,)
        pred = pred.to(target_dtype)
        return pred
    

    def condition_sampling(self, actions_hidden_states):
        device = actions_hidden_states.device
        dtype = actions_hidden_states.dtype
        llm_state_c = self.condition_adaptor( actions_hidden_states)
        
        noise_action = torch.randn(
            size=(actions_hidden_states.shape[0], self.action_horizon, self.action_dim), 
            dtype=dtype, device=device)
        # action_mask = action_mask.expand(-1, self.action_horizon, -1)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_diffusion_inference_steps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # print(t)
            noise_action_adapted = self.action_adaptor(noise_action)
            
            # print(f"****** noise_action_adapted.dtype: {noise_action_adapted.dtype}, llm_state_c.dtype: {llm_state_c.dtype}")
            m_dtype = next(self.model.parameters()).dtype
            # print(f"self.model.dtype: {model_dtype},actions_hidden_states.dtype: {dtype}")
            # Predict the model output
            model_output = self.model(noise_action_adapted.to(m_dtype), t.unsqueeze(-1).to(device),llm_state_c.to(m_dtype),)
            
            # Compute previous actions: x_t -> x_t-1
            noise_action = self.noise_scheduler_sample.step(model_output, t, noise_action).prev_sample
            noise_action = noise_action.to(dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        # noise_action = noise_action * action_mask
        return noise_action


    # adapter to action input
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector