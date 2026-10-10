"""ME-Dex-1.0 video-action-tactile model used by the evaluation runtime."""

import torch
import logging
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from wan.modules.model import sinusoidal_embedding_1d
from wan.modules.attention import flash_attention as attention
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .tactile_expert import TactileExpert, TactileExpertConfig
from .tactile_ae import TactileAE

logger = logging.getLogger(__name__)


def build_flowmatch_sigma_schedule(
    num_inference_steps: int,
    shift: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the exact shifted FM sigma nodes, including the final zero."""
    base_sigmas = torch.linspace(
        1.0,
        0.0,
        num_inference_steps + 1,
        device=device,
        dtype=dtype,
    )
    return shift * base_sigmas / (1 + (shift - 1) * base_sigmas)


@dataclass
class MEDexConfig:
    """Architecture settings required to instantiate the released checkpoint."""

    vae_path: str = ""
    wan_config_path: str = ""
    video_precision: str = "bfloat16"
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024
    action_expert_ffn_dim_multiplier: int = 4
    action_expert_norm_eps: float = 1e-5
    action_chunk_size: int = 16
    num_video_frames: int = 8
    video_height: int = 384
    video_width: int = 320
    batch_size: int = 1
    tactile_ae_checkpoint_path: str = ""
    tactile_expert_config: Optional[Dict[str, Any]] = None
    attention_topology: str = "full_joint"
    h_bridge_joint_start_layer: int = 8
    h_bridge_joint_end_layer: int = 22

    def __post_init__(self):
        """Normalize the one numeric value commonly parsed from JSON."""
        self.action_expert_norm_eps = float(self.action_expert_norm_eps)
        if self.attention_topology not in {"full_joint", "h_bridge"}:
            raise ValueError("attention_topology must be full_joint or h_bridge")
        if self.attention_topology == "h_bridge":
            if not 0 < self.h_bridge_joint_start_layer < self.h_bridge_joint_end_layer < self.num_layers:
                raise ValueError("invalid H-bridge joint layer range")

    def uses_joint_attention(self, layer_idx: int) -> bool:
        if self.attention_topology == "full_joint":
            return True
        return self.h_bridge_joint_start_layer <= layer_idx < self.h_bridge_joint_end_layer

class VideoModule(nn.Module):
    """Video processing module - handles WAN + T5 operations."""

    def __init__(
        self,
        video_model,
        dtype,
        device,
        grid_sizes,
    ):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> torch.Tensor:
        """Prepare video tokens from pre-processed noisy latent."""
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)
        return video_patched.flatten(2).transpose(1, 2)

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Pre-process T5 embeddings once for all layers."""
        text_len = self.video_model.wan_model.text_len
        padded_embeddings = []
        for embedding in language_embeddings:
            embedding = embedding[:text_len]
            padded_embeddings.append(torch.cat([
                embedding, embedding.new_zeros(text_len - embedding.shape[0], embedding.shape[1])
            ]))
        t5_context_raw = torch.stack(padded_embeddings)

        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)

        return t5_context

    def get_time_embedding(
        self,
        t_video: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WAN's time embedding using WAN's own weights."""
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()

            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))

        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor,
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """Process WAN cross attention with pre-processed T5 context."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None  # WAN uses None for fixed-length context
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out

    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for WAN (6 components)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process WAN FFN with proper AdaLN modulation."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        """Apply WAN's head + unpatchify for final video output."""
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    @staticmethod
    def _expert_self_attention(normalized_tokens, qkv_weight, norm_q, norm_k, output_projection):
        batch, token_count, _ = normalized_tokens.shape
        _, heads, _, head_dim = qkv_weight.shape
        qkv = torch.einsum("BTD,KNDE->KBTNE", normalized_tokens, qkv_weight)
        query = norm_q(qkv[0].flatten(-2)).view(batch, token_count, heads, head_dim)
        key = norm_k(qkv[1].flatten(-2)).view(batch, token_count, heads, head_dim)
        value = qkv[2].view(batch, token_count, heads, head_dim)
        return output_projection(attention(query, key, value, dtype=normalized_tokens.dtype).flatten(2))

    def process_video_action_tactile_decoupled_attention(
        self,
        video_tokens,
        action_tokens,
        tactile_tokens,
        video_adaln_modulation,
        action_adaln_modulation,
        tactile_adaln_modulation,
        layer_idx,
        action_block,
        tactile_block,
    ):
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        video_mod = video_adaln_modulation
        action_mod = action_adaln_modulation
        tactile_mod = tactile_adaln_modulation
        normalized_video = wan_layer.norm1(video_tokens).float() * (1 + video_mod[1].squeeze(2)) + video_mod[0].squeeze(2)
        normalized_action = action_block.norm1(action_tokens).float() * (1 + action_mod[1].squeeze(2)) + action_mod[0].squeeze(2)
        normalized_tactile = tactile_block.norm1(tactile_tokens).float() * (1 + tactile_mod[1].squeeze(2)) + tactile_mod[0].squeeze(2)
        batch, video_len, _ = normalized_video.shape
        video_seq_lens = torch.full((batch,), video_len, dtype=torch.long, device=video_tokens.device)
        freqs = self.video_model.wan_model.freqs
        if freqs.device != video_tokens.device:
            freqs = freqs.to(video_tokens.device)
        video_output = wan_layer.self_attn(normalized_video, video_seq_lens, self.grid_sizes, freqs)
        action_output = self._expert_self_attention(
            normalized_action,
            action_block.wan_action_qkv,
            action_block.wan_action_norm_q,
            action_block.wan_action_norm_k,
            action_block.wan_action_o,
        )
        tactile_output = self._expert_self_attention(
            normalized_tactile,
            tactile_block.wan_tactile_qkv,
            tactile_block.wan_tactile_norm_q,
            tactile_block.wan_tactile_norm_k,
            tactile_block.wan_tactile_o,
        )
        return (
            video_tokens + video_output * video_mod[2].squeeze(2),
            action_tokens + action_output * action_mod[2].squeeze(2),
            tactile_tokens + tactile_output * tactile_mod[2].squeeze(2),
        )

    def process_video_action_tactile_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        tactile_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        tactile_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ME-Dex-1.0 all-to-all Video-Action-Tactile attention in one WAN call."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation
        t_mod = tactile_adaln_modulation

        norm_video = wan_layer.norm1(video_tokens).float() * (
            1 + v_mod[1].squeeze(2)
        ) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens).float() * (
            1 + a_mod[1].squeeze(2)
        ) + a_mod[0].squeeze(2)
        norm_tactile = tactile_block.norm1(tactile_tokens).float() * (
            1 + t_mod[1].squeeze(2)
        ) + t_mod[0].squeeze(2)

        batch, video_len, channels = norm_video.shape
        action_len = norm_action.shape[1]
        tactile_len = norm_tactile.shape[1]
        heads = self.video_model.wan_model.num_heads
        head_dim = channels // heads

        action_qkv = torch.einsum(
            "BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv
        )
        action_q = action_block.wan_action_norm_q(
            action_qkv[0].flatten(-2)
        ).view(batch, action_len, heads, head_dim)
        action_k = action_block.wan_action_norm_k(
            action_qkv[1].flatten(-2)
        ).view(batch, action_len, heads, head_dim)
        action_v = action_qkv[2].view(batch, action_len, heads, head_dim)

        tactile_qkv = torch.einsum(
            "BTD,KNDE->KBTNE", norm_tactile, tactile_block.wan_tactile_qkv
        )
        tactile_q = tactile_block.wan_tactile_norm_q(
            tactile_qkv[0].flatten(-2)
        ).view(batch, tactile_len, heads, head_dim)
        tactile_k = tactile_block.wan_tactile_norm_k(
            tactile_qkv[1].flatten(-2)
        ).view(batch, tactile_len, heads, head_dim)
        tactile_v = tactile_qkv[2].view(batch, tactile_len, heads, head_dim)

        seq_lens = torch.full(
            (batch,),
            video_len + action_len + tactile_len,
            dtype=torch.long,
            device=video_tokens.device,
        )
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)
        video_out, action_out_heads, tactile_out_heads = wan_layer.self_attn(
            norm_video,
            seq_lens,
            self.grid_sizes,
            freqs,
            action_q=action_q,
            action_k=action_k,
            action_v=action_v,
            tactile_q=tactile_q,
            tactile_k=tactile_k,
            tactile_v=tactile_v,
            attn_mask=None,
        )
        action_out = action_block.wan_action_o(action_out_heads.flatten(2))
        tactile_out = tactile_block.wan_tactile_o(tactile_out_heads.flatten(2))
        return (
            video_tokens + video_out * v_mod[2].squeeze(2),
            action_tokens + action_out * a_mod[2].squeeze(2),
            tactile_tokens + tactile_out * t_mod[2].squeeze(2),
        )




class ActionModule(nn.Module):
    """Action processing module - handles Action Expert + joint attentions + masks."""

    def __init__(self, action_expert: ActionExpert, config, video_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.dtype = dtype
        self.device = device

    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action time embedding."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()

            # Create sinusoidal embedding (same pattern as VideoModule)
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, seq_len, freq_dim]

            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))  # [B, seq_len, 6, dim]


        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process Action Expert FFN with AdaLN modulation."""
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens


class MEDexModel(nn.Module):
    """Released ME-Dex-1.0 video-action-tactile inference model."""

    def __init__(self, config: MEDexConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16
        self.video_model = WanVideoModel.from_config(
            config_path=config.wan_config_path,
            vae_path=config.vae_path,
            device="cuda",
            precision=config.video_precision,
        )
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads,
            'head_dim': wan_head_dim,
        }
        action_chunk_size_for_expert = config.action_chunk_size + 1
        num_registers = 4
        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            eps=config.action_expert_norm_eps,
        )
        self.action_expert = ActionExpert(action_config, wan_config)
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)

        tactile_config = TactileExpertConfig.from_mapping(
            config.tactile_expert_config,
            num_layers=config.num_layers,
        )
        self.tactile_expert = TactileExpert(tactile_config)
        self.tactile_expert.to(device=self.device, dtype=self.dtype)
        self.tactile_expert.time_embedding.float()
        self.tactile_expert.time_projection.float()
        tactile_codec = TactileAE(
            checkpoint_path=config.tactile_ae_checkpoint_path,
            device=self.device,
            dtype=self.dtype,
        )
        object.__setattr__(self, "tactile_codec", tactile_codec)

        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0).expand(config.batch_size, -1)

        # These aliases are part of the released checkpoint state-dict layout.
        self.video_module = VideoModule(
            self.video_model,
            self.dtype,
            self.device,
            self.grid_sizes,
        )
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.dtype, self.device)

    def _joint_video_action_tactile_velocity(
        self,
        *,
        video_latent: torch.Tensor,
        noisy_actions: torch.Tensor,
        tactile_latent: torch.Tensor,
        state: torch.Tensor,
        processed_t5_context: torch.Tensor,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        tactile_timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one ME-Dex-1.0 Full-Joint V-A-T flow-velocity evaluation."""
        batch_size = int(video_latent.shape[0])
        video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
        registers = (
            self.action_expert.registers.expand(batch_size, -1, -1)
            if self.action_expert.registers is not None
            else None
        )
        action_tokens = self.action_expert.input_encoder(
            state.unsqueeze(1).to(self.dtype), noisy_actions, registers
        )
        tactile_tokens = self.tactile_expert.tokenizer(tactile_latent.to(self.dtype))
        video_head_time, video_adaln = self.video_module.get_time_embedding(
            video_timestep, video_tokens.shape[1]
        )
        action_head_time, action_adaln = self.action_module.get_time_embedding(
            action_timestep, action_tokens.shape[1]
        )
        tactile_head_time, tactile_adaln = self.tactile_expert.get_time_embeddings(
            tactile_timestep.float()
        )
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            for layer_idx in range(self.config.num_layers):
                video_mod = self.video_module.compute_adaln_modulation(
                    video_adaln, layer_idx
                )
                action_mod = self.action_module.compute_adaln_modulation(
                    action_adaln, layer_idx
                )
                tactile_mod = self.tactile_expert.modulation(
                    self.tactile_expert.blocks[layer_idx], tactile_adaln
                )
                attention_method = (
                    self.video_module.process_video_action_tactile_joint_attention
                    if self.config.attention_topology == "full_joint"
                    or self.config.h_bridge_joint_start_layer <= layer_idx < self.config.h_bridge_joint_end_layer
                    else self.video_module.process_video_action_tactile_decoupled_attention
                )
                video_tokens, action_tokens, tactile_tokens = (
                    attention_method(
                        video_tokens,
                        action_tokens,
                        tactile_tokens,
                        video_mod,
                        action_mod,
                        tactile_mod,
                        layer_idx,
                        self.action_expert.blocks[layer_idx],
                        self.tactile_expert.blocks[layer_idx],
                    )
                )
                video_tokens = self.video_module.process_cross_attention(
                    video_tokens, video_adaln, layer_idx, processed_t5_context
                )
                video_tokens = self.video_module.process_ffn(
                    video_tokens, video_mod, layer_idx
                )
                action_tokens = self.action_module.process_ffn(
                    action_tokens, action_mod, layer_idx
                )
                tactile_tokens = self.tactile_expert.apply_ffn(
                    tactile_tokens,
                    self.tactile_expert.blocks[layer_idx],
                    tactile_mod,
                )
            video_velocity = self.video_module.apply_output_head(
                video_tokens, video_head_time
            )
            action_velocity_full = self.action_expert.decoder(
                action_tokens, action_head_time
            )
            usable_length = (
                action_velocity_full.shape[1]
                - self.action_expert.config.num_registers
            )
            action_velocity = action_velocity_full[:, 1:usable_length]
            tactile_velocity = self.tactile_expert.output_head(
                tactile_tokens, tactile_head_time
            )
        return video_velocity, action_velocity, tactile_velocity



    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        language_embeddings: list[torch.Tensor],
        *,
        num_inference_steps: int = 10,
        video_schedule_shift: float = 5.0,
        action_schedule_shift: float = 5.0,
        tactile_observed_source: torch.Tensor,
        tactile_observed_support_source: torch.Tensor,
        tactile_observed_frame_times: torch.Tensor,
        tactile_future_query_times: torch.Tensor,
        tactile_schedule_shift: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict one 16-target action chunk and the accompanying video."""
        batch = first_frame.shape[0]
        first_frame = first_frame.to(self.device, dtype=self.dtype)
        state = state.to(self.device, dtype=self.dtype)
        language_embeddings = [
            embedding.to(self.device, dtype=self.dtype)
            for embedding in language_embeddings
        ]

        # WAN's VAE consumes [-1, 1]; no mean/std image normalization is used.
        vae_input = (first_frame * 2.0 - 1.0).unsqueeze(2)
        condition_frame_latent = self.video_model.encode_video(vae_input)
        _, channels, _, height, width = condition_frame_latent.shape
        latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn(
            (batch, channels, latent_frames, height, width),
            device=self.device,
            dtype=self.dtype,
        )
        video_latent[:, :, 0:1] = condition_frame_latent
        action_latent = torch.randn(
            (batch, self.config.action_chunk_size, self.config.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

        tactile_observed, _ = self.tactile_codec.encode_condition(
            tactile_observed_source,
            observed_support_source=tactile_observed_support_source,
            observed_frame_times=tactile_observed_frame_times,
            future_query_times=tactile_future_query_times,
        )
        tactile_config = self.tactile_expert.config
        tactile_future = torch.randn(
            (
                batch,
                tactile_config.future_slices,
                tactile_config.queries_per_slice,
                tactile_config.latent_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        video_sigmas = build_flowmatch_sigma_schedule(
            num_inference_steps,
            video_schedule_shift,
            device=self.device,
            dtype=self.dtype,
        )
        action_sigmas = build_flowmatch_sigma_schedule(
            num_inference_steps,
            action_schedule_shift,
            device=self.device,
            dtype=self.dtype,
        )
        tactile_sigmas = build_flowmatch_sigma_schedule(
            num_inference_steps,
            tactile_schedule_shift,
            device=self.device,
            dtype=self.dtype,
        )

        for step in range(num_inference_steps):
            video_t = video_sigmas[step]
            action_t = action_sigmas[step]
            tactile_t = tactile_sigmas[step]
            video_velocity, action_velocity, tactile_velocity = (
                self._joint_video_action_tactile_velocity(
                    video_latent=video_latent,
                    noisy_actions=action_latent,
                    tactile_latent=torch.cat((tactile_observed, tactile_future), dim=1),
                    state=state,
                    processed_t5_context=t5_context,
                    video_timestep=(video_t * 1000).expand(batch).to(self.dtype),
                    action_timestep=(action_t * 1000).expand(batch).to(self.dtype),
                    tactile_timestep=(tactile_t * 1000).expand(batch).to(self.dtype),
                )
            )
            video_latent = video_latent + video_velocity * (video_sigmas[step + 1] - video_t)
            action_latent = action_latent + action_velocity * (action_sigmas[step + 1] - action_t)
            tactile_future = tactile_future + (
                tactile_velocity[:, tactile_config.condition_slices :]
                * (tactile_sigmas[step + 1] - tactile_t)
            )
            video_latent[:, :, 0:1] = condition_frame_latent

        decoded_frames = self.video_model.decode_video(video_latent)
        predicted_frames = torch.clamp(
            (decoded_frames[:, :, 1:] + 1.0) / 2.0,
            0,
            1,
        ).float()
        return predicted_frames, action_latent.float()
