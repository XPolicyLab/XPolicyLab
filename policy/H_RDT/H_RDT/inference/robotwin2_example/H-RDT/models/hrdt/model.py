from collections import OrderedDict
from typing import List, Tuple, Optional

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hrdt.blocks import ActionDecoder, HRDTBlock, TimestepEmbedder
from models.hrdt.pos_emb import get_multimodal_pos_embed


class HRDT(nn.Module):
    """
    Robotics Diffusion Transformer model
    
    Modified to:
    1. State and noisy action chunk are processed together as input
    2. AdaLN now uses timestep only (no sentence token)
    3. Image features directly fed to cross-attention in blocks
    4. Training mode controls which cross-attention layers to use
    """
    def __init__(
        self,
        horizon: int,
        config: dict,
        x_pos_emb_config: List[Tuple],
        img_pos_emb_config: List[Tuple] = None,
        lang_pos_emb_config: List[Tuple] = None,
        max_img_len: int = None,
        max_lang_len: int = None,
        training_mode: str = 'lang',
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = config["hidden_size"]
        self.n_heads = config["num_heads"]
        self.dtype = dtype
        self.gradient_checkpointing = False
        self.training_mode = training_mode

        # Validate training mode
        if training_mode not in ['lang']:
            raise ValueError(f"training_mode must be 'lang', got {training_mode}")

        # Remove AdaLN adapter - timestep embedding goes directly to blocks

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype)

        # Create H-RDT layers with training mode
        self.depth = config["depth"]
        self.blocks = nn.ModuleList([
            HRDTBlock(layer_idx, config=config, training_mode=training_mode)
            for layer_idx in range(self.depth)
        ])
        self.action_decoder = ActionDecoder(config=config)

        # Position embeddings
        self.x_pos_emb_config = x_pos_emb_config
        self.lang_pos_emb_config = lang_pos_emb_config
        self.img_pos_emb_config = img_pos_emb_config
        self.x_pos_emb = nn.Parameter(torch.zeros(
            1, 1 + self.horizon, self.hidden_size)) # state + action
        self.lang_pos_emb = nn.Parameter(torch.zeros(
            1, max_lang_len, self.hidden_size))
        self.img_pos_emb = nn.Parameter(torch.zeros(
            1, max_img_len, self.hidden_size))

        self.initialize_weights()

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def initialize_weights(self):
        # Initialize linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize position embeddings with sincos
        x_pos_emb = get_multimodal_pos_embed(
            embed_dim=self.hidden_size,
            mm_lens=OrderedDict(self.x_pos_emb_config)
        )
        self.x_pos_emb.data.copy_(
            torch.from_numpy(x_pos_emb).float().unsqueeze(0))

        lang_pos_emb = get_multimodal_pos_embed(
            embed_dim=self.hidden_size,
            mm_lens=OrderedDict(self.lang_pos_emb_config)
        )
        self.lang_pos_emb.data.copy_(
            torch.from_numpy(lang_pos_emb).float().unsqueeze(0))

        img_pos_embed = get_multimodal_pos_embed(
            embed_dim=self.hidden_size,
            mm_lens=OrderedDict(self.img_pos_emb_config)
        )
        self.img_pos_emb.data.copy_(
            torch.from_numpy(img_pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.action_decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.action_decoder.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.action_decoder.ffn.fc2.weight, 0)
        nn.init.constant_(self.action_decoder.ffn.fc2.bias, 0)

        # Move all params to specified dtype
        self.to(self.dtype)

    def gradient_checkpointing_enable(self, value: bool = True):
        """Enable gradient checkpointing to save memory"""
        self.gradient_checkpointing = value

    def forward(self, x, t, img_c=None, lang_c=None, sentence_c=None, task_c=None, lang_attn_mask=None):
        """
        Forward pass of H-RDT

        Args:
            x: (B, 1 + T, D), state and action token sequence, T = horizon
            t: (B,) or (1,), diffusion timesteps
            img_c: (B, S_img, D), image features for cross-attention, optional
            lang_c: (B, S_lang, D), language tokens for cross-attention, optional
            sentence_c: ignored (for backward compatibility)
            lang_attn_mask: (B, S_lang), attention mask for language tokens, optional
        Returns:
            x: (B, T, D_out), predicted denoised action tokens
        """
        # Embed timestep using sinusoidal embeddings
        t_emb = self.t_embedder(t)  # (B, D) or (1, D)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(x.shape[0], -1)  # (B, D)

        # Add position embeddings
        x = x + self.x_pos_emb
        
        if img_c is not None:
            img_c = img_c + self.img_pos_emb[:, :img_c.shape[1]]
        
        if lang_c is not None:
            lang_c = lang_c + self.lang_pos_emb[:, :lang_c.shape[1]]
        
        # Pass timestep embedding directly to blocks (no sentence token)
        for i, block in enumerate(self.blocks):
            cross_contexts = {
                'img_c': img_c,
                'lang_c': lang_c,
                'lang_attn_mask': lang_attn_mask
            }
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, cross_contexts, use_reentrant=False)
            else:
                x = block(x, t_emb, cross_contexts)

        # Final layer only uses timestep (no cross-attention)
        x = self.action_decoder(x, t_emb)

        # Extract action predictions
        x = x[:, -self.horizon:]

        return x