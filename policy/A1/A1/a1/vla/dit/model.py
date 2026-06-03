# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from collections import OrderedDict

import torch
import torch.nn as nn

from a1.vla.dit.blocks import (FinalLayer, DiTBlock, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed)


class DiT(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        # dtype,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        # max_lang_cond_len=1024,
        llm_state_cond_len=1024,
        llm_state_cond_dim=3584,
        # img_cond_len=4096,
        # lang_pos_embed_config=None,
        
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        # self.max_lang_cond_len = max_lang_cond_len
        self.llm_state_cond_len = llm_state_cond_len
        # self.llm_state_cond_dim = llm_state_cond_dim

        # self.dtype = dtype
        # self.lang_pos_embed_config = lang_pos_embed_config


        self.t_embedder = TimestepEmbedder(hidden_size,)
        # self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # We will use trainable sin-cos embeddings
        # [timestep; state; action]
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                # ('ctrl_freq', 1),
                # ('state', 1),
                ('action', self.horizon),
            ])
        )
        self.x_pos_embed = nn.Parameter(torch.from_numpy(x_pos_embed).float().unsqueeze(0))
        # self.x_pos_embed = nn.Parameter(
        #     torch.zeros(1, horizon+1, hidden_size))
        

        # Language conditions
        # self.lang_cond_pos_embed = nn.Parameter(
        #     torch.zeros(1, max_lang_cond_len, hidden_size))
        
        ## LLM state conditions
        llm_state_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.hidden_size, torch.arange(self.llm_state_cond_len))
        self.llm_state_cond_pos_embed = nn.Parameter(torch.from_numpy(llm_state_cond_pos_embed).float().unsqueeze(0))
        # self.llm_state_cond_pos_embed = nn.Parameter(
        #     torch.zeros(1, llm_state_cond_len, hidden_size))


        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        # x_pos_embed = get_multimodal_cond_pos_embed(
        #     embed_dim=self.hidden_size,
        #     mm_cond_lens=OrderedDict([
        #         ('timestep', 1),
        #         # ('ctrl_freq', 1),
        #         # ('state', 1),
        #         ('action', self.horizon),
        #     ])
        # )
        # print("*"*50,f"x_pos_embed shape: {x_pos_embed.shape}")
        # self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))
        # print("*"*50,f"self.x_pos_embed shape: {self.x_pos_embed.shape}")


        # if self.lang_pos_embed_config is None:
        #     lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
        #         self.hidden_size, torch.arange(self.max_lang_cond_len))
        # else:
        #     lang_cond_pos_embed = get_multimodal_cond_pos_embed(
        #         embed_dim=self.hidden_size,
        #         mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
        #         embed_modality=False
        #     )
        # self.lang_cond_pos_embed.data.copy_(
        #     torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        
        # Initialize LLM state condition pos embed
        # llm_state_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
        #     self.hidden_size, torch.arange(self.llm_state_cond_len))
        
        # self.llm_state_cond_pos_embed.data.copy_(
        #     torch.from_numpy(llm_state_cond_pos_embed).float().unsqueeze(0))
        ##
        

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
            
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        # self.to(self.dtype)

    # def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
    def forward(self, x, t, llm_state_c, llm_state_mask=None):
        """
        Forward pass of DiT.
        
        x: (B, T, D), state + action token sequence, T = horizon + 1,
            dimension D is assumed to be the same as the hidden size.
        t: (B,) or (1,), diffusion timesteps.
        lang_c: (B, L_lang, D) or None, language condition tokens (variable length),
            dimension D is assumed to be the same as the hidden size.
        lang_mask: (B, L_lang) or None, language condition mask (True for valid).
        img_mask: (B, L_img) or None, image condition mask (True for valid).
        """
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        # freq = self.freq_embedder(freq).unsqueeze(1)    # (B, 1, D)
        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, x], dim=1)               # (B, T+1, D)
        
        # Add multimodal position embeddings
        # print("**** DiT forward pass ****")
        # print(f"x shape:{x.shape}, self.x_pos_embed.shape:{self.x_pos_embed.shape}")
        x = x + self.x_pos_embed
        # Note the lang is of variable length
        # lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        # d_model: 3584
        llm_state_c = llm_state_c + self.llm_state_cond_pos_embed

        # img_c = img_c + self.img_cond_pos_embed

        # Forward pass
        # conds = [lang_c, img_c]
        cond = llm_state_c
        mask = llm_state_mask
        for i, block in enumerate(self.blocks):
            c, mask = cond, mask
            x = block(x, c, mask)                       # (B, T+1, D)
        # Inject the language condition at the final layer
        x = self.final_layer(x)                         # (B, T+1, out_channels)

        # Only preserve the action tokens
        x = x[:, -self.horizon:]
        return x


class TextImageConditionDiT(nn.Module):
    """
    Class for Robotics Diffusion Transformers. This version is conditioned on text and image.
    """
    def __init__(
        self,
        # dtype,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        # lang_pos_embed_config=None,
        img_pos_embed_config=None,
        use_proprio=False,
       
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len

        # self.dtype = dtype
        self.img_pos_embed_config = img_pos_embed_config
        # self.lang_pos_embed_config = lang_pos_embed_config
        self.use_proprio = use_proprio

        # img_pos_embed_config=[
        #         # No initial pos embed in the last grid size
        #         # since we've already done in ViT
        #         ("image", (config["common"]["img_history_size"], 
        #             config["common"]["num_cameras"], 
        #             -vision_encoder.num_patches)),  
        #     ],


        self.t_embedder = TimestepEmbedder(hidden_size,dtype=torch.float32)
        # self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # We will use trainable sin-cos embeddings
        # [timestep; state; action]
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                # ('ctrl_freq', 1),
                *([('proprio', 1)] if self.use_proprio else []),
                ('action', self.horizon),
            ])
        )
        self.x_pos_embed = nn.Parameter(torch.from_numpy(x_pos_embed).float().unsqueeze(0))
        # self.x_pos_embed = nn.Parameter(
        #     torch.zeros(1, horizon+1, hidden_size))
        

        # Language conditions
        lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(self.max_lang_cond_len))
        self.lang_cond_pos_embed = nn.Parameter(torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        # Image conditions
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))


        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        # x_pos_embed = get_multimodal_cond_pos_embed(
        #     embed_dim=self.hidden_size,
        #     mm_cond_lens=OrderedDict([
        #         ('timestep', 1),
        #         # ('ctrl_freq', 1),
        #         # ('state', 1),
        #         ('action', self.horizon),
        #     ])
        # )
        # print("*"*50,f"x_pos_embed shape: {x_pos_embed.shape}")
        # self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))
        # print("*"*50,f"self.x_pos_embed shape: {self.x_pos_embed.shape}")

        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))
        

        # if self.lang_pos_embed_config is None:
        #     lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
        #         self.hidden_size, torch.arange(self.max_lang_cond_len))
        # else:
        #     lang_cond_pos_embed = get_multimodal_cond_pos_embed(
        #         embed_dim=self.hidden_size,
        #         mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
        #         embed_modality=False
        #     )
        # self.lang_cond_pos_embed.data.copy_(
        #     torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        # self.to(self.dtype)

    def forward(self, x, t, lang_c, img_c, lang_mask=None, img_mask=None):
        """
        Forward pass of DiT.
        
        x: (B, T, D), state + action token sequence, T = horizon + 1,
            dimension D is assumed to be the same as the hidden size.
        t: (B,) or (1,), diffusion timesteps.
        lang_c: (B, L_lang, D) or None, language condition tokens (variable length),
            dimension D is assumed to be the same as the hidden size.
        img_c: (B, L_img, D) or None, image condition tokens (fixed length),
            dimension D is assumed to be the same as the hidden size.
        lang_mask: (B, L_lang) or None, language condition mask (True for valid).
        img_mask: (B, L_img) or None, image condition mask (True for valid).
        """
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)

        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, x], dim=1)               # (B, T+1, D)
        
        # Add multimodal position embeddings

        x = x + self.x_pos_embed
        # Note the lang is of variable length
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed

        # Forward pass
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i%2], masks[i%2]
            x = block(x, c, mask)                       # (B, T+1, D)
        # Inject the language condition at the final layer
        x = self.final_layer(x)                         # (B, T+1, out_channels)

        # Only preserve the action tokens
        x = x[:, -self.horizon:]
        return x