"""
This file is based on work from open-pi-zero (https://github.com/allenzren/open-pi-zero),
licensed under the MIT License.

Modifications:
   Copyright (c) 2025 Galaxea AI.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""
import os
import glob
from typing import Optional, Tuple
from safetensors import safe_open

import torch
from torch import nn

from accelerate.logging import get_logger

from .modules import (
    ActionEncoder,
    ActionDecoder,
)
from ...utils.import_utils import get_obj_from_str
from .paligemma.siglip import PaliGemmaMultiModalProjector
from .paligemma.siglip import SiglipVisionModel

from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

import torch.nn.functional as F
import math
logger = get_logger(__name__)

def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float = 4e-3, max_period: float = 4.0
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Used for pi0 mode to embed timestep.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    device = time.device
    dtype = torch.float64 if device.type != "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class Pi(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.cond_steps = cfg.cond_steps
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps
        self.num_input_images = cfg.num_input_images

        self.image_text_hidden_size = cfg.joint.mixture.vlm.hidden_size
        # self.proprio_hidden_size = cfg.joint.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.joint.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = 0.0

        # Calculate total tokens (includes state token)
        self.total_num_tokens = (
            self.max_image_text_tokens
            + 1  # state token
            + self.num_action_tokens
        )

        # loss weights for padding actions
        self.padding_action_weight = cfg.get("padding_action_weight", 0.0)

        # train speedup
        torch.set_float32_matmul_precision("high")

        # Vision
        self.vision_tower = SiglipVisionModel(SiglipVisionConfig(**cfg.vision))
        self.multi_modal_projector = PaliGemmaMultiModalProjector(cfg.vision_projector)

        # Action expert defaults (Gemma 300M)
        self.joint_model = get_obj_from_str(cfg.joint.name)(cfg.joint)

        # text input only
        self.embed_tokens = self.joint_model.mixtures.vlm.embed_tokens
        self.joint_model.mixtures.action.embed_tokens = None

        self.action_encoder = ActionEncoder(
            self.action_dim,
            self.action_hidden_size,
            time_cond=False,
        )

        # pi0: state projection + action-time MLP
        self.state_proj = nn.Linear(self.proprio_dim, self.action_hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * self.action_hidden_size, self.action_hidden_size)
        self.action_time_mlp_out = nn.Linear(self.action_hidden_size, self.action_hidden_size)

        # Action decoder
        self.action_decoder = ActionDecoder(
            self.action_hidden_size,
            self.action_dim,
            num_layers=cfg.action_decoder_layers,
        )

        # try:  # pragma: no cover - best effort
        #     self.infer_action = torch.compile(self.infer_action, mode="max-autotune")
        # except Exception:
        #     logger.warning("Failed to compile action decoder")
        #     pass
        self.freeze_by_stage(stage=cfg.vla_training_strategy)

    @property
    def action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + self.trainable_gemma_parameters
        )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""
        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        assert len(safetensors_files) > 0, "No pre-trained weights found"
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        action_encoder_state_dict = self.action_encoder.state_dict()
        for k, v in tensors.items():
            if "action_in_proj" in k:
                new_key = k.replace("action_in_proj.", "linear_1.")
                action_encoder_state_dict[new_key] = v
        self.action_encoder.load_state_dict(action_encoder_state_dict, strict=True)

        action_decoder_state_dict = self.action_decoder.state_dict()
        for k, v in tensors.items():
            if "action_out_proj" in k:
                new_key = k.replace("action_out_proj.", "proj.0.")
                action_decoder_state_dict[new_key] = v
        self.action_decoder.load_state_dict(action_decoder_state_dict, strict=True)

        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.vision_tower.vision_model" in k:
                new_key = k.replace("paligemma_with_expert.paligemma.model.vision_tower.vision_model.", "vision_model.")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for vision tower")

        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.multi_modal_projector." in k:
                new_key = k.replace("paligemma_with_expert.paligemma.model.multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(multi_modal_projector_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for multi-modal projector")

        joint_model_state_dict = self.joint_model.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.gemma_expert.model." in k:
                new_key = k.replace("paligemma_with_expert.gemma_expert.model.", "mixtures.action.")
                joint_model_state_dict[new_key] = v
            if "paligemma_with_expert.paligemma.model.language_model." in k:
                new_key = k.replace("paligemma_with_expert.paligemma.model.language_model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        logger.info("Loaded pre-trained weights for joint model")

        # Load state_proj, action_time_mlp_in, action_time_mlp_out
        state_proj_state_dict = self.state_proj.state_dict()
        for k, v in tensors.items():
            if k.startswith("state_proj."):
                new_key = k.replace("state_proj.", "")
                state_proj_state_dict[new_key] = v
        self.state_proj.load_state_dict(state_proj_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for state_proj")

        action_time_mlp_in_state_dict = self.action_time_mlp_in.state_dict()
        for k, v in tensors.items():
            if k.startswith("action_time_mlp_in."):
                new_key = k.replace("action_time_mlp_in.", "")
                action_time_mlp_in_state_dict[new_key] = v
        self.action_time_mlp_in.load_state_dict(action_time_mlp_in_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for action_time_mlp_in")

        action_time_mlp_out_state_dict = self.action_time_mlp_out.state_dict()
        for k, v in tensors.items():
            if k.startswith("action_time_mlp_out."):
                new_key = k.replace("action_time_mlp_out.", "")
                action_time_mlp_out_state_dict[new_key] = v
        self.action_time_mlp_out.load_state_dict(action_time_mlp_out_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for action_time_mlp_out")

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in lm part of the joint model")
    
    def freeze_non_lora_weights_in_action_expert(self):
        for name, param in self.joint_model.mixtures["action"].named_parameters():
            param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in action expert part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
    
    def freeze_by_stage(self, stage: str):
        if stage in {"full-finetune", "vla-full-train"}:
            logger.info(f"[TRAINABLE]        🔥   =>> Vision Backbone `{self.vision_tower}`")  # noqa: E501
            logger.info(f"[TRAINABLE]        🔥   =>> VLM expert `{self.joint_model.mixtures['vlm']}`")  # noqa: E501
            logger.info(f"[TRAINABLE]        🔥   =>> Action expert `{self.joint_model.mixtures['action']}`")  # noqa: E501
        elif stage in {"action-expert-only"}:
            self.freeze_non_lora_weights_in_vlm()
            logger.info(f"[FROZEN]           🥶   =>> Vision Backbone `{self.vision_tower}`")  # noqa: E501
            logger.info(f"[FROZEN]           🥶   =>> VLM expert `{self.joint_model.mixtures['vlm']}`")  # noqa: E501
            logger.info(f"[TRAINABLE]        🔥   =>> Action expert `{self.joint_model.mixtures['action']}`")  # noqa: E501
        else:
            raise ValueError(f"Unknown stage: {stage}")

    # def tie_action_proprio_weights(self):
    #     """technically more than just tying weights"""
    #     self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) state action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        state       x        x        x                 x
        action      x        x        x                 x      x      x
        action      x        x        x                 x      x      x
        """
        bsz = attention_mask.size(0)
        device = attention_mask.device

        # Count actual image/text tokens per sample (ignoring padding)
        image_text_token_cnts = torch.sum(attention_mask, dim=1)

        # Initialize causal mask with -inf (avoiding softmax issues)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            -2.3819763e38,
            dtype=dtype, device=device,
        )

        state_start = self.max_image_text_tokens
        action_start = self.max_image_text_tokens + 1  # state token takes 1 position
        for idx, cnt in enumerate(image_text_token_cnts):
            # 1. Image/text self-attention (up to actual token count)
            causal_mask[idx, :cnt, :cnt] = 0
            # 2. State token attends to image/text
            causal_mask[idx, state_start, :cnt] = 0
            # 3. State token self-attention
            causal_mask[idx, state_start, state_start] = 0
            # 4. Action attends to image/text
            causal_mask[idx, action_start:, :cnt] = 0
            # 5. Action attends to state token
            causal_mask[idx, action_start:, state_start] = 0
            # 6. Action self-attention
            causal_mask[idx, action_start:, action_start:] = 0

        # Add head dimension for multi-head attention
        causal_mask = causal_mask.unsqueeze(1)

        # Position IDs
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1, device=device).repeat(bsz, 1)

        vision_lang_mask = torch.ones_like(vlm_position_ids, dtype=torch.bool)
        for idx, cnt in enumerate(image_text_token_cnts):
            vision_lang_mask[idx, cnt:] = False

        # state token + action tokens
        suffix_len = 1 + self.num_action_tokens

        action_position_ids = torch.arange(
            self.max_image_text_tokens + 1,
            self.max_image_text_tokens + suffix_len + 1,
            device=device,
        ).repeat(bsz, 1)
        action_masks = torch.ones_like(action_position_ids, dtype=torch.bool)
        position_ids = torch.cumsum(torch.cat([vision_lang_mask, action_masks], dim=1), dim=1) - 1

        return causal_mask, vlm_position_ids, action_position_ids, position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = pixel_values.device
        bsz, seq_len = input_ids.shape
        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        # NOTE: embed_tokens will use autocast dtype automatically
        inputs_embeds = self.embed_tokens(input_ids)
        lang_emb_dim = inputs_embeds.shape[-1] # 2048
        inputs_embeds = inputs_embeds * (lang_emb_dim**0.5)

        # image features from siglip and projector
        # pixel_values: [Batch_Size, T, C, H, W], T can be # of history frames or more cameras
        # Batched processing: flatten all cameras together for efficiency
        bsz_img, cam_num, c, h, w = pixel_values.shape
        # [B, T, C, H, W] -> [B*T, C, H, W]
        pixel_values_flat = pixel_values.view(bsz_img * cam_num, c, h, w)
        # Process all camera images at once
        image_output = self.vision_tower(pixel_values_flat)
        image_features = self.multi_modal_projector(image_output.last_hidden_state)
        # [B*T, num_patches, proj_dim] -> [B, T*num_patches, proj_dim]
        _, num_patches, proj_dim = image_features.shape
        image_features = image_features.view(bsz_img, cam_num * num_patches, proj_dim)

        # normalize the image features
        _, _, embed_dim = image_features.shape
        scaled_image_features = image_features 

        # AMP Best Practice: Use scaled_image_features.dtype to ensure consistency
        # In autocast context, all intermediate tensors will have the same dtype
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), 
            self.pad_token_id, 
            dtype=scaled_image_features.dtype,  # Use image features dtype for consistency
            device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index)

        image_mask = input_ids == self.image_token_index
        # Ensure dtype consistency when assigning
        final_embedding[text_mask] = inputs_embeds[text_mask].to(final_embedding.dtype)
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    @torch.no_grad()
    def infer_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        pixel_values: torch.FloatTensor,
        proprios: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        device = pixel_values.device
        bsz = pixel_values.size(0)

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # AMP Best Practice: Use dtype from model's output for consistency
        dtype = inputs_embeds.dtype

        causal_mask, vlm_position_ids, action_position_ids, position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        causal_mask_vlm = causal_mask[:,:, :self.max_image_text_tokens, :self.max_image_text_tokens]
        position_ids_vlm = position_ids[:, :self.max_image_text_tokens]
        causal_mask_suffix = causal_mask[:,:, self.max_image_text_tokens:, :]
        position_ids_suffix = position_ids[:, self.max_image_text_tokens:]

        # Prefill KV cache with VLM (image + text)
        _, past_key_values = self.joint_model(
            attention_mask=causal_mask_vlm,
            position_ids=position_ids_vlm,
            past_key_values=None,
            inputs_embeds=[inputs_embeds, None],
            use_cache=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm
        delta_t = 1.0 / self.num_inference_steps
        t = torch.ones(bsz, device=device, dtype=dtype)
        for i in range(self.num_inference_steps):
            # Embed state as a single token (use last observation step)
            # proprios: [B, num_obs_steps, proprio_dim] -> [B, proprio_dim]
            proprio_last = proprios[:, -1, :] if proprios.dim() == 3 else proprios
            state_embeds = self.state_proj(proprio_last)  # [B, proprio_dim] -> [B, action_hidden_size]
            state_embeds = state_embeds[:, None, :]  # [B, 1, action_hidden_size]

            # Embed timestep using sine-cosine positional encoding
            time_emb = create_sinusoidal_pos_embedding(t, self.action_hidden_size)
            time_emb = time_emb.to(dtype=t.dtype)

            # Embed noisy actions
            action_embeds = self.action_encoder(action)  # [B, H, action_hidden_size]

            # Fuse action + time using MLP
            time_emb_expanded = time_emb[:, None, :].expand_as(action_embeds)  # [B, H, action_hidden_size]
            action_time_embeds = torch.cat([action_embeds, time_emb_expanded], dim=-1)  # [B, H, 2*action_hidden_size]
            action_time_embeds = self.action_time_mlp_in(action_time_embeds)
            action_time_embeds = F.silu(action_time_embeds)
            action_time_embeds = self.action_time_mlp_out(action_time_embeds)

            # Concatenate state token + action tokens
            suffix_embeds = torch.cat([state_embeds, action_time_embeds], dim=1)  # [B, 1+H, action_hidden_size]

            outputs_embeds, _ = self.joint_model(
                attention_mask=causal_mask_suffix,
                position_ids=position_ids_suffix,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embeds],
                use_cache=False,
                adarms_cond=[None, None],
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.horizon_steps :]
            action_vel = self.action_decoder(suffix_out)

            action -= delta_t * action_vel
            t -= delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - t) * x1 + t * x

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        pixel_values: torch.ByteTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        action_pad_masks: torch.BoolTensor,
        action_dim_is_pad: torch.BoolTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # AMP Best Practice: Use dtype from model's output for consistency
        dtype = inputs_embeds.dtype

        causal_mask, vlm_position_ids, action_position_ids, position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # Build suffix embeddings: state token + action-time MLP
        # Embed state as a single token (use last observation step)
        # proprios: [B, num_obs_steps, proprio_dim] -> [B, proprio_dim]
        proprio_last = proprios[:, -1, :] if proprios.dim() == 3 else proprios
        state_embeds = self.state_proj(proprio_last)  # [B, proprio_dim] -> [B, action_hidden_size]
        state_embeds = state_embeds[:, None, :]  # [B, 1, action_hidden_size]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(t, self.action_hidden_size)
        time_emb = time_emb.to(dtype=t.dtype)

        # Embed noisy actions
        action_embeds = self.action_encoder(psi_t)  # [B, H, action_hidden_size]

        # Fuse action + time using MLP
        time_emb_expanded = time_emb[:, None, :].expand_as(action_embeds)  # [B, H, action_hidden_size]
        action_time_embeds = torch.cat([action_embeds, time_emb_expanded], dim=-1)  # [B, H, 2*action_hidden_size]
        action_time_embeds = self.action_time_mlp_in(action_time_embeds)
        action_time_embeds = F.silu(action_time_embeds)
        action_time_embeds = self.action_time_mlp_out(action_time_embeds)

        # Concatenate state token + action tokens
        suffix_embeds = torch.cat([state_embeds, action_time_embeds], dim=1)  # [B, 1+H, action_hidden_size]

        (_, suffix_out), _ = self.joint_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[inputs_embeds, suffix_embeds],
            use_cache=False,
            adarms_cond=[None, None],
        )
        suffix_out = suffix_out[:, -self.horizon_steps :]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(suffix_out)
        # compare to true velocity
        # d_psi = x1 - (1 - self.flow_sig_min) * x0
        # noise - action
        d_psi = x0 - x1
        l2 = F.mse_loss(d_psi, v_psi, reduction="none")

        final_mask = action_pad_masks.unsqueeze(-1) | action_dim_is_pad.unsqueeze(1)
        action_weights = torch.where(
            final_mask,
            self.padding_action_weight,
            torch.tensor(1.0, device=l2.device, dtype=l2.dtype)
        )

        weight_sum = action_weights.sum()
        weight_sum = torch.clamp(weight_sum, min=1.0)
        fm_loss = (action_weights * l2).sum() / weight_sum
        loss_dict = {"fm_loss": fm_loss}

        return loss_dict
