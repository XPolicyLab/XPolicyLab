"""
Pi0-FAST model with autoregressive action generation.

This implementation is based on the OpenPI project:
https://github.com/Physical-Intelligence/openpi

Original code: openpi/src/openpi/models/pi0_fast.py
Original implementation: JAX/Flax
Licensed under Apache License 2.0

Key differences from Pi0:
- Uses single LLM (Gemma 2B) instead of joint model with action expert
- Action tokens are part of the LLM vocabulary (mapped from FAST tokenizer)
- Standard next-token prediction instead of flow matching
- Prefix-LM attention pattern (bidirectional on prefix, causal on suffix)

Adapted to PyTorch by Galaxea AI.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from accelerate.logging import get_logger

from .paligemma.siglip import SiglipVisionModel
from .paligemma.siglip import PaliGemmaMultiModalProjector
from .mixture import Mixture, MixtureConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from omegaconf import OmegaConf

logger = get_logger(__name__)


class Pi0Fast(nn.Module):
    """
    Pi0-FAST model using single LLM with FAST tokenizer for action generation.

    Architecture:
    - Vision: SigLIP encoder + multi-modal projector
    - Language: Gemma 2B (single LLM, no separate action expert)
    - Action: Tokens in LLM vocabulary (via FAST tokenizer mapping)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.max_token_len = cfg.get("max_token_len", 256)
        self.num_input_images = cfg.get("num_input_images", 1)

        # Vision encoder
        self.vision_tower = SiglipVisionModel(SiglipVisionConfig(**cfg.vision))
        self.multi_modal_projector = PaliGemmaMultiModalProjector(cfg.vision_projector)

        # Single LLM (Gemma 2B) - reuse Mixture class
        llm_config_dict = OmegaConf.to_container(cfg.llm, resolve=True)
        llm_config = MixtureConfig(**llm_config_dict)
        self.llm = Mixture(llm_config)

        # LM head for next-token prediction
        # Weight tying: lm_head shares weights with embed_tokens (standard for Gemma/PaliGemma)
        self.lm_head = nn.Linear(cfg.llm.hidden_size, cfg.vocab_size, bias=False)
        # Tie weights: lm_head.weight points to the same tensor as embed_tokens.weight
        self.lm_head.weight = self.llm.embed_tokens.weight

        # Freeze parameters based on training strategy
        self.freeze_by_stage(stage=cfg.get("vla_training_strategy", "vla-full-train"))

    def load_pretrained_weights(self):
        """Load pretrained weights from OpenPI checkpoint."""
        import os
        import glob
        from safetensors import safe_open

        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        assert len(safetensors_files) > 0, "No pre-trained weights found"

        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # Load vision tower
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.vision_tower.vision_model" in k:
                new_key = k.replace(
                    "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.", "vision_model."
                )
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for vision tower")

        # Load multi-modal projector
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.multi_modal_projector." in k:
                new_key = k.replace(
                    "model.paligemma_with_expert.paligemma.model.multi_modal_projector.", ""
                )
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(multi_modal_projector_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for multi-modal projector")

        # Load LLM (from VLM weights)
        llm_state_dict = self.llm.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.language_model." in k:
                new_key = k.replace(
                    "model.paligemma_with_expert.paligemma.model.language_model.", ""
                )
                if new_key in llm_state_dict:
                    llm_state_dict[new_key] = v
        self.llm.load_state_dict(llm_state_dict, strict=False)
        logger.info("Loaded pre-trained weights for LLM")

        # Load lm_head/embed_tokens weights (they are tied, so only need to load once)
        # Pretrained model saves lm_head.weight, which is shared with embed_tokens.weight
        lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
        if lm_head_key in tensors:
            lm_head_weight = tensors[lm_head_key]
            # Since lm_head.weight and embed_tokens.weight are tied (same tensor),
            # we only need to update embed_tokens.weight
            embed_state_dict = self.llm.embed_tokens.state_dict()
            embed_state_dict["weight"] = lm_head_weight
            self.llm.embed_tokens.load_state_dict(embed_state_dict, strict=True)
            logger.info("Loaded pre-trained weights for embed_tokens/lm_head (weight tied)")
        else:
            logger.warning("lm_head.weight not found in pretrained weights!")

    def freeze_by_stage(self, stage: str):
        """Freeze parameters based on training stage."""
        if stage in {"full-finetune", "vla-full-train"}:
            logger.info("[TRAINABLE]        🔥   =>> Vision Backbone")
            logger.info("[TRAINABLE]        🔥   =>> LLM")
        elif stage in {"llm-only"}:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            for param in self.multi_modal_projector.parameters():
                param.requires_grad = False
            logger.info("[FROZEN]           🥶   =>> Vision Backbone")
            logger.info("[TRAINABLE]        🔥   =>> LLM")
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _embed_images(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Encode images with SigLIP and project to LLM dimension."""
        bsz, cam_num, c, h, w = pixel_values.shape
        # Batch all cameras together: [B, num_cams, C, H, W] -> [B*num_cams, C, H, W]
        pixel_values_flat = pixel_values.view(bsz * cam_num, c, h, w)
        image_output = self.vision_tower(pixel_values_flat)
        image_features = self.multi_modal_projector(image_output.last_hidden_state)
        # Reshape back: [B*num_cams, num_patches, hidden] -> [B, num_cams*num_patches, hidden]
        num_patches = image_features.shape[1]
        return image_features.view(bsz, cam_num * num_patches, -1)

    def _make_attn_mask(
        self,
        input_mask: torch.Tensor,
        ar_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create prefix-LM attention mask.

        ar_mask: 0 = bidirectional (prefix), 1 = causal (suffix)

        The attention pattern is:
        - Prefix tokens (ar_mask=0) can attend to all other prefix tokens (bidirectional)
        - Suffix tokens (ar_mask=1) can attend to all prefix tokens and previous suffix tokens (causal)
        """
        # cumsum gives position in causal sequence
        # tokens with same cumsum value can attend to each other bidirectionally
        cumsum = torch.cumsum(ar_mask, dim=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]

        # Apply input mask (padding)
        valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
        attn_mask = torch.logical_and(attn_mask, valid_mask)

        # Convert to attention mask format (0 for attend, -inf for mask)
        attn_mask = attn_mask.unsqueeze(1)  # Add head dimension
        attn_mask = torch.where(attn_mask, 0.0, -2.3819763e38).to(dtype)

        return attn_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        ar_mask: torch.LongTensor,
        loss_mask: torch.BoolTensor,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> dict:
        """
        Training forward pass with next-token prediction.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Valid token mask [B, seq_len]
            ar_mask: Autoregressive mask (0=bidirectional, 1=causal) [B, seq_len]
            loss_mask: Mask for loss computation (action tokens only) [B, seq_len]
            pixel_values: Images [B, num_cams, C, H, W]

        Returns:
            dict with "action_token_loss"
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens
        token_embeds = self.llm.embed_tokens(input_ids)
        hidden_size = token_embeds.shape[-1]
        token_embeds = token_embeds * (hidden_size ** 0.5)

        # Embed images
        image_embeds = self._embed_images(pixel_values)
        num_image_tokens = image_embeds.shape[1]

        # Merge embeddings: image tokens are always at the beginning of the sequence
        inputs_embeds = token_embeds.clone()
        inputs_embeds[:, :num_image_tokens] = image_embeds

        # Build attention mask
        attn_mask = self._make_attn_mask(attention_mask, ar_mask, dtype=inputs_embeds.dtype)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state

        # Compute loss on action tokens only
        # Optimization: only pass action token positions through lm_head
        loss_mask_shifted = loss_mask[:, 1:]  # Shift to align with next-token prediction
        targets = input_ids[:, 1:]

        # Find positions where we need to compute loss
        # loss_mask_shifted[b, t] = True means we need logits at position t to predict targets[b, t]
        # which requires hidden_states[b, t] (before shifting)
        action_positions = loss_mask_shifted.nonzero(as_tuple=False)  # [num_action_tokens, 2]

        if action_positions.shape[0] > 0:
            # Extract hidden states at action positions (hidden_states[:, :-1] for next-token prediction)
            batch_indices = action_positions[:, 0]
            seq_indices = action_positions[:, 1]
            hidden_for_loss = hidden_states[batch_indices, seq_indices]  # [num_action_tokens, hidden_size]

            # Compute logits only for action positions
            logits = self.lm_head(hidden_for_loss)  # [num_action_tokens, vocab_size]

            # Get targets at action positions
            targets_for_loss = targets[batch_indices, seq_indices]  # [num_action_tokens]

            # Compute loss
            loss = nn.functional.cross_entropy(logits, targets_for_loss)
        else:
            # No action tokens to compute loss on - create a zero loss with gradient
            # This can happen if action tokens are truncated due to max_len
            loss = hidden_states.sum() * 0.0  # Maintains gradient graph

        return {"action_token_loss": loss}

    @torch.no_grad()
    def infer_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        ar_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Autoregressive action generation.

        Args:
            input_ids: Prefix token IDs [B, seq_len]
            attention_mask: Valid token mask [B, seq_len]
            ar_mask: Autoregressive mask [B, seq_len]
            pixel_values: Images [B, num_cams, C, H, W]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Generated token IDs [B, num_generated]
        """
        bsz = input_ids.shape[0]
        device = input_ids.device

        # For inference, truncate to valid tokens only (no padding)
        # This ensures KV cache doesn't contain padding positions
        valid_lengths = attention_mask.sum(dim=1)  # [B]
        max_valid_len = valid_lengths.max().item()

        # Truncate all inputs to max valid length
        input_ids = input_ids[:, :max_valid_len]
        attention_mask = attention_mask[:, :max_valid_len]
        ar_mask = ar_mask[:, :max_valid_len]

        # Embed tokens
        token_embeds = self.llm.embed_tokens(input_ids)
        hidden_size = token_embeds.shape[-1]
        token_embeds = token_embeds * (hidden_size ** 0.5)

        # Embed images
        image_embeds = self._embed_images(pixel_values)
        num_image_tokens = image_embeds.shape[1]

        # Merge embeddings: image tokens are always at the beginning of the sequence
        inputs_embeds = token_embeds.clone()
        inputs_embeds[:, :num_image_tokens] = image_embeds

        # Build attention mask for prefix
        attn_mask = self._make_attn_mask(attention_mask, ar_mask, dtype=inputs_embeds.dtype)

        # Prefill KV cache
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # Take hidden state from each sample's last VALID position (BOS token)
        # Using gather to handle variable-length sequences in batch
        # valid_lengths: [B], each sample's valid length
        # We need position (valid_lengths - 1) for each sample
        last_pos_indices = (valid_lengths - 1).view(bsz, 1, 1).expand(-1, 1, outputs.last_hidden_state.shape[-1])  # [B, 1, H]
        last_hidden = outputs.last_hidden_state.gather(1, last_pos_indices)  # [B, 1, H]
        last_logits = self.lm_head(last_hidden)

        # Autoregressive decoding
        # Following lerobot: always generate max_new_tokens, use "|" delimiter in extraction
        output_tokens = []

        for step in range(max_new_tokens):
            # Sample or greedy decode
            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            output_tokens.append(next_token)

            # Embed next token
            next_embeds = self.llm.embed_tokens(next_token) * (hidden_size ** 0.5)

            # Build attention mask for decoding step
            # New token should only attend to:
            # 1. Valid prefix positions [0, valid_lengths[i]) for each sample
            # 2. Generated positions [max_valid_len, max_valid_len + step]
            # NOT to padding positions [valid_lengths[i], max_valid_len)
            kv_len = max_valid_len + step + 1
            position_indices = torch.arange(kv_len, device=device).view(1, -1)  # [1, kv_len]
            valid_prefix_mask = position_indices < valid_lengths.view(bsz, 1)  # [B, kv_len]
            generated_mask = position_indices >= max_valid_len  # [1, kv_len]
            attend_mask = valid_prefix_mask | generated_mask  # [B, kv_len]
            decode_attn_mask = torch.where(
                attend_mask.view(bsz, 1, 1, kv_len),
                torch.zeros((bsz, 1, 1, kv_len), dtype=inputs_embeds.dtype, device=device),
                torch.full((bsz, 1, 1, kv_len), -2.3819763e38, dtype=inputs_embeds.dtype, device=device),
            )

            outputs = self.llm(
                inputs_embeds=next_embeds,
                attention_mask=decode_attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = self.lm_head(outputs.last_hidden_state)

        if output_tokens:
            return torch.cat(output_tokens, dim=1)
        else:
            return torch.zeros((bsz, 0), dtype=torch.long, device=device)
