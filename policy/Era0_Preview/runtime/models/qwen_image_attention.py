"""Opt-in image-block visibility for scheme A's uncached policy forward."""

from __future__ import annotations

CAUSAL = "causal"
BIDIRECTIONAL_IMAGE_BLOCKS = "bidirectional_image_blocks"
CURRENT_ONLY_IMAGE_BLOCKS = "bidirectional_image_blocks_current_only"
IMAGE_ATTENTION_MODES = {CAUSAL, BIDIRECTIONAL_IMAGE_BLOCKS, CURRENT_ONLY_IMAGE_BLOCKS}


def image_attention_mode(config) -> str:
    mode = str(getattr(config, "qwen_image_attention_mode", CAUSAL)).strip().lower()
    if mode not in IMAGE_ATTENTION_MODES:
        raise ValueError(f"Invalid qwen_image_attention_mode: {mode!r}")
    return mode


def validate_image_attention(config) -> None:
    if image_attention_mode(config) == CAUSAL:
        return
    if (getattr(config, "vlm_backend", None) != "qwen3_5"
            or getattr(config, "qwen_temporal_embedding", None) != "vlm_history_current_v1"):
        raise ValueError("bidirectional_image_blocks requires Qwen3.5 vlm_history_current_v1.")


def validate_image_attention_resume(saved_config, requested_mode, *, allow_contract_migration=False) -> None:
    if requested_mode is not None and str(requested_mode).strip().lower() != image_attention_mode(saved_config):
        if (allow_contract_migration and image_attention_mode(saved_config) == BIDIRECTIONAL_IMAGE_BLOCKS
                and str(requested_mode).strip().lower() == CURRENT_ONLY_IMAGE_BLOCKS):
            return
        raise ValueError("Cannot change qwen_image_attention_mode on resume; warm-start a new experiment.")


def image_block_attention_mask(input_ids, padding_mask, *, image_token_id: int):
    """Causal visibility plus all pairs inside each contiguous image-token run.

    Image sentinels and text remain causal. Both padded keys and padded queries
    are excluded. SDPA returns zeros for these fully masked, unused queries.
    No Python per-sample loops or device-to-host transfers are needed.
    """
    import torch

    if input_ids.ndim != 2 or padding_mask.shape != input_ids.shape:
        raise ValueError("Image attention requires matching [batch, sequence] IDs/padding.")
    valid = padding_mask.bool()
    image = input_ids.eq(image_token_id) & valid
    previous = torch.nn.functional.pad(image[:, :-1], (1, 0), value=False)
    block = (image & ~previous).long().cumsum(dim=1)
    same_image = (block[:, :, None] == block[:, None, :])
    same_image &= image[:, :, None] & image[:, None, :]
    length = input_ids.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).tril()
    allowed = (causal | same_image) & valid[:, :, None] & valid[:, None, :]
    return allowed[:, None]


def current_only_image_attention_mask(packed, *, image_token_id: int):
    """Proposal (a)+(b): close direct history reads by wrists/non-image rows.

    Packing supplies roles, including missing views/short history. No inference
    from fixed frame counts or prompt lengths, and no device-to-host reads.
    """
    import torch

    image = packed["input_ids"].eq(image_token_id) & packed["attention_mask"].bool()
    current = packed["current_image_mask"] & image
    wrists = packed["wrist_image_mask"] & current
    history = image & ~current
    allowed = image_block_attention_mask(packed["input_ids"], packed["attention_mask"],
                                         image_token_id=image_token_id)
    allowed &= ~((wrists | ~image)[:, None, :, None] & history[:, None, None, :])
    # Different wrist blocks cannot read one another; same-image edges stay open.
    previous = torch.nn.functional.pad(image[:, :-1], (1, 0), value=False)
    block = (image & ~previous).long().cumsum(1)
    different = block[:, :, None] != block[:, None, :]
    allowed &= ~(wrists[:, None, :, None] & wrists[:, None, None, :] & different[:, None])
    return allowed


def policy_attention_mask(owner, packed):
    """Leave the default path untouched; route custom masks by decoder type."""
    if image_attention_mode(owner.config) == CAUSAL:
        return packed["attention_mask"]
    validate_image_attention(owner.config)
    language = owner.vlm.model.language_model
    if language.config._attn_implementation != "sdpa":
        raise ValueError("bidirectional_image_blocks requires Qwen SDPA.")
    from transformers.masking_utils import create_recurrent_attention_mask

    padding = packed["attention_mask"]
    return {
        "full_attention": (current_only_image_attention_mask(
            packed, image_token_id=int(owner.vlm.config.image_token_id))
            if image_attention_mode(owner.config) == CURRENT_ONLY_IMAGE_BLOCKS else
            image_block_attention_mask(packed["input_ids"], padding,
                                      image_token_id=int(owner.vlm.config.image_token_id))),
        "linear_attention": create_recurrent_attention_mask(
            config=language.config, inputs_embeds=packed["inputs_embeds"],
            attention_mask=padding, past_key_values=None,
        ),
    }
