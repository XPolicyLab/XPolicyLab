"""Fill unused T5 positions with vlm instruction/summary tokens.

Keep the original 512-position, unmasked Wan condition, including its padding
semantics. A zero-initialized bias-free vlm projector is exactly T5-only.
"""
from dataclasses import replace

import torch


def instruction_lengths(t5, instructions):
    """Untruncated T5 lengths, with original whitespace cleaning and EOS.

    This loads only a tokenizer for offline encoders, never T5 weights. Cache
    Python lengths because the same task prompts recur on every training step.
    """
    if instructions is None:
        raise ValueError('T5 + vlm requires language_instruction')
    cache = getattr(t5, '_concat_length_cache', None)
    if cache is None:
        cache = t5._concat_length_cache = {}
    missing = list(dict.fromkeys(text for text in instructions if text not in cache))
    if missing:
        tokenizer = getattr(t5, '_concat_tokenizer', None)
        if tokenizer is None:
            tokenizer = getattr(t5, '_tokenizer', None)
            if tokenizer is None:
                from .text.tokenizer import HuggingfaceTokenizer
                tokenizer = HuggingfaceTokenizer(name=t5.tokenizer_path, clean='whitespace')
            t5._concat_tokenizer = tokenizer
        _, mask = tokenizer(missing, return_mask=True, add_special_tokens=True,
                            padding=True, truncation=False, max_length=None)
        cache.update(zip(missing, mask.sum(-1).tolist(), strict=True))
    return [int(cache[text]) for text in instructions]


def append_to_t5_padding(t5_tokens, projected, prefix_mask, lengths):
    if t5_tokens.ndim != 3 or t5_tokens.shape[1] != 512:
        raise ValueError('T5 + vlm requires [B,512,D] T5 features')
    if (projected.ndim != 3 or projected.shape[0] != t5_tokens.shape[0]
            or projected.shape[-1] != t5_tokens.shape[-1]
            or prefix_mask is None or prefix_mask.dtype != torch.bool
            or prefix_mask.shape != projected.shape[:2]
            or len(lengths) != len(t5_tokens)):
        raise ValueError('T5 + vlm feature/mask/batch dimensions differ')
    counts = prefix_mask.sum(-1)
    if not torch.equal(prefix_mask, torch.arange(prefix_mask.shape[1], device=prefix_mask.device)[None] < counts[:, None]):
        raise ValueError('vlm valid tokens must be a contiguous prefix')
    projected = projected.to(device=t5_tokens.device, dtype=t5_tokens.dtype)
    rows = []
    for i, (length, count) in enumerate(zip(lengths, counts.tolist(), strict=True)):
        if length <= 0 or count <= 0 or length + count > 512:
            raise ValueError(f'T5 + vlm token budget exceeded/empty: T5={length}, vlm={count}, limit=512; no truncation allowed')
        # No masking or shifting of EOS. Retain baseline tail values exactly.
        # Baseline T5 guarantees zero padding; adding preserves zero-init parity
        # even if a future encoder uses nonzero padding representations.
        rows.append(torch.cat((t5_tokens[i, :length],
                               t5_tokens[i, length:length + count] + projected[i, :count],
                               t5_tokens[i, length + count:]), dim=0))
    return torch.stack(rows)


def fuse_t5_vlm(baseline, prefix, t5, prompts):
    if baseline.cross_attention_mask is not None:
        raise ValueError('T5 + vlm requires the original unmasked T5 condition')
    lengths = instruction_lengths(t5, prompts.language_instruction)
    tokens = append_to_t5_padding(baseline.cross_attention_tokens,
                                 prefix.cross_attention_tokens, prefix.cross_attention_mask, lengths)
    negative = tokens
    if prompts.negative_language_instruction is not None:
        negative = append_to_t5_padding(baseline.negative_cross_attention_tokens,
                                       prefix.cross_attention_tokens, prefix.cross_attention_mask,
                                       instruction_lengths(t5, prompts.negative_language_instruction))
    return replace(baseline, cross_attention_tokens=tokens,
                   negative_cross_attention_tokens=negative)
