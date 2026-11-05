"""Chunked, differentiable full-frame head history encoding."""

import math
import os

import torch

from focus_vlwa.data.head_history import HISTORY_FRAMES, IMAGE_SIZE, POOL_SIZE, TOKENS_PER_FRAME


def pool_history(features):
    """Average-pool a square feature grid into 4 x 4 cells."""
    batch, count, width = features.shape
    side = math.isqrt(count)
    if side * side != count or side % POOL_SIZE:
        raise ValueError(f"Expected square grid divisible by 4, got {count} tokens")
    return features.reshape(batch, POOL_SIZE, side // POOL_SIZE, POOL_SIZE, side // POOL_SIZE, width).mean(
        dim=(2, 4)
    ).reshape(batch, TOKENS_PER_FRAME, width)


def encode_head_history(encode_image, history, mask, reference):
    """Encode adjacent slots in batches, then zero and mask invalid slots."""
    batch = reference.shape[0]
    expected = (batch, HISTORY_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
    if tuple(history.shape) != expected:
        raise ValueError(f"Expected full main-view history {expected}, got {tuple(history.shape)}")
    if mask is None or tuple(mask.shape) != (batch, HISTORY_FRAMES):
        raise ValueError("Full-frame history requires a [batch, 20] validity mask")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("History validity mask must be binary")
    chunk_size = int(os.environ.get("FOCUS_VLWA_HISTORY_CHUNK_SIZE", "4"))
    if chunk_size < 1 or HISTORY_FRAMES % chunk_size:
        raise ValueError(f"History chunk size must divide {HISTORY_FRAMES}, got {chunk_size}")
    valid = mask.to(device=reference.device, dtype=torch.bool).repeat_interleave(TOKENS_PER_FRAME, dim=1)
    history = history.to(device=reference.device)
    features = []
    for start in range(0, HISTORY_FRAMES, chunk_size):
        images = history[:, start:start + chunk_size].reshape(
            batch * chunk_size, IMAGE_SIZE, IMAGE_SIZE, 3
        ).permute(0, 3, 1, 2)
        pooled = pool_history(encode_image(images))
        features.append(pooled.reshape(batch, chunk_size * TOKENS_PER_FRAME, -1))
    tokens = torch.cat(features, dim=1).to(reference.dtype)
    return torch.where(valid[..., None], tokens, 0), valid
