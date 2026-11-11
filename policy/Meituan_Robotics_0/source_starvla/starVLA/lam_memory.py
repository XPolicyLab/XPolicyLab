"""Causal, episode-long 1 Hz memory from frozen stride-4 CD-LAM means.

Cache row s encodes images (s, s+stride). Bin by END time, never by start.
Only complete seconds are exposed. Clamped episode-tail pairs are excluded.
"""
import numpy as np
import torch
from torch import nn

MEMORY_MARKER = '<|fim_pad|>'


def validate_window_seconds(window_sec):
    if isinstance(window_sec, bool) or not np.isfinite(float(window_sec)) or float(window_sec) <= 0 or int(window_sec) != float(window_sec):
        raise ValueError('LAM window_sec must be a positive integer number of completed seconds')
    return int(window_sec)


def limit_memory_window(memory, window_sec=None):
    """Shared train/inference window contract; retain absolute episode timestamps.

    A 20-second window contains the latest 20 completed one-second motion bins.
    First-frame RGB is separately controlled by visual_history.include_first.
    """
    features, ends, now = validate_memory(memory)
    declared = memory.get('window_sec')
    if window_sec is None:
        if declared is not None:
            raise ValueError('Episode-long model requires the full memory prefix')
        return memory
    width = validate_window_seconds(window_sec)
    if declared is not None and validate_window_seconds(declared) != width:
        raise ValueError('Input and model LAM windows differ')
    return dict(features=features[-width:].copy(), end_times_sec=ends[-width:].copy(),
                current_time_sec=now, window_sec=width)


def validate_memory(memory):
    if memory is None:
        raise ValueError('LAM memory is required; use an explicit empty prefix at episode start')
    features = np.asarray(memory['features'], dtype=np.float32)
    ends = np.asarray(memory['end_times_sec'], dtype=np.float32)
    now = float(memory['current_time_sec'])
    if features.shape != (len(ends), 32) or ends.ndim != 1:
        raise ValueError('Invalid LAM memory feature/time shape')
    if not np.isfinite(features).all() or not np.isfinite(ends).all() or not np.isfinite(now) or now < 0:
        raise ValueError('Nonfinite or negative LAM memory time/features')
    expected = np.arange(1, int(np.floor(now + 1e-6)) + 1, dtype=np.float32)
    if memory.get('window_sec') is not None:
        width = validate_window_seconds(memory['window_sec'])
        expected = expected[-width:]
    if not np.array_equal(ends, expected):
        raise ValueError('LAM memory must contain every completed second in its declared window')
    return features, ends, now


class LAMMemoryProjection(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.feature_proj = nn.Sequential(nn.Linear(32, 256), nn.SiLU(), nn.Linear(256, hidden_dim))
        self.time_proj = nn.Linear(32, hidden_dim, bias=False)

    def forward(self, features, end_times, now):
        # Absolute episode time AND age; no learned length limit.
        freq = torch.exp(torch.arange(8, device=features.device, dtype=torch.float32) * (-np.log(10000) / 8))
        times = torch.stack([end_times.float(), (now - end_times.float()).clamp_min(0)], dim=-1)
        phase = times[..., None] * freq
        time_features = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)
        return self.feature_proj(features) + self.time_proj(time_features.to(features.dtype))


def inject_memory_embeddings(model, projection, batch_inputs, memories, marker_id):
    """Replace reserved text slots; retain native image embeddings and M-RoPE."""
    result = dict(batch_inputs)
    ids = result.pop('input_ids')
    embeds = model.get_input_embeddings()(ids)
    if len(memories) != ids.shape[0]:
        raise ValueError('LAM memory batch size mismatch')
    output = []
    for i, memory in enumerate(memories):
        features, ends, now = validate_memory(memory)
        slots = torch.nonzero(ids[i] == marker_id, as_tuple=False).flatten()
        if len(slots) != len(features):
            raise ValueError('LAM placeholder count differs from feature count')
        dtype = projection.feature_proj[0].weight.dtype
        projected = projection(torch.as_tensor(features, device=embeds.device, dtype=dtype),
                               torch.as_tensor(ends, device=embeds.device), now).to(embeds.dtype)
        # Empty prefixes still participate in autograd / DDP with zero gradients.
        output.append(embeds[i].index_copy(0, slots, projected) + projected.sum() * 0)
    result['inputs_embeds'] = torch.stack(output)
    # inputs_embeds alone cannot reconstruct the multimodal positions correctly.
    result['position_ids'], _ = model.model.get_rope_index(
        input_ids=ids, mm_token_type_ids=result['mm_token_type_ids'],
        image_grid_thw=result.get('image_grid_thw'), attention_mask=result.get('attention_mask'))
    return result


class LAMMemoryBank:
    """Online counterpart: caller supplies each completed frozen-teacher pair.

    At 25 FPS, append rows (s, s+4) in order, starting with (0,4).
    This class retains pooled vectors, never raw RGB or model KV caches.
    Call reset on each episode. Teacher extraction belongs to the observation loop.
    """
    def __init__(self, window_sec=None):
        self.window_sec = window_sec
        self.reset()

    def reset(self):
        self.next_end_frame = 4
        self.sums = {}
        self.counts = {}

    def append(self, end_frame, latent):
        if end_frame != self.next_end_frame:
            raise ValueError('LAM bank requires every completed stride-4 pair in order')
        value = np.asarray(latent, dtype=np.float32)
        if value.shape != (32,) or not np.isfinite(value).all():
            raise ValueError('LAM bank expects one finite 32D teacher mean')
        bucket = (end_frame + 24) // 25
        self.sums[bucket] = self.sums.get(bucket, np.zeros(32, dtype=np.float32)) + value
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        self.next_end_frame += 1

    def select(self, current_frame):
        if current_frame < 0 or self.next_end_frame != max(4, current_frame + 1):
            raise ValueError('LAM bank must be synchronized with the current observation')
        seconds = current_frame // 25
        features = [self.sums[k] / self.counts[k] for k in range(1, seconds + 1)]
        memory = dict(features=np.asarray(features, dtype=np.float32).reshape(-1, 32),
                      end_times_sec=np.arange(1, seconds + 1, dtype=np.float32),
                      current_time_sec=current_frame / 25)
        return limit_memory_window(memory, self.window_sec)
