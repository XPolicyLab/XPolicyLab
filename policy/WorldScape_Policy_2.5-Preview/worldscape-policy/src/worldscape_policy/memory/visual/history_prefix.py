"""Nine-frame visual history adapted from ActiveMemory-WAM."""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, fields
import numpy as np
import torch
from torch import Tensor, nn
from .history import CausalVisualHistory, prepare_history_frames

@dataclass(frozen=True)
class HistoryMemoryConfig:
    enabled: bool = False
    history_frames: int = 8
    stride: int = 24
    train_stride_jitter: bool = True
    encode_microbatch: int = 1
    image_layout: str = "mosaic_T"
    include_current: bool = True  # False: episode first + eight strictly past slots.

    def __post_init__(self):
        if self.image_layout not in {"mosaic_T"}:
            raise ValueError("unsupported history image_layout")
        for name in ('enabled', 'train_stride_jitter', 'include_current'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'history_memory.{name} must be bool')
        if type(self.history_frames) is not int or self.history_frames != 8:
            raise ValueError('history_memory.history_frames must be 8 (nine RGB clip slots)')
        if type(self.stride) is not int or self.stride != 24:
            raise ValueError('history_memory.stride must be 24 (one action chunk)')
        if type(self.encode_microbatch) is not int or self.encode_microbatch < 1:
            raise ValueError('history_memory.encode_microbatch must be a positive integer')

    @classmethod
    def from_dict(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError('history_memory must be a mapping')
        raw = dict(value)
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown history_memory fields: {sorted(unknown)}')
        return cls(**raw)

def sample_history_ids(anchor: int, config: HistoryMemoryConfig, *, training=False, rng=None) -> list[int]:
    """Return nine native-step slots, retaining clamped / duplicate first frames.

Training uses the upstream categorical stride and independent 10% +/-1 jitter
on each past frame. Supplying a NumPy Generator or RandomState makes the draw
reproducible; dataset callers seed it with their existing seed + sample index.
"""
    if type(anchor) is not int or anchor < 0:
        raise ValueError('history anchor must be a nonnegative integer')
    if not isinstance(config, HistoryMemoryConfig):
        raise TypeError('config must be HistoryMemoryConfig')
    stride = config.stride
    augment = bool(training and config.train_stride_jitter)
    if augment:
        rng = np.random.default_rng() if rng is None else rng
        stride = int(rng.choice([config.stride + delta for delta in (-4, -2, 0, 2, 4)], p=[.07, .07, .72, .07, .07]))
    past_slots = config.history_frames - int(config.include_current)
    offsets = (np.arange(config.history_frames, dtype=np.int64) - past_slots) * stride
    if augment:
        selected = rng.random(past_slots) < .1
        if bool(selected.any()):
            draw = rng.integers if hasattr(rng, 'integers') else rng.randint
            magnitude = draw(1, 2, size=past_slots)
            sign = rng.choice(np.asarray([-1, 1], dtype=np.int64), size=past_slots)
            offsets[:past_slots] += selected.astype(np.int64) * magnitude * sign
    valid_last = offsets[-1] == 0 if config.include_current else offsets[-1] < 0
    if not valid_last or bool((np.diff(offsets) <= 0).any()):
        raise RuntimeError('history augmentation must preserve ordered history and its current-frame policy')
    return [0, *np.maximum(anchor + offsets, 0).tolist()]


def select_history_frames(history: CausalVisualHistory) -> Tensor:
    """Select the stored nine slots, retaining clamped episode-first repeats."""
    history.validate()
    ids = history.step_ids.detach().cpu().tolist()
    masks = history.mask.detach().cpu().tolist()
    slots = history.clip_step_ids.detach().cpu().tolist()
    clips = []
    for batch, (row, mask, clip) in enumerate(zip(ids, masks, slots, strict=True)):
        positions = {step: i for i, (step, valid) in enumerate(zip(row, mask, strict=True)) if valid}
        selection = torch.tensor([positions[step] for step in clip], device=history.frames.device)
        clips.append(history.frames[batch].index_select(0, selection))
    return torch.stack(clips)

class HistoryEncoder(nn.Module):
    """Use the frozen VAE on nine-frame clips, never on independent frames."""
    def __init__(self, config: HistoryMemoryConfig):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def encode_history(self, history: CausalVisualHistory, codec) -> Tensor:
        history.validate()
        if history.include_current != self.config.include_current:
            raise ValueError('history payload and model include_current settings differ')
        if any(p.requires_grad for p in codec.vae.parameters()):
            raise ValueError('visual history encoding requires a frozen VAE')
        clips = list(select_history_frames(history).unbind(0))
        parameter = next(codec.vae.parameters(), None)
        device_type = parameter.device.type if parameter is not None else history.frames.device.type
        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
        elif parameter is not None and parameter.dtype in (torch.bfloat16, torch.float16):
            dtype = parameter.dtype
        else:
            dtype = torch.float32
        outputs = []
        for start in range(0, len(clips), self.config.encode_microbatch):
            inputs = torch.stack(clips[start:start + self.config.encode_microbatch])
            with torch.autocast(device_type=device_type, dtype=dtype,
                                enabled=dtype in (torch.bfloat16, torch.float16)):
                encoded = codec.encode_normalized(inputs)
            if encoded.ndim != 5 or encoded.shape[:3] != (inputs.shape[0], getattr(codec.vae, "z_dim", 48), 3):
                raise ValueError('history nine-frame encoding must produce [B,vae.z_dim,3,H,W]')
            outputs.append(encoded)
        return torch.cat(outputs, dim=0).detach()


class EpisodeHistoryArchive:
    """Keep the first image and recent real observations needed by history.

    Dense capture remains the default. An explicit observation_stride=24 is
    valid for evaluation that replans only at native steps 0, 24, 48, ...:
    every sampler slot then falls on that same grid. Step IDs remain native
    control steps, and missing capture boundaries are rejected. Repeated current
    observations are idempotent; prediction/commit do not alter the archive.
    """
    def __init__(self, config: HistoryMemoryConfig, *, max_bytes=2 * 1024**3,
                 observation_stride=1):
        if not isinstance(config, HistoryMemoryConfig):
            raise TypeError('config must be HistoryMemoryConfig')
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('history max_bytes must be a positive integer')
        if type(observation_stride) is not int or observation_stride not in (1, config.stride):
            raise ValueError('history observation_stride must be 1 or the configured history stride (24)')
        self.config, self.max_bytes = config, max_bytes
        self.observation_stride = observation_stride
        self.reset('uninitialized')

    def reset(self, episode_id: str):
        self.episode_id = str(episode_id)
        self.frames = OrderedDict()
        self.bytes = 0

    def append(self, step_id: int, views: Tensor):
        previous = next(reversed(self.frames)) if self.frames else -self.observation_stride
        if (type(step_id) is not int or step_id < 0
                or step_id not in (previous, previous + self.observation_stride)):
            raise ValueError(
                'history requires every real control step on its '
                f'observation_stride={self.observation_stride} grid in order, starting at zero'
            )
        if step_id == previous:
            return
        options = ({"layout": self.config.image_layout}
                   if self.config.image_layout != "mosaic_T" else {})
        frame = prepare_history_frames(views.detach().cpu().unsqueeze(0), **options)[0]
        past_slots = self.config.history_frames - int(self.config.include_current)
        recent_length = past_slots * self.config.stride + 2
        expired = [step for step in self.frames if step != 0 and step < step_id - recent_length + 1]
        reclaimed = sum(self.frames[step].numel() * self.frames[step].element_size() for step in expired)
        size = frame.numel() * frame.element_size()
        if self.bytes - reclaimed + size > self.max_bytes:
            raise RuntimeError('visual history byte limit reached; increase max_bytes explicitly')
        for step in expired:
            del self.frames[step]
        self.frames[step_id] = frame
        self.bytes += size - reclaimed

    def snapshot(self, anchor: int, *, device='cpu') -> CausalVisualHistory:
        if not self.frames or anchor != next(reversed(self.frames)):
            raise ValueError('append the exact current observation before reading visual history')
        slots = sample_history_ids(anchor, self.config)
        unique = sorted(set(slots))
        if any(step not in self.frames for step in unique):
            raise ValueError('required visual history frame is missing')
        result = CausalVisualHistory(
            torch.stack([self.frames[step] for step in unique])[None].to(device),
            torch.tensor([unique], dtype=torch.long, device=device),
            torch.ones(1, len(unique), dtype=torch.bool, device=device),
            torch.tensor([anchor], dtype=torch.long, device=device), (self.episode_id,),
            clip_step_ids=torch.tensor([slots], dtype=torch.long, device=device),
            include_current=self.config.include_current,
        )
        result.validate()
        return result
