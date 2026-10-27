from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any
import zlib
import numpy as np


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = False
    n_previous: int = 8
    config_path: str = ''

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> 'MemoryConfig':
        for feature in ('bbox_enabled', 'memory_enabled'):
            if config.get(feature, False):
                raise NotImplementedError(f'{feature} is not implemented in this release')
        enabled = config.get('memory_enabled', False)
        if not isinstance(enabled, bool):
            raise ValueError('memory_enabled must be a YAML boolean')
        n = config.get('memory_n_previous', 8)
        if isinstance(n, bool) or not isinstance(n, (int, str)):
            raise ValueError('memory_n_previous must be a positive integer')
        try:
            n = int(n)
        except (ValueError, TypeError) as e:
            raise ValueError('memory_n_previous must be a positive integer') from e
        if n < 1:
            raise ValueError('memory_n_previous must be a positive integer')
        if enabled and (config.get('ttc_enabled', False) or config.get('bbox_enabled', False)):
            raise ValueError('memory inference requires ttc_enabled=false and bbox_enabled=false')
        return cls(enabled, n, str(config.get('memory_config_path') or cls.config_path))

    def resolve_config(self, original: str, deploy_path: str = '') -> str:
        if not self.enabled:
            return original
        path = Path(self.config_path).expanduser()
        if not path.is_absolute():
            path = (Path(deploy_path).resolve().parent if deploy_path else Path(__file__).parent) / path
        if not path.is_file():
            raise FileNotFoundError(f'memory_config_path not found: {path}')
        return str(path.resolve())

    def metadata(self) -> dict[str, Any]:
        return dict(memory_enabled=self.enabled, memory_n_previous=self.n_previous,
                    memory_sampling='uniform_history_with_current' if self.enabled else None)

    def validate_backend(self, metadata: Mapping[str, Any]) -> None:
        if metadata.get('memory_enabled', False) is not self.enabled:
            raise ValueError('Memory enabled mismatch: restart proxy and backend with the same deploy.yml')
        if self.enabled and (metadata.get('memory_n_previous') != self.n_previous or
                             metadata.get('memory_sampling') != 'uniform_history_with_current'):
            raise ValueError('Memory parameters mismatch between proxy and backend')

    def validate_task_flow(self, task_flow) -> None:
        from policy_core.memory_inference import MemoryWAM_policyTaskFlow
        is_memory = isinstance(task_flow, MemoryWAM_policyTaskFlow)
        if is_memory != self.enabled:
            raise ValueError('memory_enabled does not match the inference config task flow')
        if self.enabled:
            actual = int(task_flow.cfg.data.val.n_previous)
            rope = int(task_flow.cfg.model.video.memory_order_rope_num_mem_frames)
            if actual != self.n_previous or rope != self.n_previous:
                raise ValueError(f'memory_n_previous={self.n_previous} differs from model config ({actual}, {rope})')

    def validate_checkpoint(self, checkpoint: str, task_flow) -> None:
        if not self.enabled:
            return
        import json
        saved = json.loads((Path(checkpoint) / 'config.json').read_text())
        for key in ('base_num_frames', 'memory_order_rope_num_mem_frames',
                    'memory_order_rope_theta', 'memory_order_rope_scale',
                    'block_memory_to_future_attention'):
            configured = getattr(task_flow.cfg.model.video, key)
            if saved.get(key) != configured:
                raise ValueError(f'Memory checkpoint/config mismatch for {key}: {saved.get(key)} != {configured}')

    def validate_request(self, params: Mapping[str, Any]) -> None:
        images = np.asarray(params['obs'])
        if self.enabled:
            if images.ndim != 6 or images.shape[1] != self.n_previous:
                raise ValueError(f'memory requires obs [B,{self.n_previous},V,H,W,C], got {images.shape}')
        elif images.ndim == 6:
            raise ValueError('multi-frame observation received while memory_enabled=false')


class ObservationMemory:
    def __init__(self, n_previous: int):
        self.n_previous = n_previous
        self.histories: dict[int, list[tuple[tuple[int, ...], str, bytes]]] = {}

    def append(self, env_idx: int, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame)
        if frame.min() >= 0 and frame.max() <= 255 and np.equal(frame, np.floor(frame)).all():
            frame = frame.astype(np.uint8)
        entry = (frame.shape, frame.dtype.str, zlib.compress(frame.tobytes(), level=1))
        self.histories.setdefault(env_idx, []).append(entry)

    def sample_indices(self, env_idx: int) -> list[int]:
        current = len(self.histories[env_idx]) - 1
        count = min(self.n_previous - 1, current)
        sampled = np.rint(np.linspace(0, current - 1, num=count)).astype(np.int64).tolist() if count else []
        indices = sorted(sampled + [current])
        return [0] * (self.n_previous - len(indices)) + indices

    def sample(self, env_idx: int) -> np.ndarray:
        history = self.histories[env_idx]
        frames = []
        for index in self.sample_indices(env_idx):
            shape, dtype, data = history[index]
            frames.append(np.frombuffer(zlib.decompress(data), dtype=dtype).reshape(shape))
        return np.ascontiguousarray(np.stack(frames)[None], dtype=np.float32)

    def clear(self) -> None:
        self.histories.clear()
