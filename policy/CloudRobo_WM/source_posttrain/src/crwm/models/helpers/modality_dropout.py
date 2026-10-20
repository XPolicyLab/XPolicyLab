from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModalityDropoutConfig:
    enabled: bool = False
    p_present_video: float = 1.0
    p_present_dino: float = 1.0
    p_present_pointmap: float = 1.0
    p_jv: float = 0.5
    p_jd: float = 0.5
    p_jp: float = 0.5
    cross_modal_predict_video: bool = False
    cross_modal_predict_dino: bool = False
    cross_modal_predict_pointmap: bool = False

    def __post_init__(self):
        for name in ('p_present_video', 'p_present_dino', 'p_present_pointmap', 'p_jv', 'p_jd', 'p_jp'):
            val = float(getattr(self, name))
            if not 0.0 <= val <= 1.0:
                raise ValueError(f'ModalityDropoutConfig.{name} must be in [0, 1], got {val}')
            setattr(self, name, val)

@dataclass
class ModalityBatchFlags:
    present_v: torch.Tensor
    present_d: torch.Tensor
    present_p: torch.Tensor
    j_v: torch.Tensor
    j_d: torch.Tensor
    j_p: torch.Tensor
    cm_v: bool
    cm_d: bool
    cm_p: bool

    @property
    def B(self) -> int:
        return int(self.present_v.shape[0])

def sample_modality_flags(cfg: ModalityDropoutConfig, batch_size: int, device: torch.device, generator: Optional[torch.Generator]=None, pointmap_off: bool=False) -> ModalityBatchFlags:

    def _bern(p: float) -> torch.Tensor:
        if p >= 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if p <= 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        u = torch.rand(batch_size, device=device, generator=generator)
        return u < p
    present_v = _bern(cfg.p_present_video)
    present_d = _bern(cfg.p_present_dino)
    present_p = _bern(0.0 if pointmap_off else cfg.p_present_pointmap)
    positive_streams: list[str] = []
    if cfg.p_present_video > 0:
        positive_streams.append('v')
    if cfg.p_present_dino > 0:
        positive_streams.append('d')
    if cfg.p_present_pointmap > 0 and (not pointmap_off):
        positive_streams.append('p')
    if positive_streams:
        all_false = ~present_v & ~present_d & ~present_p
        if all_false.any():
            n = len(positive_streams)
            pick = torch.randint(0, n, (batch_size,), device=device, generator=generator)
            for i, name in enumerate(positive_streams):
                flip = all_false & (pick == i)
                if name == 'v':
                    present_v = present_v | flip
                elif name == 'd':
                    present_d = present_d | flip
                else:
                    present_p = present_p | flip
    j_v = _bern(cfg.p_jv)
    j_d = _bern(cfg.p_jd)
    j_p = _bern(0.0 if pointmap_off else cfg.p_jp)
    cm_p = bool(cfg.cross_modal_predict_pointmap) and (not pointmap_off)
    if not cfg.cross_modal_predict_video:
        j_v = j_v & present_v
    if not cfg.cross_modal_predict_dino:
        j_d = j_d & present_d
    if not cm_p:
        j_p = j_p & present_p
    return ModalityBatchFlags(present_v=present_v, present_d=present_d, present_p=present_p, j_v=j_v, j_d=j_d, j_p=j_p, cm_v=bool(cfg.cross_modal_predict_video), cm_d=bool(cfg.cross_modal_predict_dino), cm_p=cm_p)
