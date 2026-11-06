# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import concurrent.futures

import numpy as np
import torch

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

__all__ = ['get_mesh_id', 'sample_timestep_id']



def get_mesh_id(f, h, w, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij')
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    grid_id = torch.cat([grid_id], dim=0)
    return grid_id



def sample_timestep_id(
    batch_size: int = 1,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
):
    u = torch.rand(size=[batch_size], generator=generator, device=device)
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    timestep_id = (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)
    return timestep_id
