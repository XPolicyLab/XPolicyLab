from __future__ import annotations
import torch
import torch.nn as nn

def get_dino_mesh_id(n_frames: int, cam_regions: list[tuple[int, int, int, int]], cam_patches: list[tuple[int, int]], frame_indices: list[int] | None=None) -> torch.Tensor:
    cam_h_list = []
    cam_w_list = []
    for i, (h_start, h_end, w_start, w_end) in enumerate(cam_regions):
        nh, nw = cam_patches[i]
        i_idx = torch.arange(nh, dtype=torch.float32)
        j_idx = torch.arange(nw, dtype=torch.float32)
        h_pos = (h_start + i_idx * (h_end - h_start) / nh).round().long()
        w_pos = (w_start + j_idx * (w_end - w_start) / nw).round().long()
        h_pos = h_pos.clamp(h_start, h_end - 1)
        w_pos = w_pos.clamp(w_start, w_end - 1)
        hh, ww = torch.meshgrid(h_pos, w_pos, indexing='ij')
        cam_h_list.append(hh.flatten())
        cam_w_list.append(ww.flatten())
    cam_h = torch.cat(cam_h_list)
    cam_w = torch.cat(cam_w_list)
    n_total = cam_h.shape[0]
    if frame_indices is None:
        f_idx = torch.arange(n_frames)
    else:
        f_idx = torch.tensor(frame_indices, dtype=torch.long)
    F_used = f_idx.shape[0]
    ff = f_idx[:, None].expand(F_used, n_total).reshape(-1)
    hh = cam_h[None].expand(F_used, n_total).reshape(-1)
    ww = cam_w[None].expand(F_used, n_total).reshape(-1)
    return torch.stack([ff, hh, ww], dim=0)

def compute_dino_freqs(freqs_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor], n_frames: int, cam_regions: list[tuple[int, int, int, int]], cam_patches: list[tuple[int, int]], device: torch.device, frame_indices: list[int] | None=None) -> torch.Tensor:
    grid = get_dino_mesh_id(n_frames=n_frames, cam_regions=cam_regions, cam_patches=cam_patches, frame_indices=frame_indices)
    f_idx = grid[0].long()
    h_idx = grid[1].long()
    w_idx = grid[2].long()
    freqs_f, freqs_h, freqs_w = freqs_3d
    freqs = torch.cat([freqs_f[f_idx], freqs_h[h_idx], freqs_w[w_idx]], dim=-1)
    return freqs.unsqueeze(1).to(device)

def select_aux_frame_slots(num_latent: int, stride: int, keep_far: bool=False) -> list[int]:
    if keep_far:
        return [0] + sorted(range(num_latent - 1, 0, -stride))
    return [0] + list(range(1, num_latent, stride))
_DINO_X0_SIGMA_MIN = 0.05

def _dino_x0_to_velocity(x_t: torch.Tensor, x0_pred: torch.Tensor, timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    sigma = (timestep.float() / float(num_train_timesteps)).clamp(min=_DINO_X0_SIGMA_MIN)
    return (x_t.float() - x0_pred.float()) / sigma

def _zero_init_dino_x0_head_(head: nn.Module) -> None:
    last = head[-1] if isinstance(head, nn.Sequential) else head
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
