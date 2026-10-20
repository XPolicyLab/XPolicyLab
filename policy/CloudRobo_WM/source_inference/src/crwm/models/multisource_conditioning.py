import torch
from torch import nn
import torch.nn.functional as F

class MultiSourceConditioning(nn.Module):

    def __init__(self, text_dim, branch_dims, condition_count=256):
        super().__init__()
        self.state_valid = nn.Linear(96, text_dim, bias=False)
        self.action_valid = nn.Linear(90, branch_dims['action'], bias=False)
        self.conditions = nn.Embedding(condition_count, text_dim)
        self.time_encoder = nn.Sequential(nn.Linear(8 * 9, 128), nn.SiLU(), nn.Linear(128, 128))
        self.time_projection = nn.ModuleDict({k: nn.Linear(128, d, bias=False) for k, d in branch_dims.items()})
        for layer in self.time_projection.values():
            nn.init.zeros_(layer.weight)
        self.register_buffer('frequencies', torch.tensor([0.25, 1.0, 4.0, 16.0]), persistent=True)

    def time(self, branch, grid, valid, dtype):
        grid = torch.where(valid[..., None], grid, 0).float()
        phase = grid[..., None] * self.frequencies * torch.pi
        features = torch.cat([grid[..., None], phase.sin(), phase.cos()], -1).flatten(-2)
        hidden = self.time_encoder(features.to(self.time_encoder[0].weight.dtype))
        return self.time_projection[branch](hidden).to(dtype) * valid[..., None]

def conservative_latent_frames(image_valid):
    if image_valid.shape[-1] != 9:
        raise ValueError('Expected nine visual frames')
    return torch.stack((image_valid[:, 0], image_valid[:, 1:5].all(1), image_valid[:, 5:9].all(1)), 1)

def camera_canvas_mask(camera_valid, height, width):
    B = camera_valid.shape[0]
    mask = torch.zeros(B, 1, 384, 320, device=camera_valid.device)
    mask[:, :, 256:, :160] = camera_valid[:, 1, None, None, None]
    mask[:, :, 256:, 160:] = camera_valid[:, 2, None, None, None]
    mask[:, :, :256, :] = camera_valid[:, 0, None, None, None]
    return F.adaptive_avg_pool2d(mask, (height, width))[:, 0] >= 1 - 1e-07

def label_masks(sample, device, dino_patches=(196, 49, 49)):
    cam = sample['camera_valid'].to(device).bool()
    image = ~sample['image_is_pad'].to(device).bool()
    latent = conservative_latent_frames(image)
    modality = sample['modality_valid'].to(device).bool()
    spatial = camera_canvas_mask(cam, 24, 20)
    video = latent[:, :, None, None] & spatial[:, None] & modality[:, 0, None, None, None]
    point = video & modality[:, 3, None, None, None]
    dcam = torch.cat([cam[:, i, None].expand(-1, n) for i, n in enumerate(dino_patches)], 1)
    dino = image[:, [0, 4, 8], None] & dcam[:, None] & modality[:, 2, None, None]
    frames = sample.get('branch_frame_valid', {})
    for name, mask in (('pointmap', point), ('dino', dino)):
        if name in frames:
            valid = frames[name].to(device).bool()
            if valid.shape != (cam.shape[0], 3):
                raise ValueError(f'{name} branch_frame_valid must be [B,3]')
            mask &= valid.reshape(*valid.shape, *([1] * (mask.ndim - 2)))
    return dict(video=video, pointmap=point, dino=dino, action=sample['action_valid'].to(device).bool())

def restrict_attention(mask, valid):
    if mask.ndim == 2:
        mask = mask[None].expand(valid.shape[0], -1, -1)
    if mask.ndim == 4:
        key = valid[:, None, None, :]
        query = valid[:, None, :, None]
        eye = torch.eye(valid.shape[-1], device=mask.device, dtype=torch.bool)[None, None]
    else:
        key = valid[:, None, :]
        query = valid[:, :, None]
        eye = torch.eye(valid.shape[-1], device=mask.device, dtype=torch.bool)[None]
    return mask & key & query | eye & ~query
