import copy
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .helpers.gradient import gradient_checkpoint_forward
from .multisource_conditioning import restrict_attention

@dataclass(frozen=True)
class PerceptionConfig:
    enabled: bool = False
    lambda_pointmap: float = 0.1
    lambda_dino: float = 0.1
    warmup_steps: int = 500

    def weight(self, step: int) -> float:
        return min((step + 1) / self.warmup_steps, 1.0) if self.warmup_steps else 1.0

def camera_feature_grids(tokens, grid_hw, layout, target_side):
    batch, _, dim = tokens.shape
    height, width = grid_hw
    canvas_h, canvas_w = layout.composite_hw
    features = tokens.transpose(1, 2).reshape(batch, dim, height, width)
    views = []
    for slot in layout.cam_slots():
        numerators = (slot.top * height, (slot.top + slot.h) * height, slot.left * width, (slot.left + slot.w) * width)
        divisors = (canvas_h, canvas_h, canvas_w, canvas_w)
        if any((n % d for n, d in zip(numerators, divisors))):
            raise ValueError('Perception camera boundaries must align with RGB DiT patches')
        top, bottom, left, right = (n // d for n, d in zip(numerators, divisors))
        view = features[:, :, top:bottom, left:right]
        view = F.interpolate(view.float(), size=(target_side, target_side), mode='bilinear', align_corners=False).to(tokens.dtype)
        views.append(view.permute(0, 2, 3, 1))
    return torch.stack(views, dim=1)

class PerceptionHeads(nn.Module):

    def __init__(self, video_expert, layout, native_side, native_dim):
        super().__init__()
        self.layout = layout
        self.native_side = int(native_side)
        self.native_dim = int(native_dim)
        self.pointmap_head = copy.deepcopy(video_expert.head)
        text_dim = video_expert.text_embedding[0].in_features
        self.task_token = nn.Parameter(torch.zeros(1, 1, text_dim))
        self.dino_head = nn.Sequential(nn.LayerNorm(video_expert.hidden_dim), nn.Linear(video_expert.hidden_dim, self.native_dim))
        nn.init.xavier_uniform_(self.dino_head[1].weight)
        nn.init.zeros_(self.dino_head[1].bias)

    def contract(self, config):
        return dict(version=1, config=asdict(config), layout=self.layout.name, native_dino_side=self.native_side, native_dino_dim=self.native_dim, pointmap_output='negative_wan_head_x0', dino_output='native_x0', context='text_and_perception_token', timestep=0)

    def forward(self, video_expert, latents, context, context_mask, rgb_valid, *, checkpoint_layers=False, checkpoint_every=1):
        if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every <= 0:
            raise ValueError('perception checkpoint_every must be a positive integer')
        if latents.ndim != 5 or latents.shape[2] != 1:
            raise ValueError('Perception requires one clean latent frame')
        batch = latents.shape[0]
        context = torch.cat((context, self.task_token.to(context).expand(batch, -1, -1)), dim=1)
        context_mask = torch.cat((context_mask, torch.ones(batch, 1, device=context.device, dtype=torch.bool)), dim=1)
        pre = video_expert.pre_dit(x=latents, timestep=latents.new_zeros(batch), context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=True)
        _, height, width = pre['meta']['grid_size']
        valid = F.adaptive_avg_pool2d(rgb_valid[:, None].float(), (height, width))[:, 0] >= 1 - 1e-07
        valid = valid.flatten(1)
        tokens = pre['tokens'] * valid[..., None]
        mask = restrict_attention(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool), valid)
        checkpoint_layers = checkpoint_layers and torch.is_grad_enabled()
        for index, block in enumerate(video_expert.blocks):
            tokens = gradient_checkpoint_forward(block, checkpoint_layers and index % checkpoint_every == 0, tokens, pre['context'], pre['t_mod'], pre['freqs'], context_mask=pre['context_mask'], self_attn_mask=mask)
        pointmap = -video_expert.unpatchify(self.pointmap_head(tokens, pre['t']), pre['meta']['grid_size'])
        dino = self.dino_head(camera_feature_grids(tokens, (height, width), self.layout, self.native_side))
        return {'pointmap': pointmap, 'dino': dino}
