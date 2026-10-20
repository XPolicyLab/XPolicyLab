"""Fixed current pointmap masking for CR-WM inference."""
import torch

def branch_frame_valid(device):
    return dict(pointmap=torch.tensor([[False, True, True]], device=device),
                dino=torch.ones(1, 3, device=device, dtype=torch.bool))
