"""Online frozen teacher: exact preprocessing/loader used for training caches."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch

def load_teacher(args: argparse.Namespace):
    sys.path.insert(0, str(args.acwm_root.resolve()))
    os.environ["CDLAM_ACWM_ROOT"] = str(args.acwm_root.resolve())
    from external.lam.modules import LatentActionModel
    from external.lam.modules.blocks import SelfAttention

    if not getattr(SelfAttention, "_sdpa_patched", False):
        def scaled_dot_product_attention(self, query, key, value, is_causal=False):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=is_causal,
            )

        SelfAttention.scaled_dot_product_attention = scaled_dot_product_attention
        SelfAttention._sdpa_patched = True
    teacher = LatentActionModel(
        in_dim=3,
        model_dim=1024,
        latent_dim=32,
        patch_size=16,
        enc_blocks=24,
        dec_blocks=24,
        num_heads=16,
    )
    checkpoint = torch.load(
        args.teacher_checkpoint.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Released CD-LAM checkpoint must contain a top-level model state dict"
        )
    teacher_state = {
        key.removeprefix("lam."): value
        for key, value in state_dict.items()
        if key.startswith("lam.")
    }
    if len(teacher_state) != len(state_dict):
        raise ValueError("Expected every CD-LAM checkpoint key to start with 'lam.'")
    teacher.load_state_dict(teacher_state, strict=True)
    teacher = teacher.to(args.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher

def center_crop_resize(frame_rgb: np.ndarray) -> np.ndarray:
    """Match CD-LAM's 4:3 center-crop and 480x640 -> 240x320 resize."""
    height, width = frame_rgb.shape[:2]
    target_ratio = 640 / 480
    current_ratio = width / height
    if current_ratio > target_ratio:
        crop_height, crop_width = height, max(1, int(height * target_ratio))
    elif current_ratio < target_ratio:
        crop_height, crop_width = max(1, int(width / target_ratio)), width
    else:
        crop_height, crop_width = height, width
    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    cropped = frame_rgb[top : top + crop_height, left : left + crop_width]
    if cropped.shape[:2] != (480, 640):
        cropped = cv2.resize(cropped, (640, 480), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(cropped, (320, 240), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


class OnlineTeacher:
    def __init__(self, acwm_root, checkpoint, device='cuda', batch_size=16):
        self.device = device
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('teacher_batch_size must be positive')
        self.model = load_teacher(SimpleNamespace(acwm_root=Path(acwm_root),
                                  teacher_checkpoint=Path(checkpoint), device=device))

    @torch.inference_mode()
    def encode(self, pairs):
        result=[]
        for start in range(0,len(pairs),self.batch_size):
            batch=np.stack(pairs[start:start+self.batch_size])
            if batch.shape[1:]!=(2,240,320,3) or batch.dtype!=np.uint8:
                raise ValueError('Teacher pairs must be RGB uint8 [B,2,240,320,3]')
            inputs=torch.from_numpy(batch).to(device=self.device,dtype=torch.float32)/255.0
            with torch.autocast(torch.device(self.device).type,dtype=torch.bfloat16):
                values=self.model.encode(inputs)['z_mu']
            if values.shape!=(len(batch),32) or not torch.isfinite(values).all():
                raise ValueError('Teacher returned invalid 32D means')
            # Match float16 training caches before the float32 mean pooling.
            result.extend(values.float().cpu().numpy().astype(np.float16).astype(np.float32))
        return result
