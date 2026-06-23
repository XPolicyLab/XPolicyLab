import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as torch_F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class MaskGenerator:
    def __init__(self, max_ref_frames, factor=8, start=1):
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames):
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames
        num_latents = 1 + (num_frames - 1) // self.factor
        num_ref_latents = random.randint(self.start, self.max_ref_latents)
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0
        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1
        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1
        return ref_masks, ref_latent_masks


class WATransforms:
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        norm_path=None,
        image_cfg=None,
        num_views=1,
        t5_len=32,
    ):
        self.fps = fps
        self.is_train = is_train
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.dst_size = dst_size
        self.num_frames = num_frames
        self.image_cfg = image_cfg
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])
        self.num_views = num_views
        self.t5_len = int(t5_len)

        json_path = norm_path
        with open(json_path, "r", encoding="utf-8") as f:
            self.stats_dict = json.load(f)
        if os.environ.get("RANK", "0") == "0":
            print("Loading stats dict from:", json_path)
        self.use_delta = True

    def __call__(self, data_dict):
        raise NotImplementedError("WATransforms.__call__ requires giga_datasets video pipeline; use WATransformsLerobot instead.")
