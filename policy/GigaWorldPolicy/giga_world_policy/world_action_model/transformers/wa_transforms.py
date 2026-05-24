import copy
import random

import numpy as np
import torch
from decord import VideoReader
from giga_datasets import video_utils
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
import torch.nn.functional as torch_F 

from giga_train import TRANSFORMS
from decord import VideoReader
import decord

import h5py
from PIL import Image
from typing import List, Tuple, Optional, Union
import re
import ftfy
import html
from transformers import AutoTokenizer, UMT5EncoderModel

from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from giga_datasets import utils as gd_utils
import pandas as pd

from decord import VideoReader as DecordVideoReader
from torchvision.io import VideoReader as TorchVideoReader
import torch


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def save_video_per_frame(images, path):
    import cv2
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, 16, (width, height))
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    video_writer.release()


import json

import os
import pickle
@TRANSFORMS.register
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
        print("Loading stats dict from:", json_path)
        self.use_delta = True
        print("Using delta mode")

    def __call__(self, data_dict):
        dst_width, dst_height = data_dict['video_info']
        if self.num_views == 1:
            if 'front_video_path' in data_dict:
                front_images = DecordVideoReader(data_dict['front_video_path'])
                video_legnth = len(front_images)
                assert len(front_images)
        elif self.num_views == 3:

            def read_video(video_path):
                from torchvision.io import VideoReader as TorchVideoReader
                reader = TorchVideoReader(video_path, 'video')
                cv_frame_list = []
                frame_count = 0
                for frame in reader:
                    frame_count += 1
                    last_pts = frame['pts']
                    frame_np = frame['data'].cpu().numpy()
                    # Convert frame to a NumPy array (HWC format)
                    # (3, 480, 640) -> (480, 640, 3)
                    frame_np = np.transpose(frame_np, (1, 2, 0))
                    cv_frame = frame_np
                    cv_frame_list.append(cv_frame)  # hdf5 -> mp4
                cv_frame_list = np.array(cv_frame_list)
                return cv_frame_list

            front_view_images = read_video(data_dict['front_video_path'])
            left_view_images = read_video(data_dict['left_video_path'])
            right_view_images = read_video(data_dict['right_video_path'])

            assert len(front_view_images) == len(left_view_images) == len(right_view_images)
            video_legnth = len(front_view_images)


        start_frame = random.randint(0, max(0, video_legnth - self.num_frames - 2))
        end_frame = start_frame + self.num_frames
        assert end_frame < video_legnth
        sample_indexes = np.linspace(start_frame, end_frame, num=5, dtype=int)

        action = data_dict['action'][start_frame:end_frame]
        state = data_dict['state'][start_frame:start_frame+1]
        mask = np.array([True, True, True, True, True, True, False, True, True, True, True, True, True, False])
        delta = action.copy()
        delta[:, mask] = action[:, mask] -  state[:, mask] 

        state_max = np.array(self.stats_dict['norm_stats']['observation.state' ]['max'])[:14]
        state_min = np.array(self.stats_dict['norm_stats']['observation.state' ]['min'])[:14]
        delta_max =   np.array(self.stats_dict['norm_stats']['action' ]['max'])[:14]
        delta_min =   np.array(self.stats_dict['norm_stats']['action']['min'])[:14]
        eps = 1e-8  # small epsilon to avoid division by zero
        state_range = state_max - state_min + eps
        delta_range = delta_max - delta_min + eps

        norm_state = (state - state_min) / state_range
        norm_delta = (delta - delta_min) / delta_range
        norm_state = norm_state * 2.0 - 1.0
        norm_delta = norm_delta * 2.0 - 1.0
        def get_input_images(video):
            if isinstance(video, VideoReader):
                input_images = video_utils.sample_video(video, sample_indexes, method=2)
            else:
                input_images = video[sample_indexes]
            data_dict['input_fps'] = self.fps
            input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()
            height = input_images.shape[2]
            width = input_images.shape[3]
            if float(dst_height) / height < float(dst_width) / width:
                new_height = int(round(float(dst_width) / width * height))
                new_width = dst_width
            else:
                new_height = dst_height
                new_width = int(round(float(dst_height) / height * width))
            input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
            x1 = random.randint(0, new_width - dst_width)
            y1 = random.randint(0, new_height - dst_height)
            input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
            input_images = input_images / 255.0
            input_images = self.normalize(input_images)
            return input_images

        if self.num_views == 1:
            data_dict['input_front_images'] = get_input_images(front_images)

        elif self.num_views == 3:
            front_images = get_input_images(front_view_images)
            left_images = get_input_images(left_view_images)
            right_images = get_input_images(right_view_images)
            data_dict['input_images'] = torch.cat([front_images, left_images, right_images], dim=-1)

        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict['input_images'].shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = copy.deepcopy(data_dict['input_images'])
            ref_images = ref_images * ref_masks
            data_dict['input_ref_images'] = ref_images
            data_dict['input_ref_masks'] = ref_latent_masks

        if self.is_train:
            new_data_dict = {}
            if 'input_fps' in data_dict:
                new_data_dict['fps'] = data_dict['input_fps']
            if 'input_images' in data_dict:
                new_data_dict['images'] = data_dict['input_images']
            if 'input_ref_images' in data_dict:
                new_data_dict['ref_images'] = data_dict['input_ref_images']
                new_data_dict['ref_masks'] = data_dict['input_ref_masks']
                t5_len = int(self.t5_len)
                prompt = data_dict["t5_embedding"][:t5_len]
                new_data_dict["prompt_embeds"] = torch_F.pad(prompt, (0, 0, 0, t5_len - prompt.shape[0]), value=0)
                if self.use_delta:
                    new_data_dict['action'] = norm_delta
                else:
                    new_data_dict['action'] = norm_action
                new_data_dict['state'] = norm_state

        else:
            assert False
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict

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
