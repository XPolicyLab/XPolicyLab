# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations
from re import U
import numpy as np
import torch
import random
from mmengine import fileio
from scipy.interpolate import interp1d
from ..utils import open_h5, quat_to_rotate6d
from PIL import Image
from .base import DomainHandler
import json

# Human Labeled Split file
SPLITFILE = None
USE_WRIST_VIEW = False

# Task instruction per domain
DOMAIN2INS = {
    "agiworld-on-site-pack": "Pick up the object and place it in the bag.",
    "agiworld-on-site-pack-extra": "Pick up the object and place it in the bag.",
    "agiworld-on-site-conveyor": "Pick objects from the conveyor belt and place them in the box.",
    "agiworld-on-site-conveyor-extra": "Pick objects from the conveyor belt and place them in the box.",
    "agiworld-on-site-restock": "Hang the snacks on the shelf.",
    "agiworld-on-site-pour": "pour the water into the cup.", # "stop and place the cup on the table."
    "agiworld-on-site-microwave": "Open the microwave, put the food in", # close and start it.
    "agiworld-on-site-cloth": "fold the clothes.",
}


# Max chunk length per domain

# shorter chunk version
DOMAIN2CHUNKSIZE = {
    "agiworld-on-site-pack": 61,
    "agiworld-on-site-pack-extra": 61,
    "agiworld-on-site-conveyor": 61,
    "agiworld-on-site-conveyor-extra": 61,
    "agiworld-on-site-restock": 61,
    "agiworld-on-site-pour": 61,
    "agiworld-on-site-microwave": 121,
    "agiworld-on-site-cloth": 121
}


import torch


class AGIWolrdHandler(DomainHandler):  # Note: "Wolrd" looks like a typo; kept for compatibility
    def read_action(self, item: str):
        """Read actions and end-effector states; return joint+gripper and 6D EE features."""
        # Different file layout for "extra" datasets
        action_path = (fileio.join_path(item, 'aligned_joints.h5')
                       if 'extra' not in self.meta['dataset_name']
                       else fileio.join_path(item, 'proprio_stats.h5'))
        
        with open_h5(str(action_path)) as f:
            try:
                # Some versions: grippers under action/effector/position (two columns: L, R)
                gripper_left = f['action']['effector']['position'][:, 0]   # [T]
                gripper_right = f['action']['effector']['position'][:, 1]  # [T]
            except Exception:
                # Fallback: split under state/left_effector and state/right_effector
                gripper_left = f['action']['left_effector']['position'][:, 0]
                gripper_right = f['action']['right_effector']['position'][:, 0]

            joints = f['state']['joint']['position'][:]                  # [T, 14]
            assert len(gripper_left) == joints.shape[0], "gripper/joint length mismatch"

            xyz_position_left = f['state']['end']['position'][:, 0]      # [T, 3]
            xyz_position_right = f['state']['end']['position'][:, 1]     # [T, 3]
            
            orientation_left = f['state']['end']['orientation'][:, 0]    # [T, 4]
            orientation_right = f['state']['end']['orientation'][:, 1]   # [T, 4]
            
            # Concatenate joints and grippers -> [T, 16]
            abs_joint = np.concatenate([joints,
                                        gripper_left[:, None],
                                        gripper_right[:, None]], axis=-1)

            # Build 6D rotations + XYZ + gripper for both arms
            abs_ee6d = np.concatenate([
                xyz_position_left, quat_to_rotate6d(orientation_left), gripper_left[:, None],
                xyz_position_right, quat_to_rotate6d(orientation_right), gripper_right[:, None]
            ], axis=-1)

        return abs_joint, abs_ee6d

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool, action_mode,
                     image_aug, lang_aug_map: dict | None):
        """
        Sample multiple start indices within a trajectory, crop a window, resample to fixed steps,
        and yield model-ready samples.
        Windows are limited around gripper-state changes to avoid spanning multiple sub-tasks.
        """
        item = self.meta["datalist"][traj_idx]

        abs_joint, abs_ee6d = self.read_action(item)

        # Mark gripper changes across time (boolean over T-1)
        grippers = abs_joint[:, -2:]
        chg = np.any(grippers[1:] != grippers[:-1], axis=1)
        gripper_change_idx = np.flatnonzero(chg)
        
        # Domain instruction
        ins = DOMAIN2INS[self.meta['dataset_name']]

        # Build (start, end) segment list; allow external split_list if provided
        
        current_ep_idx = item.split('/')[-1]
        try:
            with open(SPLITFILE, "r") as f: 
                split_data = json.load(f) 
        except: split_data = {}
        split_list = [0]
        if current_ep_idx in split_data.keys():
            # print("current_ep_idx:", current_ep_idx)
            split_list.extend(split_data[current_ep_idx])
        split_list.append(len(abs_joint))
        # Drop very short segments
        split_list = [(a, b) for a, b in zip(split_list[:-1], split_list[1:])]
        
        random.shuffle(split_list)

        for traj_start_idx, traj_end_idx in split_list:
            # Candidate start indices; keep tail room
            index_list = list(range(traj_start_idx, 
                                        max(traj_start_idx + 1, 
                                        traj_end_idx - DOMAIN2CHUNKSIZE[self.meta['dataset_name']])))
            if self.meta['dataset_name'] == 'agiworld-on-site-pour':
                for idx in gripper_change_idx: 
                    for i in range(idx-DOMAIN2CHUNKSIZE[self.meta['dataset_name']], \
                                idx+DOMAIN2CHUNKSIZE[self.meta['dataset_name']]):
                        if i in index_list: index_list.append(i)
                    
            random.shuffle(index_list)
            for idx in index_list:
                # Skip near-static consecutive frames (by EE6D)
                if np.abs(abs_ee6d[idx + 1] - abs_ee6d[idx]).max() < 5e-4: continue
                
                # Tricks for pour water
                if self.meta['dataset_name'] == 'agiworld-on-site-pour':
                    if idx > len(abs_joint) // 2: 
                        if random.random() < idx / len(abs_joint): 
                            ins = "stop and place the cup on the table."
                
                if self.meta['dataset_name'] == 'agiworld-on-site-microwave':
                    if idx > 500: 
                        if random.random() < idx / len(abs_joint): 
                            ins = "close and start it."
                            
                # Initial window upper bound
                rel = min(DOMAIN2CHUNKSIZE[self.meta['dataset_name']] + 1, traj_end_idx - idx)
                
                # Choose representation
                seg = abs_ee6d[idx:idx + rel] if 'ee6d' in action_mode else abs_joint[idx:idx + rel]

                # Linear interpolate to fixed length (inclusive endpoints -> +1)
                t_old = np.linspace(0.0, 1.0, seg.shape[0])
                t_new = np.linspace(0.0, 1.0, num_actions + 1)
                abs_trajectory = interp1d(t_old, seg, axis=0, kind='linear', bounds_error=False)(t_new)

                # Build image paths (different layouts for "extra")
                if 'extra' not in self.meta['dataset_name']:
                    image_names = ['head_color.jpg', 'hand_left_color.jpg', 'hand_right_color.jpg']
                    image_path = [fileio.join_path(item, f'camera/{idx}/{name}') for name in image_names]
                else:
                    image_names = ['head_color', 'hand_left_color', 'hand_right_color']
                    image_path = [fileio.join_path(item, f'videos/{name}/frame_{idx}.jpg') for name in image_names]

                if random.random() < 0.5:
                    # Load, convert to RGB, apply augmentation, and stack
                    image_mask = torch.tensor([1, 1, 1]).to(torch.bool)
                    imgs = [image_aug(Image.open(p).convert('RGB')) for p in image_path]
                else:
                    image_mask = torch.tensor([1, 0, 0]).to(torch.bool)
                    imgs = [image_aug(Image.open(image_path[0]).convert('RGB'))]
                    while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
                
                image_input = torch.stack(imgs, dim=0)
                
                yield {
                    "language_instruction": ins,
                    "image_input": image_input,
                    "image_mask": image_mask,
                    "abs_trajectory": torch.from_numpy(abs_trajectory).float(),
                }