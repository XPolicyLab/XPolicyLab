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
import numpy as np, torch, random
from mmengine import fileio
from scipy.interpolate import interp1d
from ..utils import read_video_to_frames, read_parquet, quat_to_rotate6d
from PIL import Image
from .base import DomainHandler

class AGIBOTLeRobotHandler(DomainHandler):
    dataset_name = "AGIBOT"

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool,
                     image_aug, lang_aug_map: dict | None, **kwargs):
        item = self.meta["datalist"][traj_idx]
        ep = item["episode_index"]; chunk = f"chunk-{ep//1000:03d}"; key = f"episode_{ep:06d}"

        pq_path = fileio.join_path(item["top_path"], "data", chunk, key + ".parquet")
        vkeys = ["observation.images.head", "observation.images.hand_left", "observation.images.hand_right"]
        vid_paths = [fileio.join_path(item["top_path"], "videos", chunk, k, key + ".mp4") for k in vkeys]
        images = [read_video_to_frames(p) for p in vid_paths]
        image_mask = torch.ones(self.num_views, dtype=torch.bool)

        data = read_parquet(pq_path)
        pos = np.asarray(data["actions.end.position"])     # [T,2,3]
        ori = np.asarray(data["actions.end.orientation"])  # [T,2,4]
        grip = np.asarray(data["actions.effector.position"])  # [T,2]
        left = np.concatenate([pos[:,0], quat_to_rotate6d(ori[:,0]), grip[:, :1]], -1)
        right = np.concatenate([pos[:,1], quat_to_rotate6d(ori[:,1]), grip[:, 1:]], -1)

        freq = 30.0; qdur = 4.0; t = np.arange(left.shape[0], dtype=np.float64) / freq
        L = interp1d(t, left,  axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(t, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        start = item["action_config"][0]["start_frame"]; end = item["action_config"][-1]["end_frame"] - 30
        idxs = list(range(start, end, 4 if training else 120))
        if training: random.shuffle(idxs)

        ins = item["tasks"][0].split(" | ")[0]

        for idx in idxs:
            imgs = []
            for v in range(min(self.num_views, len(images))):
                imgs.append(image_aug(Image.fromarray(images[v][idx])))
            while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, 0)
            cur = t[idx]
            q = np.linspace(cur, min(cur + qdur, float(t.max())), num_actions + 1, dtype=np.float32)
            lseq, rseq = torch.tensor(L(q)), torch.tensor(R(q))
            if (lseq[1]-lseq[0]).abs().max() < 1e-5 and (rseq[1]-rseq[0]).abs().max() < 1e-5:continue
            if lang_aug_map is not None and ins in lang_aug_map: ins = random.choice(lang_aug_map[ins])
            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float(),
            }
