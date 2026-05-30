#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch
import torch.utils.data
from torch.utils.data._utils.collate import default_collate
import tqdm
from PIL import Image as PILImage

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD


def collate_skip_none(batch: list[dict]):
    """Collate a batch of dict items while tolerating `None` values.

    When a key contains `None` for at least one sample, we return a Python list for that key
    (so downstream code can skip those per-sample values).
    """
    if not batch:
        return {}

    if not isinstance(batch[0], dict):
        return default_collate(batch)

    # Only collate keys present in all items (keeps shapes predictable).
    common_keys = set(batch[0].keys())
    for item in batch[1:]:
        common_keys &= item.keys()

    collated: dict = {}
    for key in common_keys:
        values = [item[key] for item in batch]
        if any(v is None for v in values):
            collated[key] = values
            continue
        try:
            collated[key] = default_collate(values)
        except Exception:
            # Fallback: return raw values to avoid crashing the visualizer.
            collated[key] = values
    return collated


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 tensor in [0,1] to HWC uint8 numpy array."""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def downscale_hwc_image(
    image: np.ndarray,
    max_height: int = 128,
    max_width: int = 128,
) -> np.ndarray:
    """Downscale HWC uint8 image to at most (max_height, max_width) while keeping aspect ratio.

    This is applied on the frames used for visualization only (both images and video frames),
    so it does not affect the underlying stored dataset.
    """
    # Only handle standard HWC images
    if image.ndim != 3:
        return image

    h, w = image.shape[:2]

    # already small enough
    if h <= max_height and w <= max_width:
        return image

    scale = min(max_height / h, max_width / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Be robust to unusual channel layouts / dtypes: if PIL fails, just return original
    try:
        # Ensure uint8 for PIL
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)

        # Handle single-channel explicitly
        if image.shape[2] == 1:
            pil_img = PILImage.fromarray(image[:, :, 0], mode="L")
        else:
            pil_img = PILImage.fromarray(image)

        pil_img = pil_img.resize((new_w, new_h), resample=PILImage.BILINEAR)
        return np.asarray(pil_img, dtype=np.uint8)
    except Exception:
        # Fallback: no downscaling if anything goes wrong
        return image


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id


    # 从 meta.info 中拿出各维度的名字（如果有）
    state_names = None
    action_names = None
    # 存储所有字段的 names 信息，用于可视化
    field_names = {}
    try:
        feats = dataset.features  # 即 info["features"]
        if "observation.state" in feats:
            state_names = feats["observation.state"].get("names")
            if "motors" in state_names:
                state_names = state_names["motors"]
        if "action" in feats:
            action_names = feats["action"].get("names")
            if "motors" in action_names:
                action_names = action_names["motors"]
        
        # 为所有字段提取 names 信息
        for key, feat in feats.items():
            if feat.get("dtype") in ["video", "image"]:
                continue  # 跳过图像/视频字段
            names = feat.get("names")
            if names is not None:
                # 处理嵌套的 names 结构（如 {"motors": [...]}）
                if isinstance(names, dict) and "motors" in names:
                    field_names[key] = names["motors"]
                elif isinstance(names, list):
                    field_names[key] = names
    except Exception:
        # 出现异常时退回到原来的 index 写法
        state_names = None
        action_names = None
        field_names = {}


    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_skip_none,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        rr.serve_web_viewer(open_browser=False, web_port=web_port)

    logging.info("Logging to Rerun")
    
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            frame_index = batch["frame_index"][i]
            timestamp = batch["timestamp"][i]
            if frame_index is None or timestamp is None:
                continue
            rr.set_time_sequence("frame_index", sequence=int(frame_index.item()))
            rr.set_time_seconds("timestamp", seconds=float(timestamp.item()))

            #YYD add video visualize
            # display each camera image (downscaled for lighter visualization)
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                if key not in batch:
                    continue
                cam_val = batch[key][i]
                if cam_val is None:
                    continue
                img = to_hwc_uint8_numpy(cam_val)
                img = downscale_hwc_image(img)
                rr.log(key, rr.Image(img))

            # display each dimension of action space (e.g. actuators command)
            if ACTION in batch:
                action_vec = batch[ACTION][i]
                if action_vec is None:
                    continue
                # for dim_idx, val in enumerate(batch[ACTION][i]):
                #     rr.log(f"{ACTION}/{dim_idx}", rr.Scalar(val.item()))
                for dim_idx, val in enumerate(action_vec):
                    if val is None:
                        continue
                    if action_names is not None and dim_idx < len(action_names):
                        name = action_names[dim_idx]
                        rr.log(f"action/{name}", rr.Scalar(val.item()))
                    else:
                        rr.log(f"{ACTION}/{dim_idx}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                obs_vec = batch[OBS_STATE][i]
                if obs_vec is None:
                    continue
                # for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                #     rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))
                for dim_idx, val in enumerate(obs_vec):
                    if val is None:
                        continue
                    if state_names is not None and dim_idx < len(state_names):
                        name = state_names[dim_idx]
                        rr.log(f"state/{name}", rr.Scalar(val.item()))
                    else:
                        rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            if DONE in batch:
                done_val = batch[DONE][i]
                if done_val is not None:
                    rr.log(DONE, rr.Scalar(done_val.item()))

            if REWARD in batch:
                reward_val = batch[REWARD][i]
                if reward_val is not None:
                    rr.log(REWARD, rr.Scalar(reward_val.item()))

            if "next.success" in batch:
                next_success_val = batch["next.success"][i]
                if next_success_val is not None:
                    rr.log("next.success", rr.Scalar(next_success_val.item()))

            # 可视化所有其他字段（除了已经处理的相机图像、action、observation.state、done、reward等）
            # 需要跳过的字段
            skip_keys = {
                ACTION, OBS_STATE, DONE, REWARD, "next.success",
                "index", "episode_index", "frame_index", "timestamp", "task_index"
            }
            skip_keys.update(dataset.meta.camera_keys)  # 跳过所有相机图像字段
            
            # 遍历所有 features 中的字段
            for key in dataset.features:
                if key in skip_keys:
                    continue
                
                # 如果 batch 中没有这个字段，跳过
                if key not in batch:
                    continue
                
                try:
                    value = batch[key][i]
                    feat = dataset.features[key]
                    dtype = feat.get("dtype", "")
                    shape = feat.get("shape", [])
                    
                    # 跳过视频和图像字段（已经处理过）
                    if dtype in ["video", "image"]:
                        continue
                    
                    # 处理标量值
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            # 标量
                            rr.log(key, rr.Scalar(value.item()))
                        elif value.ndim == 1:
                            # 1D 数组，按维度记录
                            names = field_names.get(key)
                            for dim_idx, val in enumerate(value):
                                if names is not None and dim_idx < len(names):
                                    name = names[dim_idx]
                                    rr.log(f"{key}/{name}", rr.Scalar(val.item()))
                                else:
                                    rr.log(f"{key}/{dim_idx}", rr.Scalar(val.item()))
                        elif value.ndim == 2:
                            # 2D 数组，按行和列记录
                            names = field_names.get(key)
                            # 检查原始 features 中的 names 结构
                            feat_names = feat.get("names")
                            for row_idx in range(value.shape[0]):
                                row_name = None
                                # 尝试从 names 中获取行名称
                                if names is not None and isinstance(names, list) and row_idx < len(names):
                                    row_name = names[row_idx]
                                elif feat_names is not None:
                                    # 处理 {"motors": [...]} 这种结构
                                    if isinstance(feat_names, dict) and "motors" in feat_names:
                                        motors = feat_names["motors"]
                                        if isinstance(motors, list) and row_idx < len(motors):
                                            row_name = motors[row_idx]
                                
                                for col_idx in range(value.shape[1]):
                                    val = value[row_idx, col_idx]
                                    if row_name is not None:
                                        # 使用行名称和列索引
                                        rr.log(f"{key}/{row_name}/{col_idx}", rr.Scalar(val.item()))
                                    else:
                                        # 默认使用索引
                                        rr.log(f"{key}/{row_idx}/{col_idx}", rr.Scalar(val.item()))
                        else:
                            # 更高维的数组，展平处理
                            flat_value = value.flatten()
                            for dim_idx, val in enumerate(flat_value):
                                rr.log(f"{key}/{dim_idx}", rr.Scalar(val.item()))
                    elif isinstance(value, (int, float)):
                        # Python 原生标量
                        rr.log(key, rr.Scalar(float(value)))
                except Exception as e:
                    # 如果某个字段处理失败，记录警告但继续处理其他字段
                    logging.warning(f"Failed to visualize field '{key}': {e}")
                    continue

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="debug.rrd",
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
