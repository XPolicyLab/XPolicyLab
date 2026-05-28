"""
This script is adapted from the Hugging Face 🤗 LeRobot project:
https://github.com/huggingface/lerobot

Original file:
https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/visualize_dataset.py

The original script was developed as part of the LeRobot project for dataset visualization.
This version adds support for depth map visualization.
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"Expect channel first images, but instead {chw_float32_torch.shape}"

    if c == 1:
        # If depth image, clip and normalize the depth map just for visualization
        min_depth = 0.4
        max_depth = 3
        clipped_depth = torch.clamp(chw_float32_torch, min=min_depth, max=max_depth)
        normalized_depth = (clipped_depth - min_depth) / (max_depth - min_depth)
        depth_image = np.sqrt(normalized_depth.squeeze().cpu().numpy())

        colormap = plt.get_cmap("jet")
        colored_depth_image = colormap(depth_image)
        hwc_uint8_numpy = (colored_depth_image[:, :, :3] * 255).astype(np.uint8)
    else:
        # If RGB image
        hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()

    return hwc_uint8_numpy


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
        assert (
            output_dir is not None
        ), "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
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
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

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
        "--task-id",
        type=int,
        default=None,
        help="Index of the AgiBot World task.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to visualize.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Root directory for the converted LeRobot dataset stored locally.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
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

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = f"agibotworld/task_{kwargs.pop('task_id')}"
    root = f"{kwargs.pop('dataset_path')}/{repo_id}"

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, root=root)

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
