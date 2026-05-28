import os
import shutil
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

from agilex_features import FEATURES
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_hdf5_dataset(
    episode_path: str | Path,
) -> tuple[list, dict] | None:
    """Load hdf5 dataset and return a dict with observations and actions"""

    with h5py.File(episode_path) as f:
        state_images_cam_high = np.array(f["observations/images/cam_high"])
        state_images_cam_left_wrist = np.array(f["observations/images/cam_left_wrist"])
        state_images_cam_right_wrist = np.array(f["observations/images/cam_right_wrist"])
        state_qpos = np.array(f["observations/qpos"])

    assert (
        state_images_cam_high.shape[0]
        == state_images_cam_left_wrist.shape[0]
        == state_images_cam_right_wrist.shape[0]
        == state_qpos.shape[0]
    )

    frames = [
        {
            "observation.state": state_qpos[i].reshape(-1),
            "observation.images.cam_high": state_images_cam_high[i],
            "observation.images.cam_left_wrist": state_images_cam_left_wrist[i],
            "observation.images.cam_right_wrist": state_images_cam_right_wrist[i],
            "action": state_qpos[i].reshape(-1),
        }
        for i in range(len(state_qpos))
    ]

    return frames


def main(
    src_path: str,
    tgt_path: str,
    repo_ids: List[str],
    save_repoid: str,
):
    # check if the target path exists, if exists, delete it
    target_dir = f"{tgt_path}/{save_repoid}"
    if os.path.exists(target_dir):
        print(f"target folder {target_dir} exists, deleting...")
        shutil.rmtree(target_dir)
        print(f"deleted folder: {target_dir}")

    dataset = LeRobotDataset.create(
        repo_id=save_repoid,
        root=f"{tgt_path}/{save_repoid}",
        fps=30,
        robot_type="agilex",
        features=FEATURES,
    )
    exclude_files = [
        # list of hdf5 files to exclude
    ]

    for repo_id in repo_ids:
        # find all .hdf5 files in src_path/repo_id
        hdf5_source_path = f"{src_path}/{repo_id}"
        hdf5_files = []

        if os.path.exists(hdf5_source_path):
            hdf5_files = sorted([f.as_posix() for f in Path(hdf5_source_path).glob("*.hdf5")])
            print(f"found {len(hdf5_files)} .hdf5 files:")

        for exclude_file in exclude_files:
            if exclude_file in hdf5_files:
                hdf5_files.remove(exclude_file)

        print(len(hdf5_files))

        for hdf5_file in tqdm(hdf5_files, total=len(hdf5_files), desc="Processing episodes"):
            frames = load_hdf5_dataset(hdf5_file)
            for frame in tqdm(frames, total=len(frames), desc=f"Processing episode {hdf5_file}"):
                dataset.add_frame(frame, task="test")

            dataset.save_episode()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="src path for the original dataset")
    parser.add_argument("--tgt_path", type=str, required=True, help="tgt path to save the converted dataset")
    parser.add_argument("--repo_ids", nargs="+", required=True, help="repo ids")
    parser.add_argument(
        "--save_repoid", type=str, default=None, help="save repoid, default as the first repo_id in the list"
    )
    args = parser.parse_args()
    repo_ids = args.repo_ids
    save_repoid = args.save_repoid
    if save_repoid is None:
        save_repoid = repo_ids[0]

    main(
        src_path=args.src_path,
        tgt_path=args.tgt_path,
        repo_ids=repo_ids,
        save_repoid=save_repoid,
    )
