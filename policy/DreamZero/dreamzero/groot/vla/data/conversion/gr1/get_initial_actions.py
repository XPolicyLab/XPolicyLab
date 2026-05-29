from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import groot.vla.common.utils as U
from groot.vla.data.conversion.gr1.constants import (
    INITIAL_ACTIONS_FILENAME,
    TRAINABLE_HDF5_FILENAME,
)
from groot.vla.data.dataset.macro import (
    LE_ROBOT_EPISODE_FILENAME,
    LE_ROBOT_INFO_FILENAME,
    LE_ROBOT_METADATA_DIR,
    LE_ROBOT_MODALITY_FILENAME,
)


def get_initial_actions(data_dir: str | Path):
    hdf5_file = h5py.File(Path(data_dir) / TRAINABLE_HDF5_FILENAME, "r")
    initial_actions = []

    """
    initial_actions: dict[str, dict[str, np.ndarray]]
    0: (the dataset dimension)
        trajectory_name:
          action_key:
            action: np.ndarray
    """
    initial_actions = {}
    for demo_name in hdf5_file["data"].keys():
        demo_group = hdf5_file["data"][demo_name]
        initial_actions[demo_name] = {}
        action_keys = list(demo_group["action"].keys())
        for action_key in action_keys:
            initial_actions[demo_name][action_key] = demo_group["action"][action_key][0]
    return [initial_actions]


def get_initial_actions_from_lerobot(data_dir: str | Path):
    data_dir = Path(data_dir)

    # 1. Get modality for slicing action
    meta_modality_path = data_dir / LE_ROBOT_METADATA_DIR / LE_ROBOT_MODALITY_FILENAME
    meta_modality = U.load_json(meta_modality_path)
    action_keys = meta_modality["action"].keys()

    # 2. Get episode paths
    # 2.1. Get data_path_pattern
    meta_info_path = data_dir / LE_ROBOT_METADATA_DIR / LE_ROBOT_INFO_FILENAME
    meta_info = U.load_json(meta_info_path)
    data_path_pattern = meta_info["data_path"]
    chunk_size = meta_info["chunks_size"]

    # 2.2. Get episode info
    episode_metadata_path = data_dir / LE_ROBOT_METADATA_DIR / LE_ROBOT_EPISODE_FILENAME
    episode_metadata = U.load_jsonl(episode_metadata_path)

    initial_actions = {}
    for episode_info in episode_metadata:
        episode_index = episode_info["episode_index"]
        episode_chunk = episode_index // chunk_size
        episode_path = data_dir / data_path_pattern.format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        if not episode_path.exists():
            raise ValueError(f"Episode path {episode_path} does not exist")

        episode_data = pd.read_parquet(episode_path)

        initial_action_concat = episode_data["action"].iloc[0]
        trajectory_id = episode_info["episode_index"]
        initial_actions[trajectory_id] = {}
        for action_key in action_keys:
            start = meta_modality["action"][action_key]["start"]
            end = meta_modality["action"][action_key]["end"]
            initial_actions[trajectory_id][action_key] = initial_action_concat[start:end]
    return [initial_actions]


def save_initial_actions(
    initial_actions: dict[str, dict[str, np.ndarray]], initial_actions_path: str | Path
):
    np.savez(str(initial_actions_path), initial_actions)


def load_initial_actions(initial_actions_path: str | Path):
    """
    initial_actions: list[dict[str, dict[str, np.ndarray]]]
    0: (the first dataset)
        trajectory_name:
          action_key:
            action: np.ndarray
    1: (the second dataset)
        ...
    """
    initial_actions_npz = np.load(str(initial_actions_path), allow_pickle=True)
    initial_actions = []
    initial_actions_array = initial_actions_npz[
        "arr_0"
    ]  # This is the default key when np.savez saves a list
    for dataset_initial_actions in initial_actions_array:
        initial_actions_for_this_dataset = {}
        for trajectory_name, action_dict in dataset_initial_actions.items():
            initial_actions_for_this_dataset[trajectory_name] = action_dict
        initial_actions.append(initial_actions_for_this_dataset)
    return initial_actions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate initial_actions.npz for a LeRobot dataset")
    parser.add_argument("data_dir", type=str, help="Path to LeRobot dataset directory")
    args = parser.parse_args()

    initial_actions = get_initial_actions_from_lerobot(args.data_dir)
    save_initial_actions(
        initial_actions,
        Path(args.data_dir) / LE_ROBOT_METADATA_DIR / INITIAL_ACTIONS_FILENAME,
    )

    # Verify
    loaded_initial_actions = load_initial_actions(
        Path(args.data_dir) / LE_ROBOT_METADATA_DIR / INITIAL_ACTIONS_FILENAME
    )
    print(f"Saved initial actions for {len(loaded_initial_actions[0])} trajectories")
