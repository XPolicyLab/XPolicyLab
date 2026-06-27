# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path
from typing import Sequence
from copy import deepcopy
from omegaconf import OmegaConf

from eventvla.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from eventvla.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from eventvla.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from eventvla.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

_INITIAL_FRAME_INDEX = 0


def _get_nested_cfg_value(config, path: str, default=None):
    current = config
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            get_method = getattr(current, "get", None)
            if callable(get_method):
                sentinel = object()
                next_value = get_method(key, sentinel)
                if next_value is not sentinel:
                    current = next_value
                    continue
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def _build_temporal_image_indices(data_cfg: dict | None) -> tuple[list[int], list[int]]:
    default_delta_indices = [0]
    default_absolute_indices: list[int] = []
    if data_cfg is None:
        return default_delta_indices, default_absolute_indices

    explicit_delta = _get_nested_cfg_value(data_cfg, "temporal.image.delta_indices", default=None)
    explicit_absolute = _get_nested_cfg_value(data_cfg, "temporal.image.absolute_indices", default=None)
    if explicit_delta is not None or explicit_absolute is not None:
        delta_indices = list(explicit_delta) if explicit_delta is not None else default_delta_indices
        absolute_indices = list(explicit_absolute) if explicit_absolute is not None else default_absolute_indices
        return [int(x) for x in delta_indices], [int(x) for x in absolute_indices]

    temporal_enabled = bool(_get_nested_cfg_value(data_cfg, "temporal.enabled", default=False))
    include_initial = bool(_get_nested_cfg_value(data_cfg, "temporal.image.include_initial", default=False))
    history_steps = int(_get_nested_cfg_value(data_cfg, "temporal.image.history_steps", default=0) or 0)
    history_stride = int(_get_nested_cfg_value(data_cfg, "temporal.image.history_stride", default=1) or 1)
    include_current = bool(_get_nested_cfg_value(data_cfg, "temporal.image.include_current", default=True))

    if history_steps < 0:
        raise ValueError(f"`temporal.image.history_steps` must be >= 0, got {history_steps}.")
    if history_stride <= 0:
        raise ValueError(f"`temporal.image.history_stride` must be > 0, got {history_stride}.")

    use_temporal_images = temporal_enabled or include_initial or history_steps > 0
    if not use_temporal_images:
        return default_delta_indices, default_absolute_indices

    delta_indices = [-(step_idx * history_stride) for step_idx in range(history_steps, 0, -1)]
    if include_current or (not include_initial and not delta_indices):
        delta_indices.append(0)

    absolute_indices = [_INITIAL_FRAME_INDEX] if include_initial else []
    return delta_indices, absolute_indices


def _apply_temporal_video_overrides(modality_config: dict, data_cfg: dict | None) -> None:
    if "video" not in modality_config:
        return
    delta_indices, absolute_indices = _build_temporal_image_indices(data_cfg)
    modality_config["video"].delta_indices = delta_indices
    modality_config["video"].absolute_indices = absolute_indices

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = deepcopy(ROBOT_TYPE_CONFIG_MAP[robot_type])
    modality_config = data_config.modality_config()
    _apply_temporal_video_overrides(modality_config, data_cfg)
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"
    
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":

    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./eventvla/config/training/starvla_cotrain_behavior.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "./examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.datasets.vla_data.data_mix = "robotwin"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass