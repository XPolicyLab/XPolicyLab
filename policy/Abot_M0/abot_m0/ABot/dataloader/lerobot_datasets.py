

from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from ABot.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset, ValidationLeRobotMixtureDataset
from ABot.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from ABot.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from ABot.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

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
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
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
    for dataset_item in mixture_spec:
        if len(dataset_item) == 3:
            d_name, d_weight, robot_type = dataset_item
            dataset_config = {}
        elif len(dataset_item) == 4:
            d_name, d_weight, robot_type, dataset_config = dataset_item
        else:
            raise ValueError(f"Invalid dataset item format: {dataset_item}")
        

        if robot_type == "unknown" or robot_type not in ROBOT_TYPE_CONFIG_MAP:
            print(f"Skipping dataset with invalid robot_type '{robot_type}': `{(d_name, d_weight, robot_type)}`")
            continue

        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type, dataset_config))

    dataset_mixture = []
    for dataset_item in filtered_mixture_spec:
        d_name, d_weight, robot_type, dataset_config = dataset_item
        
        if data_cfg is not None:
            if hasattr(data_cfg, 'copy'):
                dataset_specific_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
            else:
                dataset_specific_cfg = data_cfg.copy()
            if dataset_config:
                dataset_specific_cfg = OmegaConf.merge(dataset_specific_cfg, OmegaConf.create(dataset_config))
        else:
            dataset_specific_cfg = OmegaConf.create(dataset_config) if dataset_config else {}
        
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=dataset_specific_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )

def get_vla_dataset_test(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> ValidationLeRobotMixtureDataset:
    """
    Get a ValidationLeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for dataset_item in mixture_spec:
        if len(dataset_item) == 3:
            d_name, d_weight, robot_type = dataset_item
            dataset_config = {}
        elif len(dataset_item) == 4:
            d_name, d_weight, robot_type, dataset_config = dataset_item
        else:
            raise ValueError(f"Invalid dataset item format: {dataset_item}")

        if robot_type == "unknown" or robot_type not in ROBOT_TYPE_CONFIG_MAP:
            print(f"Skipping dataset with invalid robot_type '{robot_type}': `{(d_name, d_weight, robot_type)}`")
            continue

        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type, dataset_config))

    dataset_mixture = []
    for dataset_item in filtered_mixture_spec:
        d_name, d_weight, robot_type, dataset_config = dataset_item
        
        if data_cfg is not None:
            if hasattr(data_cfg, 'copy'):
                dataset_specific_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
            else:
                dataset_specific_cfg = data_cfg.copy()
            if dataset_config:
                dataset_specific_cfg = OmegaConf.merge(dataset_specific_cfg, OmegaConf.create(dataset_config))
        else:
            dataset_specific_cfg = OmegaConf.create(dataset_config) if dataset_config else {}
        
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=dataset_specific_cfg), d_weight))

    return ValidationLeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Pretrain/ABot_pretrain.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
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

    from tqdm import tqdm
    count = 1
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass