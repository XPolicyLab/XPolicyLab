import os
import gc
import sys
import logging
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from data.robot_dataset import RobotDataset
from runners.xwam_runner import XWAMRunner
from utils.console_logger import ConsoleLogger


def main():
    model_config = OmegaConf.load("configs/model/wan22_5b_sft.yaml")
    config_override = OmegaConf.from_cli()
    config = OmegaConf.merge(model_config, config_override)

    data_config = OmegaConf.load(f"configs/data/{config.dataset}.yaml")
    config["dataset"] = data_config

    pprint(OmegaConf.to_container(config))

    os.makedirs(os.path.join(config.exp_root, config.exp_name), exist_ok=True)
    OmegaConf.save(
        config,
        os.path.join(config.exp_root, config.exp_name, "config.yaml"),
        resolve=True,
    )
    L.seed_everything(config.seed, workers=True)

    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            dirpath=os.path.join(config.exp_root, config.exp_name, "checkpoints"),
            save_top_k=-1,
            save_last=True,
            every_n_train_steps=config.save_interval,
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    tb_path = os.getenv("TENSORBOARD_LOG_PATH", None)
    loggers = [
        TensorBoardLogger(
            save_dir=tb_path if tb_path else os.path.join(config.exp_root, config.exp_name, "tb_logs"),
            name=config.exp_name,
        ),
        ConsoleLogger(max_steps=config.num_training_steps),
    ]
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    train_dataset = RobotDataset(
        dataset_path=config.dataset.dataset_path,
        sequence_length=config.dataset.sequence_length,
        frame_skip=config.dataset.frame_skip,
        video_size=config.dataset.video_size,
        action_skip=config.dataset.action_skip,
        augment=config.dataset.augment,
        crop_ratio=config.dataset.crop_ratio,
        brightness=config.dataset.brightness,
        contrast=config.dataset.contrast,
        saturation=config.dataset.saturation,
        hue=config.dataset.hue,
        inverse_gripper=config.dataset.inverse_gripper,
        normalize_depths_per_view=config.dataset.normalize_depths_per_view,
        shuffle_view_order=config.dataset.shuffle_view_order,
        statistics=OmegaConf.to_container(config.dataset.statistics, resolve=True),
    )
    config.action_num = train_dataset.action_num

    val_dataset = RobotDataset(
        dataset_path=config.dataset.dataset_path,
        sequence_length=config.dataset.sequence_length,
        frame_skip=config.dataset.frame_skip,
        video_size=config.dataset.video_size,
        action_skip=config.dataset.action_skip,
        augment=False,
        inverse_gripper=config.dataset.inverse_gripper,
        normalize_depths_per_view=config.dataset.normalize_depths_per_view,
        shuffle_view_order=config.dataset.shuffle_view_order,
        statistics=OmegaConf.to_container(config.dataset.statistics, resolve=True),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        num_workers=config.num_workers_per_gpu,
        prefetch_factor=config.prefetch_factor,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        multiprocessing_context="forkserver" if config.num_workers_per_gpu > 0 else None,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        multiprocessing_context="forkserver",
    )

    model = XWAMRunner(config)
    if config.get("pretrained_checkpoint", None) is not None:
        print(f"Loading pretrained checkpoint from {config.pretrained_checkpoint}...")
        ckpt = torch.load(
            os.path.join(config.pretrained_checkpoint, "checkpoint/mp_rank_00_model_states.pt"), map_location="cpu"
        )
        model.load_state_dict(ckpt["module"])

    devices = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", devices))
    num_nodes = world_size // devices

    print(f"Runner v2, num_nodes: {num_nodes}, world_size: {world_size}, devices: {devices}")

    trainer = L.Trainer(
        accelerator="auto",
        strategy=DeepSpeedStrategy(
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        ),
        precision="bf16-mixed",
        num_nodes=num_nodes,
        max_steps=config.num_training_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        enable_progress_bar=False,
        callbacks=callbacks,
        logger=loggers,
        val_check_interval=config.val_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=config.limit_val_batches,
        log_every_n_steps=config.log_interval,
        default_root_dir=os.path.join(config.exp_root, config.exp_name),
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
