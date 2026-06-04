# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import json
import logging
import os
import sys
from multiprocessing import set_start_method
from pathlib import Path

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.state import AcceleratorState, is_initialized  # noqa: E402
from accelerate.utils import (  # noqa: E402
    DataLoaderConfiguration,
    ProjectConfiguration,
)

from projects.holobrain.utils import (  # noqa: E402
    ActionMetric,
    load_checkpoint,
    load_config,
)
from robo_orchard_lab.dataset.collates import collate_batch_dict  # noqa: E402
from robo_orchard_lab.dataset.dataset_wrapper import (  # noqa: E402
    DistributedBatchFlagSampler,
)
from robo_orchard_lab.pipeline import SimpleTrainer  # noqa: E402
from robo_orchard_lab.pipeline.hooks import (  # noqa: E402
    LossTrackerConfig,
    SaveCheckpointConfig,
    StatsMonitorConfig,
)
from robo_orchard_lab.processing.step_processor import (  # noqa: E402
    SimpleStepProcessor,
)
from robo_orchard_lab.utils import log_basic_config  # noqa: E402

logger = logging.getLogger(__file__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"


class MyBatchProcessor(SimpleStepProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, model, batch):
        output = model(batch)
        loss = sum([y.mean() for x, y in output.items() if "loss" in x])
        return output, loss


def main(args, accelerator):
    if_cluster = os.environ.get("CLUSTER") is not None
    if accelerator.is_main_process:
        import shutil

        shutil.copytree(
            CONFIGS_DIR,
            os.path.join(args.workspace, "configs"),
            dirs_exist_ok=True,
        )

    config = load_config(args.config)
    build_model = config.build_model
    build_dataset = config.build_training_dataset
    build_validation_dataset = config.build_validation_dataset
    build_optimizer = config.build_optimizer
    build_processors = config.build_processors
    config = config.config

    # export data processors
    if accelerator.is_main_process:
        processors = build_processors(config)
        for dataset_name, processor in processors.items():
            processor.save(args.workspace, f"{dataset_name}_processor.json")

    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            kwargs = json.load(open(args.kwargs, "r"))
        else:
            kwargs = json.loads(args.kwargs)
        config.update(kwargs)

    if accelerator.is_main_process:
        logger.info("\n" + json.dumps(config, indent=4))

    model = build_model(config)

    num_workers = config.get("num_workers", 4)
    if not args.eval_only:
        train_dataset = build_dataset(config)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_batch_dict,
            persistent_workers=num_workers > 0,
            batch_sampler=DistributedBatchFlagSampler(
                train_dataset,
                config["batch_size"],
                drop_last=True,
                dataset_sample_weights=config.get("dataset_sample_weights"),
            ),
            # in_order=False,
        )
        optimizer, lr_scheduler = build_optimizer(config, model)
    else:
        train_dataloader = optimizer = lr_scheduler = None

    trainable_param = 0
    non_trainable_param = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_param += param.numel()
        else:
            non_trainable_param += param.numel()
    total_param = trainable_param + non_trainable_param
    logger.info(
        f"number of parameters: {total_param / 10**6:.2f}M, "
        f"trainable: {trainable_param / 10**6:.2f}M, "
        f"non-trainable: {non_trainable_param / 10**6:.2f}M"
    )

    accelerator.register_save_state_pre_hook(
        model.accelerator_save_state_pre_hook
    )
    load_checkpoint(model, config.get("checkpoint"), accelerator)

    val_dataset = build_validation_dataset(config)
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=False,
            batch_size=config["batch_size"],
            collate_fn=collate_batch_dict,
            persistent_workers=num_workers > 0,
        )
        pred_steps = config.get("pred_steps", 64)
        metric = ActionMetric(
            eval_horizons=[pred_steps // 4, pred_steps // 2, pred_steps],
        )
    else:
        val_dataloader = None
        metric = None

    trainer = SimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        grad_clip_mode="norm",
        grad_max_norm=10,
        batch_processor=MyBatchProcessor(need_backward=True),
        hooks=[
            StatsMonitorConfig(
                step_log_freq=config["step_log_freq"],
            ),
            LossTrackerConfig(
                step_log_freq=config["step_log_freq"],
                log_total_loss=True,
            ),
            SaveCheckpointConfig(
                save_step_freq=config.get("save_step_freq"),
                save_epoch_freq=config.get("save_epoch_freq"),
            ),
        ],
        max_step=config.get("max_step"),
        step_eval_freq=config.get("save_step_freq"),
        lr_scheduler_step_at="step",
        resume_from=config.get("resume_from"),
        resume_share_dir=(
            "/job_data/resume_from" if if_cluster else "./resume_from"
        ),
        val_dataloader=val_dataloader,
        metric=metric,
    )
    if args.eval_only:
        assert val_dataset is not None, (
            "The validation dataset must be specified when eval_only=True."
        )
        trainer.eval()
    else:
        trainer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()

    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.workspace, "logs")

    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    accelerator = Accelerator(
        log_with="tensorboard",
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(
            project_dir=args.workspace,
            logging_dir=args.logging_dir,
            automatic_checkpoint_naming=True,
            total_limit=3,
        ),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
        ),
    )
    accelerator.init_trackers("tensorboard")
    logger.info(f"Save config to workspace dir {args.workspace}")

    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    logger.info(f"if accelerator initialized:{is_initialized()}")
    logger.info(f"accelerator state: {AcceleratorState._shared_state}")
    set_start_method("spawn", force=True)
    main(args, accelerator)
