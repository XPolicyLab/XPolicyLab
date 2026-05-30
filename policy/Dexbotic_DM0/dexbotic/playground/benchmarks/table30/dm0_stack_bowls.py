import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import megfile
import torch
from loguru import logger
from transformers import AutoTokenizer

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddAction,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import (
    Pipeline,
    ToDict,
    ToList,
    ToNumpy,
    ToTensor,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import ActionDenorm, AbsoluteAction
from dexbotic.exp.dm0_exp import DM0Exp as _DM0Exp
from dexbotic.exp.dm0_exp import DM0ModelConfig as _DM0ModelConfig
from dexbotic.exp.dm0_exp import DM0OptimizerConfig as _DM0OptimizerConfig
from dexbotic.exp.dm0_exp import DM0TrainerConfig as _DM0TrainerConfig
from dexbotic.exp.dm0_exp import DM0ComputeNormActionConfig as _DM0ComputeNormActionConfig
from dexbotic.exp.dm0_exp import DM0ActionConfig as _DM0ActionConfig
from dexbotic.exp.dm0_exp import DM0DataConfig as _DM0DataConfig
from dexbotic.exp.dm0_exp import DM0TokenizerConfig as _DM0TokenizerConfig
from dexbotic.exp.dm0_exp import DM0InferenceConfig as _DM0InferenceConfig
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference"],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class DM0OptimizerConfig(_DM0OptimizerConfig):
    base_lr: float = field(default=5e-5)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=1e-10)


@dataclass
class DM0TrainerConfig(_DM0TrainerConfig):
    wandb_project: str = field(default="dm0_sft_table30")
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=40_000)
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=20)
    per_device_train_batch_size: int = field(default=4)
    gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=2)
    model_max_length: int = field(default=100)
    output_dir: str = field(
        default=f"./user_checkpoints/dexbotic/table30_dm0/stack_bowls-{datetime.now().strftime('%m%d')}"
    )
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(
        default_factory=lambda: {"min_lr": 5e-6}
    )
    logging_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)


class DM0ComputeNormActionConfig(_DM0ComputeNormActionConfig):
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ToList(),
            ]
        )

        return action_config


@dataclass
class DM0ActionConfig(_DM0ActionConfig):
    statistic_mapping: str = field(default=None)
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ActionNorm(statistic_mapping=statistic_mapping, use_quantiles=True),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )
        return action_config


@dataclass
class DM0DataConfig(_DM0DataConfig):
    dataset_name: str = field(default="table30_stack_bowls")
    num_images: int = field(default=3)
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
            "state",
            "image_masks",
        ]
    )
    aug_policy: str | list[str] = field(
        default_factory=lambda: ["dm0", "color_dm0", "color_dm0"]
    )
    action_config: DM0ActionConfig = field(default_factory=DM0ActionConfig)


@dataclass
class DM0ModelConfig(_DM0ModelConfig):
    model_name_or_path: str = field(
        default="./checkpoints/DM0-base"
    )

    def build_model(self) -> DM0ForCausalLM:
        model = DM0ForCausalLM.from_pretrained(self.model_name_or_path)
        return model


@dataclass
class DM0TokenizerConfig(_DM0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class DM0InferenceConfig(_DM0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default="./user_checkpoints/dexbotic/table30_dm0/stack_bowls"
    )
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [6])
    action_dim: int = field(default=7)

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        
        model = DM0ForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=False
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = DM0Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(statistic_mapping=self.norm_stats, strict=False, use_quantiles=True),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False, use_quantiles=True),
                AbsoluteAction(),
            ]
        )


@dataclass
class DM0Exp(_DM0Exp):
    model_config: DM0ModelConfig = field(default_factory=DM0ModelConfig)
    optimizer_config: DM0OptimizerConfig = field(default_factory=DM0OptimizerConfig)
    trainer_config: DM0TrainerConfig = field(default_factory=DM0TrainerConfig)
    data_config: DM0DataConfig = field(default_factory=DM0DataConfig)
    tokenizer_config: DM0TokenizerConfig = field(default_factory=DM0TokenizerConfig)
    inference_config: DM0InferenceConfig = field(default_factory=DM0InferenceConfig)

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = DM0ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        if self.local_rank == 0:
            print(
                f"Action config before auto compute norm: {self.data_config.action_config}"
            )
        _action_config = self.data_config.action_config
        norm_config = DM0ComputeNormActionConfig()
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing norm stats on rank0")
            self.compute_norm_stats()
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f"Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}"
                )
        _action_config.statistic_mapping = norm_file_path
        self.data_config.action_config = _action_config
        if self.local_rank == 0:
            print(
                f"Action config after auto compute norm: {self.data_config.action_config}"
            )


if __name__ == "__main__":
    args = parse_args()
    exp = DM0Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
