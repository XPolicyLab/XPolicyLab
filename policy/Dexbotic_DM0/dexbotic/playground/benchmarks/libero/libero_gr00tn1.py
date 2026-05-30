import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import deepspeed.runtime.zero.utils as ds_zero_utils

ds_zero_utils.warned = False
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dexbotic.exp.gr00tn1_exp import (
    GR00TN1DataConfig,
    GR00TN1Exp,
    GR00TN1ModelConfig,
    GR00TN1OptimizerConfig,
    GR00TN1TrainerConfig,
    InferenceConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "inference_single"],
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class LiberoGR00TN1OptimizerConfig(GR00TN1OptimizerConfig):
    base_lr: float = field(default=1e-4)
    weight_decay: float = field(default=1e-5)
    warmup_ratio: float = field(default=0.05)

    adam_beta1: float = field(default=0.95)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)


@dataclass
class LiberoGR00TN1DataConfig(GR00TN1DataConfig):
    dataset_name: str = field(default="libero_object")


@dataclass
class LiberoGR00TModelConfig(GR00TN1ModelConfig):
    model_name_or_path: str = field(
        default="/mlp_vepfs/share/ldx/model/dexbotic_gr00tn1"
    )


@dataclass
class LiberoGR00TN1TrainerConfig(GR00TN1TrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/libero_object_gr00tn1/step40k_bs128_lr1e-4_{datetime.now().strftime("%m%d%H%M%S")}'
    )
    wandb_project: str = field(default="dexbotic_gr00tn1")
    num_train_steps: int = field(default=40000)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    save_steps: int = field(default=1000)

    save_total_limit: int = field(default=50)


@dataclass
class LiberoGR00TN1InferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(default="")
    port: int = field(default=7891)


@dataclass
class LiberoGR00TN1Exp(GR00TN1Exp):
    data_config: LiberoGR00TN1DataConfig = field(
        default_factory=LiberoGR00TN1DataConfig
    )
    model_config: LiberoGR00TModelConfig = field(default_factory=LiberoGR00TModelConfig)
    trainer_config: LiberoGR00TN1TrainerConfig = field(
        default_factory=LiberoGR00TN1TrainerConfig
    )
    optimizer_config: LiberoGR00TN1OptimizerConfig = field(
        default_factory=LiberoGR00TN1OptimizerConfig
    )
    inference_config: LiberoGR00TN1InferenceConfig = field(
        default_factory=LiberoGR00TN1InferenceConfig
    )

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = LiberoGR00TN1Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "inference_single":
        exp.inference_single(args.image_path, args.prompt)
