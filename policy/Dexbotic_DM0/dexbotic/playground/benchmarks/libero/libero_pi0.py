import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.pi0_exp import Pi0InferenceConfig as _Pi0InferenceConfig
from dexbotic.exp.pi0_exp import Pi0Exp as _Pi0Exp
from dexbotic.exp.pi0_exp import Pi0DataConfig as _Pi0DataConfig
from dexbotic.exp.pi0_exp import Pi0ModelConfig as _Pi0ModelConfig
from dexbotic.exp.pi0_exp import Pi0OptimizerConfig as _Pi0OptimizerConfig
from dexbotic.exp.pi0_exp import Pi0TrainerConfig as _Pi0TrainerConfig
from dexbotic.exp.pi0_exp import (
    Pi0ComputeNormActionConfig as _Pi0ComputeNormActionConfig,
)
from dexbotic.exp.pi0_exp import Pi0ActionConfig as _Pi0ActionConfig
from dexbotic.exp.pi0_exp import Pi0TokenizerConfig as _Pi0TokenizerConfig
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    parser.add_argument(
        "--train-backend",
        type=str,
        default=None,
        choices=["deepspeed", "fsdp", "fsdp2", "ddp"],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class Pi0OptimizerConfig(_Pi0OptimizerConfig):
    base_lr: float = field(default=2.5e-5)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=1e-10)


@dataclass
class Pi0TrainerConfig(_Pi0TrainerConfig):
    deepspeed: str = field(default=None)
    use_raw_backward: bool = field(default=True)
    use_raw_warmup: bool = field(default=True)
    seed: int = field(default=42)
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=10000)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    model_max_length: int = field(default=48)
    output_dir: str = field(
        default=f"./user_checkpoints/dexbotic/libero_all_pi0/all-{datetime.now().strftime('%m%d')}"
    )
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})


class Pi0ComputeNormActionConfig(_Pi0ComputeNormActionConfig):
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ToList(),
            ]
        )

        return action_config


@dataclass
class Pi0ActionConfig(_Pi0ActionConfig):
    statistic_mapping: str = field(default=None)
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ActionNorm(statistic_mapping=statistic_mapping),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )
        return action_config


@dataclass
class Pi0DataConfig(_Pi0DataConfig):
    dataset_name: str = field(default="libero_pi0_all")
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
        default_factory=lambda: ["pi0", "color", "identity"]
    )
    action_config: Pi0ActionConfig = field(default_factory=Pi0ActionConfig)


@dataclass
class Pi0ModelConfig(_Pi0ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/Dexbotic-PI0")

    def build_model(self) -> Pi0ForCausalLM:
        model = Pi0ForCausalLM.from_pretrained(self.model_name_or_path)
        return model


@dataclass
class Pi0TokenizerConfig(_Pi0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class Pi0InferenceConfig(_Pi0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default="./user_checkpoints/dexbotic/libero_all_pi0/all-{datetime.now().strftime('%m%d')}"
    )
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [6])
    action_dim: int = field(default=7)


@dataclass
class Pi0Exp(_Pi0Exp):
    model_config: Pi0ModelConfig = field(default_factory=Pi0ModelConfig)
    optimizer_config: Pi0OptimizerConfig = field(default_factory=Pi0OptimizerConfig)
    trainer_config: Pi0TrainerConfig = field(default_factory=Pi0TrainerConfig)
    data_config: Pi0DataConfig = field(default_factory=Pi0DataConfig)
    tokenizer_config: Pi0TokenizerConfig = field(default_factory=Pi0TokenizerConfig)
    inference_config: Pi0InferenceConfig = field(default_factory=Pi0InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = Pi0ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)


if __name__ == "__main__":
    args = parse_args()
    exp = Pi0Exp()
    if args.train_backend is not None:
        exp.trainer_config.train_backend = args.train_backend
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
