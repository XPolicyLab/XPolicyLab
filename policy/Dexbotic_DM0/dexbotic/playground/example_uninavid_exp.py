"""Example UniNaVid experiment config.

Train::
    torchrun --nproc_per_node=8 playground/example_uninavid_exp.py --task train

Inference server::
    python playground/example_uninavid_exp.py --task inference

Single image (one RGB frame, navigation instruction)::
    python playground/example_uninavid_exp.py --task inference_single --image_path test_data/uninavid_test.png --prompt "Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug."
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dexbotic.exp.uninavid_exp import (
    UniNaVidDataConfig,
    UniNaVidExp,
    UniNaVidInferenceConfig,
    UniNaVidModelConfig,
    UniNaVidOptimizerConfig,
    UniNaVidTokenizerConfig,
    UniNaVidTrainerConfig,
    parse_args,
)


@dataclass
class OptimizerConfig(UniNaVidOptimizerConfig):
    # Parent OptimizerConfig: base_lr 2e-5; UniNaVidOptimizerConfig: 1e-5.
    base_lr: float = field(default=5e-6)


@dataclass
class TrainerConfig(UniNaVidTrainerConfig):
    """Only overrides that differ from ``UniNaVidTrainerConfig`` / defaults."""

    num_train_epochs: int = field(default=5)
    save_steps: int = field(default=1000)
    deepspeed: str | None = field(default="./script/deepspeed/zero2.json")
    output_dir: str = field(
        default=f"./user_checkpoints/dexbotic/uninavid/{datetime.now().strftime('%m%d')}"
    )


@dataclass
class DataConfig(UniNaVidDataConfig):
    dataset_name: str = field(default="uninavid_objnav")
    video_fps: int = field(default=1)
    dex_use_nav_augment: bool = field(default=True)
    image_aspect_ratio: str = field(default="pad")


@dataclass
class ModelConfig(UniNaVidModelConfig):
    """Example paths; other fields use ``UniNaVidModelConfig`` defaults."""

    model_name_or_path: str = field(
        default="./checkpoints/dexbotic-uninavid"
    )


@dataclass
class TokenizerConfig(UniNaVidTokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class InferenceConfig(UniNaVidInferenceConfig):
    model_name_or_path: Optional[str] = field(
        default="./checkpoints/dexbotic-uninavid"
    )


@dataclass
class Exp(UniNaVidExp):
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)


if __name__ == "__main__":
    args = parse_args()
    exp = Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "inference_single":
        if not args.image_path or not args.prompt:
            raise SystemExit("inference_single requires --image_path and --prompt")
        exp.inference_single(args.image_path, args.prompt)
