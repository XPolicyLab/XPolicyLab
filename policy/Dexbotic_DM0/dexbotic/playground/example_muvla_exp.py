
import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.muvla_exp import MUVLAActionConfig as _MUVLAActionConfig
from dexbotic.exp.muvla_exp import MUVLADataConfig as _MUVLADataConfig
from dexbotic.exp.muvla_exp import MUVLAExp as _MUVLAExp
from dexbotic.exp.muvla_exp import MUVLAModelConfig as _MUVLAModelConfig
from dexbotic.exp.muvla_exp import MUVLATrainerConfig as _MUVLATrainerConfig
from dexbotic.exp.muvla_exp import InferenceConfig as _MUVLAInferenceConfig

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference',
            'compute_norm_stats'])
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class MUVLATrainerConfig(_MUVLATrainerConfig):

    output_dir: str = field(
        default=f'/checkpoints/muvla/{datetime.now().strftime("%m%d")}-stage2')

    wandb_project: str = field(default='dexbotic_muvla')

    num_train_epochs: int = field(default=5)

    per_device_train_batch_size: int = field(default=8)

    gradient_accumulation_steps: int = field(default=2)

    save_strategy: str = field(default='steps')

    save_steps: int = field(default=5000)

    save_total_limit: int = field(default=3)

    save_only_model: bool = field(default=True)

    debug_mode: bool = field(default=False)


@dataclass
class MUVLAActionConfig(_MUVLAActionConfig):

    pass


@dataclass
class MUVLADataConfig(_MUVLADataConfig):

    dataset_name: str = field(default='muvla_stage3')

    action_config: MUVLAActionConfig = field(default_factory=MUVLAActionConfig)

    num_images: int = field(default=5)
    auto_norm: bool = field(default=False)

    data_keys: list[str] = field(
        default_factory=lambda: [
            'input_ids',
            'labels',
            # 'reward',
            'image'])


@dataclass
class MUVLAModelConfig(_MUVLAModelConfig):
    
    model_name_or_path: str = field(
        default='muvla/pretrain_stage1')

    freeze_llm: bool = field(default=False)

    freeze_mm_projector: bool = field(default=False)

    freeze_mm_vision: bool = field(default=False)

@dataclass
class MUVLAInferenceConfig(_MUVLAInferenceConfig):

    model_name_or_path: str = field(default='checkpoints/inference/')

    port: int = field(default=7892)


@dataclass
class MUVLAExp(_MUVLAExp):

    model_config: MUVLAModelConfig = field(default_factory=MUVLAModelConfig)

    trainer_config: MUVLATrainerConfig = field(default_factory=MUVLATrainerConfig)

    data_config: MUVLADataConfig = field(default_factory=MUVLADataConfig)

    inference_config: MUVLAInferenceConfig = field(
        default_factory=MUVLAInferenceConfig)


if __name__ == "__main__":

    args = parse_args()
    exp = MUVLAExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'compute_norm_stats':
        exp.compute_norm_stats()
