import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.cogact_exp import (CogACTDataConfig, CogACTExp,
                                     CogACTModelConfig, CogACTTrainerConfig,
                                     InferenceConfig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference',
            'inference_single'])
    parser.add_argument(
        '--image_path',
        type=str,
        default=None)
    parser.add_argument(
        '--prompt',
        type=str,
        default=None)
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class ManiskillCogActTrainerConfig(CogACTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/maniskill_cogact/{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_maniskill_cogact')
    num_train_epochs: int = field(default=10)
    save_strategy: str = field(default='epoch')
    save_total_limit: int = field(default=2)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)


@dataclass
class ManiskillCogActActionConfig(CogACTDataConfig):
    pass


@dataclass
class ManiskillCogActDataConfig(CogACTDataConfig):
    dataset_name: str = field(default='maniskill_pickcube+maniskill_stackcube+maniskill_picksingleycb+maniskill_picksingleegad+maniskill_pickclutterycb')


@dataclass
class ManiskillCogActModelConfig(CogACTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class ManiskillCogActInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/maniskill/maniskill_cogact')
    port: int = field(default=7892)


@dataclass
class ManiskillCogActExp(CogACTExp):
    model_config: ManiskillCogActModelConfig = field(
        default_factory=ManiskillCogActModelConfig)
    trainer_config: ManiskillCogActTrainerConfig = field(
        default_factory=ManiskillCogActTrainerConfig)
    data_config: ManiskillCogActDataConfig = field(
        default_factory=ManiskillCogActDataConfig)
    inference_config: ManiskillCogActInferenceConfig = field(
        default_factory=ManiskillCogActInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions = self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = ManiskillCogActExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
