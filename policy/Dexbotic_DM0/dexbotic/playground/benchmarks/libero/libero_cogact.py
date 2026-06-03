import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.data.dataset.transform.action import ActionNormAnd2String
from dexbotic.data.dataset.transform.language import ReplaceAnswer
from dexbotic.exp.cogact_exp import (CogACTActionConfig, CogACTDataConfig, CogACTExp,
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
    parser.add_argument(
        '--train-backend',
        type=str,
        default=None,
        choices=['deepspeed', 'fsdp', 'fsdp2', 'ddp'],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class LiberoCogActTrainerConfig(CogACTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/libero_all_cogact/all-{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_libero_cogact')
    num_train_epochs: int = field(default=25)


@dataclass
class LiberoCogActActionConfig(CogACTActionConfig):
    def build_action_process_func(self):
        action_process_func = super().build_action_process_func()
        for transform in action_process_func.transforms:
            if isinstance(transform, ActionNormAnd2String):
                transform.add_answer = False
            elif isinstance(transform, ReplaceAnswer):
                transform.replace_existing = True
        return action_process_func


@dataclass
class LiberoCogActDataConfig(CogACTDataConfig):
    dataset_name: str = field(default='libero_goal+libero_10+libero_spatial+libero_object')
    action_config: CogACTActionConfig = field(default_factory=LiberoCogActActionConfig)

@dataclass
class LiberoCogActModelConfig(CogACTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class LiberoCogActInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/libero/libero_cogact')
    port: int = field(default=7891)
        

@dataclass
class LiberoCogActExp(CogACTExp):
    model_config: LiberoCogActModelConfig = field(
        default_factory=LiberoCogActModelConfig)
    trainer_config: LiberoCogActTrainerConfig = field(
        default_factory=LiberoCogActTrainerConfig)
    data_config: LiberoCogActDataConfig = field(
        default_factory=LiberoCogActDataConfig)
    inference_config: LiberoCogActInferenceConfig = field(
        default_factory=LiberoCogActInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions =self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = LiberoCogActExp()
    if args.train_backend is not None:
        exp.trainer_config.train_backend = args.train_backend
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
