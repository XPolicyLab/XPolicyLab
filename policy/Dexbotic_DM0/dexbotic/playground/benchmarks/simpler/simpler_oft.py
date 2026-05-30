import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.oft_exp import (OFTDataConfig, OFTExp,
                                  OFTModelConfig, OFTTrainerConfig,
                                  InferenceConfig, OFTActionConfig)


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
class SimplerOFTTrainerConfig(OFTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/simpler_oft/{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_simpler_oft')
    num_train_epochs: int = field(default=5)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=16)


@dataclass
class SimplerOFTActionConfig(OFTActionConfig):
    padding_action: bool = field(default=True)

@dataclass
class SimplerOFTDataConfig(OFTDataConfig):
    dataset_name: str = field(default='simpler_all')
    action_config: SimplerOFTActionConfig = field(default_factory=SimplerOFTActionConfig)

@dataclass
class SimplerOFTModelConfig(OFTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class SimplerOFTInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/simpler/simpler_oft')
    port: int = field(default=7891)
        

@dataclass
class SimplerOFTExp(OFTExp):
    model_config: SimplerOFTModelConfig = field(
        default_factory=SimplerOFTModelConfig)
    trainer_config: SimplerOFTTrainerConfig = field(
        default_factory=SimplerOFTTrainerConfig)
    data_config: SimplerOFTDataConfig = field(
        default_factory=SimplerOFTDataConfig)
    inference_config: SimplerOFTInferenceConfig = field(
        default_factory=SimplerOFTInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions =self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = SimplerOFTExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
