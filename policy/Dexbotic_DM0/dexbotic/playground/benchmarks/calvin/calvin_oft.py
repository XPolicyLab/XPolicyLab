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
class CalvinOFTTrainerConfig(OFTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/calvin_all_cogact/ABC-{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_calvin_cogact')
    num_train_epochs: int = field(default=5)


@dataclass
class CalvinOFTDataConfig(OFTDataConfig):
    dataset_name: str = field(default='calvin_ABC')
    auto_norm_method: str = field(default='minmax')


@dataclass
class CalvinOFTModelConfig(OFTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class CalvinOFTInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/calvin/calvin_oft')
    port: int = field(default=7891)
        

@dataclass
class CalvinOFTExp(OFTExp):
    model_config: CalvinOFTModelConfig = field(
        default_factory=CalvinOFTModelConfig)
    trainer_config: CalvinOFTTrainerConfig = field(
        default_factory=CalvinOFTTrainerConfig)
    data_config: CalvinOFTDataConfig = field(
        default_factory=CalvinOFTDataConfig)
    inference_config: CalvinOFTInferenceConfig = field(
        default_factory=CalvinOFTInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions =self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    exp = CalvinOFTExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
