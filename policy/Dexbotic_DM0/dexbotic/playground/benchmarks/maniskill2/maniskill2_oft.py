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
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class ManiskillOFTTrainerConfig(OFTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/maniskill_oft/{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_maniskill_oft')
    num_train_epochs: int = field(default=30)


@dataclass
class ManiskillOFTActionConfig(OFTActionConfig):
    delta: bool = field(default=False)


@dataclass
class ManiskillOFTDataConfig(OFTDataConfig):
    dataset_name: str = field(default='maniskill_pickcube+maniskill_stackcube+maniskill_picksingleycb+maniskill_picksingleegad+maniskill_pickclutterycb')
    action_config: ManiskillOFTActionConfig = field(default_factory=ManiskillOFTActionConfig)
    data_keys: list[str] = field(default_factory=lambda: [
        'input_ids', 'labels', 'action', 'state', 'image'])


@dataclass
class ManiskillOFTModelConfig(OFTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')
    use_proprio: bool = field(default=True)
    proprio_dim: int = field(default=11)


@dataclass
class ManiskillOFTInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(
        default='./checkpoints/maniskill/maniskill_oft')
    port: int = field(default=7892)


@dataclass
class ManiskillOFTExp(OFTExp):
    model_config: ManiskillOFTModelConfig = field(
        default_factory=ManiskillOFTModelConfig)
    trainer_config: ManiskillOFTTrainerConfig = field(
        default_factory=ManiskillOFTTrainerConfig)
    data_config: ManiskillOFTDataConfig = field(
        default_factory=ManiskillOFTDataConfig)
    inference_config: ManiskillOFTInferenceConfig = field(
        default_factory=ManiskillOFTInferenceConfig)

    def inference_single(self, image_path: str, prompt: str):
        self.inference_config._initialize_inference()
        actions = self.inference_config._get_response(prompt, [image_path])


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        import debugpy
        try:
            port = 9501
            debugpy.listen(("0.0.0.0", port))
            print(f"Waiting for debugger attach: {port}")
            debugpy.wait_for_client()
        except Exception as e:
            print(f"error when attach: {e}")
    exp = ManiskillOFTExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'inference_single':
        exp.inference_single(args.image_path, args.prompt)
