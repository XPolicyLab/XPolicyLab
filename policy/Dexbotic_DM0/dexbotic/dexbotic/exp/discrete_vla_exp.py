import argparse
import time
from dataclasses import dataclass, field

from loguru import logger
from PIL import Image
import torch
from transformers import AutoTokenizer

from dexbotic.model.discrete_vla.discrete_vla_arch import DiscreteVLAForCausalLM
from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.exp.base_exp import (ActionConfig, BaseExp,
                                   ComputeNormActionConfig, DataConfig,
                                   ModelConfig, OptimizerConfig, TrainerConfig,
                                   InferenceConfig)
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token
from dexbotic.data.dataset.transform.action import (ActionNormAnd2String,
                                                    AddAction, AddTrajectory,
                                                    DeltaAction)
from dexbotic.data.dataset.transform.common import (Pipeline, ToDict, ToList,
                                                    ToNumpy)
from dexbotic.data.dataset.transform.language import AddPromptTemplate
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference'])
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class DiscreteVLAOptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=2e-5)


@dataclass
class DiscreteVLATrainerConfig(TrainerConfig):
    num_train_epochs: int = field(default=5)
    save_steps: int = field(default=20000)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)


@dataclass
class DiscreteVLAActionConfig(ActionConfig):

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline([
            ToDict(),
            ToNumpy(),
            AddAction(predict_length=1),
            DeltaAction(enable=self.delta),
            ActionNormAnd2String(statistic_mapping=statistic_mapping,
                                 vocab_size=self.vocab_size,
                                 string_format=self.string_format),
            LoadMultiModal(),
            AddPromptTemplate(prompt_template=self.prompt_template),
            ToList(),
        ])

        return action_config


@dataclass
class DiscreteVLADataConfig(DataConfig):
    action_config: ActionConfig = field(default_factory=DiscreteVLAActionConfig)


@dataclass
class DiscreteVLAModelConfig(ModelConfig):
    pass


@dataclass
class DiscreteVLAInferenceConfig(InferenceConfig):
    vocab_size: int = field(default=255)

    def _load_model(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = DiscreteVLAForCausalLM.from_pretrained(self.model_name_or_path,
                                                  torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True,
                                                  trust_remote_code=True,
                                                  device_map={"": "cuda:0"}).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        logger.info(f"Model loaded successfully")

    def _get_response(self, text: str, images: list[str]) -> str:
        t0 = time.monotonic()
        if len(images) == 1:
            images = [Image.open(images[0]).convert('RGB')]
            image_tensor = self.model.process_images(images).to(dtype=self.model.dtype)
        else:
            images = [Image.open(image).convert('RGB') for image in images]
            image_tensor = self.model.process_images(
                images).to(dtype=self.model.dtype).unsqueeze(0)

        self._save_image(images, text)

        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to(
            self.model.device)
        logger.debug(f'input_ids: {input_ids}')
        inference_args = {
            'conv': conv,
            'tokenizer': self.tokenizer,
            'vocab_size': self.vocab_size,
            'action_norms': self.norm_stats}

        outputs = self.model.inference_action(input_ids, image_tensor, inference_args)
        logger.info(f'prompt: <start>{prompt}<end>\naction: {outputs}')
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs

@dataclass
class DiscreteVLAExp(BaseExp):
    model_config: DiscreteVLAModelConfig = field(default_factory=DiscreteVLAModelConfig)
    optimizer_config: DiscreteVLAOptimizerConfig = field(
        default_factory=DiscreteVLAOptimizerConfig)
    trainer_config: DiscreteVLATrainerConfig = field(default_factory=DiscreteVLATrainerConfig)
    data_config: DiscreteVLADataConfig = field(default_factory=DiscreteVLADataConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)


if __name__ == "__main__":
    args = parse_args()
    exp = DiscreteVLAExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
