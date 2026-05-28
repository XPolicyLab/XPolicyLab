import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import transformers
from easydict import EasyDict
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer, BaseImageProcessor

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.transform.action import (
    ActionNormAnd2String,
    AddAction,
    AddTrajectory,
    DeltaAction,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.language import (
    InsertImageTokenPrefix,
    ReplaceAnswer,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.base_exp import ActionConfig, BaseExp, DataConfig
from dexbotic.exp.base_exp import InferenceConfig as BaseInferenceConfig
from dexbotic.exp.base_exp import (
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    TrainerConfig,
)
from dexbotic.model.gr00tn1.gr00tn1_arch import GR00TN1ForCausalLM
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.process import GR00TN1Tokenization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class GR00TN1ModelConfig(ModelConfig):
    action_model_type: str = field(default="fm")
    chat_template: str = field(default="qwen2-chat")
    action_head_cfg: Dict[str, Any] = field(default=None)

    tune_llm: bool = field(default=False)
    tune_visual: bool = field(default=True)
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)

    def build_model(self) -> GR00TN1ForCausalLM:
        if self.from_llm:
            raise ValueError("GR00TN1 does not support from_llm")
        else:
            model = GR00TN1ForCausalLM.from_pretrained(self.model_name_or_path)
            model.model.set_trainable_parameters(
                tune_projector=self.tune_projector,
                tune_diffusion_model=self.tune_diffusion_model,
                tune_llm=self.tune_llm,
                tune_visual=self.tune_visual,
            )
        return model


@dataclass
class GR00TN1TrainerConfig(TrainerConfig):
    pass


@dataclass
class GR00TN1OptimizerConfig(OptimizerConfig):
    pass


@dataclass
class GR00TN1ActionConfig(ActionConfig):
    replace_with_default_answer: str = field(default=None)
    num_images: int = field(default=2)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                DeltaAction(enable=self.delta),
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    padding_mode=self.trajectory_padding_model,
                    padding_action=self.padding_action,
                ),
                ActionNormAnd2String(
                    statistic_mapping=statistic_mapping,
                    vocab_size=self.vocab_size,
                    string_format=self.string_format,
                ),
                LoadMultiModal(),
                InsertImageTokenPrefix(num_images=self.num_images),
                ReplaceAnswer(default_answer=self.replace_with_default_answer),
                ToList(),
            ]
        )

        return action_config


@dataclass
class GR00TN1DataConfig(DataConfig):
    action_config: ActionConfig = field(default_factory=GR00TN1ActionConfig)
    data_keys: list[str] = field(
        default_factory=lambda: ["input_ids", "labels", "action", "image"]
    )
    num_images: int = field(default=2)

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexDataset:
        # FIXME: DO NOT USE EASYDICT IN NEXT VERSION
        data_args = EasyDict(
            {
                "dataset_name": self.dataset_name,
                "num_images": self.num_images,
                "data_keys": self.data_keys,
                "images_keys": self.images_keys,
                "aug_policy": self.aug_policy,
                "image_aspect_ratio": self.image_aspect_ratio,
                "image_processor": image_processor,
                "chat_template": chat_template,
            }
        )
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = GR00TN1Tokenization(tokenizer, data_args)
        dataset = DexDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )
        return dataset


@dataclass
class InferenceConfig(BaseInferenceConfig):
    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = GR00TN1ForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        logger.info("Model loaded successfully")

    def _get_response(self, text: str, images: list[str]) -> str:
        t0 = time.monotonic()
        if len(images) == 1:
            images = [Image.open(images[0]).convert("RGB")]
            image_tensor = self.model.process_images(images).to(dtype=self.model.dtype)
        else:
            images = [Image.open(image).convert("RGB") for image in images]
            image_tensor = (
                self.model.process_images(images)
                .to(dtype=self.model.dtype)
                .unsqueeze(0)
            )

        self._save_image(images, text)

        prefix = ""
        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        if self.model_config.chat_template == "step":
            conv.append_message(
                conv.roles[0], text + "<im_start>" + DEFAULT_IMAGE_TOKEN + "<im_end>"
            )
        elif self.model_config.chat_template == "qwen2-chat":
            if len(images) == 1:
                prefix = "<image>\n"
            else:
                for idx in range(len(images)):
                    prefix = prefix + f"<image {idx + 1}><img><image></img>\n"
            conv.append_message(conv.roles[0], prefix + text)
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            GR00TN1Tokenization.tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        logger.debug(f"input_ids: {input_ids}")
        inference_args = {
            "cfg_scale": 1.5,
            "num_ddim_steps": 10,
            "action_norms": self.norm_stats,
        }
        outputs = self.model.inference_action(input_ids, image_tensor, inference_args)

        logger.info(f"prompt: <start>{prompt}<end>\naction: {outputs}")
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs


class GR00TN1TokenizerConfig(TokenizerConfig):
    def build_tokenizer(
        self, model_name_or_path: str, **kwargs
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        tokens_to_keep = ["<box>", "</box>", "<ref>", "</ref>"]
        tokenizer.additional_special_tokens = [
            item
            for item in tokenizer.additional_special_tokens
            if item not in tokens_to_keep
        ]
        tokenizer.padding_side = "left"
        return tokenizer


@dataclass
class GR00TN1Exp(BaseExp):
    data_config: GR00TN1DataConfig = field(default_factory=GR00TN1DataConfig)
    model_config: GR00TN1ModelConfig = field(default_factory=GR00TN1ModelConfig)
    trainer_config: GR00TN1TrainerConfig = field(default_factory=GR00TN1TrainerConfig)
    optimizer_config: GR00TN1OptimizerConfig = field(
        default_factory=GR00TN1OptimizerConfig
    )
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    tokenizer_config: GR00TN1TokenizerConfig = field(
        default_factory=GR00TN1TokenizerConfig
    )

    def inference(self) -> None:
        self.inference_config.run()


if __name__ == "__main__":
    args = parse_args()
    exp = GR00TN1Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
