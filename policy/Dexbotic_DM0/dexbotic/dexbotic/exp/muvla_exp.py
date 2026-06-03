import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, cast
from typing import Callable, Dict, Optional, Union
import megfile
import torch
from flask import Flask, jsonify, request
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.exp.base_exp import (ActionConfig, BaseExp,
                                   ComputeNormActionConfig, Config, DataConfig,
                                   ModelConfig, OptimizerConfig, TrainerConfig,
                                   InferenceConfig)
from dexbotic.model.muvla.muvla_arch import (MUVLAConfig, MUVLAForCausalLM,
                                               MUVLAModel)
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.conversation import SeparatorStyle
from dexbotic.tokenization.tokenization import tokenizer_image_token
from dexbotic.tokenization.conversation import KeywordsStoppingCriteria
from dexbotic.data.dataset.transform.language import (AddPromptTemplate,
                                                      ReplaceAnswer,
                                                      defalut_prompt_template)
from dexbotic.data.dataset.transform.common import (Pipeline, ToDict, ToList,
                                                    ToNumpy)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.trainer import DexboticTrainer
OPENAI_CLIP_PATH = os.environ.get(
    'OPENAI_CLIP_PATH',
    'openai/clip-vit-large-patch14-336')
                                                
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
class MUVLAOptimizerConfig(OptimizerConfig):

    base_lr: float = field(default=2e-5)


@dataclass
class MUVLATrainerConfig(TrainerConfig):

    num_train_epochs: int = field(default=5)
    save_steps: int = field(default=20000)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)
    debug_mode: bool = field(default=False)
    
        

@dataclass
class MUVLAActionConfig(ActionConfig):

    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline([
            ToDict(),
            ToNumpy(),
            LoadMultiModal(),
            AddPromptTemplate(prompt_template=self.prompt_template),
            ToList(),
        ])

        return action_config


@dataclass
class MUVLADataConfig(DataConfig):

    num_images: int = field(default=5)
    action_config: ActionConfig = field(default_factory=MUVLAActionConfig)


@dataclass
class MUVLAModelConfig(ModelConfig):

    mm_vision_tower: str = field(
        default=OPENAI_CLIP_PATH)
    obs_vision_tower: str = field(
        default=OPENAI_CLIP_PATH)
    mm_projector_type: str = field(default='mlp2x_gelu')

    def build_model(self) -> MUVLAForCausalLM:

        if self.from_llm:
            model_config_args = {
                "llm_config": self.model_name_or_path,
                "chat_template": self.chat_template,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "obs_vision_tower": self.obs_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "init_llm_weights": True,
            }
            model_config = MUVLAConfig(**model_config_args)
            model = MUVLAForCausalLM(model_config)
        else:
            model_config_args = {
                "model_name_or_path": self.model_name_or_path,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "obs_vision_tower": self.obs_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
            }
            model = MUVLAForCausalLM.from_pretrained(self.model_name_or_path)

            model.model.initialize_model(model_config_args)

        self._freeze_model(model)

        return model

    def _freeze_model(self, model: MUVLAForCausalLM):
        model.model = cast(MUVLAModel, model.model)

        for param in model.model.parameters():
            param.requires_grad = True

        if self.freeze_llm:
            for param in model.model.backbone.parameters():
                param.requires_grad = False
        if self.freeze_mm_projector:
            for param in model.model.mm_projector_module.parameters():
                param.requires_grad = False
        if self.freeze_mm_vision:
            for param in model.model.mm_vision_module.parameters():
                param.requires_grad = False
        if self.freeze_action_head:
            for param in model.model.action_head_module.parameters():
                param.requires_grad = False


@dataclass
class InferenceConfig(InferenceConfig):

    model_name_or_path: Optional[str] = field(default=None)

    port: int = field(default=7892)

    save_image: bool = field(default=False)

    save_image_dir: str = field(default='./debug_data')

    norm_stats: Optional[dict] = field(default=None)

    def process_frame(self) -> None:
        results = self._get_response(
            text=request.form.get('text', ''),
            images=request.files.getlist('image', None),
        )
        print(results)
        return jsonify({'response': results})

    def run(self) -> None:
        self._initialize_inference()
        self.app = Flask(__name__)
        self.app.add_url_rule(
            '/process_frame',
            'process_frame',
            self.process_frame,
            methods=['POST'])
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=False)

    def _load_model(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = MUVLAForCausalLM.from_pretrained(self.model_name_or_path,
                                                  torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True,
                                                  trust_remote_code=True,
                                                  device_map='cuda:0').to(self.device)
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
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        print('input preprocessing time', time.monotonic() - t0)
        temperature=0.7
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=5)
        
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if hasattr(conv, 'sep2'):
            outputs = outputs.replace(conv.sep2, '')
        
        logger.info(f'prompt: <start>{prompt}<end>\naction: {outputs}')
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs

    def _save_image(self, images: list[str], text: str) -> None:
        if not self.save_image:
            return
        if text == self.prev_text:
            self.timestep += 1
        else:
            self.timestep = 0
            self.prev_text = text
            self.episode += 1
        save_image_dir_episode = os.path.join(self.save_image_dir, str(self.episode))
        os.makedirs(save_image_dir_episode, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(
                os.path.join(
                    save_image_dir_episode,
                    f'{self.timestep}_{idx}.png'))
        if self.timestep == 0:
            with open(os.path.join(save_image_dir_episode, 'text.txt'), 'w') as f:
                f.write(text)

    def _initialize_inference(self) -> None:
        self._load_model()
        self.prev_prompt = None
        self.timestep = 0
        self.episode = 0

        if self.norm_stats is None:
            norm_stats_file = os.path.join(self.model_name_or_path, 'norm_stats.json')
            self.norm_stats = self.read_normalization_stats(norm_stats_file)
        logger.info(f"Normalization stats: {self.norm_stats}")

    def read_normalization_stats(self, action_norm_file):
        logger.info(f"Reading normalization stats from {action_norm_file}")
        if action_norm_file is None or not megfile.smart_exists(action_norm_file):
            return {'min': -1, 'max': 1}
        with megfile.smart_open(action_norm_file, 'r') as f:
            norm_stats = json.load(f)
            if 'norm_stats' in norm_stats:
                norm_stats = norm_stats['norm_stats']
            norm_stats = norm_stats['default']
        return norm_stats


@dataclass
class MUVLAExp(BaseExp):
    model_config: MUVLAModelConfig = field(default_factory=MUVLAModelConfig)
    optimizer_config: MUVLAOptimizerConfig = field(
        default_factory=MUVLAOptimizerConfig)
    trainer_config: MUVLATrainerConfig = field(default_factory=MUVLATrainerConfig)
    data_config: MUVLADataConfig = field(default_factory=MUVLADataConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)
    
    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)

        self._auto_compute_norm_stats()

        tokenizer_kwargs = {
            "model_max_length": self.trainer_config.model_max_length,
            "padding_side": "right",
            "use_fast": self.tokenizer_config.use_fast_tokenizer,
        }
        tokenizer = self.tokenizer_config.build_tokenizer(
            self.model_config.model_name_or_path, **tokenizer_kwargs)
        self.tokenizer = tokenizer

        model = self.model_config.build_model()
        self.model = model
        self.tokenizer = self.tokenizer_config.add_special_tokens(
            self.data_config.action_config.string_format,
            self.data_config.action_config.vocab_size,
            self.tokenizer,
            self.model)
        self.model.config.use_cache = False
        self.model.model.llm.config.use_cache = False

        train_dataset, data_collator = self.data_config.build_data(
            self.tokenizer, self.model_config.chat_template, self.model.model.mm_vision_module.image_processor, )

        trainer_kwargs = {
            "model": self.model,
            "processing_class": self.tokenizer,
            "exp_config": self,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
        }
        trainer = DexboticTrainer(**trainer_kwargs)
        self.trainer = trainer

        os.makedirs(self.trainer_config.output_dir, exist_ok=True)



if __name__ == "__main__":
    args = parse_args()
    exp = MUVLAExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'compute_norm_stats':
        exp.compute_norm_stats()
