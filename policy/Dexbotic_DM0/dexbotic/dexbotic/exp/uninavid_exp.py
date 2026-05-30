"""Experiment definitions for UniNaVid navigation fine-tuning."""

from __future__ import annotations

import argparse
import io
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from distutils.util import strtobool
from typing import Any, List, Optional

import numpy as np
import torch
import transformers
from easydict import EasyDict
from flask import jsonify, request
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer, BaseImageProcessor

from dexbotic.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.dex_uninavid_dataset import DexUniNaVidDataset
from dexbotic.exp.base_exp import (
    ActionConfig,
    BaseExp,
    DataConfig,
    InferenceConfig,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    TrainerConfig,
)
from dexbotic.exp.uninavid_trainer import DexboticUniNaVidTrainer
from dexbotic.model.uninavid.constants import (
    IMAGE_END_TOKEN,
    IMAGE_SEPARATOR,
    IMAGE_START_TOKEN,
    NAVIGATION_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
)
from dexbotic.model.uninavid.uninavid_arch import (
    DexboticUniNaVidConfig,
    DexboticUniNaVidForCausalLM,
)
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.conversation import KeywordsStoppingCriteria
from dexbotic.tokenization.process import UniNaVidTokenization
from dexbotic.tokenization.tokenization import tokenizer_image_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "inference_single"],
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args, _unknown = parser.parse_known_args()
    return args

@dataclass
class UniNaVidOptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=1e-5)


@dataclass
class UniNaVidTrainerConfig(TrainerConfig):
    output_dir: str = field(
        default=f"./checkpoints/uninavid-{datetime.now().strftime('%m%d')}"
    )
    # Differs from base TrainerConfig: save_steps/logging_steps/dataloader for UniNaVid runs.
    save_steps: int = field(default=1000)
    logging_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    group_by_modality_length: bool = field(default=True)
    lr_multi: Optional[str] = field(default=None)
    freeze_mm_mlp_adapter: bool = field(default=False)
    tune_vision_encoder: bool = field(default=False)

    def added_args_dict(self) -> EasyDict:
        return EasyDict({"tune_mm_mlp_adapter": self.tune_mm_mlp_adapter})


@dataclass
class UniNaVidModelConfig(ModelConfig):
    """
    Model configuration for UniNaVid (LLaMA + vision tower + projector).

    - image_processor: Relative processor directory inside the integrated checkpoint.
    - compress_type: Video token compression strategy (e.g. ``grid:2``).
    - conversation_version: Tokenization template; UniNaVid uses Vicuna-style prompts.
    - tune_mm_mlp_adapter: When True, freeze all params except mm_projector.
    - freeze_backbone: Freeze the LLM backbone.
    """

    model_name_or_path: str = field(default="")
    mm_vision_tower: str = field(default="")
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: int = field(default=-2)
    mm_vision_select_feature: str = field(default="patch")
    compress_type: Optional[str] = field(default="grid:2")
    run_type: str = field(default="train")
    conversation_version: str = field(default="vicuna")
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    tokenizer_model_max_length: Optional[int] = field(default=None)

    def build_model(self) -> DexboticUniNaVidForCausalLM:
        config = DexboticUniNaVidConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

        orig_rope_scaling = getattr(config, "rope_scaling", None) or {"factor": 1}
        orig_rope_scaling_factor = orig_rope_scaling.get("factor", 1)
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if (
            self.tokenizer_model_max_length is not None
            and orig_ctx_len
            and self.tokenizer_model_max_length
            > orig_ctx_len * orig_rope_scaling_factor
        ):
            scaling_factor = float(
                math.ceil(
                    self.tokenizer_model_max_length
                    / (orig_ctx_len * orig_rope_scaling_factor)
                )
            )
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        if getattr(config, "model_type", None) != DexboticUniNaVidConfig.model_type:
            raise ValueError(
                "UniNaVid training runtime only accepts dexbotic checkpoints."
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = DexboticUniNaVidForCausalLM.from_pretrained(
            self.model_name_or_path,
            config=config,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self._freeze_model(model)
        return model

    def _freeze_model(self, model: DexboticUniNaVidForCausalLM):
        if self.freeze_backbone:
            model.model.requires_grad_(False)
        if self.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True


@dataclass
class UniNaVidTokenizerConfig(TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class UniNaVidActionConfig(ActionConfig):
    def build_action_process_func(self) -> Pipeline:
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                LoadMultiModal(),
                ToList(),
            ]
        )


@dataclass
class UniNaVidDataConfig(DataConfig):
    """
    Navigation data config for UniNaVid.

    - dataset_name: ``CONVERSATION_DATA`` key; jsonl root and video directory are read
      from the registry (``annotations`` / ``data_path_prefix`` in
      ``data_source/uninavid_official.py``).
    - dex_jsonl_suffix: Extension used to glob jsonl files under the annotations root.
    - video_fps: Target fps for temporal frame subsampling.
    - dex_use_nav_augment: Enable frame dropout and color jitter on history prefixes.
    """

    dataset_name: str = field(default="uninavid_objnav")
    aug_policy: str = field(default="")
    images_keys: list[str] = field(default_factory=lambda: ["images_1"])
    data_keys: list = field(
        default_factory=lambda: ["input_ids", "labels", "image", "prompt"]
    )
    auto_norm: bool = field(default=False)
    dex_jsonl_suffix: str = field(default=".jsonl")
    video_fps: int = field(default=1)
    dex_use_nav_augment: bool = field(default=True)
    image_grid_pinpoints: Optional[str] = field(default=None)
    is_multimodal: bool = field(default=True)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: bool = field(default=False)
    action_config: ActionConfig = field(default_factory=UniNaVidActionConfig)

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexUniNaVidDataset:
        data_args = EasyDict(
            {
                "dataset_name": self.dataset_name,
                "aug_policy": self.aug_policy,
                "num_images": self.num_images,
                "data_keys": self.data_keys,
                "images_keys": self.images_keys,
                "dex_jsonl_suffix": self.dex_jsonl_suffix,
                "video_fps": self.video_fps,
                "dex_use_nav_augment": self.dex_use_nav_augment,
                "image_aspect_ratio": self.image_aspect_ratio,
                "image_grid_pinpoints": self.image_grid_pinpoints,
                "is_multimodal": self.is_multimodal,
                "input_prompt": self.input_prompt,
                "refine_prompt": self.refine_prompt,
                "image_processor": image_processor,
                "chat_template": chat_template,
                "image_pad_mode": self.image_pad_mode,
            }
        )
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = UniNaVidTokenization(tokenizer, data_args)
        return DexUniNaVidDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )

    def _build_data_collator(self, tokenizer):
        return UniNaVidDataCollator(tokenizer=tokenizer)


class UniNaVidDataCollator:
    """Pads input_ids / labels; stacks image tensors under ``images``."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        from dexbotic.constants import IGNORE_INDEX

        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if (
                all(x is not None and x.shape == images[0].shape for x in images)
                and len(images) > 1
            ):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        return batch


def load_pretrained_model(
    model_path: str,
    device: str | torch.device = "cuda",
) -> tuple[object, DexboticUniNaVidForCausalLM, object, int]:
    """Load tokenizer + full dexbotic UniNaVid model."""
    device_t = torch.device(device) if not isinstance(device, torch.device) else device

    config = DexboticUniNaVidConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config, "model_type", None) != DexboticUniNaVidConfig.model_type:
        raise ValueError("UniNaVid runtime only accepts dexbotic checkpoints.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    model = DexboticUniNaVidForCausalLM.from_pretrained(
        model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": str(device_t)} if device_t.type == "cuda" else "auto",
    )
    model.eval()

    vision_tower = model.get_vision_tower()
    model.to(dtype=torch.float16)
    if hasattr(vision_tower, "to"):
        vision_tower.to(device=device_t, dtype=torch.float16)
    image_processor = getattr(vision_tower, "image_processor", None)
    context_len = getattr(
        model.config,
        "tokenizer_model_max_length",
        getattr(model.config, "max_sequence_length", 2048),
    )
    return tokenizer, model, image_processor, context_len


@dataclass
class UniNaVidInferenceConfig(InferenceConfig):
    """
    Flask inference preserving the original UniNaVid online inference behavior:
    append incoming frames, flush buffered frames on generation, and rely on
    feat_cache for long-horizon navigation memory.
    """

    model_name_or_path: Optional[str] = field(default=None)
    port: int = field(default=7892)
    policy_name: str = field(default="uninavid")
    conversation_template: str = field(default="vicuna")
    temperature: float = field(default=0.2)
    max_new_tokens: int = field(default=1024)
    navigation_prompt_template: str = field(
        default=(
            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
            "of historical observations and an image of the current observation <image>. Your assigned task is: '{0}'. "
            "Analyze this series of images to determine your next four actions. The predicted action should be one of "
            "the following: forward, left, right, or stop."
        )
    )
    max_return_actions: int = field(default=2)

    def _parse_bool(self, value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        try:
            return bool(strtobool(str(value)))
        except Exception:
            return default

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.model_name_or_path:
            raise ValueError("UniNaVidInferenceConfig.model_name_or_path is required.")
        logger.info("Loading UniNaVid model from {}", self.model_name_or_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                self.model_name_or_path,
                device=str(self.device),
            )
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval()
        self.model.config.run_type = "eval"
        self.model_config = self.model.config
        logger.info("UniNaVid model loaded on {}", self.device)

    def _initialize_inference(self) -> None:
        super()._initialize_inference()
        self.rgb_list: deque[np.ndarray] = deque()
        self.prev_text = None  # required by base _save_image for episode tracking
        self._reset_nav_session()
        if self.save_image:
            self.save_image_dir = os.path.join(
                self.save_image_dir,
                self.policy_name,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            os.makedirs(self.save_image_dir, exist_ok=True)

    def _reset_nav_session(self) -> None:
        self.rgb_list.clear()
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def _build_navigation_input_ids(self, qs: str) -> torch.Tensor:
        conv = conversation_lib.conv_templates[self.conversation_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        token_prompt = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]

        def _lookup_special_token(token: str) -> torch.Tensor:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Special token {token!r} is missing from tokenizer vocab.")
            return torch.tensor([token_id], dtype=token_prompt.dtype)

        image_start_special_token = _lookup_special_token(IMAGE_START_TOKEN)
        image_end_special_token = _lookup_special_token(IMAGE_END_TOKEN)
        video_start_special_token = _lookup_special_token(VIDEO_START_SPECIAL_TOKEN)
        video_end_special_token = _lookup_special_token(VIDEO_END_SPECIAL_TOKEN)
        navigation_special_token = _lookup_special_token(NAVIGATION_SPECIAL_TOKEN)
        image_separator = _lookup_special_token(IMAGE_SEPARATOR)
        new_list: List[torch.Tensor] = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_separator)
            new_list.append(token_prompt[idx : idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1 :]
            indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
        num_emb = self.model.get_input_embeddings().weight.shape[0]
        bad = (input_ids != IMAGE_TOKEN_INDEX) & (
            (input_ids < 0) | (input_ids >= num_emb)
        )
        if bad.any():
            bad_vals = input_ids.masked_select(bad).unique().tolist()
            raise ValueError(
                f"input_ids contain id(s) {bad_vals} outside embedding rows [0, {num_emb}); "
                "ensure special tokens were added (initialize_vision_tokenizer) and the "
                "checkpoint matches the tokenizer."
            )
        return input_ids.to(self.device)

    def _actions_from_navigation_text(self, navigation: str) -> List[int]:
        queue_actions: List[int] = []
        for action in navigation.split(" "):
            if action == "stop":
                queue_actions.append(0)
            elif action == "forward":
                queue_actions.append(1)
            elif action == "left":
                queue_actions.append(2)
            elif action == "right":
                queue_actions.append(3)
            else:
                raise ValueError(
                    f"wrong actions! got {action!r}, please check the code and data"
                )
            if len(queue_actions) == self.max_return_actions:
                break
        return queue_actions

    def _prepare_images(self, image_bytes_list: List[bytes]) -> List[np.ndarray]:
        reset_memory = getattr(self, "meta_data", {}).get("reset_memory", False)
        run_model = getattr(self, "meta_data", {}).get("run_model", True)

        if reset_memory:
            self._reset_nav_session()

        for image_bytes in image_bytes_list:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.asarray(pil)[..., ::-1]
            self.rgb_list.append(np.ascontiguousarray(arr))

        if not run_model:
            return []

        buffered_frames = list(self.rgb_list)
        self.rgb_list.clear()
        return buffered_frames

    def _build_navigation_prompt(self, text: str) -> tuple[str, str]:
        prompt_full = self.navigation_prompt_template.format(text)
        question = prompt_full.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_full.replace("<image>", "")
        return question, qs

    def _predict_navigation_text(self, text: str, rgb_list: List[np.ndarray]) -> str:
        if not rgb_list:
            return ""
        question, qs = self._build_navigation_prompt(text)
        input_ids = self._build_navigation_input_ids(qs)
        conv = conversation_lib.conv_templates[self.conversation_template].copy()
        stop_str = (
            conv.sep
            if conv.sep_style != conversation_lib.SeparatorStyle.TWO
            else conv.sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        batch = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        dtype = next(self.model.parameters()).dtype
        pixel = self.image_processor.preprocess(batch, return_tensors="pt")[
            "pixel_values"
        ].to(device=self.device, dtype=dtype)
        self.model.update_prompt([[question]])

        temp = float(self.temperature) if self.temperature is not None else 0.2
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                images=[pixel],
                do_sample=False,
                temperature=temp,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()
        logger.info("UniNaVid raw output: {!r}", outputs)
        return outputs

    def _get_response(
        self, text: str, images: List[np.ndarray] | None = None, **kwargs
    ) -> List[int]:
        if not images:
            return []
        pil_images = [Image.fromarray(image).convert("RGB") for image in images]
        self._save_image(pil_images, text)
        outputs = self._predict_navigation_text(text=text, rgb_list=images)
        return self._actions_from_navigation_text(outputs)

    def process_frame(self):
        text = request.form.get("text", "")
        episode_first_frame = request.form.get("episode_first_frame")
        run_model = self._parse_bool(request.form.get("run_model"), default=True)
        reset_memory = self._parse_bool(episode_first_frame, default=False)
        temp_raw = request.form.get("temperature")
        if temp_raw is not None and str(temp_raw).strip() != "":
            try:
                self.temperature = float(temp_raw)
            except ValueError:
                pass
        self.meta_data = {"reset_memory": reset_memory, "run_model": run_model}
        images = request.files.getlist("image")
        image_bytes_list = [image.read() for image in images]
        rgb_list = self._prepare_images(image_bytes_list)
        results = self._get_response(
            text=text, images=rgb_list, reset_memory=reset_memory
        )
        return jsonify({"response": results})


@dataclass
class UniNaVidExp(BaseExp):
    """Training experiment: UniNaVid vision modules, imgsp tokenizer, DexUniNaVidDataset."""

    model_config: UniNaVidModelConfig = field(default_factory=UniNaVidModelConfig)
    optimizer_config: UniNaVidOptimizerConfig = field(
        default_factory=UniNaVidOptimizerConfig
    )
    trainer_config: UniNaVidTrainerConfig = field(default_factory=UniNaVidTrainerConfig)
    data_config: UniNaVidDataConfig = field(default_factory=UniNaVidDataConfig)
    tokenizer_config: UniNaVidTokenizerConfig = field(
        default_factory=UniNaVidTokenizerConfig
    )
    inference_config: UniNaVidInferenceConfig = field(
        default_factory=UniNaVidInferenceConfig
    )

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda _msg: None)

        tc = self.trainer_config
        mc = self.model_config
        dc = self.data_config

        if mc.conversation_version not in {"vicuna"}:
            logger.warning(
                "UniNaVid training uses Vicuna-style tokenization only; "
                "conversation_version={!r} is ignored.",
                mc.conversation_version,
            )

        # Step 1: build tokenizer
        tokenizer_kwargs = {
            "model_max_length": tc.model_max_length,
            "padding_side": "right",
            "use_fast": self.tokenizer_config.use_fast_tokenizer,
        }
        self.tokenizer = self.tokenizer_config.build_tokenizer(
            mc.model_name_or_path, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        mc.tokenizer_model_max_length = tc.model_max_length

        # Step 2: build model
        model = mc.build_model()
        model.config.use_cache = False
        self.model = model

        # Trainer-config-driven parameter freezes (depend on both mc and tc)
        model.config.freeze_mm_mlp_adapter = tc.freeze_mm_mlp_adapter
        if tc.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        vision_tower = model.get_vision_tower()
        if tc.tune_vision_encoder:
            vision_tower.requires_grad_(True)

        # Step 3: sync model config fields that data / trainer configs depend on
        model.config.image_aspect_ratio = dc.image_aspect_ratio
        model.config.image_grid_pinpoints = dc.image_grid_pinpoints
        model.config.tune_mm_mlp_adapter = mc.tune_mm_mlp_adapter
        tc.tune_mm_mlp_adapter = mc.tune_mm_mlp_adapter

        # Step 4: build dataset and collator
        train_dataset, data_collator = dc.build_data(
            self.tokenizer,
            mc.conversation_version,
            vision_tower.image_processor,
        )

        # Step 5: build trainer
        self.trainer = DexboticUniNaVidTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            exp_config=self,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

    def inference(self) -> None:
        self.inference_config.run()

    def inference_single(
        self,
        image_path: str,
        prompt: str,
        reset_memory: bool = True,
    ):
        self.inference_config._initialize_inference()
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            logger.error("Image not found: {}", image_path)
            return None
        self.inference_config.meta_data = {
            "reset_memory": reset_memory,
            "run_model": True,
        }
        rgb_list = self.inference_config._prepare_images([image_bytes])
        result = self.inference_config._get_response(
            text=prompt,
            images=rgb_list,
            reset_memory=reset_memory,
        )
        logger.info("Inference result: {}", result)
        return result


if __name__ == "__main__":
    _args = parse_args()
    exp = UniNaVidExp()
    if _args.task == "train":
        exp.train()
    elif _args.task == "inference":
        exp.inference()
    elif _args.task == "inference_single":
        if not _args.image_path or not _args.prompt:
            raise SystemExit("inference_single requires --image_path and --prompt")
        exp.inference_single(_args.image_path, _args.prompt)
