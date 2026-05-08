# --------------------------------------------------------
# GO1 based on InternVL
# Copyright (c) 2024 AgiBot
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import argparse
import functools
import importlib
import json
import logging
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import transformers
from accelerate import PartialState
from PIL import Image, ImageFile, PngImagePlugin
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import enable_default_handler, enable_explicit_format, set_verbosity

from go1.configs.go1_base_cfg import BaseDatasetArguments, BaseModelArguments, BaseSpaceArguments
from go1.internvl.dist_utils import init_dist
from go1.internvl.model.go1 import GO1Model, GO1ModelConfig
from go1.internvl.model.go1.configuration_action_expert import ActionExpertConfig
from go1.internvl.patch import concat_pad_data_collator_go1
from go1.internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from go1.lerobot.dataset_lerobot import WrappedLeRobotDataset
from go1.tools.env_parse import get_bool_env

warnings.simplefilter(action="ignore", category=FutureWarning)

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def exp_info_init(exp_dir: str, cfg_path: str, dist_state: PartialState):
    if dist_state.is_main_process and cfg_path is not None:
        cfg_path = Path(cfg_path)
        exp_dir = Path(exp_dir)
        backup_path = exp_dir / cfg_path.name
        if os.path.exists(backup_path):
            time_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = exp_dir / Path(cfg_path.stem + f"_{time_suffix}.py")
        shutil.copyfile(cfg_path, backup_path)


def build_datasets(
    tokenizer,
    num_image_token: int,
    dataset_args: BaseDatasetArguments,
    model_args: BaseModelArguments,
    is_train: bool = True,
    space_args: BaseSpaceArguments = None,
    stats_save_path: str = None,
):
    if dataset_args.dataset_type == "lerobot":
        dataset = WrappedLeRobotDataset(
            root=dataset_args.data_root_dir,
            action_chunk_size=model_args.action_chunk_size,
            transforms=dataset_args.transforms,
            is_train=is_train,
            text_tokenizer=tokenizer,
            num_image_token=num_image_token,
            image_size=model_args.force_image_size,
            pad2square=model_args.pad2square,
            normalize_type=model_args.normalize_type,
            use_thumbnail=model_args.use_thumbnail,
            min_dynamic_patch=model_args.min_dynamic_patch,
            max_dynamic_patch=model_args.max_dynamic_patch,
            space_args=space_args,
        )

        if stats_save_path is not None:
            # Try to get stats, fallback to meta.stats if not available
            if hasattr(dataset, "stats"):
                dataset_stats = deepcopy(dataset.stats)
            else:
                raise ValueError("Cannot find stats in dataset.dataset or dataset.dataset.meta")
            for k, v in space_args.space_repack.items():
                if v in dataset_stats:
                    dataset_stats[k] = dataset_stats.pop(v)
            # save the stats as json in checkpoint
            if not dist.is_initialized() or dist.get_rank() == 0:
                with open(os.path.join(stats_save_path, "dataset_stats.json"), "w") as f:
                    json.dump(convert(dataset_stats), f)
            if dist.is_initialized():
                dist.barrier()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_args.dataset_type}")

    return dataset


def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    else:
        return obj


def build_ae_config(
    model_args: BaseModelArguments,
    go1_config: GO1ModelConfig,
    space_args: BaseSpaceArguments,
) -> ActionExpertConfig:
    """Build the action expert config from the given model and data arguments."""
    # LLM architecture related config update
    llm_config_dict: Dict = go1_config.llm_config.to_dict()
    action_config_dict = deepcopy(llm_config_dict)
    # Remove unnecessary keys from the action config
    for key in (
        "_name_or_path",
        "bos_token_id",
        "chunk_size_feed_forward",
        "cross_attention_hidden_size",
        "decoder_start_token_id",
        "diversity_penalty",
        "do_sample",
        "early_stopping",
        "encoder_no_repeat_ngram_size",
        "eos_token_id",
        "exponential_decay_length_penalty",
        "finetuning_task",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "id2label",
        "num_beam_groups",
        "num_beams",
        "sep_token_id",
        "suppress_tokens",
        "task_specific_params",
        "tie_encoder_decoder",
        "tie_word_embeddings",
        "tokenizer_class",
        "top_k",
        "top_p",
        "typical_p",
        "vocab_size",
    ):
        action_config_dict.pop(key, None)

    action_config_dict["architectures"] = ["ActionExpertModel"]
    action_config_dict["attn_implementation"] = "eager"
    action_config_dict["auto_map"] = {
        "AutoConfig": "configuration_action_expert.ActionExpertConfig",
        "AutoModel": "modeling_action_expert.ActionExpertModel",
    }

    action_config_dict["input_hidden_size"] = llm_config_dict["hidden_size"]
    action_config_dict["hidden_size"] = llm_config_dict["hidden_size"] // 2
    action_config_dict["intermediate_size"] = llm_config_dict["intermediate_size"] // 2
    action_config_dict["model_type"] = "internlm2_300m"
    action_config_dict["use_flash_attn"] = False

    # Action related config update
    action_config_dict["action_dim"] = space_args.action_dim
    action_config_dict["state_dim"] = space_args.state_dim
    action_config_dict["action_chunk_size"] = model_args.action_chunk_size
    action_config_dict["state_token_num"] = 3  # time + ctrl_freq + proprioception

    return ActionExpertConfig(**action_config_dict)


def build_noise_scheduler_config(model_args: BaseModelArguments) -> Dict:
    noise_scheduler_config = {
        "num_train_timesteps": model_args.num_train_timesteps,
        "num_inference_timesteps": model_args.num_inference_timesteps,
        "prediction_type": model_args.prediction_type,
        "clip_sample": model_args.clip_sample,
        "beta_schedule": model_args.beta_schedule,
    }
    return noise_scheduler_config


def get_config_args(cfg_path: str):
    file_path = Path(cfg_path)
    sys.path.insert(0, str(file_path.parent))
    cfg = importlib.import_module(file_path.stem)

    return (
        cfg,
        cfg.DatasetArguments(),
        cfg.GOModelArguments(),
        cfg.GOTrainingArguments(),
        cfg.SpaceArguments(),
    )


def build_go1_model(dataset_args, model_args, training_args, space_args):
    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f"Loading Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = model_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # Determin training backend data dtype and model weights dtype
    if training_args.bf16 is True:
        torch_dtype = torch.bfloat16
    elif training_args.fp16 is True:
        torch_dtype = torch.float16
    else:
        raise ValueError(f"InternVision only supports bfloat16/float16, but trian args no specified!")

    # Load model state dict from given model safetensor directory
    logger.info("Loading GO1Model...")
    config = GO1ModelConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    config.llm_config.attn_implementation = "flash_attention_2"  # for InternLM
    config.pad_token_id = tokenizer.pad_token_id
    config.template = model_args.conv_style
    config.select_layer = model_args.vision_select_layer
    config.dynamic_image_size = model_args.dynamic_image_size
    config.use_thumbnail = model_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.min_dynamic_patch = model_args.min_dynamic_patch
    config.max_dynamic_patch = model_args.max_dynamic_patch
    config.img_context_token_id = img_context_token_id

    # rewrite the config for GO1
    ae_config = build_ae_config(model_args, config, space_args)
    config.action_config = ae_config
    config.action_chunk_size = ae_config.action_chunk_size
    noise_scheduler_config = build_noise_scheduler_config(model_args)
    config.noise_scheduler_config = noise_scheduler_config
    config.norm = any(t["type"] == "Normalize" for t in dataset_args.transforms)

    # Add latent planner related config
    if model_args.latent_planning:
        assert config.latent_planner_config.state_token_num == 0
        config.latent_planner_config.action_dim = 1  # codebook size is 32, and we need to do cross-entropy loss
        config.latent_planning = model_args.latent_planning

    model = GO1Model.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        _fast_init=get_bool_env(name="DEBUG_MODE"),
        ignore_mismatched_sizes=True,
    )

    assert model.config.downsample_ratio == model_args.down_sample_ratio
    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"model_args.force_image_size: {model_args.force_image_size}")
    logger.info(f"model.config.vision_config.image_size: {model.config.vision_config.image_size}")
    if model.config.vision_config.image_size != model_args.force_image_size:
        logger.info(
            f"Resizing position embedding from "
            f"{model.config.vision_config.image_size} "
            f"to {model_args.force_image_size}..."
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=model_args.force_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = model_args.force_image_size
    model.config.force_image_size = model_args.force_image_size
    model.num_image_token = int((model_args.force_image_size // patch_size) ** 2 * (model_args.down_sample_ratio**2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = True
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    model.language_model._set_output_logits(model_args.output_logits)
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
        model.latent_planner._set_gradient_checkpointing()
        model.action_model._set_gradient_checkpointing()

    # Model freeze params operation
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.freeze_latent_planner:
        _freeze_params(model.latent_planner)

    return tokenizer, model


def main(
    cfg_path: str,
    cfg: object,
    dataset_args: BaseDatasetArguments,
    model_args: BaseModelArguments,
    training_args: TrainingArguments,
    space_args: BaseSpaceArguments,
):
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    init_dist(launcher="pytorch", backend="nccl")

    # Mkdir for training outputs
    os.makedirs(f"{training_args.output_dir}/log", exist_ok=True)
    distributed_state = PartialState()
    exp_info_init(training_args.output_dir, cfg_path, distributed_state)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer, model = build_go1_model(dataset_args, model_args, training_args, space_args)

    # Dataset initialization
    train_dataset = build_datasets(
        tokenizer,
        model.num_image_token,
        dataset_args=dataset_args,
        model_args=model_args,
        is_train=True,
        space_args=space_args,
        stats_save_path=training_args.output_dir,
    )
    logger.info(f"Train dataset {cfg} initialized!")

    # Set seed for torch dataloaders
    set_seed(training_args.seed)

    # Trianer initialization
    collator = functools.partial(concat_pad_data_collator_go1, pad_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        processing_class=tokenizer,
        data_collator=collator,
    )

    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params/1e6:.2f}M")
        logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")
        logger.info(f"Frozen parameters: {(total_params - trainable_params)/1e6:.2f}M")

        module_params = {}
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                module_params[name] = {"total": params, "trainable": trainable, "frozen": params - trainable}

        sorted_modules = sorted(module_params.items(), key=lambda x: x[1]["total"], reverse=True)
        for name, stats in sorted_modules:
            logger.info(f"{name}:")
            logger.info(
                f"  Total: {stats['total'] / 1e6:.2f}M  Trainable: {stats['trainable'] / 1e6:.2f}M  Frozen: {stats['frozen'] / 1e6:.2f}M"
            )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None and last_checkpoint is None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(train_dataset)
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", help="config path", required=True)
    args = parser.parse_args()

    cfg, dataset_args, model_args, training_args, space_args = get_config_args(args.cfg_path)

    main(
        cfg_path=args.cfg_path,
        cfg=cfg,
        dataset_args=dataset_args,
        model_args=model_args,
        training_args=training_args,
        space_args=space_args,
    )
