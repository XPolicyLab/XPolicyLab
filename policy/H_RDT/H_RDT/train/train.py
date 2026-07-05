#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import math
import os
import copy
from pathlib import Path
import sys
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.nn as nn
import safetensors.torch
from PIL import Image
from models.hrdt.pos_emb import get_multimodal_pos_embed
from collections import OrderedDict

from models.hrdt_runner import HRDTRunner
from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from datasets.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")


if is_wandb_available():
    import wandb

def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- hrdt
---
    """
    model_card = f"""
# H-RDT - {repo_id}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    accelerator.init_trackers(
        project_name="h-rdt",
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    args.weight_dtype = weight_dtype
    
    # Create vision encoder
    vision_encoder = DinoSigLIPViTBackbone(
        vision_backbone_id=args.pretrained_vision_encoder_name_or_path,
        image_resize_strategy="letterbox"
            if config["dataset"]["image_aspect_ratio"] == "pad"
            else "resize-naive",
        default_image_size=384
    )
    image_transform = vision_encoder.get_image_transform()

    # Create H-RDT model
    hrdt = HRDTRunner(
        state_dim=config["common"]["state_dim"],
        action_dim=config["common"]["action_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        act_pos_emb_config= [
            ('state', 1),
            ('action', config["common"]["action_chunk_size"]),
        ],
        img_pos_emb_config=[
            # No initial pos embed in the last grid size
            # since we've already done in ViT
            ("image", (config["common"]["img_history_size"], 
                config["common"]["num_cameras"], 
                -vision_encoder.num_patches)),  
        ],
        lang_pos_emb_config=[
            # No initial pos embed in the last grid size
            # since we've already done in ViT
            ("language", -config["dataset"]["tokenizer_max_length"],),
        ],
        max_img_len=config["common"]["img_history_size"] * config["common"]["num_cameras"] * vision_encoder.num_patches,
        max_lang_len=config["dataset"]["tokenizer_max_length"],
        training_mode=args.training_mode,
        mode=args.mode,
        pretrained_backbone_path=args.pretrained_backbone_path if hasattr(args, 'pretrained_backbone_path') else None,
        dtype=weight_dtype,
    )
    
    # Load from a pretrained checkpoint if provided (for pretrain mode)
    if (
        args.mode == 'pretrain' and
        args.pretrained_model_name_or_path is not None
        and not os.path.isfile(args.pretrained_model_name_or_path)
    ):
        logger.info("Constructing model from pretrained checkpoint in pretrain mode.")
        hrdt = HRDTRunner.from_pretrained(args.pretrained_model_name_or_path)
    elif args.mode == 'finetune' and args.pretrained_backbone_path is not None:
        logger.info(f"Model initialized in finetune mode with pretrained backbone from {args.pretrained_backbone_path}")
    else:
        logger.info("Constructing model from provided config.")

    # Move encoders to device and proper dtype
    vision_encoder.to(accelerator.device, dtype=weight_dtype)
    
    for param in vision_encoder.parameters():
        param.requires_grad = False

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(hrdt))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    # Enable gradient checkpointing if needed
    if args.gradient_checkpointing:
        hrdt.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for H-RDT model")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate if needed
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Create optimizer for trainable parameters
    params_to_optimize = hrdt.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Create dataset and dataloader
    train_dataset = VLAConsumerDataset(
        config=config,
        image_transform=image_transform,
        num_cameras=config["common"]["num_cameras"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        image_corrupt_severity=args.image_corrupt_severity if hasattr(args, 'image_corrupt_severity') else None,
        upsample_rate=args.upsample_rate,
        val=False,
        use_precomp_lang_embed=args.precomp_lang_embed,
        task_name=args.task_name,
        bench_name=args.bench_name,
    )
    
    val_dataset = VLAConsumerDataset(
        config=config,
        image_transform=image_transform,
        num_cameras=config["common"]["num_cameras"],
        dataset_type=args.dataset_type,
        image_aug=False,
        image_corrupt_severity=None,
        upsample_rate=args.upsample_rate if hasattr(args, 'upsample_rate') else None,
        val=True,
        use_precomp_lang_embed=args.precomp_lang_embed,
        task_name=args.task_name,
        bench_name=args.bench_name,
    )

    # Create data collator for batching
    data_collator = DataCollatorForVLAConsumerDataset(use_precomp_lang_embed=args.precomp_lang_embed)

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    # Set up learning rate scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`
    hrdt = hrdt.to(dtype=weight_dtype)
    hrdt, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        hrdt, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Recalculate number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = {
            key: value if isinstance(value, (int, float, str, bool, torch.Tensor)) else str(value)
            for key, value in vars(args).items()
        }
        accelerator.init_trackers("hrdt", config=tracker_config)

    if args.report_to == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=logging_dir)

    # Training loop setup
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            initial_global_step = int(path.split("-")[1])
            logger.info(f"Resuming training from global step {initial_global_step}")
    else:
        initial_global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        initial=initial_global_step,
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="Steps"
    )

    # Training loop
    global_step = initial_global_step
    for epoch in range(args.num_train_epochs):
        # Skip training if we've already reached max steps
        if global_step >= args.max_train_steps:
            break

        hrdt.train()
        
        for step, batch in enumerate(train_dataloader):
            # Skip training if we've already reached max steps
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(hrdt):      
                # Process image data
                if isinstance(batch["images"], dict):
                    # {"dino": (B, T, C, H, W), "dino": (B, T, C, H, W)}
                    images = {k: v.to(dtype=weight_dtype) for k, v in batch["images"].items()}
                else:
                    raise ValueError(f"Unsupported `batch[\"images\"]` type = {type(batch['images'])}")
                with torch.no_grad():
                    k = next(iter(images))
                    batch_size, _, C, H, W = images[k].shape
                    for k in images:
                        images[k] = images[k].view(-1, C, H, W)
                    image_features = vision_encoder(images).detach()
                    image_features = image_features.view((batch_size, -1, vision_encoder.embed_dim))

                # Process language data based on training mode
                lang_embeds = None
                lang_attn_mask = None
                if args.training_mode == "lang":
                    lang_embeds = batch["lang_embeds"].to(dtype=weight_dtype)
                    lang_attn_mask = batch["lang_attn_mask"].to(dtype=weight_dtype)
                
                print(batch["lang_embeds"].shape)

                # Compute loss
                loss_dict = hrdt.compute_loss(
                    state_tokens=batch["states"].to(dtype=weight_dtype),
                    action_gt=batch["actions"].to(dtype=weight_dtype),
                    image_tokens=image_features,
                    lang_tokens=lang_embeds,
                    lang_attn_mask=lang_attn_mask,
                )
                
                loss = loss_dict["loss"]
                
                # Backward pass and optimization
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(hrdt.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint periodically
                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                # Evaluate periodically
                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        hrdt=hrdt,
                        args=args,
                        config=config,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        dataset_id2name=val_dataset.get_dataset_id2name(),
                        dataloader=val_dataloader,
                        logger=logger,
                        vision_encoder=vision_encoder,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            # Log training metrics
            for k in loss_dict:
                loss_dict[k] = loss_dict[k].detach().item()
            logs = {**loss_dict, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(hrdt).save_pretrained(args.output_dir)

        logger.info(f"Saved Model to {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()
