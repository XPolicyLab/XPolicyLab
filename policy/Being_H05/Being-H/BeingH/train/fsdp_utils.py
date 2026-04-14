# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by BeindBeyond Ltd. and/or its affiliates. on 2026-01-10.


import functools
import json
import os
from typing import Optional, Tuple, Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file
 
from BeingH.model.vit_model.internvit.modeling_intern_vit import (
    InternVisionEncoderLayer,
    InternVisionModel,
)
from BeingH.model.layers import (
    ProprioProjector,
    SinusoidalPositionalEncoding,
    FlowMatchingHead,
    MLPResNetBlock,
    MLPResNet,
    InternVLConnector,
    CategorySpecificLinear,
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    SimpleMLP,
    ActionEncoder,
)


# ==============================================================================
# FSDP Configuration
# ==============================================================================

class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
        activation_checkpointing=False
    ):
        """
        Args:
            sharding_strategy: FSDP sharding strategy (FULL_SHARD, HYBRID_SHARD, etc.)
            backward_prefetch: Backward prefetch strategy
            cpu_offload: Whether to offload parameters to CPU
            num_replicate: Number of replicas for HYBRID_SHARD
            num_shard: Number of shards for HYBRID_SHARD
            activation_checkpointing: Whether to use activation checkpointing
        """
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.activation_checkpointing = activation_checkpointing


# ==============================================================================
# FSDP Model Wrapper
# ==============================================================================

# Module classes to wrap with FSDP
FSDP_WRAPPER_CLASSES = {
    ProprioProjector,
    SinusoidalPositionalEncoding,
    FlowMatchingHead,
    MLPResNetBlock,
    MLPResNet,
    CategorySpecificLinear,
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    SimpleMLP,
    ActionEncoder,
    InternVLConnector,
    InternVisionEncoderLayer,
    InternVisionModel,
}

def fsdp_wrapper(model: torch.nn.Module, fsdp_config: FSDPConfig) -> FSDP:
    """
    Wrap a model with FSDP for distributed training.
    
    Args:
        model: Model to wrap
        fsdp_config: FSDP configuration
        
    Returns:
        FSDP-wrapped model
    """

    # Initialize device mesh for HYBRID_SHARD strategy
    device_mesh = None
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )

    return FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=FSDP_WRAPPER_CLASSES,
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


def grad_checkpoint_check_fn(module):
    """
    Determine if a module should use gradient checkpointing.
    
    Gradient checkpointing is applied to compute-intensive layer-level modules
    rather than entire Transformer blocks to balance memory and compute.
    
    Args:
        module: Module to check
        
    Returns:
        True if module should use gradient checkpointing
    """
    checkpoint_modules = (
        InternVisionEncoderLayer,
        InternVLConnector,
        ProprioProjector,
        CategorySpecificLinear,
        CategorySpecificMLP,
        MultiEmbodimentActionEncoder
    )
    return isinstance(module, checkpoint_modules)


# ==============================================================================
# FSDP Checkpoint Management
# ==============================================================================

class FSDPCheckpoint:
    """Utilities for saving and loading FSDP checkpoints."""

    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir: str,
        train_steps: int,
        model: FSDP,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        fsdp_config: FSDPConfig,
        logger: Any,
        dataset_name: str = None,
    ):
        """
        Save FSDP checkpoint.

        Args:
            ckpt_dir: Checkpoint directory
            train_steps: Current training step
            model: FSDP-wrapped model
            tokenizer: Tokenizer to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            fsdp_config: FSDP configuration
            logger: Logger instance
            dataset_name: Dataset name for copying metadata file to checkpoint subdirectory
        """
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        
        # Save model (rank 0 only for full state dict)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                # Save config
                config_dict = model.module.config.to_dict()
                with open(os.path.join(save_path, "config.json"), 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                # Save model weights     
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        # Save optimizer (sharded)
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            # if fsdp_config.sharding_strategy == "FULL_SHARD":
            if fsdp_config.sharding_strategy in ["FULL_SHARD", "SHARD_GRAD_OP"]:
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            if optimizer is not None:
                optimizer_path = os.path.join(
                    save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
                )

                # Only save optimizer for relevant ranks in HYBRID_SHARD
                should_save = (
                    fsdp_config.sharding_strategy in ["FULL_SHARD", "SHARD_GRAD_OP"] or
                    (fsdp_config.sharding_strategy == "HYBRID_SHARD" and 
                     dist.get_rank() < fsdp_config.num_shard)
                )

                if should_save:
                    torch.save(optimizer.state_dict(), optimizer_path)

        # Save scheduler (rank 0 only)
        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        # Copy metadata file to checkpoint subdirectory (rank 0 only)
        # This makes each checkpoint self-contained for easier distribution
        if dist.get_rank() == 0 and dataset_name is not None:
            exp_cfg_dir = os.path.join(ckpt_dir, "experiment_cfg")
            metadata_filename = f"{dataset_name}_metadata.json"
            src_metadata_path = os.path.join(exp_cfg_dir, metadata_filename)
            dst_metadata_path = os.path.join(save_path, metadata_filename)

            if os.path.exists(src_metadata_path):
                import shutil
                shutil.copy2(src_metadata_path, dst_metadata_path)
                logger.info(f"Copied metadata to checkpoint: {dst_metadata_path}")
            else:
                logger.warning(f"Metadata file not found: {src_metadata_path}")

        dist.barrier()

    @staticmethod
    def try_load_ckpt(
        resume_from: Optional[str],
        logger: Any,
        model: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Load model checkpoint if available.
        
        Args:
            resume_from: Checkpoint path to resume from
            logger: Logger instance
            model: Model to load weights into
            
        Returns:
            Model with loaded weights
        """
        if resume_from is None or not os.path.exists(resume_from):
            logger.info("Training from scratch")
            return model
        
        logger.info(f"Loading checkpoint from {resume_from}.")

        # Load checkpoint
        model_path = os.path.join(resume_from, f"model.safetensors")
        model_state_dict = load_file(model_path, device="cpu")

        # Remap keys for compatibility
        logger.info("Remapping checkpoint keys for compatibility...")
        model_state_dict, remapped = FSDPCheckpoint._remap_checkpoint_keys(
            model_state_dict, model, logger
        )

        # Handle size mismatches
        logger.info("Checking for size mismatches between checkpoint and model...")
        model_state_dict, skipped, resized = FSDPCheckpoint._handle_size_mismatch(
            model_state_dict, model, logger
        )

        # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
        # which makes it easier to adapt to different resolutions.
        # model_state_dict.pop('latent_pos_embed.pos_embed')
        # model_state_dict.pop('vit_pos_embed.pos_embed')
        msg = model.load_state_dict(model_state_dict, strict=False)
        logger.info(msg)

        del model_state_dict
        return model

    @staticmethod
    def _remap_checkpoint_keys(model_state_dict, model, logger):
        """
        Remap checkpoint keys from legacy naming to current naming.

        Key mappings:
        - state_encoder.* -> proprio_encoder_robot.* (same SimpleMLP architecture)

        Note: action_context_encoder has no equivalent in current model (will be ignored)
        Note: proprio_encoder (99-dim input) has no equivalent in legacy model (random init)
        """
        current_state_dict = model.state_dict()
        remapped_keys = []

        # Define key mappings: old_prefix -> new_prefix
        key_mappings = {
            'state_encoder.': 'proprio_encoder_robot.',
        }

        keys_to_process = list(model_state_dict.keys())

        for old_key in keys_to_process:
            for old_prefix, new_prefix in key_mappings.items():
                if old_key.startswith(old_prefix):
                    new_key = old_key.replace(old_prefix, new_prefix, 1)

                    # Only remap if the new key exists in the model and shapes match
                    if new_key in current_state_dict:
                        old_tensor = model_state_dict[old_key]
                        new_tensor = current_state_dict[new_key]

                        if old_tensor.shape == new_tensor.shape:
                            # Remap the key
                            model_state_dict[new_key] = model_state_dict.pop(old_key)
                            remapped_keys.append((old_key, new_key))
                            logger.info(f"  Remapped '{old_key}' -> '{new_key}'")
                        else:
                            logger.warning(f"  Cannot remap '{old_key}' -> '{new_key}': "
                                         f"shape mismatch {list(old_tensor.shape)} vs {list(new_tensor.shape)}")
                    break  # Only try one mapping per key

        if remapped_keys:
            logger.info(f"  Total remapped keys: {len(remapped_keys)}")

        return model_state_dict, remapped_keys
    
    @staticmethod
    def _handle_size_mismatch(model_state_dict, model, logger):
        """
        Handle size mismatches between checkpoint and current model.

        This function handles two types of mismatches:
        1. Token embeddings (embed_tokens, lm_head): Copy compatible vocab, random init new tokens
        2. Position embeddings: Skip and let the model's resize_pos_embeddings handle it later

        Returns: Modified state_dict with size-compatible tensors
        """
        current_state_dict = model.state_dict()
        skipped_keys = []
        resized_keys = []

        keys_to_process = list(model_state_dict.keys())

        for key in keys_to_process:
            if key not in current_state_dict:
                continue

            ckpt_tensor = model_state_dict[key]
            model_tensor = current_state_dict[key]

            if ckpt_tensor.shape == model_tensor.shape:
                continue

            # Position embeddings: skip and let resize happen later
            if 'position_embedding' in key or 'pos_embed' in key:
                logger.info(f"  Skipping position embedding '{key}': "
                           f"ckpt={list(ckpt_tensor.shape)} vs model={list(model_tensor.shape)}")
                model_state_dict.pop(key)
                skipped_keys.append(key)
                continue

            # Token embeddings (vocab size mismatch): copy what we can
            if 'embed_tokens' in key or 'lm_head' in key:
                ckpt_vocab, hidden = ckpt_tensor.shape
                model_vocab, _ = model_tensor.shape

                if hidden == model_tensor.shape[1]:  # Same hidden dim
                    # Copy min(ckpt_vocab, model_vocab) rows
                    min_vocab = min(ckpt_vocab, model_vocab)
                    new_tensor = model_tensor.clone()  # Start with model's init
                    new_tensor[:min_vocab] = ckpt_tensor[:min_vocab]
                    model_state_dict[key] = new_tensor

                    if model_vocab > ckpt_vocab:
                        logger.info(f"  Resized token embedding '{key}': "
                                   f"copied {min_vocab}/{model_vocab} tokens, "
                                   f"random init for {model_vocab - ckpt_vocab} new tokens")
                    else:
                        logger.info(f"  Truncated token embedding '{key}': "
                                   f"copied {min_vocab} tokens (ckpt had {ckpt_vocab})")
                    resized_keys.append(key)
                    continue

            # Other size mismatches: skip with warning
            logger.warning(f"  Size mismatch for '{key}': "
                          f"ckpt={list(ckpt_tensor.shape)} vs model={list(model_tensor.shape)}, skipping")
            model_state_dict.pop(key)
            skipped_keys.append(key)

        if skipped_keys:
            logger.info(f"  Total skipped due to size mismatch: {len(skipped_keys)} keys")
        if resized_keys:
            logger.info(f"  Total resized for compatibility: {len(resized_keys)} keys")

        return model_state_dict, skipped_keys, resized_keys

    @staticmethod
    def try_load_train_state(
        resume_from: Optional[str],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        fsdp_config: FSDPConfig
    ) -> Tuple[torch.optim.Optimizer, Any, int]:
        """
        Load optimizer and scheduler state if resuming training.
        
        Args:
            resume_from: Checkpoint path to resume from
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            fsdp_config: FSDP configuration
            
        Returns:
            Tuple of (optimizer, scheduler, train_steps)
        """
        if resume_from is None or not os.path.exists(resume_from):
            return optimizer, scheduler, 0
            
        if fsdp_config.sharding_strategy in ["FULL_SHARD", "SHARD_GRAD_OP"]:
            shard_index = dist.get_rank()
            total_shards = dist.get_world_size()
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            shard_index = dist.get_rank() % fsdp_config.num_shard
            total_shards = fsdp_config.num_shard
        else:
            # Fallback for SHARD_GRAD_OP, FULL_SHARD, NO_SHARD, etc.
            shard_index = dist.get_rank()
            total_shards = dist.get_world_size()

        # Load optimizer state
        optimizer_path = os.path.join(
            resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
        )
        optimizer_state_dict = torch.load(optimizer_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict

        # Load scheduler state
        scheduler_path = os.path.join(resume_from, "scheduler.pt")
        scheduler_state_dict = torch.load(scheduler_path, weights_only=True, map_location="cpu")
        scheduler.load_state_dict(scheduler_state_dict)
        del scheduler_state_dict

        # Get training step
        train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1    

        return optimizer, scheduler, train_steps


