#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from typing import cast, Dict, Any
from PIL import Image
from dataclasses import replace

from omegaconf import omegaconf, OmegaConf

from launch_scripts.utils import VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS, select_checkpoint
from a1.torch_util import get_world_size, get_global_rank, get_local_rank
from a1.config import ModelConfig
from a1.model import Molmo
from a1.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment,
)
from a1.checkpoint import load_model_state

from a1.data import build_mm_preprocessor
from a1.data.collator import MMCollatorForAction
from a1.data.vla.utils import NormalizationType
from a1.data.vla.lerobot_datasets import LeRobotDataset

# Enable memory efficiency optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger("infer_vla")


def load_model(checkpoint_path: str, model_cfg: ModelConfig, device: str = "cuda") -> Molmo:
    """Load VLA model from checkpoint"""
    log.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Create model
    model = Molmo(model_cfg)
    
    # Load checkpoint
    checkpoint_path = select_checkpoint(checkpoint_path)
    
    # Try loading as a directory with model.pt file
    if Path(checkpoint_path).is_dir():
        model_file = Path(checkpoint_path) / "model.pt"
        if model_file.exists():
            log.info(f"Loading from model.pt file: {model_file}")
            checkpoint = torch.load(model_file, map_location=device)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        else:
            # Try FSDP format
            try:
                load_model_state(checkpoint_path, model)
            except Exception as e:
                log.error(f"Failed to load FSDP checkpoint: {e}")
                raise
    else:
        # Direct file
        log.info(f"Loading from checkpoint file: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    log.info("Model loaded successfully")
    return model

def get_lerobot_dataset_norm_stats(dataset_path):
    dataset_demo = LeRobotDataset(repo_id=os.path.basename(dataset_path),root=dataset_path, video_backend='pyav')
    stats = dataset_demo.meta.stats
    return stats


def create_mock_input_data(seq_len: int = 768, use_wrist_image: bool = True, use_proprio: bool = True) -> Dict[str, Any]:
    """Create mock input data for inference"""
    
    # Mock images (batch_size=1)
    batch_size = 1
    image_height, image_width = 336, 336
    
    # Create random RGB images
    images = []
    
    # Main camera image
    main_image = torch.randint(0, 256, (3, image_height, image_width), dtype=torch.uint8)
    images.append(main_image)
    
    # Wrist camera image (if enabled)
    if use_wrist_image:
        wrist_image = torch.randint(0, 256, (3, image_height, image_width), dtype=torch.uint8)
        images.append(wrist_image)
    
    # Stack images
    images_tensor = torch.stack(images).float() / 255.0  # Convert to [0,1] range
    images_tensor = images_tensor.unsqueeze(0)  # Add batch dimension
    
    # Mock text instruction
    instruction = "Pick up the red block and place it on the blue block"
    
    # Mock proprioceptive data (7-DOF arm + gripper)
    if use_proprio:
        proprio_dim = 8  # 7 joint angles + 1 gripper state
        proprio_data = torch.randn(batch_size, seq_len, proprio_dim)
    else:
        proprio_data = None
    
    # Mock previous actions (for action history)
    action_dim = 7  # 6-DOF pose + 1 gripper
    previous_actions = torch.randn(batch_size, seq_len, action_dim)
    
    mock_data = {
        'images': images_tensor,
        'instruction': [instruction],  # List for batch
        'proprio': proprio_data,
        'previous_actions': previous_actions,
        'seq_len': seq_len
    }
    
    return mock_data

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any], normalization_type: NormalizationType) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if normalization_type == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif normalization_type == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    elif normalization_type == NormalizationType.NORMAL:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["mean"], dtype=bool))
        mean = np.array(norm_stats["mean"])  # E[x]
        std = np.array(norm_stats["std"])    # sqrt(Var[x])
        normalized_proprio = np.where(mask, (proprio - mean) / (std + 1e-8), proprio)
        return normalized_proprio
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio

def _unnormalize_actions(normalized_actions, norm_stats, normalization_type):
    """Unnormalize actions using dataset statistics"""
    action_norm_stats = norm_stats['actions']

    if normalization_type == NormalizationType.BOUNDS:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions
    elif normalization_type == NormalizationType.BOUNDS_Q99:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions
    elif normalization_type == NormalizationType.NORMAL:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["mean"], dtype=bool))
        mean = np.array(action_norm_stats["mean"])  # E[x]
        std = np.array(action_norm_stats["std"])    # sqrt(Var[x])
        actions = np.where(mask, normalized_actions * (std + 1e-8) + mean, normalized_actions)
        return actions
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")


def run_inference(model: Molmo, 
                input_data: Dict[str, Any], 
                sequence_length: int,
                norm_stats: Dict[str, Any],
                normalization_type: NormalizationType,
                use_proprio: bool,
                use_wrist_image: bool,
                no_norm: bool = False) -> Dict[str, Any]:
    """Run inference on mock data"""

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    
    with torch.inference_mode():
        # Prepare inputs for the model
        images = input_data['images']  # b, n, c, h, w
        prompt = input_data['instruction']
        proprio = input_data['proprio']

        
        proprio = proprio.squeeze()
        if not no_norm:
            proprio_norm_stats = norm_stats["state"] ## 
            proprio = normalize_proprio(proprio, proprio_norm_stats, normalization_type)
        proprio = torch.tensor(proprio, dtype=torch.float32).to(device).unsqueeze(0)  # 添加batch维度
        proprio = proprio.unsqueeze(1)  # 添加时间步维度，变为 (batch_size, 1, proprio_dim)
        proprio = proprio.cpu().numpy()  # 转换为numpy数组
        

        # 使用与训练时相同的预处理器
        preprocessor = build_mm_preprocessor(
            model_config=model.config,
            # for_inference=True,  # 设置为推理模式
            shuffle_messages=False,
            # is_training=False,
            is_training=True,  # 这里设置为True是因为我们需要使用训练时的预处理方式
        )
        # 构建输入数据 - 模拟训练时的数据格式
        action_len = proprio.shape[-1]
        proprio = np.pad(proprio, ((0, 0), (0, 0), (0, model.config.fixed_action_dim - proprio.shape[-1])), mode='constant')
        dummy_action = np.zeros((model.config.num_actions_chunk, model.config.fixed_action_dim), dtype=np.float32)  # dummy action for inference
        action_pad_mask = np.zeros((model.config.num_actions_chunk, model.config.fixed_action_dim), dtype=bool)
        input_data_model = {
            # "image": np.array(primary_image),
            "question": prompt,
            "proprio": proprio,  
            "action": dummy_action,
            "action_pad_mask": action_pad_mask,
            "answer": "Action",  # 不起作用
            "style": "action",
            "metadata": {},
            
        }
    
        if use_wrist_image:
            # input_data["images"] = [image_primary, image_wrist]
            input_data_model["images"] = images
        else:
            input_data_model["image"] = images[0]

       # 通过预处理器处理
        processed_input = preprocessor(input_data_model)
        
        # 创建collator进行批处理
        collator = MMCollatorForAction(
                model_config=model.config,
                use_proprio=use_proprio,
                max_sequence_length=sequence_length,
                include_metadata=False,
            pad="to_max", max_crops=model.config.get_max_crops()
        )
        # 批处理数据
        batch_data = collator([processed_input])
        
        # 移动到设备
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device)

        # 准备模型输入
        model_inputs = {
            "input_ids": batch_data["input_ids"],
            "images": batch_data.get("images"),
            "image_masks": batch_data.get("image_masks"),
            "image_input_idx": batch_data.get("image_input_idx"),
            "attention_mask": batch_data.get("attention_mask"),
            "subsegment_ids": batch_data.get("subsegment_ids"),
            "position_ids": batch_data.get("position_ids"),
            
            "action_proprio":batch_data.get("proprio"),  
            "proprio_token_idx":  batch_data.get("proprio_token_idx"), 
            "output_hidden_states": False,
        }

        normalized_actions  = model.predict_actions(**model_inputs)
        
        normalized_actions = normalized_actions.to(torch.float32)  # 确保是float32格式
        normalized_actions = normalized_actions.cpu().numpy()
        normalized_actions = normalized_actions[..., :action_len]
        if not no_norm:
            actions = _unnormalize_actions(normalized_actions, norm_stats, normalization_type)
        else:
            actions = normalized_actions


    # Format output
    results = {
        'predicted_actions': actions,
        'instruction': prompt,
        # 'action_head_type': action_head_type,
        'input_shape': {
            'images': list(images[0].shape),
            'seq_len': sequence_length
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(prog="Inference for VLA model")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--llm", choices=["debug"] + list(LLMS.keys()), default="qwen2_7b")
    parser.add_argument("--vision_backbone", choices=list(VISION_BACKBONES.keys()), default="openai")
    parser.add_argument("--seq_len", default=768, type=int)
    parser.add_argument("--device_batch_size", default=1, type=int)
    
    # VLA specific arguments
    parser.add_argument("--action_head", default="l1_regression", type=str, 
                       choices=["l1_regression", "diffusion"])
    parser.add_argument("--use_proprio", default=True, type=bool)
    parser.add_argument("--use_wrist_image", default=True, type=bool)
    
    # Diffusion parameters (if using diffusion action head)
    parser.add_argument("--action_head_diffusion_inference_steps", default=30, type=int)
    parser.add_argument("--action_head_dit_depth", default=28, type=int)
    parser.add_argument("--action_head_dit_hidden_size", default=1152, type=int)
    parser.add_argument("--action_head_dit_num_heads", default=16, type=int)
    
    # Output options
    parser.add_argument("--save_results", action="store_true", help="Save inference results to file")
    parser.add_argument("--output_dir", default="./inference_results", help="Directory to save results")
    
    args, other_args = parser.parse_known_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")
    
    # Initialize distributed training if using multiple GPUs
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl")
            log.info("Process group initialized")
        except:
            log.info("Running in single GPU mode")
    
    prepare_cli_environment()
    add_cached_path_clients()
    
    # Load model configuration
    seq_len = args.seq_len
    debug = args.llm == "debug"
    
    if debug:
        from launch_scripts.utils import DEBUG_MODEL
        model_cfg = DEBUG_MODEL
        model_cfg.system_prompt_kind = 'demo_or_style'
    else:
        # Check if checkpoint has config
        checkpoint_path = Path(args.checkpoint)
        if (checkpoint_path / "model.yaml").exists():
            model_cfg = ModelConfig.load(checkpoint_path / "model.yaml")
        elif (checkpoint_path / "config.yaml").exists():
            model_cfg = ModelConfig.load(checkpoint_path / "config.yaml", key="model")
        else:
            # Use default configuration
            vit_layers = [-2, -9] if args.vision_backbone == "openai" else [-3, -9]
            model_cfg = replace(
                LLMS[args.llm],
                vision_backbone=VISION_BACKBONES[args.vision_backbone],
                llm_load_path=DEFAULT_LOAD_PATHS.get(args.llm, omegaconf.MISSING),
                vit_load_path=DEFAULT_LOAD_PATHS.get(args.vision_backbone, omegaconf.MISSING),
                crop_mode="overlap-and-resize-c2",
                system_prompt_kind='demo_or_style',
                residual_dropout=0.0,
                response_residual_dropout=0.1,
                max_crops=12,
                vit_layers=vit_layers,
                additional_vocab_size=128,
                action_head=args.action_head,
                num_diffusion_inference_steps=args.action_head_diffusion_inference_steps,
                use_proprio=args.use_proprio,
                action_head_dit_depth=args.action_head_dit_depth,
                action_head_dit_hidden_size=args.action_head_dit_hidden_size,
                action_head_dit_num_heads=args.action_head_dit_num_heads,
                action_use_left_eef=True,
            )
        
        # Update configuration with inference parameters
        model_cfg = replace(
            model_cfg,
            action_head=args.action_head,
            num_diffusion_inference_steps=args.action_head_diffusion_inference_steps,
            use_proprio=args.use_proprio,
            action_head_dit_depth=args.action_head_dit_depth,
            action_head_dit_hidden_size=args.action_head_dit_hidden_size,
            action_head_dit_num_heads=args.action_head_dit_num_heads,
        )
    
    # Override with command line arguments
    if other_args:
        conf = OmegaConf.create(model_cfg)
        overrides = [clean_opt(arg) for arg in other_args]
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
        model_cfg = cast(ModelConfig, OmegaConf.to_object(conf))
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, model_cfg, device)
    
    log.info(f"Model configuration:")
    log.info(f"  - Action head: {model_cfg.action_head}")
    log.info(f"  - Use proprioception: {model_cfg.use_proprio}")
    log.info(f"  - Vision backbone: {args.vision_backbone}")
    log.info(f"  - Sequence length: {seq_len}")
    
    # Create mock input data
    log.info("Creating mock input data...")
    input_data = create_mock_input_data(
        seq_len=seq_len,
        use_wrist_image=args.use_wrist_image,
        use_proprio=args.use_proprio
    )
    
    log.info(f"Input data shapes:")
    log.info(f"  - Images: {input_data['images'].shape}")
    log.info(f"  - Instruction: {input_data['instruction']}")
    if input_data['proprio'] is not None:
        log.info(f"  - Proprioception: {input_data['proprio'].shape}")
    # log.info(f"  - Previous actions: {input_data['previous_actions'].shape}")
    
    # Run inference
    log.info("Running inference...")
    results = run_inference(model, input_data, args.action_head)
    
    # Print results
    log.info("Inference completed!")
    log.info(f"Results:")
    log.info(f"  - Instruction: {results['instruction']}")
    log.info(f"  - Action head type: {results['action_head_type']}")
    log.info(f"  - Predicted actions shape: {results['predicted_actions'].shape}")
    log.info(f"  - Predicted actions: {results['predicted_actions'].cpu().numpy()}")
    
    # Save results if requested
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(args.output_dir, f"inference_results_{timestamp}.json")
        
        # Convert tensors to lists for JSON serialization
        results_json = {
            'instruction': results['instruction'],
            'action_head_type': results['action_head_type'],
            'predicted_actions': results['predicted_actions'].cpu().numpy().tolist(),
            'input_shape': results['input_shape'],
            'model_config': {
                'action_head': model_cfg.action_head,
                'use_proprio': model_cfg.use_proprio,
                'seq_len': seq_len,
            },
            'timestamp': timestamp
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        log.info(f"Results saved to: {output_file}")
    
    log.info("Inference completed successfully!")


if __name__ == "__main__":
    # main()
    dataset_path = "/vast/users/xiaodan/zhangjian/HuggingFace/dataset/FR3_A1/franka_data_v2.1_n"
    stats = get_lerobot_dataset_norm_stats(dataset_path)
    print(stats)