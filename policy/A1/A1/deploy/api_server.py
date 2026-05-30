#!/usr/bin/env python3
"""
VLA Inference API Server
Provides REST API endpoints for VLA model inference on port 6789
"""

import os
import sys
import json
import base64
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import threading
from dataclasses import replace

import torch
import numpy as np
from PIL import Image
from io import BytesIO
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from omegaconf import OmegaConf

# Add project root to Python path
# PROJECT_ROOT = "/home/xurongtao/guominghao/code/A1"
# sys.path.insert(0, PROJECT_ROOT)

# from launch_scripts.utils import VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS, select_checkpoint
from a1.config import ModelConfig
# from a1.model import Molmo
from a1.vla.affordvla import AffordVLA
from a1.util import resource_path
from a1.checkpoint import load_model_state
from a1.config import EvalConfig, FSDPConfig, FSDPWrapStrategy, FSDPPrecision, TrainConfig
from a1.torch_util import get_local_rank, seed_all
from a1.data.vla.utils import NormalizationType

from deploy.infer_vla import create_mock_input_data, run_inference

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vla_api_server")

# Global model instance
model_instance = None
model_config = None
norm_stats = None
normalization_type = None
use_proprio = True
use_wrist_image = True
sequence_length = 768
no_norm = False
device = "cuda" if torch.cuda.is_available() else "cpu"

# Enable memory efficiency optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# FastAPI app
app = FastAPI(
    title="VLA Inference API",
    description="REST API for Vision-Language-Action model inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class InferenceRequest(BaseModel):
    instruction: str
    images: Optional[List[str]] = None  # Base64 encoded images
    # action_head: str = "l1_regression"  # "l1_regression" or "diffusion"
    # use_proprio: bool = True
    # use_wrist_image: bool = True
    # seq_len: int = 768
    # normalization_type: str = "bounds"

    proprio_data: Optional[List[List[float]]] = None  # Custom proprioception data
    # previous_actions: Optional[List[List[float]]] = None  # Custom action history

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

class InferenceResponse(BaseModel):
    predicted_actions: List[List[float]]
    instruction: str
    action_head_type: str
    processing_time_ms: float
    model_info: Dict[str, Any]
    timestamp: str

def load_vla_model(checkpoint_path: str, fsdp: bool = False,seed=6198) -> tuple:
    """Load VLA model from checkpoint"""
    global model_instance, model_config, use_proprio, use_wrist_image, sequence_length
    
    log.info(f"Loading VLA model from checkpoint: {checkpoint_path}")

    cfg = EvalConfig(
        max_crops_override=None,
        # evaluations=[eval_config], 不要这个数据集加载的参数
        load_path=checkpoint_path,
        seed=seed,
        device_inf_eval_batch_size=4,
        pbar=True,
        console_log_interval=10,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
        ) if fsdp else None,
    )


    # Prepare environment
    # prepare_cli_environment()
    # add_cached_path_clients()

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")
    
    if not fsdp:
        # Load model configuration
        checkpoint_path_obj = Path(checkpoint_path)

        config_path = checkpoint_path_obj / "config.yaml"
        assert config_path.exists(), "config.yaml not found"
        config = TrainConfig.load(config_path, validate_paths=False)
        use_proprio = config.data.use_proprio
        use_wrist_image = config.data.use_wrist_image
        sequence_length = config.data.sequence_length

        model_cfg = config.model
        
        # 去掉原来路径pretrained_image_encoders和pretrained_llms之前的部分，改成环境变量
        
        model_cfg.vit_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_image_encoders",os.path.basename(model_cfg.vit_load_path))
        model_cfg.llm_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_llms",os.path.basename(model_cfg.llm_load_path))
        model_cfg.tokenizer.tokenizer_dir = os.environ.get("HF_HOME", "")
        
        # Create and load model
        model = AffordVLA(model_cfg)

        # print('config.use_lora', config.use_lora)
        # if config.use_lora:
        #     if config.lora_llm:
        #         custom_lora_config = {
        #             torch.nn.Linear: {
        #                 "weight": partial(LoRAParametrization.from_linear, rank=config.lora_rank),
        #             },
        #         }
                    
        #         add_lora(olmo_model.transformer,custom_lora_config)
        #         log.info("Added LoRA to model.transformer")

        # checkpoint_path = select_checkpoint(checkpoint_path)
        # state = load_model_state(checkpoint_path, device=device)
        model_state_dict_path = resource_path(checkpoint_path, "model.pt")
        model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=True)

        # if config.use_lora:
        #     if config.lora_llm:
        #         merge_lora(olmo_model.transformer) # merge olmo_model会怎么样

        
        model = model.to(device)
        model.eval()
    
    seed_all(seed)

    model_instance = model
    model_config = model_cfg
    
    log.info("VLA model loaded successfully")
    return model, model_cfg
    

def get_norm_stats_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        stats = json.load(f)
    return stats
    
def decode_base64_image(base64_str: str) -> torch.Tensor:
    """Decode base64 string to image tensor"""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        # Resize to expected dimensions
        # image = image.resize((336, 336))
        
        # Convert to tensor [C, H, W] in range [0, 1]
        image_array = np.array(image).astype(np.uint8)
        # image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        return image_array
        
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def prepare_inference_data(request: InferenceRequest) -> Dict[str, Any]:
    """Prepare input data from API request"""
    
    # Handle images
    if request.images:
        image_tensors = []
        for img_b64 in request.images:
            img_tensor = decode_base64_image(img_b64)
            image_tensors.append(img_tensor)
        # images_tensor = torch.stack(image_tensors).unsqueeze(0)  # Add batch dimension
        images_tensor = image_tensors
    else:
        # Use mock images if none provided
        print('Error: No images provided!!')
        mock_data = create_mock_input_data(
            seq_len=sequence_length,
            use_wrist_image=request.use_wrist_image,
            use_proprio=request.use_proprio
        )
        images_tensor = mock_data['images']
    
    # Handle proprioception data
    if request.proprio_data:
        proprio_tensor = torch.tensor(request.proprio_data, dtype=torch.float32)
    # elif request.use_proprio:
    #     # Generate mock proprioception
    #     print('Error: No proprioception data provided!!')
    #     batch_size = 1
    #     proprio_dim = 8  # 7 joint angles + 1 gripper state
    #     proprio_tensor = torch.randn(batch_size, request.seq_len, proprio_dim)
    else:
        proprio_tensor = None
    
    # Handle previous actions
    # if request.previous_actions:
        # prev_actions_tensor = torch.tensor(request.previous_actions, dtype=torch.float32)
    # else:
        # print('Error: No previous actions provided!!')
        # Generate mock previous actions
        # batch_size = 1
        # action_dim = 7  # 6-DOF pose + 1 gripper
        # prev_actions_tensor = torch.randn(batch_size, request.seq_len, action_dim)
        # prev_actions_tensor = None
    
    input_data = {
        'images': images_tensor,
        'instruction': request.instruction,
        'proprio': proprio_tensor,
        # 'previous_actions': prev_actions_tensor,
        # 'seq_len': request.seq_len
    }
    
    return input_data

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_instance is not None,
        device=device,
        timestamp=datetime.now().isoformat()
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "VLA Inference API Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "inference": "/inference",
            "model_info": "/model_info"
        },
        "docs": "/docs"
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "device": device,
        "config": {
            "action_head": getattr(model_config, 'action_head', 'unknown'),
            "use_proprio": getattr(model_config, 'use_proprio', False),
            "vision_backbone": getattr(model_config.vision_backbone, 'name', 'unknown'),
        }
    }

@app.post("/inference", response_model=InferenceResponse)
async def run_vla_inference(request: InferenceRequest):
    """Run VLA inference"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Prepare input data
        input_data = prepare_inference_data(request)
        

        # Run inference
        results = run_inference(model_instance, 
                    input_data,
                    sequence_length, 
                    norm_stats, 
                    normalization_type, 
                    use_proprio, 
                    use_wrist_image, 
                    no_norm=no_norm
        )
        print(f"***results['predicted_actions'].shape: {results['predicted_actions'].shape}")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Format response
        response = InferenceResponse(
            predicted_actions=results['predicted_actions'].squeeze().tolist(),
            instruction=results['instruction'],
            action_head_type=model_instance.config.action_head,
            processing_time_ms=processing_time,
            model_info={
                "device": device,
                "input_shapes": results['input_shape'],
                "model_config": {
                    "action_head": getattr(model_config, 'action_head', 'unknown'),
                    "use_proprio": getattr(model_config, 'use_proprio', False),
                }
            },
            timestamp=end_time.isoformat()
        )
        
        log.info(f"Inference completed in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        log.error(f"Inference failed: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/inference/mock")
async def run_mock_inference():
    """Run inference with mock data for testing"""
    try:
        start_time = datetime.now()
        
        # Create mock data
        input_data = create_mock_input_data(seq_len=sequence_length, use_wrist_image=True, use_proprio=True)
        
        # Mock inference results (since we might not have a real model)
        batch_size = input_data['images'].shape[0]
        predicted_actions = torch.randn(batch_size, 7)  # 6-DOF pose + 1 gripper
        
        results = {
            'predicted_actions': predicted_actions,
            'instruction': input_data['instruction'][0],
            'action_head_type': 'mock',
            'input_shape': {
                'images': list(input_data['images'].shape),
                'seq_len': sequence_length
            }
        }
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        response = InferenceResponse(
            predicted_actions=results['predicted_actions'].cpu().numpy().tolist(),
            instruction=results['instruction'],
            action_head_type=results['action_head_type'],
            processing_time_ms=processing_time,
            model_info={
                "device": device,
                "input_shapes": results['input_shape'],
                "model_config": {"action_head": "mock", "use_proprio": True}
            },
            timestamp=end_time.isoformat()
        )
        
        return response
        
    except Exception as e:
        log.error(f"Mock inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Mock inference failed: {str(e)}")

def main():
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="VLA Inference API Server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--normalization_type", default="bounds", help="Normalization type")
    parser.add_argument("--norm_stats_json_path", help="Path to norm stats json file")
    parser.add_argument("--no_norm", default=False, action="store_true", help="If normalize the input data")
    parser.add_argument("--fsdp", default=False, action="store_true", help="Enable FSDP")
    parser.add_argument("--seed", default=6198, type=int, help="Seed")

    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", default=6789, type=int, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", default=1, type=int, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Load model
    log.info("Starting VLA Inference API Server...")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Device: {device}")

    if not args.no_norm:
        global norm_stats
        norm_stats = get_norm_stats_from_json(args.norm_stats_json_path)
        log.info(f"Norm stats loaded from {args.norm_stats_json_path}")

        global normalization_type
        normalization_type = NormalizationType(args.normalization_type)
    
    global no_norm
    no_norm = args.no_norm
    # try:
    load_vla_model(args.checkpoint, args.fsdp, args.seed)
    log.info("Model loaded successfully")
    # except Exception as e:
    #     log.error(f"Failed to load model: {e}")
    #     log.info("Starting server without model (mock inference only)")
    
    # Start server
    log.info(f"Starting server on {args.host}:{args.port}")
    
    # 使用已创建的 app 实例启动，以确保模型保持在提供服务的同一进程中
    # 注意：当传入 app 实例时，uvicorn 不支持多进程 workers>1，且 reload 需要导入字符串。
    if args.reload:
        log.warning("--reload 仅在使用导入字符串时可用，已自动禁用 reload。")
    if args.workers != 1:
        log.warning("workers>1 需要使用导入字符串并在每个 worker 内自行加载模型，已强制为 1。")

    uvicorn.run(
        # "api_server:app",
        app,
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
        log_level="info"
        # reload=args.reload,
        # workers=args.workers,
    )

if __name__ == "__main__":
    main()