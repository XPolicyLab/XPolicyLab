import os
import time
import atexit
import signal
import sys
import torch
import torch.distributed as dist
from torch import nn
import logging

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator

logger = logging.getLogger(__name__)


def register_graceful_exit(accelerator: Accelerator):
    """
    Register cleanup handlers to ensure accelerator.end_training() is called on exit.
    This flushes experiment trackers (wandb/swanlab) metadata.json.

    Handles: normal exit, Ctrl+C (SIGINT), SIGTERM (SLURM/K8s), uncaught exceptions.

    Note: When using torchrun, the elastic agent intercepts signals first and forwards
    them to workers. Trackers like SwanLab may finish before our atexit runs, so we
    wrap end_training() with exception handling to avoid errors on double-finish.
    """
    def _safe_end_training():
        try:
            accelerator.end_training()
        except Exception:
            pass  # Tracker already finished (e.g., SwanLab caught SIGINT first)

    atexit.register(_safe_end_training)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def get_mlp_task_id():
    """
    Get the MLP task ID like `t-20260202105509-djbbd`
    """
    task_name = os.uname().nodename.split('-worker')[0]
    return task_name


def get_distributed_topology() -> dict:
    """
    Get distributed training topology information.
    Compatible with single-GPU, single-node multi-GPU, and multi-node multi-GPU setups.

    Supports environment variables from:
    - MLP cluster (MLP_WORKER_NUM, MLP_WORKER_GPU)
    - SLURM (SLURM_NNODES)
    - torchrun (WORLD_SIZE, LOCAL_WORLD_SIZE)

    Returns:
        Dictionary containing:
        - num_nodes: Number of machines
        - gpus_per_node: GPUs per machine
        - world_size: Total number of processes
    """
    # Get world size from distributed context or env
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Get num_nodes: MLP_WORKER_NUM > SLURM_NNODES > NNODES > 1
    num_nodes = int(os.environ.get(
        "MLP_WORKER_NUM",
        os.environ.get("SLURM_NNODES", os.environ.get("NNODES", 1))
    ))

    # Get gpus_per_node: MLP_WORKER_GPU > LOCAL_WORLD_SIZE > world_size / num_nodes
    gpus_per_node_env = os.environ.get(
        "MLP_WORKER_GPU",
        os.environ.get("LOCAL_WORLD_SIZE")
    )
    if gpus_per_node_env is not None:
        gpus_per_node = int(gpus_per_node_env)
    else:
        gpus_per_node = world_size // num_nodes if num_nodes > 0 else world_size

    return {
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
    }

_GLOBAL_MONITOR = None
class GlobalMonitor:
    """
    Hack for inner model layers to pass some monitoring info to the trainer.
    """
    def __init__(self):
        self.log_dict = {}
    def reset(self):
        self.log_dict = {}
    
    def log(self, update_dict):
        self.log_dict.update(update_dict)
    
    def get_metrics(self):
        return self.log_dict

def set_global_monitor():
    global _GLOBAL_MONITOR
    _GLOBAL_MONITOR = GlobalMonitor()

def get_global_monitor() -> GlobalMonitor:
    return _GLOBAL_MONITOR

def _log_attention_stats(attn_weights: torch.Tensor, layer_idx: int, mixtures: nn.ModuleList, active_names: list):
    """
    Helper function to log attention statistics. 
    Keeps the main forward pass clean.
    """
    # 1. Check if this is the target layer (Total - 2)
    # Using the last active mixture to determine the architecture depth
    total_layers = len(mixtures[active_names[-1]].layers)
    if layer_idx != total_layers - 2:
        return
    # 2. Check if monitor is available
    monitor = get_global_monitor()
    if monitor is None:
        return

    # 3. Log the statistics
    # Use .detach() to ensure no gradient tracking for logging
    monitor.log({
        f"monitor/softmax_maxlogits_layer_{layer_idx}": attn_weights.max().detach(),
    })

def _log_hidden_states_stats(
    hidden_states_dict: dict[str, torch.FloatTensor], 
    layer_idx: int, 
    mixtures: nn.ModuleList
):
    """
    Private hook to log statistics (absmax, absmean, std) of hidden states.
    Triggered only at the penultimate layer (Total - 2).
    """
    active_names = list(hidden_states_dict.keys())
    if not active_names:
        return

    # 1. Condition check: only log for the penultimate layer
    total_layers = len(mixtures[active_names[-1]].layers)
    if layer_idx != total_layers - 2:
        return
    # 2. Monitor availability check
    monitor = get_global_monitor()
    if monitor is None:
        return

    # 3. Batch logging of stats for each active mixture
    stats_to_log = {}
    for name in active_names:
        # Move to CPU/Float and detach to avoid graph overhead
        x = hidden_states_dict[name].detach()
        stats_to_log.update({
            f"monitor/{name}/x_out_absmax_layer_{layer_idx}": x.abs().max(),
            f"monitor/{name}/x_out_absmean_layer_{layer_idx}": x.abs().mean(),
            f"monitor/{name}/x_out_std_layer_{layer_idx}": x.std(),
        })
    
    monitor.log(stats_to_log)

def init_experiment_tracker(cfg: DictConfig, accelerator: Accelerator, output_dir: Path):
    """
    Initialize experiment tracker (SwanLab or WandB) using Accelerator's unified API.
    
    Args:
        cfg: Hydra configuration
        accelerator: Accelerator instance
        output_dir: Output directory for logs
        
    Returns:
        tracker_type: Type of tracker initialized ('swanlab', 'wandb', or 'none')
    """
    tracker_type = cfg.logger.type.lower()
    
    if tracker_type == "none":
        logger.info("Logger disabled (type=none)")
        return tracker_type
    
    # Set project and experiment name from task if not specified
    task_name = cfg.logger.task if cfg.logger.task else cfg.hydra.runtime.choices.task
    project_name = cfg.logger.project if cfg.logger.project else task_name.split('/')[0]
    experiment_name = cfg.logger.experiment_name if cfg.logger.experiment_name else task_name.split('/')[-1]
    dir = cfg.logger.dir if cfg.logger.dir else str(output_dir / "swanlab") if tracker_type == "swanlab" else str(output_dir / "wandb")
    
    init_kwargs = {}
    
    if tracker_type == "swanlab":
        init_kwargs["swanlab"] = {
            "workspace": cfg.logger.workspace,
            "experiment_name": experiment_name,
            "logdir": dir,
            "mode": cfg.logger.mode,
        }
    elif tracker_type == "wandb":
        init_kwargs["wandb"] = {
            "name": experiment_name,
            "dir": dir,
            "mode": cfg.logger.mode,
        }
        # For wandb, workspace field is entity
        if cfg.logger.workspace:
            init_kwargs["wandb"]["entity"] = cfg.logger.workspace
    elif tracker_type is None:
        logger.info("Logger disabled (type=none)")
        return tracker_type
    else:
        raise ValueError(f"Unsupported logger type: {tracker_type}. Choose 'swanlab', 'wandb', or 'none'.")
    
    # Build config with distributed topology
    config = OmegaConf.to_container(cfg, resolve=True)
    topology = get_distributed_topology()

    # Calculate effective batch size
    batch_size = cfg.model.batch_size
    grad_accumulation_steps = cfg.model.grad_accumulation_steps
    effective_batch_size = batch_size * grad_accumulation_steps * topology["world_size"]
    task_id = get_mlp_task_id()

    # Add distributed topology to config
    config["distributed"] = {
        **topology,
        "effective_batch_size": effective_batch_size,
        "MLP_task_id": task_id,
    }

    accelerator.init_trackers(
        project_name=project_name,
        config=config,
        init_kwargs=init_kwargs,
    )
    logger.info(f"MLP task ID: {task_id}")
    logger.info(
        f"Initialized {tracker_type} tracker | "
        f"Topology: {topology['num_nodes']} nodes × {topology['gpus_per_node']} GPUs = {topology['world_size']} processes | "
        f"Effective batch size: {effective_batch_size}"
    )

    return tracker_type


class MFUTracker:
    """
    Model FLOPS Utilization (MFU) Tracker
    
    Tracks and calculates the hardware utilization during training by comparing
    actual FLOPS achieved vs theoretical peak FLOPS of the GPU.
    """
    
    def __init__(self, model,
        batch_size,
        device_id=0,
        update_interval=10,
        world_size=1,
        dtype=None,
    ):
        """
        Initialize MFU tracker.
        
        Args:
            model: The model to track
            batch_size: Effective batch size (batch_size * grad_accumulation * world_size)
            device_id: GPU device ID
            seq_length: Sequence length (optional, for sequence models)
            world_size: Number of GPUs (default: 1)
            dtype: Training dtype for MFU calculation (torch.float32, torch.bfloat16, torch.float16)
            
        Important Notes on dtype:
            - For pure FP32/BF16 training: Pass the model's dtype
            - For AMP training: Pass the autocast dtype (e.g., torch.bfloat16), NOT the weight dtype
              * In AMP, weights are stored in FP32, but compute uses lower precision
              * ~95%+ of FLOPs come from matmul/conv which use the autocast precision
              * Example: with torch.autocast("cuda", dtype=torch.bfloat16), pass torch.bfloat16
        """
        self.device_id = device_id
        self.batch_size = batch_size
        self.world_size = world_size
        
        # Auto-detect dtype from model if not specified
        if dtype is None:
            dtype = next(model.parameters()).dtype
            logger.warning(f"Auto-detected training dtype from model weights: {dtype}. "
                          f"For AMP training, please explicitly pass amp_dtype for accurate MFU calculation.")
        self.dtype = dtype
        
        # Get GPU peak FLOPS (single GPU) for the specified dtype
        # Note: For AMP training, this should be the autocast dtype (e.g., bf16), 
        # not the model weight dtype, since compute-intensive ops use the autocast dtype
        self.gpu_peak_flops = self._get_gpu_peak_flops(dtype)
        
        # Total peak FLOPS across all GPUs
        self.total_peak_flops = self.gpu_peak_flops * world_size
        
        # Estimate model FLOPS per step
        self.model_flops_per_step = self._estimate_model_flops(model)
        
        # Tracking variables
        self.start_time = time.time()
        self.start_step = 0
        self.update_interval = update_interval  # Update MFU metrics every N steps for recent performance
        
        # Detect if this is likely AMP training (weights FP32 but compute dtype is lower precision)
        weight_dtype = next(model.parameters()).dtype
        is_amp_training = (weight_dtype == torch.float32 and dtype in [torch.bfloat16, torch.float16])
        training_mode = f"AMP ({dtype})" if is_amp_training else f"{dtype}"
        
        logger.info(f"MFU Tracker initialized:")
        logger.info(f"  Training mode: {training_mode}")
        logger.info(f"  GPU Peak: {self.gpu_peak_flops/1e12:.1f} TFLOPS/GPU ({dtype})")
        logger.info(f"  Total Peak ({world_size} GPUs): {self.total_peak_flops/1e12:.1f} TFLOPS")
        logger.info(f"  Model FLOPs/step: {self.model_flops_per_step/1e12:.2f} TFLOPs")
    
    def _get_gpu_peak_flops(self, dtype):
        """
        Estimate peak FLOPS for the GPU based on dtype.
        
        Args:
            dtype: torch.float32, torch.bfloat16, or torch.float16
        """
        device_name = torch.cuda.get_device_name(self.device_id)
        device_capability = torch.cuda.get_device_capability(self.device_id)
        
        # Peak FLOPS estimates for common GPUs
        # Format: {GPU_name: {'bf16': TFLOPS, 'fp16': TFLOPS, 'fp32': TFLOPS}}
        gpu_peak_flops_db = {
            'H100': {'bf16': 1979e12, 'fp16': 1979e12, 'fp32': 67e12, 'tf32': 989e12},
            'H20': {'bf16': 148e12, 'fp16': None, 'fp32': 44e12, 'tf32': 74e12},
            'A100': {'bf16': 312e12, 'fp16': 624e12, 'fp32': 19.5e12, 'tf32': 312e12},
            'A800': {'bf16': 312e12, 'fp16': 624e12, 'fp32': 19.5e12, 'tf32': 312e12},
            '4090': {'bf16': 82.6e12, 'fp16': 82.6e12, 'fp32': 82.6e12}, # RTX 4090
        }
        
        # Determine precision type
        if dtype == torch.bfloat16:
            dtype_key = 'bf16'
        elif dtype == torch.float16:
            dtype_key = 'fp16'
        elif dtype == torch.float32:
            dtype_key = 'fp32'
        else:
            logger.warning(f"Unknown dtype {dtype}, defaulting to fp32")
            dtype_key = 'fp32'
        
        # Try to match GPU name
        for key, flops_dict in gpu_peak_flops_db.items():
            if key in device_name:
                peak_flops = flops_dict[dtype_key]
                logger.info(f"Detected GPU: {device_name}, dtype: {dtype}, "
                          f"peak FLOPS: {peak_flops/1e12:.1f} TFLOPS")
                return peak_flops
        
        # Default estimate based on compute capability
        if device_capability[0] >= 9:      # Hopper (H100)
            default_flops = {'bf16': 500e12, 'fp16': 500e12, 'fp32': 50e12}
        elif device_capability[0] >= 8:    # Ampere (A100, A30, RTX 30xx/40xx)
            default_flops = {'bf16': 150e12, 'fp16': 150e12, 'fp32': 20e12}
        elif device_capability[0] >= 7:    # Volta/Turing (V100, T4)
            default_flops = {'bf16': 50e12, 'fp16': 100e12, 'fp32': 15e12}
        else:
            default_flops = {'bf16': 25e12, 'fp16': 50e12, 'fp32': 10e12}
        
        peak_flops = default_flops[dtype_key]
        logger.warning(f"Unknown GPU: {device_name}, dtype: {dtype}, "
                      f"estimated peak FLOPS: {peak_flops/1e12:.1f} TFLOPS")
        return peak_flops
    
    def _estimate_model_flops(self, model):
        """
        Estimate FLOPs per training step (forward + backward).
        
        Formula: ~6 * params * batch_size * seq_length
            - 2x for forward pass (one matmul for input, one for weight)
            - 4x for backward pass (2x for gradient computation, 2x for weight gradients)
        
        For AMP training:
            - This formula counts total FLOPs regardless of precision
            - ~95-98% of FLOPs come from matmul/conv operations
            - These operations automatically use the autocast dtype
            - Small ops like layernorm stay in FP32, but contribute <5% of total FLOPs
            - Therefore, using autocast dtype for peak FLOPS is accurate
        """
        # vision ops
        num_vision_params = sum(p.numel() for p in model.model.vision_tower.parameters())
        num_proj_params = sum(p.numel() for p in model.model.multi_modal_projector.parameters())
        vision_seq_length = model.model.cfg.vision.num_image_tokens * model.model.num_input_images
        vision_num_hidden_layers = model.model.cfg.vision.num_hidden_layers
        vision_hidden_size = model.model.cfg.vision.hidden_size
        vision_attention_heads = model.model.cfg.vision.num_attention_heads
        vision_head_dim = vision_hidden_size // vision_attention_heads
        vision_attn_flops = 12 * vision_num_hidden_layers * vision_attention_heads * vision_head_dim * vision_seq_length
        vision_flops_per_token = 6 * (num_vision_params + num_proj_params) + vision_attn_flops 
        vision_flops_per_seq = vision_flops_per_token * vision_seq_length

        # vlm ops
        num_vlm_params = sum(p.numel() for p in model.model.joint_model.mixtures["vlm"].parameters())
        vlm_seq_length = model.model.cfg.max_image_text_tokens
        vlm_flops_per_seq = 6 * num_vlm_params * vlm_seq_length
        
        # proprio ops
        num_proprio_params = sum(p.numel() for p in model.model.joint_model.mixtures["proprio"].parameters())
        num_proprio_encoder_params = sum(p.numel() for p in model.model.proprio_encoder.parameters())
        proprio_seq_length = model.model.num_proprio_tokens
        proprio_flops_per_seq = 6 * (num_proprio_params + num_proprio_encoder_params) * proprio_seq_length
        
        # action ops
        num_action_params = sum(p.numel() for p in model.model.joint_model.mixtures["action"].parameters())
        num_action_encoder_params = sum(p.numel() for p in model.model.action_encoder.parameters())
        num_action_decoder_params = sum(p.numel() for p in model.model.action_decoder.parameters())
        action_seq_length = model.model.num_action_tokens
        action_flops_per_seq = 6 * (num_action_params + num_action_encoder_params + num_action_decoder_params) * action_seq_length
        
        # joint ops
        num_params = sum(p.numel() for p in model.parameters())
        total_seq_length = model.model.total_num_tokens
        num_hidden_layers = model.model.cfg.joint.num_hidden_layers
        num_attention_heads = model.model.cfg.joint.num_attention_heads
        head_dim = model.model.cfg.joint.head_dim
        joint_attn_flops_per_token = 12 * num_hidden_layers * num_attention_heads * head_dim * total_seq_length
        joint_attn_flops_per_seq = joint_attn_flops_per_token * total_seq_length

        flops_per_step = (vision_flops_per_seq + vlm_flops_per_seq + proprio_flops_per_seq + action_flops_per_seq + joint_attn_flops_per_seq) * self.batch_size
        
        logger.info(f"  Model parameters: {num_params/1e6:.2f}M")
        logger.info(f"  Estimated FLOPs/step: {flops_per_step/1e12:.2f} TFLOPs")
        return flops_per_step
    
    def reset(self, current_step):
        """Reset the timer for tracking recent performance."""
        self.start_time = time.time()
        self.start_step = current_step
    
    def compute_metrics(self, current_step):
        """
        Compute MFU and throughput metrics.
        
        Returns:
            dict: Dictionary containing mfu, samples_per_sec, steps_per_sec
        """
        elapsed_time = time.time() - self.start_time
        steps_completed = current_step - self.start_step
        
        if elapsed_time > 0 and steps_completed > 0:
            # Actual FLOPS = (FLOPs per step * steps) / time
            actual_flops = (self.model_flops_per_step * steps_completed) / elapsed_time
            # Use total_peak_flops since model_flops_per_step accounts for all GPUs
            mfu = actual_flops / self.total_peak_flops
            
            # Throughput metrics
            samples_per_sec = (self.batch_size * steps_completed) / elapsed_time
            steps_per_sec = steps_completed / elapsed_time
        else:
            mfu = 0.0
            samples_per_sec = 0.0
            steps_per_sec = 0.0
        
        # Reset timer periodically for recent performance tracking
        if steps_completed >= self.update_interval:
            self.reset(current_step)
        
        return {
            "performance/mfu": mfu,
            "performance/samples_per_sec": samples_per_sec,
            "performance/steps_per_sec": steps_per_sec,
        }
