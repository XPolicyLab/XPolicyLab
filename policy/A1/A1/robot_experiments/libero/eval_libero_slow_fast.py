import os
import json
import logging
from collections import deque
import random
import time

import draccus
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Optional, Union
from enum import Enum
import tqdm
from functools import partial

from packaging import version
import torch
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
import numpy as np

# from omegaconf import omegaconf, OmegaConf

from libero.libero import benchmark

import wandb

# from a1.model import Molmo
from a1.vla.affordvla import AffordVLA
from a1.checkpoint import load_model_state
# from launch_scripts.utils import DEBUG_MODEL, VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS

from a1.config import EvalConfig, TokenizerConfig, ModelConfig,TrainConfig
from a1.config import FSDPConfig, FSDPWrapStrategy, FSDPPrecision

from a1.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    peak_gpu_memory,
    seed_all, get_world_size,
)

from a1.util import (
    resource_path,
    # add_cached_path_clients,
    # clean_opt,
    # prepare_cli_environment, log_metrics_to_console,
)

from robot_experiments.utils import log_message,setup_logging#,initialize_and_load_model
from transformers import AutoModelForCausalLM
from transformers import SiglipVisionModel, SiglipTextModel, AutoTokenizer, AutoProcessor

from robot_experiments.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)

from robot_experiments.robot_utils import (
    DATE_TIME,
    # get_action,
    get_image_resize_size,
    # get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


from robot_experiments.vla_utils import resize_image_for_policy,get_vla_action,get_dit_action
from a1.vla.model_selector import ModelSelector
from a1.vla.dit_action import DiffusionTransformerAction
from a1.config import DiTActionConfig

from a1.vla.constants import NUM_ACTIONS_CHUNK, NormalizationType


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "a1"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/mnt/data3/zhangjian/a1/libero_4_Slow_qwen3-4b_dit-l_ft_ah_lora_r8_llm_Fast_dit-l_wrist_proprio_bs32/ckpt_step57500.pt"     # Pretrained checkpoint path (Slow)
    pretrained_checkpoint_dir: Union[str, Path] = os.path.dirname(pretrained_checkpoint)   # Pretrained checkpoint path 
    slow_pretrained_checkpoint: Union[str, Path] = None   # If set, overrides pretrained_checkpoint for slow model
    fast_pretrained_checkpoint: Union[str, Path] = None   # Fast model (DiT Action) checkpoint directory
    selector_checkpoint: Union[str, Path] = None          # Selector checkpoint file path (ckpt_step*.pt or latest.pt)
    fsdp: bool = False                               # Whether to use FSDP for model loading
    # llm: str = "qwen2_7b"                            # LLM model name (e.g., "qwen2_7b", "olmoe", "qwen2_72b")
    # vision_backbone: str = "openai"                  # Vision backbone name (e.g., "openai", "siglip", "dinov2_large_336")
    sequence_length: int = 768                       # Sequence length for the model

    use_l1_regression: bool = False                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = True                       # If True, uses continuous action head with diffusion modeling objective (DDIM)

    action_head_diffusion_inference_steps: int = 30
    llm_causal_attention: bool = False                # default: False, as openvla-oft's parallel decoding, If True, uses causal attention in the transformer model

    # Fast model backbones
    siglip_ckpt_path: Union[str, Path] = "/mnt/data/zhangjian/google/siglip-so400m-patch14-384"
    text_model: str = "Qwen3-0.6B"                   # ["SigLip","Qwen3-0.6B","Qwen3-1.7B","SmolLM3-3B","Qwen3-4B"]
    text_model_path: Union[str, Path] = "/mnt/data3/zhangjian/Qwen3/Qwen3-0.6B"

    # Fast DiT architecture (fallback if no config in checkpoint)
    dit_hidden_dim: int = 1024
    dit_depth: int = 24
    dit_num_heads: int = 16

    # Selector
    selector_max_history_len: int = 520

    # num_diffusion_steps: int = 10                    # (When `diffusion==True`) Number of diffusion steps for inference
    # use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_wrist_image: bool = False                     # Whether to use wrist image in input
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    normalization_type: str = NormalizationType.BOUNDS_Q99                # Action un-normalization key

    # load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    # load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 30 #50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    # local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    local_log_dir: str = os.path.join( pretrained_checkpoint , "eval_logs")        # Local directory for eval logs

    save_rollout_video: bool = True                  # Whether to save rollout videos
    save_rollout_video_path: str = pretrained_checkpoint  # Path to save rollout videos

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "demo0"                      # Name of WandB entity
    wandb_project: str = "a1-vla-eval"               # Name of WandB project

    seed: int = 6199                                 # Random Seed (for reproducibility)


def validate_config(cfg) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    # if "image_aug" in str(cfg.pretrained_checkpoint):
    #     assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    # assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    # selector ckpt (required - can be file or directory containing latest.pt)
    # assert cfg.selector_checkpoint is not None, "selector_checkpoint is required (file ckpt_step*.pt/latest.pt or directory containing it)"
    # if os.path.isdir(cfg.selector_checkpoint):
    #     candidate = os.path.join(cfg.selector_checkpoint, "latest.pt")
    #     assert os.path.exists(candidate), f"latest.pt not found in directory: {cfg.selector_checkpoint}"
    #     cfg.selector_checkpoint = candidate
    # else:
    #     assert os.path.exists(cfg.selector_checkpoint), f"Checkpoint file not found: {cfg.selector_checkpoint}"
    # default slow path: fallback to pretrained_checkpoint
    if cfg.slow_pretrained_checkpoint is None:
        cfg.slow_pretrained_checkpoint = cfg.pretrained_checkpoint
    # fast path is optional if we load from joint ckpt; if provided, must be valid directory or ckpt file
    if cfg.fast_pretrained_checkpoint is not None:
        if os.path.isdir(cfg.fast_pretrained_checkpoint):
            # ok: directory expected to contain model.pt or latest.pt
            assert os.path.exists(cfg.fast_pretrained_checkpoint), "fast_pretrained_checkpoint directory does not exist"
        else:
            # joint ckpt file
            assert os.path.exists(cfg.fast_pretrained_checkpoint), f"fast_pretrained_checkpoint file not found: {cfg.fast_pretrained_checkpoint}"
    
def initialize_and_load_model(generate_cfg, logger):
    cfg = EvalConfig(
        max_crops_override=None,
        # evaluations=[eval_config], 不要这个数据集加载的参数
        load_path=generate_cfg.pretrained_checkpoint,
        seed=generate_cfg.seed,
        device_inf_eval_batch_size=4,
        pbar=True,
        console_log_interval=10,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
        ) if generate_cfg.fsdp else None,
    )


    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device(f"cuda:{get_local_rank()}")

    if cfg.load_path == "debug":
        logging.warning("Loading debugging model")
        raise NotImplementedError("Debugging model loading not implemented yet")

    elif cfg.load_path.startswith("hf-"):
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.load_path[3:], trust_remote_code=True, torch_dtype='fp32', device_map='cpu')
        import pdb; pdb.set_trace()
    elif cfg.fsdp is None:
        # 这里从qwen2-7b加载模型参数
        model_cfg_path = resource_path(generate_cfg.pretrained_checkpoint_dir, "config.yaml")
        config = TrainConfig.load(model_cfg_path, validate_paths=False)
        # model_cfg = ModelConfig.load(model_cfg_path, key="model", validate_paths=False)
        model_cfg = config.model
        model_cfg.vit_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_image_encoders/vit-l-14-336.pt")
        model_cfg.llm_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_llms/qwen2-7b.pt")
        model_cfg.action_head_diffusion_inference_steps = generate_cfg.action_head_diffusion_inference_steps
        model_cfg.llm_causal_attention = generate_cfg.llm_causal_attention

        # model_cfg.use_proprio = generate_cfg.use_proprio ##

        olmo_model = AffordVLA(model_cfg)
        # olmo_model.reset_with_pretrained_weights(False) # 有影响吗?

        # print(f"Loading default llm and vit state dict done.")

        # olmo_model = AffordVLA.from_checkpoint(cfg.load_path, device=device)
        # model_cfg = olmo_model.config

        model_state_dict_path = resource_path(cfg.load_path, "model.pt")
        model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
        olmo_model.load_state_dict(model_state_dict, strict=True)
        print(f"Load model state dict done.")

        olmo_model.to(device)
        olmo_model.eval()
        
    else:
        logger.info("Building FSDP model...")
        model_cfg_path = resource_path(cfg.load_path, "config.yaml")
        model_cfg = ModelConfig.load(model_cfg_path, key="model", validate_paths=False)
        olmo_model = AffordVLA(model_cfg)

        # We always have only rank0 load the checkpoint, and then use `sync_module_states`
        # in FSDP to broadcast the weights to the other processes
        if get_global_rank() == 0:
            # is_unsharded = resource_path(cfg.load_path, "model.pt").is_file()
            is_unsharded = resource_path(cfg.load_path).is_file()
            if is_unsharded:
                logger.info("Loading state dict...")
                state_dict_path = resource_path(cfg.load_path)
                olmo_model.to_empty(device="cpu")
                state_dict = torch.load(state_dict_path, map_location="cpu")
                print(f"******** keys in state_dict: {list(state_dict.keys())}")
                print("*"*15)
                
                olmo_model.load_state_dict(state_dict, assign=True)
            else:
                olmo_model.to_empty(device="cpu")
                load_model_state(cfg.load_path, olmo_model)

        logger.info("Wrapping model with FDSP...")
        wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)
        hybrid_sharding_fsdp_kwargs = {}
        if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            raise NotImplementedError()
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise NotImplementedError()

        def dummy_init_fn(module: torch.nn.Module) -> None:
            # Prevent FSDP from re-initializing the parameters
            module.to_empty(device=get_default_device(), recurse=False)

        param_init_fn = dummy_init_fn
        olmo_model = FSDP(
            olmo_model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=MixedPrecision(
                param_dtype=cfg.autocast_precision,
                buffer_dtype=cfg.autocast_precision
            ),
            auto_wrap_policy=wrap_policy,
            use_orig_params=False,
            limit_all_gathers=True,
            device_id=get_local_rank(),
            sync_module_states=True,
            param_init_fn=param_init_fn,
            **hybrid_sharding_fsdp_kwargs,
        )
        olmo_model.eval()
        torch.cuda.empty_cache()  # For the 70B this can prevent OOMs by reduce memory fragmentation

    if cfg.max_crops_override:
        logging.info(f"Overriding max crops from {olmo_model.config.max_crops} to {cfg.max_crops_override}")
        olmo_model.config.max_crops = cfg.max_crops_override

    seed_all(cfg.seed)

    dtype = olmo_model.transformer.wte.embedding.dtype
    logger.info(f"Model weight dtype: {dtype}")
    logger.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    logger.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    logger.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")
    barrier()
    return olmo_model, device


def _resolve_ckpt_file(path_or_dir: str) -> Optional[str]:
    """Return a checkpoint file path from a file or a directory containing latest.pt/ckpt_step*.pt."""
    if path_or_dir is None:
        return None
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        latest = os.path.join(path_or_dir, "latest.pt")
        if os.path.exists(latest):
            return latest
        # fallback: pick max step ckpt
        try:
            cands = [p for p in os.listdir(path_or_dir) if p.startswith("ckpt_step") and p.endswith(".pt")]
            if cands:
                def _step(p):
                    import re
                    m = re.search(r"ckpt_step(\d+)\.pt$", p)
                    return int(m.group(1)) if m else -1
                cands.sort(key=_step)
                return os.path.join(path_or_dir, cands[-1])
        except Exception:
            pass
    return None


def initialize_and_load_fast_model(generate_cfg):
    """Load DiffusionTransformerAction fast model; supports separate model.pt dir or joint ckpt file."""
    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")

    # Attempt to resolve joint ckpt first from fast_pretrained_checkpoint; if not provided, use selector ckpt
    joint_ckpt = _resolve_ckpt_file(generate_cfg.fast_pretrained_checkpoint) or _resolve_ckpt_file(generate_cfg.pretrained_checkpoint)

    # Try load DiT config
    config = None
    # prefer fast_config.yaml in the directory of joint ckpt or provided dir
    search_dir = None
    if joint_ckpt is not None:
        search_dir = os.path.dirname(joint_ckpt)
    elif generate_cfg.fast_pretrained_checkpoint and os.path.isdir(generate_cfg.fast_pretrained_checkpoint):
        search_dir = generate_cfg.fast_pretrained_checkpoint
    if search_dir is not None:
        cfg_path_alt_yaml = os.path.join(search_dir, "fast_config.yaml")
        cfg_path_fast_yaml = os.path.join(search_dir, "config.yaml")
        try:
            if os.path.exists(cfg_path_alt_yaml):
                config = DiTActionConfig.load(cfg_path_alt_yaml, validate_paths=False)
            elif os.path.exists(cfg_path_fast_yaml):
                config = DiTActionConfig.load(cfg_path_fast_yaml, validate_paths=False)
        except Exception:
            config = None
    if config is None:
        config = DiTActionConfig()
        config.dit_hidden_dim = generate_cfg.dit_hidden_dim
        config.dit_depth = generate_cfg.dit_depth
        config.dit_num_heads = generate_cfg.dit_num_heads
        config.use_wrist_image = generate_cfg.use_wrist_image
        config.use_proprio = generate_cfg.use_proprio
        config.sequence_length = generate_cfg.sequence_length
        config.text_model = generate_cfg.text_model

    # Load text/vision backbones
    if generate_cfg.text_model == "SigLip":
        text_model = SiglipTextModel.from_pretrained(generate_cfg.siglip_ckpt_path)
    else:
        text_model = AutoModelForCausalLM.from_pretrained(generate_cfg.text_model_path, torch_dtype=torch.float32)
    vision_model = SiglipVisionModel.from_pretrained(generate_cfg.siglip_ckpt_path)
    text_model = text_model.to(device).eval()
    vision_model = vision_model.to(device).eval()

    # Complete config dims
    config.lang_cond_dim = text_model.config.hidden_size
    config.img_cond_dim = vision_model.config.hidden_size
    try:
        config.num_patches = (vision_model.config.image_size // vision_model.config.patch_size) ** 2
    except Exception:
        config.num_patches = (384 // 14) ** 2

    # Build model
    fast_model = DiffusionTransformerAction(config).to(device)

    # Load weights
    if joint_ckpt and os.path.isfile(joint_ckpt):
        ckpt = torch.load(joint_ckpt, map_location="cpu")
        state = ckpt.get("dit_action_model", {}) if isinstance(ckpt, dict) else {}
        fast_model.load_state_dict(state, strict=False)
    else:
        # separate model.pt style
        checkpoint_dir = generate_cfg.fast_pretrained_checkpoint
        assert checkpoint_dir and os.path.isdir(checkpoint_dir), f"Invalid fast_pretrained_checkpoint: {checkpoint_dir}"
        model_pt = os.path.join(checkpoint_dir, "model.pt")
        assert os.path.exists(model_pt), f"model.pt not found in {checkpoint_dir}"
        ckpt = torch.load(model_pt, map_location=device)
        state = ckpt.get("model_state_dict", None) if isinstance(ckpt, dict) else None
        if state is None:
            try:
                fast_model.load_state_dict(ckpt, strict=False)
            except Exception:
                raise RuntimeError("Failed to load fast model state dict; expecting key 'model_state_dict' or a plain state_dict")
        else:
            fast_model.load_state_dict(state, strict=False)

    fast_model.eval()
    return fast_model, text_model, vision_model, config, device


def _load_joint_ckpt(ckpt_path: str):
    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt, device


def initialize_and_load_slow_model_from_joint_ckpt(generate_cfg, logger):
    """Load AffordVLA slow model from slow_config.yaml and weights from joint ckpt."""
    ckpt, device = _load_joint_ckpt(generate_cfg.pretrained_checkpoint)
    save_dir = os.path.dirname(generate_cfg.pretrained_checkpoint)
    slow_cfg_path = os.path.join(save_dir, "slow_config.yaml")
    assert os.path.exists(slow_cfg_path), f"slow_config.yaml not found in {save_dir}"
    # model_cfg = ModelConfig.load(slow_cfg_path, key="model")
    model_cfg = ModelConfig.load(slow_cfg_path)
    model_cfg.action_head_diffusion_inference_steps = generate_cfg.action_head_diffusion_inference_steps
    model_cfg.llm_causal_attention = generate_cfg.llm_causal_attention
    olmo_model = AffordVLA(model_cfg)
    state = ckpt.get("affordvla_model", {})
    olmo_model.load_state_dict(state, strict=False)
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device(f"cuda:{get_local_rank()}")
    olmo_model = olmo_model.to(device)
    olmo_model.eval()
    seed_all(generate_cfg.seed)
    barrier()
    return olmo_model, device


def initialize_and_load_selector(generate_cfg, text_model, vision_model, device):
    """Load ModelSelector and its weights from selector checkpoint."""
    image_feature_dim = vision_model.config.hidden_size
    text_feature_dim = text_model.config.hidden_size
    selector = ModelSelector(
        image_feature_dim=image_feature_dim,
        text_feature_dim=text_feature_dim,
        history_embedding_dim=32,
        hidden_dim=128,
    ).to(device)

    sel_ckpt_path = generate_cfg.pretrained_checkpoint
    assert sel_ckpt_path is not None and os.path.exists(sel_ckpt_path), f"Invalid selector_checkpoint: {sel_ckpt_path}"
    sel_ckpt = torch.load(sel_ckpt_path, map_location="cpu")
    state = sel_ckpt.get("model_selector", None) if isinstance(sel_ckpt, dict) else None
    if state is None:
        # Allow loading a raw state_dict
        state = sel_ckpt
    try:
        selector.load_state_dict(state, strict=False)
    except Exception:
        raise RuntimeError("Failed to load selector state dict from checkpoint")
    selector.eval()
    return selector


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    
    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    # img_resized = img
    # wrist_img_resized = wrist_img

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "a1":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    slow_model,
    fast_model,
    selector,
    text_model,
    vision_model,
    device,
    resize_size,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    # Selector history for current episode: 0=none, 1=slow, 2=fast
    selector_history_codes = []

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    # try:
    while t < max_steps + cfg.num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, slow_model.config.vision_backbone.image_default_input_size)
        replay_images.append(img)

        # If action queue is empty, requery model
        if len(action_queue) == 0:
            # Use selector to choose between slow (AffordVLA) and fast (DiT)
            # Prepare features for selector
            with torch.no_grad():
                # Text features
                tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_path if cfg.text_model != "SigLip" else cfg.siglip_ckpt_path)
                text_inputs = tokenizer(task_description.lower(), return_tensors="pt", padding="max_length", max_length=cfg.sequence_length)
                input_ids = text_inputs["input_ids"].to(device)
                attn_mask = text_inputs.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                if cfg.text_model == "SigLip":
                    text_embeds = text_model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state.detach()
                else:
                    outputs = text_model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True, output_hidden_states=True)
                    text_embeds = outputs.hidden_states[-1].detach()

                # Image features
                processor = AutoProcessor.from_pretrained(cfg.siglip_ckpt_path)
                all_images = [observation["full_image"]]
                if cfg.num_images_in_input > 1:
                    all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])
                pil_images = []
                for im in all_images:
                    if isinstance(im, np.ndarray):
                        if im.dtype != np.uint8:
                            im = (im * 255).astype(np.uint8) if im.max() <= 1.0 else im.astype(np.uint8)
                        from PIL import Image as PILImage
                        pil_images.append(PILImage.fromarray(im))
                    else:
                        pil_images.append(im)
                img_inputs = processor(images=pil_images, return_tensors="pt")
                pixel_values = img_inputs["pixel_values"].to(device)
                image_embeds = vision_model(pixel_values=pixel_values).last_hidden_state.detach()
                # Aggregate across patches & cams to (B, F)
                bs_agg = input_ids.shape[0]
                image_embeds = image_embeds.contiguous().view(bs_agg, -1, image_embeds.shape[-1])

                # Build history tensors
                max_hist = cfg.selector_max_history_len
                hist_arr = np.zeros((bs_agg, max_hist), dtype=np.int64)
                L = min(len(selector_history_codes), max_hist)
                if L > 0:
                    hist_arr[0, :L] = np.array(selector_history_codes[-L:], dtype=np.int64)
                history = torch.from_numpy(hist_arr).to(device)
                history_lengths = torch.tensor([max(1, L)], device=device)

                selection_logits = selector(image_embeds.float(), text_embeds.float(), attn_mask, history.long(), history_lengths.long())
                selection_prob_fast = torch.sigmoid(selection_logits.view(-1))[0].item()

            # Choose branch
            choose_fast = selection_prob_fast >= 0.5
            if choose_fast:
                actions = get_dit_action(cfg, fast_model, text_model, vision_model, device, observation, task_description)
                selector_history_codes.append(2)
            else:
                actions = get_vla_action(cfg, slow_model, device, observation, task_description)
                selector_history_codes.append(1)
            action_queue.extend(actions)

        # Get action from queue
        action = action_queue.popleft()

        # Process action
        action = process_action(action, cfg.model_family)

        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file)

    return success, replay_images

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    slow_model,
    fast_model,
    selector,
    text_model,
    vision_model,
    device,
    num_tasks,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # record task start time
    task_start_time = time.time()

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask {task_id}: {task_description}", log_file)

        # record episode start time
        episode_start_time = time.time()

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}/{cfg.num_trials_per_task}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            slow_model,
            fast_model,
            selector,
            text_model,
            vision_model,
            device,
            resize_size,
            initial_state,
            log_file,
        )

        # compute episode duration time
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        if cfg.save_rollout_video:
            save_rollout_video(
                replay_images, total_episodes, success=success, task_description=task_description, save_path=cfg.save_rollout_video_path,log_file=log_file
            )

        # 计算当前总耗时和预测剩余时间
        current_total_time = time.time() - task_start_time
        avg_time_per_episode = current_total_time / task_episodes
        remaining_episodes = cfg.num_trials_per_task - task_episodes
        estimated_remaining_time = avg_time_per_episode * remaining_episodes
        estimated_total_time = current_total_time + estimated_remaining_time
        estimated_total_evaluation_time = avg_time_per_episode*(cfg.num_trials_per_task*num_tasks - total_episodes)

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}/{cfg.num_trials_per_task*num_tasks}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
        # log time statistics
        log_message("")
        log_message(f"Episode duration: {episode_duration:.2f}s", log_file)
        log_message(f"Current task time so far: {current_total_time:.2f}s ({current_total_time/60:.1f}min)", log_file)
        # log_message(f"Average time per episode: {avg_time_per_episode:.2f}s", log_file)
        log_message(f"Estimated remaining time for this task: {estimated_remaining_time:.2f}s ({estimated_remaining_time/60:.1f}min)", log_file)
        # print(f"Estimated total time for this task: {estimated_total_time:.2f}s ({estimated_total_time/60:.1f}min)", log_file)
        log_message(f"Estimated remaining time for evaulation: {estimated_total_evaluation_time:.2f}s ({estimated_total_evaluation_time/60:.1f}min)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # set config: prefer using the directory of selector ckpt for logs, if available
    ckpt_dir = os.path.dirname(cfg.pretrained_checkpoint) if os.path.isfile(cfg.pretrained_checkpoint) else cfg.pretrained_checkpoint
    cfg.local_log_dir = os.path.join(ckpt_dir, "eval_logs")
    cfg.save_rollout_video_path = ckpt_dir

    # Initialize models: slow (AffordVLA), fast (DiT Action), selector
    # Prefer loading slow from joint ckpt if slow_config.yaml exists next to selector ckpt
    save_dir = os.path.dirname(cfg.pretrained_checkpoint)
    slow_cfg_path = os.path.join(save_dir, "slow_config.yaml")
    if os.path.exists(slow_cfg_path):
        slow_model, device = initialize_and_load_slow_model_from_joint_ckpt(cfg, logger)
    else:
        cfg_for_slow = replace(cfg, pretrained_checkpoint=cfg.slow_pretrained_checkpoint)
        slow_model, device = initialize_and_load_model(cfg_for_slow, logger)
    fast_model, text_model, vision_model, fast_config, device_fast = initialize_and_load_fast_model(cfg)
    assert str(device_fast) == str(device), "Slow/Fast models must be on the same device"
    selector = initialize_and_load_selector(cfg, text_model, vision_model, device)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg) # 336

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # 从 num_tasks-1 到 0 倒序遍历
    # for task_id in tqdm.tqdm(range(num_tasks-1, -1, -1)):
    # 方式2：随机打乱顺序
    task_ids = list(range(num_tasks))
    random.shuffle(task_ids)
    for task_id in tqdm.tqdm(task_ids):
    # 正序
    # for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            slow_model,
            fast_model,
            selector,
            text_model,
            vision_model,
            device,
            num_tasks,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()