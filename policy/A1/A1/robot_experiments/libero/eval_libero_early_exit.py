import os
import gc
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
from a1.vla.affordvla_early_exit import AffordVLAEarlyExit
from a1.vla.value_net import ExitController, ActionValueNet
from a1.checkpoint import load_model_state
# from launch_scripts.utils import DEBUG_MODEL, VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS

from a1.config import EvalConfig, TokenizerConfig, ModelConfig,TrainConfig,DataConfig
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

from transformers import AutoModelForCausalLM

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


from robot_experiments.libero.exit_vla_utils import resize_image_for_policy,get_vla_action

from a1.vla.constants import NormalizationType

from a1.data import build_rlds_train_dataloader


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
    # pretrained_checkpoint: Union[str, Path] = "/mnt/data3/zhangjian/hf_cache/hub/models--JianZhangAI--libero_trained_model_exit_flow/snapshots/30849fd1603aa61f49ceb4fb8ebe346e044e1cac/libero-4_extra_10_task_8_MolmoE-7B-10131629-5000_openai_seq368_flow_matching-qwen2_early_exit_two_images_proprio-8_ft_ah_fullyft_llm_bs176/step16500-unsharded"     # Pretrained checkpoint path
    # pretrained_checkpoint: Union[str, Path] = "/mnt/data3/zhangjian/hf_cache/hub/models--JianZhangAI--trained_model_early_exit/snapshots/250746ac29c8850549a95f9565d7004f0c19a2af/libero_4_molmo-7b-09242207_clip_l1_regression_early_exit_wrist_proprio_ft_ah_fullyft_llm_bs224/step41500-unsharded"
    pretrained_checkpoint: Union[str, Path] = None
    fsdp: bool = False                               # Whether to use FSDP for model loading
    # llm: str = "qwen2_7b"                            # LLM model name (e.g., "qwen2_7b", "olmoe", "qwen2_72b")
    # vision_backbone: str = "openai"                  # Vision backbone name (e.g., "openai", "siglip", "dinov2_large_336")
    # sequence_length: int = 768                       # Sequence length for the model
    sequence_length: int = 1024                       # Sequence length for the model

    action_head_diffusion_inference_steps: int = 20
    action_head_flow_matching_inference_steps: int = 30

    # llm_causal_attention: bool = False                # default: False, as openvla-oft's parallel decoding, If True, uses causal attention in the transformer model

    # num_diffusion_steps: int = 10                    # (When `diffusion==True`) Number of diffusion steps for inference
    # use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_wrist_image: bool = True                     # Whether to use wrist image in input
    use_proprio: bool = True                         # Whether to include proprio state in input
    # libero_proprio_dim: int = 8

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    normalization_type: NormalizationType = NormalizationType.BOUNDS_Q99               # Action normalization type

    # load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    # load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite LIBERO_10, LIBERO_SPATIAL,LIBERO_OBJECT,LIBERO_GOAL
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 30 #50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    # local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    local_log_dir: str = None        # Local directory for eval logs

    save_rollout_video: bool = True                  # Whether to save rollout videos
    save_rollout_video_path: str = None  # Path to save rollout videos

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "demo0"                      # Name of WandB entity
    wandb_project: str = "a1-vla-eval"               # Name of WandB project

    seed: int = 6198                                 # Random Seed (for reproducibility)

    exit_interval: int = 2                          # The interval to exit the model
    threshold_type: str = "cosine"                    # The type of threshold to use for exiting the model
    steps_per_stage: int = 1                          
    exit_dist: str = "exp"                            # The distribution to use for exiting the model
    max_layer: int = -1                               # The maximum layer to exit the model

    # for threshold calibration 
    rlds_data_root_dir: str = "data/libero_rlds"
    device_batch_size: int = 20               # this param is used for get dataloader for threshold calibration
    calib_max_batches: int = None 
    thresholds: float = None                         # directly set thresholds API (for bayesian optimization)
    exit_ratio: float = 1.0                          # the ratio of exit
    # 配合exp
    # 首次评测：1.0 作为基线。
    # 想更省算力：0.6–0.9（如 0.8）。
    # 想更稳性能：1.1–1.4（如 1.2）。
    # 配合 gauss：是“中心层”的位置（越靠近中心配额越大）。
    # 配合 gamma：形状参数，控制偏度。
    load_threshold: bool = True                     # load cached value distribution
    llm_name: str = "qwen2_7b"                       # the name of the LLM

    amp: bool = True                        # Whether to use AMP
    # 如果设置了，模型直接从该层退出
    exit_layer_id: int = None


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    # if "image_aug" in str(cfg.pretrained_checkpoint):
    #     assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    # assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    # 根据最终的 pretrained_checkpoint 动态补全依赖配置（若用户未显式指定）
    if not cfg.local_log_dir or str(cfg.local_log_dir).strip() == "":
        cfg.local_log_dir = os.path.join(cfg.pretrained_checkpoint, "eval_logs")
    if not cfg.save_rollout_video_path or str(cfg.save_rollout_video_path).strip() == "":
        cfg.save_rollout_video_path = cfg.pretrained_checkpoint

def setup_logging(cfg: GenerateConfig, action_head: str):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    if action_head == "flow_matching":
        if cfg.exit_layer_id is not None:
            run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-fm_steps{cfg.action_head_flow_matching_inference_steps}-exit_layer_id-{cfg.exit_layer_id}-{DATE_TIME}"
        else:
            run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-fm_steps{cfg.action_head_flow_matching_inference_steps}-exit_dist-{cfg.exit_dist}-exit_ratio-{cfg.exit_ratio}-{DATE_TIME}"
    else:
        run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-exit_dist-{cfg.exit_dist}-exit_ratio-{cfg.exit_ratio}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id

def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

def get_dataloader(generate_cfg,train_config):
    # mapping task_suite_name to rlds_dataset_name
    rlds_dataset_name_mapping = {
        TaskSuite.LIBERO_SPATIAL: "libero_spatial_no_noops",
        TaskSuite.LIBERO_OBJECT: "libero_object_no_noops",
        TaskSuite.LIBERO_GOAL: "libero_goal_no_noops",
        TaskSuite.LIBERO_10: "libero_10_no_noops",
        TaskSuite.LIBERO_90: "libero_90_no_noops",
    }
    train_config.data.rlds_dataset_name = rlds_dataset_name_mapping[generate_cfg.task_suite_name]
    train_config.data.rlds_data_root_dir = generate_cfg.rlds_data_root_dir
    train_config.data.seed = generate_cfg.seed
    train_config.device_train_batch_size = generate_cfg.device_batch_size

    return build_rlds_train_dataloader(train_config)

def initialize_and_load_model(generate_cfg):
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
    device = torch.device("cuda")

    if cfg.load_path == "debug":
        logging.warning("Loading debugging model")
        raise NotImplementedError("Debugging model loading not implemented yet")

    elif cfg.load_path.startswith("hf-"):
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.load_path[3:], trust_remote_code=True, torch_dtype='fp32', device_map='cpu')
        import pdb; pdb.set_trace()
    elif cfg.fsdp is None:
        logger.info("Loading model without FSDP...")
        # 这里从qwen2-7b加载模型参数
        model_cfg_path = resource_path(cfg.load_path, "config.yaml")
        config = TrainConfig.load(model_cfg_path, validate_paths=False)

        generate_cfg.sequence_length = config.data.sequence_length
        generate_cfg.use_proprio = config.data.use_proprio
        generate_cfg.use_wrist_image = config.data.use_wrist_image

        # model_cfg = ModelConfig.load(model_cfg_path, key="model", validate_paths=False)
        # model_cfg = config.model
        config.model.vit_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_image_encoders/vit-l-14-336.pt")
        config.model.llm_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_llms/qwen2-7b.pt")
        config.model.tokenizer.tokenizer_dir = os.environ.get("HF_HOME", "")
        config.model.num_diffusion_inference_steps = generate_cfg.action_head_flow_matching_inference_steps if config.model.action_head == "flow_matching" else generate_cfg.action_head_diffusion_inference_steps
        # config.model.llm_causal_attention = generate_cfg.llm_causal_attention
        # 显式设置模型初始化设备，保证嵌入权重等参数在 CUDA 上初始化
        config.model.init_device = f"cuda:{get_local_rank()}"

        # model_cfg.use_proprio = generate_cfg.use_proprio ##

        # olmo_model = AffordVLA(model_cfg)
        olmo_model = AffordVLAEarlyExit(config.model)
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
        # olmo_model = AffordVLA(model_cfg)
        olmo_model = AffordVLAEarlyExit(model_cfg)

        # We always have only rank0 load the checkpoint, and then use `sync_module_states`
        # in FSDP to broadcast the weights to the other processes
        if get_global_rank() == 0:
            is_unsharded = resource_path(cfg.load_path, "model.pt").is_file()
            if is_unsharded:
                logger.info("Loading state dict...")
                state_dict_path = resource_path(cfg.load_path, "model.pt")
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
    return olmo_model, device, config

def initialize_exit_controller(generate_cfg, model,dataloader,device):
    cfg = generate_cfg
    value_net = ActionValueNet(exit_list=model.get_all_exit_idx(cfg.exit_interval), exit_head=model.action_head, model=model, 
                                interval=cfg.exit_interval, threshold_type=cfg.threshold_type)
    exit_controller = ExitController(value_net, exit_id_list=model.get_all_exit_idx(cfg.exit_interval), steps_per_stage=generate_cfg.steps_per_stage,
                                                leq=True, exit_dist=generate_cfg.exit_dist, max_layer=model.config.n_layers)
    exit_controller.to(device)
    # 要进行离线校准，需要设置value_net的threshold
    
    state_dict_path = resource_path(generate_cfg.pretrained_checkpoint, "model.pt")
    checkpoint = torch.load(state_dict_path, map_location="cpu")

    # find threshold
    # 各出口的“动作差异分布”（形状约为 (n_exit, n_sample)），用于之后根据预算把分位数选为阈值。
    # thresholds = f(exit_action_delta_matrix, exit_ratio, exit_dist, leq, max_layer)
    # 过程：先统计各出口的动作差异分布 exit_action_delta_matrix (形状≈(n_exit, n_sample)) → 按 exit_ratio 与 exit_dist 生成每层配额 → 取对应分位数作为该层 threshold。最后一层设为 +∞ 兜底。
    values = checkpoint['exit_action_delta_matrix'] if checkpoint is not None and  "exit_action_delta_matrix" in checkpoint else None
    if not generate_cfg.thresholds:
        if model.config.action_head == "flow_matching":
            json_file_path = state_dict_path.with_name(f"exit_action_delta_matrix_{cfg.task_suite_name}_fm_steps{generate_cfg.action_head_flow_matching_inference_steps}.json")
        else:
            json_file_path = state_dict_path.with_name(f"exit_action_delta_matrix_{cfg.task_suite_name}.json")
        print(f"**** json_file_path: {json_file_path}")
        # assert json_file_path.exists(), f"json file {json_file_path} does not exist"

        if generate_cfg.load_threshold and values is not None: # load cached value distribution
            print(f'load values for threshold')
            exit_controller.set_threshold(cfg, model, dataloader, cfg.exit_ratio, cfg.llm_name, values)
        # read from json file
        elif generate_cfg.load_threshold and json_file_path.exists():
            print(f'load values for threshold from json file {json_file_path}')
            with open(json_file_path, "r") as f:
                values = json.load(f)
                # convert to torch tensor
                values = torch.tensor(values)
            exit_controller.set_threshold(cfg, model, dataloader, cfg.exit_ratio, cfg.llm_name, values)
        else:
            values = exit_controller.set_threshold(cfg, model, dataloader, cfg.exit_ratio, cfg.llm_name)
            # 缓存到 checkpoint，避免下次重复计算
            checkpoint["exit_action_delta_matrix"] = values
            if get_global_rank() == 0:
                # print("save new values for threshold to ckpt.")
                # torch.save(checkpoint, state_dict_path)
                # set values to a single json file
                with open(json_file_path, "w") as f:
                    json.dump(values.tolist(), f)
                print(f"save new values for threshold to json file {json_file_path} done.")
    else:
        exit_controller._set_threshold_value(cfg.thresholds)
    
    # save thresholds to json file
    thresholds_json_file_path = state_dict_path.with_name(f"exit_thresholds_{cfg.task_suite_name}_{cfg.exit_dist}_{cfg.exit_ratio}.json")
    with open(thresholds_json_file_path, "w") as f:
        # ensure thresholds are JSON serializable (convert Tensors to float)
        thresholds = exit_controller.get_threshold()
        thresholds = {int(k): float(v) for k, v in thresholds.items()}
        json.dump(thresholds, f)
    print(f"save thresholds to json file {thresholds_json_file_path} done.")
            
    del checkpoint
    # clear GPU memory used by finding thresholds        
    gc.collect()
    torch.cuda.empty_cache()

    return exit_controller


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
    model,
    exit_controller,
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


    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    exit_layers_episode = []

    def _exit_log_fn(message: str):
        # 始终写入统一日志
        # log_message(message, log_file)
        # 解析并收集退出层
        try:
            key = "Exit by exit_controller, block_idx:"
            if key in message:
                idx_str = message.split(key)[-1].strip()
                # 去掉可能的尾随逗号/空格
                idx = int(idx_str.split()[0].strip().strip(','))
                exit_layers_episode.append(idx)
        except Exception:
            pass
    # try:
    while t < max_steps + cfg.num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, model.config.vision_backbone.image_default_input_size)
        replay_images.append(img)

        # If action queue is empty, requery model
        if len(action_queue) == 0:
            # set timestep for exit_controller
            if exit_controller is not None:
                exit_controller.set_timestep(t)
            # Query model to get action
            # 将日志回调传入模型，收集并记录早退信息
            actions = get_vla_action(
                cfg,
                model,
                device,
                observation,
                task_description,
                exit_controller,
                output_hidden_states=True,
                log_fn=_exit_log_fn if log_file is not None else None,
            )
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

    # 统计本 episode 的退出层信息
    exit_mean_ratio_episode = None
    # try:
    if len(exit_layers_episode) > 0:
        n_layers = getattr(model.config, 'n_layers', None)
        if isinstance(n_layers, int) and n_layers > 1:
            denom = float(n_layers - 1)
            exit_sum = float(sum(exit_layers_episode))
            exit_count = len(exit_layers_episode)
            # exit_sum_ratio = exit_sum / denom
            reduced_count = denom * exit_count - exit_sum
            reduction_mean_ratio = (reduced_count / exit_count) / float(n_layers)
            executed_mean_ratio = ((exit_sum + exit_count) / exit_count) / float(n_layers)
            log_message(f"Exit layers this episode: {exit_layers_episode}", log_file)
            # log_message(f"Exit count this episode: {exit_count}", log_file)
            # log_message(f"Exit sum ratio (sum/({int(denom)})): {exit_sum_ratio:.4f}", log_file)
            log_message(f"Executed layers mean ratio (mean/({int(denom)})): {executed_mean_ratio:.4f}", log_file)
            log_message(f"Computation reduction layers ratio: {reduction_mean_ratio:.4f}", log_file)
            exit_mean_ratio_episode = executed_mean_ratio
        else:
            log_message(f"Exit layers this episode: {exit_layers_episode}", log_file)
            # log_message(f"Exit count this episode: {len(exit_layers_episode)}", log_file)
    else:
        log_message("No early exits recorded this episode.", log_file)
    # except Exception:
    #     pass

    return success, replay_images, exit_mean_ratio_episode

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    exit_controller,
    device,
    num_tasks,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    total_exit_mean_sum=0.0,
    total_exit_mean_count=0,
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
        success, replay_images, exit_mean_ratio = run_episode(
            cfg,
            env,
            task_description,
            model,
            exit_controller,
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
        # 累加本 episode 的执行层比例均值
        if exit_mean_ratio is not None:
            total_exit_mean_sum += exit_mean_ratio
            total_exit_mean_count += 1

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

    return total_episodes, total_successes, total_exit_mean_sum, total_exit_mean_count


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    # model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model, device, config = initialize_and_load_model(cfg)
    print("**** initialize model done!")

    # 如果设置了强制退出层，则跳过阈值标定与 ExitController 初始化
    if cfg.exit_layer_id is not None:
        dataloader = None
        exit_controller = None
        print(f"**** force exit at layer id = {cfg.exit_layer_id}, skip threshold calibration and ExitController.")
    else:
        dataloader = get_dataloader(cfg, config)
        print("**** initialize dataloader done!")
        exit_controller = initialize_exit_controller(cfg, model, dataloader, device)
        print("**** initialize exit_controller done!")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg) # 336

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg,model.config.action_head)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Exit dist: {cfg.exit_dist}", log_file)
    log_message(f"Exit ratio: {cfg.exit_ratio}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_exit_mean_sum, total_exit_mean_count = 0.0, 0
    # 从 num_tasks-1 到 0 倒序遍历
    # for task_id in tqdm.tqdm(range(num_tasks-1, -1, -1)):
    # 方式2：随机打乱顺序
    task_ids = list(range(num_tasks))
    rng = random.Random(cfg.seed)
    rng.shuffle(task_ids)
    # random.shuffle(task_ids)
    for task_id in tqdm.tqdm(task_ids):
    # 正序
    # for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, total_exit_mean_sum, total_exit_mean_count = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            exit_controller,
            device,
            num_tasks,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
            total_exit_mean_sum,
            total_exit_mean_count,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    final_mean_execute_ratio = (
        float(total_exit_mean_sum) / float(total_exit_mean_count)
        if total_exit_mean_count > 0 else 0.0
    )

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message(f"Overall executed mean ratio: {final_mean_execute_ratio:.4f}", log_file)

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