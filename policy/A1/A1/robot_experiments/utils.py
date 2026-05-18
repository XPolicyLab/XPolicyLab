
import time
import os
import logging
import wandb
import torch
from packaging import version
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from transformers import AutoModelForCausalLM
from functools import partial
from a1.util import (
    resource_path,
    # add_cached_path_clients,
    # clean_opt,
    # prepare_cli_environment, log_metrics_to_console,
)
from a1.vla.affordvla import AffordVLA
from a1.config import EvalConfig, TokenizerConfig, ModelConfig,TrainConfig
from a1.config import FSDPConfig, FSDPWrapStrategy, FSDPPrecision
from a1.checkpoint import load_model_state
from a1.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    peak_gpu_memory,
    seed_all, get_world_size,
)

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

def setup_logging(cfg):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

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
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

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
    device = torch.device("cuda")

    if cfg.load_path == "debug":
        logging.warning("Loading debugging model")
        raise NotImplementedError("Debugging model loading not implemented yet")

    elif cfg.load_path.startswith("hf-"):
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.load_path[3:], trust_remote_code=True, torch_dtype='fp32', device_map='cpu')
        import pdb; pdb.set_trace()
    elif cfg.fsdp is None:
        # 这里从qwen2-7b加载模型参数
        model_cfg_path = resource_path(cfg.load_path, "config.yaml")
        config = TrainConfig.load(model_cfg_path, validate_paths=False)
        # model_cfg = ModelConfig.load(model_cfg_path, key="model", validate_paths=False)
        model_cfg = config.model
        model_cfg.vit_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_image_encoders/vit-l-14-336.pt")
        model_cfg.llm_load_path = os.path.join(os.environ.get("DATA_DIR", ""), "pretrained_llms/qwen2-7b.pt")
        model_cfg.tokenizer.tokenizer_dir = os.environ.get("HF_HOME", "")
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
    return olmo_model, device