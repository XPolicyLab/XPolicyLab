import os
import argparse
import logging
from dataclasses import replace
from typing import cast
from os.path import join, exists
import random
from omegaconf import omegaconf, OmegaConf  # type: ignore[import-not-found]

from launch_scripts.utils import DEBUG_MODEL, VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS
from a1.vla.config_loader import read_vla_yaml_config
from a1.torch_util import get_world_size


from a1 import TrainConfig, WandbConfig, DataConfig, OptimizerConfig, OptimizerType, \
    SchedulerConfig, SchedulerType, FSDPConfig, FSDPPrecision, FSDPWrapStrategy
from a1.config import BatchDivisor, SpeedMonitorConfig, ActivationCheckpointingStrategy, \
    DatasetEvaluatorConfig, ModelConfig
from a1.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment,
)
import torch.multiprocessing as mp
import torch.distributed as dist
import torch

from scripts.train_for_action import main as train

# 启用内存效率优化
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] ='max_split_size_mb:256,expandable_segments:True'  # 'max_split_size_mb:512'

from datetime import datetime, timedelta
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# from a1.data.vla.rlds_datasets import DummyRLDS

log = logging.getLogger(__name__)



if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        log.info(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Initialize process group.
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        timeout_s = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", "1800"))
        init_kwargs = {
            "backend": "nccl",
            "timeout": timedelta(seconds=timeout_s),
        }
        try:
            dist.init_process_group(**init_kwargs, device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(**init_kwargs)

    try:
        log.info(
            "Process group initialized on local_rank=%s with timeout=%ss",
            local_rank,
            os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", "1800"),
        )


        prepare_cli_environment()
        log.info("CLI environment prepared")
        
        add_cached_path_clients()
        
        parser = argparse.ArgumentParser(prog="Train a VLA model")
        parser.add_argument("llm", choices=["debug"] + list(LLMS.keys()))
        parser.add_argument("--checkpoint", help="Path to checkpoint to start from", default=None)
        parser.add_argument("--vision_backbone", choices=list(VISION_BACKBONES.keys()), default="siglip") #default="openai"
        parser.add_argument("--global_batch_size", default=96, type=int)
        parser.add_argument("--device_train_microbatch_size", default=24, type=int)
        parser.add_argument("--train_steps", default=500000, type=int, help="Number of training steps")
        
        parser.add_argument("--n_eval_examples", default=2048, type=int)
        parser.add_argument("--device_eval_batch_size", default=4, type=int)
        # parser.add_argument("--seq_len", default=2304, type=int)
        parser.add_argument("--seq_len", default=1024, type=int) # 768 for two images,512 for one image

        parser.add_argument("--crop_mode", default='overlap-and-resize-c2', type=str) # overlap-and-resize-c2
        parser.add_argument("--max_crops", default=3, type=int) # 12 for molmo default

        parser.add_argument("--llm_causal_attention", default=False, type=bool, help="Whether to use causal attention in the LLM")

        parser.add_argument("--llm_learning_rate", default=5e-5, type=float) # default 2e-5
        parser.add_argument("--vit_learning_rate", default=6e-6, type=float) # default 6e-6
        parser.add_argument("--connector_learning_rate", default=2e-4, type=float) # default 2e-4
        parser.add_argument("--action_head_learning_rate", default=5e-5, type=float) # default 5e-5
        parser.add_argument("--connector_weight_decay", default=0.0, type=float)
        parser.add_argument("--vit_weight_decay", default=0.0, type=float)
        parser.add_argument("--llm_weight_decay", default=0.0, type=float)
        parser.add_argument("--action_head_weight_decay", default=0.0, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.95, type=float)
        parser.add_argument("--warmup_steps", default=2000, type=int, help="Warmup steps for VLA learning rate scheduler") #2000
        parser.add_argument("--freeze_steps", default=0, type=int, help="Freeze steps for VLA learning rate scheduler")
        parser.add_argument("--scheduler_alpha_f", default=0.1, type=float, help="Final LR fraction for cosine decay")
        parser.add_argument("--warmup_min_lr", default=None, type=float, help="Absolute LR to start warmup from; default is 10 percent of each LR")
        parser.add_argument("--save_interval", default=500, type=int, help="Interval (in steps) to save checkpoints")
        parser.add_argument("--save_interval_unsharded", default=500, type=int, help="Interval (in steps) to save unsharded checkpoints")
        parser.add_argument("--save_num_checkpoints_to_keep", default=1, type=int, help="Number of sharded checkpoints to keep; 0 disables sharded checkpoint saves")
        parser.add_argument("--save_num_unsharded_checkpoints_to_keep", default=1, type=int, help="Number of unsharded checkpoints to keep")

        parser.add_argument("--ft_connector", default=False, action="store_true")
        parser.add_argument("--ft_llm", default=False, action="store_true")
        parser.add_argument("--ft_vit", default=False, action="store_true")
        parser.add_argument("--ft_embedding", default="lm_head", choices=["all", "lm_head", "ln_f", "wte"])
        parser.add_argument("--early_exit", default=False, action="store_true")
        parser.add_argument("--train_exit_random_layer", default=False, action="store_true")

        parser.add_argument("--allow_resume", default=False, action="store_true", help="Allow resuming from the latest checkpoint in the save folder.")
        parser.add_argument("--load_path", type=str, default=None, help="Path to the model checkpoint to load.")
        #"/mnt/data/zhangjian/a1/libero_spatial_qwen3-4b_dit-xl_wrist_proprio_ft_ah_lora_llm_bs102/latest"
        parser.add_argument("--keep_lr_on_load", default=True, action="store_true", help="Keep LR/WD from checkpoint instead of resetting")
        parser.add_argument("--save_overwrite", default=False, action="store_true", help="Overwrite existing checkpoint step folders when saving")

        parser.add_argument("--action_loss_weight", default=1.0, type=float)
        parser.add_argument("--state_mask_prob", default=0.5, type=float, help="Probability of randomly masking state to 0 during training (0.0-1.0)")
        parser.add_argument("--action_head",default="flow_matching",type=str,choices=["l1_regression", "diffusion", "diffusion_openvla","flow_matching","flow_matching_joint"])

        parser.add_argument("--action_head_diffusion_train_steps", default=1000, type=int, help="Number of diffusion steps for the action head")
        parser.add_argument("--action_head_diffusion_inference_steps", default=10, type=int) # 10

        parser.add_argument("--action_head_dit_depth", default=28, type=int,) # 28
        parser.add_argument("--action_head_dit_hidden_size", default=1152 , type=int,) # 2048
        parser.add_argument("--action_head_dit_num_heads", default=16, type=int,)

        parser.add_argument("--action_head_flow_matching_dim", default=1024, type=int)
        # 若为 -1，则默认对齐主 VLM 的 model_cfg.n_layers
        parser.add_argument("--action_head_flow_matching_layers", default=-1, type=int)
        # FM 里小 Qwen2 的注意力头数，可与主 VLM 不同，默认为 8
        parser.add_argument("--action_head_flow_matching_heads", default=8, type=int)
        parser.add_argument("--action_head_flow_matching_intermediate_size", default=2048, type=int)
        # KV 头数：若为 -1，则默认对齐主 VLM 的 n_kv_heads（若为 None 则回退到 n_heads）
        parser.add_argument("--action_head_flow_matching_kv_heads", default=-1, type=int)
        parser.add_argument("--action_head_flow_matching_pvf_function", default="2d_attn_mask", type=str, choices=["2d_attn_mask", "4d_attn_mask"])

        parser.add_argument("--use_proprio",default=True,type=bool)
        parser.add_argument("--use_wrist_image",default=True,type=bool,help="Whether to use wrist image in the dataset")
        parser.add_argument("--dataset", default="vla_dataset_realworld")


        parser.add_argument("--wandb_debug", default=False, action="store_true", help="Whether to use wandb")
        parser.add_argument("--wandb_entity", type=str, default=None)
        parser.add_argument("--wandb_project", type=str, default='a1-vla-camd')
        parser.add_argument("--wandb_run_name", type=str, default="libero_4_qwen3-8b_siglip_flow-matching_wrist_proprio_ft_ah_lora_r8_llm_bs96", )
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--num_workers", type=int, default=2)
        # minghao args
        parser.add_argument(
            "--vla_config_path",
            type=str,
            default="libero_simulation.yaml",
            help="VLA config file name. Auto-searches in: configs/experiments/, configs/, launch_scripts/"
        )


        args, other_args = parser.parse_known_args()
        
        tmp_rng = random.Random()
        tmp_rng.seed(int(os.urandom(4).hex(), 16))

        yaml_name = args.vla_config_path
        # Configure VLA constants from the chosen YAML path
        os.environ["VLA_CONFIG_YAML"] = yaml_name

        vla_cfg = read_vla_yaml_config(yaml_name)
        action_head_cfg = vla_cfg["model"]["action_head"]
        fixed_action_dim = action_head_cfg["fixed_action_dim"]
        proprio_dim = fixed_action_dim
        action_dim = fixed_action_dim
        num_actions_chunk = action_head_cfg["num_actions_chunk"]

        # Optional parameters with defaults
        action_use_left_eef = action_head_cfg.get("use_left_eef", False)
        action_use_mobile_base = action_head_cfg.get("use_mobile_base", False)

        # Get action_tokens_mapping with defaults
        mapping = action_head_cfg.get("action_tokens_mapping", {})
        mobile_base_dim = mapping.get("mobile_base", 0)
        left_end_effector_dim = mapping.get("left_end_effector", 7 if action_use_left_eef else 0)
        right_end_effector_dim = mapping.get("right_end_effector", fixed_action_dim - left_end_effector_dim - mobile_base_dim)
        ##
        
        
        seq_len = args.seq_len
        debug = args.llm in ["debug", "debug-12crop"]
        if debug:
            model_cfg = DEBUG_MODEL
            if args.llm == "debug-12crop":
                model_cfg.max_crops = 12
                # model_cfg.crop_mode = "overlap-and-resize-c2"
                model_cfg.crop_mode = "resize"
            model_cfg.system_prompt_kind = 'demo_or_style'

            global_batch_size = 8
            model_init = None
            # eval_interval = 20
            eval_interval = 0
            log_interval = 2 #5
            eval_examples = 64
            duration = 100000
        else:
            eval_examples = args.n_eval_examples
            # log_interval = 20
            log_interval = args.log_interval
            global_batch_size = args.global_batch_size
            # n = len(DummyRLDS("train"))
            # duration = 4 * (n + global_batch_size - 1) // global_batch_size
            duration = args.train_steps
            log.info(f"!Total training steps: {duration}")
            # eval_interval = 1000
            eval_interval = 0
            if args.checkpoint:
                if exists(join(args.checkpoint, "model.yaml")):
                    model_cfg = ModelConfig.load(join(args.checkpoint, "model.yaml"))
                else:
                    model_cfg = ModelConfig.load(join(args.checkpoint, "config.yaml"), key="model")

                # 根据主 VLM 的配置，确定 FM head 的层数与 KV 头数（可被 CLI 覆盖）
                fm_layers = args.action_head_flow_matching_layers if args.action_head_flow_matching_layers > 0 else model_cfg.n_layers
                fm_kv_heads = (
                    args.action_head_flow_matching_kv_heads
                    if args.action_head_flow_matching_kv_heads > 0
                    else (model_cfg.n_kv_heads if model_cfg.n_kv_heads is not None else model_cfg.n_heads)
                )
                model_cfg = replace(
                    model_cfg,
                    crop_mode=args.crop_mode,
                    # residual_dropout=0.0,
                    # response_residual_dropout=0.1,
                    max_crops=args.max_crops,
                    # vit_layers=vit_layers,
                    # overlap_margins=(2, 2),
                    # additional_vocab_size=128,
                    
                    llm_causal_attention=args.llm_causal_attention,
                    action_head =  args.action_head,##vla action head
                    fixed_action_dim=action_dim,
                    proprio_dim=proprio_dim,
                    action_dim=action_dim,
                    action_use_left_eef=action_use_left_eef,
                    action_use_mobile_base=action_use_mobile_base,
                    right_end_effector_dim=right_end_effector_dim,
                    left_end_effector_dim=left_end_effector_dim,
                    mobile_base_dim=mobile_base_dim,
                    num_actions_chunk=num_actions_chunk,
                    use_proprio= args.use_proprio,

                    action_head_dit_depth = args.action_head_dit_depth,
                    action_head_dit_hidden_size = args.action_head_dit_hidden_size,
                    action_head_dit_num_heads = args.action_head_dit_num_heads,
                    
                    num_diffusion_steps=args.action_head_diffusion_train_steps,
                    num_diffusion_inference_steps=args.action_head_diffusion_inference_steps,
                    action_head_flow_matching_dim = args.action_head_flow_matching_dim,
                    action_head_flow_matching_layers = fm_layers,
                    action_head_flow_matching_heads = args.action_head_flow_matching_heads,
                    action_head_flow_matching_intermediate_size = args.action_head_flow_matching_intermediate_size,
                    action_head_flow_matching_kv_heads = fm_kv_heads,
                    action_head_flow_matching_pvf_function = args.action_head_flow_matching_pvf_function,
                )
            else:
                vit_layers = [-2, -9] if args.vision_backbone == "openai" else [-3, -9]
                base_cfg = LLMS[args.llm]
                # 未从 checkpoint 加载时，同样按主 VLM 的配置确定默认层数与 KV 头数
                fm_layers = args.action_head_flow_matching_layers if args.action_head_flow_matching_layers > 0 else base_cfg.n_layers
                fm_kv_heads = (
                    args.action_head_flow_matching_kv_heads
                    if args.action_head_flow_matching_kv_heads > 0
                    else (base_cfg.n_kv_heads if base_cfg.n_kv_heads is not None else base_cfg.n_heads)
                )
                model_cfg = replace(
                    base_cfg,
                    vision_backbone=VISION_BACKBONES[args.vision_backbone],
                    llm_load_path=DEFAULT_LOAD_PATHS.get(args.llm, omegaconf.MISSING),
                    vit_load_path=DEFAULT_LOAD_PATHS.get(args.vision_backbone, omegaconf.MISSING),
                    crop_mode=args.crop_mode,
                    system_prompt_kind='demo_or_style',
                    residual_dropout=0.0,
                    response_residual_dropout=0.1,
                    max_crops=args.max_crops, #12
                    vit_layers=vit_layers,
                    # overlap_margins=(2, 2),
                    additional_vocab_size=128,

                    llm_causal_attention=args.llm_causal_attention,
                    action_head =  args.action_head,##vla action head
                    fixed_action_dim=action_dim,
                    proprio_dim=proprio_dim,
                    action_dim=action_dim,
                    action_use_left_eef=action_use_left_eef,
                    action_use_mobile_base=action_use_mobile_base,
                    right_end_effector_dim=right_end_effector_dim,
                    left_end_effector_dim=left_end_effector_dim,
                    mobile_base_dim=mobile_base_dim,
                    num_actions_chunk=num_actions_chunk,
                    use_proprio= args.use_proprio,

                    action_head_dit_depth = args.action_head_dit_depth,
                    action_head_dit_hidden_size = args.action_head_dit_hidden_size,
                    action_head_dit_num_heads = args.action_head_dit_num_heads,
                    
                    num_diffusion_steps=args.action_head_diffusion_train_steps,
                    num_diffusion_inference_steps=args.action_head_diffusion_inference_steps,
                    action_head_flow_matching_dim = args.action_head_flow_matching_dim,
                    action_head_flow_matching_layers = fm_layers,
                    action_head_flow_matching_heads = args.action_head_flow_matching_heads,
                    action_head_flow_matching_intermediate_size = args.action_head_flow_matching_intermediate_size,
                    action_head_flow_matching_kv_heads = fm_kv_heads,
                    action_head_flow_matching_pvf_function = args.action_head_flow_matching_pvf_function,
                )
                

        evaluator = DatasetEvaluatorConfig(
            label="val",
            subset_num_batches=eval_examples//(args.device_eval_batch_size*get_world_size()),
            data=DataConfig(
                dataset=args.dataset,
                use_wrist_image=args.use_wrist_image,  ##
                use_proprio=args.use_proprio, ## whether to use proprioceptive data
                for_inference=False,
                shuffle=False,
                split="validation",
                drop_last=True,
                sequence_length=seq_len,
                num_workers=0, #2
                pin_memory=True,
                persistent_workers=True,
                shuffle_messages=False,
            ),
        )
        log.info("before cfg")

        cfg = TrainConfig(
            run_name=f"{args.wandb_run_name}_{timestamp}",
            no_pre_train_checkpoint=True,
            save_folder="debug_run" if debug else omegaconf.MISSING,
            seed=6198,
            dry_run=False,
            early_exit=args.early_exit, ##
            train_exit_random_layer=args.train_exit_random_layer, ##
            wandb=None if args.wandb_debug else WandbConfig(
            # wandb= WandbConfig(
                name="${run_name}",
                project=args.wandb_project,
                group=None,
                # entity="${oc.env:WANDB_ENTITY}",
                entity=args.wandb_entity,
                log_interval=log_interval
            ),
            model=model_cfg,
            data=DataConfig(
                dataset=args.dataset,
                use_wrist_image=args.use_wrist_image,  # Set to True if you want to use wrist images
                use_proprio=args.use_proprio,  # Set to True if you want to use proprioceptive data
                for_inference=False,
                shuffle=True,
                split="train",
                drop_last=True,
                sequence_length=seq_len,
                seed=tmp_rng.randint(1, 1000000),
                num_workers=args.num_workers,
                pad="to_max",
                pin_memory=True,
                shuffle_messages=False,
            ),
            ft_connector=args.ft_connector,
            ft_llm=args.ft_llm,
            ft_vit=args.ft_vit,
            ft_embedding=args.ft_embedding,
            optimizer=OptimizerConfig(
                name=OptimizerType.adamw,
                connector_learning_rate=args.connector_learning_rate,
                vit_learning_rate=args.vit_learning_rate,
                llm_learning_rate=args.llm_learning_rate, #2e-5
                action_head_learning_rate=args.action_head_learning_rate, #5e-5
                connector_weight_decay=args.connector_weight_decay,
                vit_weight_decay=args.vit_weight_decay,
                llm_weight_decay=args.llm_weight_decay,
                action_head_weight_decay=args.action_head_weight_decay,
                connector_betas=[args.adam_beta1, args.adam_beta2],
                vit_betas=[args.adam_beta1, args.adam_beta2],
                llm_betas=[args.adam_beta1, args.adam_beta2],
                action_head_betas=[args.adam_beta1, args.adam_beta2],
                connector_eps=1e-6,
                vit_eps=1e-6,
                llm_eps=1e-6,
                action_head_eps=1e-6,
                metrics_log_interval=log_interval
            ),
            scheduler=SchedulerConfig(
                name=SchedulerType.multimodal,
                connector_t_warmup=args.warmup_steps,
                vit_t_warmup=args.warmup_steps,
                action_head_t_warmup=args.warmup_steps,
                llm_t_warmup=args.warmup_steps, 
                connector_freeze_steps=args.freeze_steps,
                vit_freeze_steps=args.freeze_steps,
                llm_freeze_steps=args.freeze_steps,
                action_head_freeze_steps=0,
                alpha_f=args.scheduler_alpha_f,
                warmup_min_lr=args.warmup_min_lr,
            ),
            fsdp=FSDPConfig(
                use_orig_params=True,
                wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
                # wrapping_strategy=FSDPWrapStrategy.by_block,
                precision=FSDPPrecision.float
                # precision=FSDPPrecision.mixed
            ),
            load_path=args.load_path,
            checkpoint_dir=args.checkpoint,
            initial_model_checkpoint=args.checkpoint,
            allow_resume=args.allow_resume, ## add
            keep_lr_on_load=args.keep_lr_on_load,
            save_overwrite=args.save_overwrite,
            save_dataloader_state=False,
            # save_interval="${max_duration}", # 4000
            save_interval=args.save_interval, # 4000
            save_num_checkpoints_to_keep=args.save_num_checkpoints_to_keep,
            # save_interval_unsharded="${max_duration}",
            save_interval_unsharded=args.save_interval_unsharded,
            save_num_unsharded_checkpoints_to_keep = args.save_num_unsharded_checkpoints_to_keep,
            save_interval_action_head = None,
            save_num_action_head_checkpoints_to_keep = 0,
            global_train_batch_size=global_batch_size,
            device_eval_batch_size=args.device_eval_batch_size,
            # device_train_microbatch_size=4,
            device_train_microbatch_size=args.device_train_microbatch_size,
            time_limit=None,
            max_duration=duration,
            stop_at="${max_duration}",
            max_grad_norm=1,
            batch_divisor=BatchDivisor.global_batch,
            precision='amp_bf16',
            # precision='fp32',
            # precision=precision_str,
            console_log_interval=log_interval,
            speed_monitor=SpeedMonitorConfig(window_size=20),
            softmax_auxiliary_loss=True,
            softmax_auxiliary_loss_scale=1e-4,
            activation_checkpointing=ActivationCheckpointingStrategy.whole_layer,
            # activation_checkpointing=None,
            eval_interval=eval_interval,
            evaluators=[
                # Evaluate loss on data with and without the transcripts
                evaluator,
                # replace(
                #     evaluator,
                #     label="caption_val",
                #     data=replace(
                #         evaluator.data,
                #         dataset="pixmo_cap"
                #     )
                # )
            ]
        )

        conf = OmegaConf.create(cfg)
        if other_args:
            overrides = [clean_opt(arg) for arg in other_args]
            conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
        cfg = cast(TrainConfig, OmegaConf.to_object(conf))

        # Add the specific training logic of VLA
        cfg.action_loss_weight = args.action_loss_weight
        cfg.state_mask_prob = args.state_mask_prob
        train(cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
