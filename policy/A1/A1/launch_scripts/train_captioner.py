import argparse
import logging
from dataclasses import replace
from typing import cast

from omegaconf import omegaconf, OmegaConf

from a1.data import PixMoCap
from launch_scripts.utils import DEBUG_MODEL, VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS
from a1.torch_util import get_world_size
from scripts.train import main as train

from a1 import TrainConfig, WandbConfig, DataConfig, OptimizerConfig, OptimizerType, \
    SchedulerConfig, SchedulerType, FSDPConfig, FSDPPrecision, FSDPWrapStrategy
from a1.config import BatchDivisor, SpeedMonitorConfig, ActivationCheckpointingStrategy, \
    DatasetEvaluatorConfig
from a1.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment,
)
import torch.multiprocessing as mp
import torch.distributed as dist


log = logging.getLogger("train")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Set CUDA device correctly before init
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.cuda.set_device(local_rank)
    # # Initialize process group.
    # if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    parser = argparse.ArgumentParser(prog="Train a captioner")
    parser.add_argument("llm", choices=["debug"] + list(LLMS.keys()))
    parser.add_argument("--vision_backbone", choices=list(VISION_BACKBONES.keys()), default="openai")
    parser.add_argument("--global_batch_size", default=2, type=int)
    parser.add_argument("--n_eval_examples", default=2048, type=int)
    parser.add_argument("--device_eval_batch_size", default=1, type=int)
    parser.add_argument("--seq_len", default=2304, type=int)
    parser.add_argument("--dataset", default="pixmo_cap_with_transcripts")
    args, other_args = parser.parse_known_args()

    seq_len = args.seq_len
    debug = args.llm in ["debug", "debug-12crop"]
    if debug:
        model_cfg = DEBUG_MODEL
        if args.llm == "debug-12crop":
            model_cfg.max_crops = 12
            model_cfg.crop_mode = "overlap-and-resize-c2"
        model_cfg.system_prompt_kind = 'style_and_length'

        global_batch_size = 8
        model_init = None
        eval_interval = 20
        log_interval = 5
        eval_examples = 64
        duration = 200
    else:
        eval_examples = args.n_eval_examples
        log_interval = 20
        global_batch_size = args.global_batch_size
        n = len(PixMoCap("train", "captions"))
        duration = 4 * (n + global_batch_size - 1) // global_batch_size
        eval_interval = 1000
        vit_layers = [-2, -9] if args.vision_backbone == "openai" else [-3, -9]
        model_cfg = replace(
            LLMS[args.llm],
            vision_backbone=VISION_BACKBONES[args.vision_backbone],
            llm_load_path=DEFAULT_LOAD_PATHS.get(args.llm, omegaconf.MISSING),
            vit_load_path=DEFAULT_LOAD_PATHS.get(args.vision_backbone, omegaconf.MISSING),
            crop_mode="overlap-and-resize-c2",
            system_prompt_kind='style_and_length',
            residual_dropout=0.0,
            response_residual_dropout=0.1,
            max_crops=12,
            vit_layers=vit_layers,
            # overlap_margins=(2, 2),
            additional_vocab_size=128,
        )

    evaluator = DatasetEvaluatorConfig(
        label="val",
        subset_num_batches=eval_examples//(args.device_eval_batch_size*get_world_size()),
        data=DataConfig(
            dataset=args.dataset,
            for_inference=False,
            shuffle=False,
            split="validation",
            drop_last=True,
            sequence_length=seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            shuffle_messages=False,
        ),
    )

    cfg = TrainConfig(
        run_name="multitask_train",
        no_pre_train_checkpoint=True,
        save_folder="debug_run" if debug else omegaconf.MISSING,
        seed=6198,
        dry_run=False,
        wandb=None if debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval
        ),
        model=model_cfg,
        data=DataConfig(
            dataset=args.dataset,
            for_inference=False,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            seed=95818,
            num_workers=2,
            pad="to_max",
            pin_memory=True,
            shuffle_messages=False,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=2e-4,
            vit_learning_rate=6e-6,
            llm_learning_rate=2e-5,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            metrics_log_interval=20
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=2000,
            llm_t_warmup=2000,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=None,
        save_overwrite=debug,
        save_dataloader_state=False,
        save_interval=4000,
        save_num_checkpoints_to_keep=1,
        save_interval_unsharded="${max_duration}",
        global_train_batch_size=global_batch_size,
        device_eval_batch_size=args.device_eval_batch_size,
        device_train_microbatch_size=1,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        activation_checkpointing=ActivationCheckpointingStrategy.whole_layer,
        eval_interval=eval_interval,
        evaluators=[
            # Evaluate loss on data with and without the transcripts
            evaluator,
            replace(
                evaluator,
                label="caption_val",
                data=replace(
                    evaluator.data,
                    dataset="pixmo_cap"
                )
            )
        ]
    )

    conf = OmegaConf.create(cfg)
    if other_args:
        overrides = [clean_opt(arg) for arg in other_args]
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    train(cfg)

