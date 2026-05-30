import argparse
import logging
import re
from pathlib import Path
from typing import cast

import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from a1.config import EvalConfig, FSDPConfig, FSDPWrapStrategy, FSDPPrecision, DatasetEvaluatorConfig, \
    EvaluatorConfig, DataConfig
from a1.torch_util import get_world_size
from a1.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment, )
from scripts.mm_eval import ModelEvaluator

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(prog="Script to generate dense captions")
    parser.add_argument("checkpoint")
    parser.add_argument("--task", required=True)
    # parser.add_argument("--split", default="validation")
    parser.add_argument("--max_crops", type=int, default=None)
    parser.add_argument("--seed", default=6198, type=int)
    parser.add_argument("--seq_len", default=512, type=int)
    parser.add_argument("--max_examples", default=None, type=int)
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--eval_name")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=448,
                        help="Override max new tokens, otherwise use task-specific default")
    args, other_args = parser.parse_known_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    add_cached_path_clients()
    prepare_cli_environment()

    if args.max_examples:
        batch_size = get_world_size()*args.device_batch_size
        n_batches = args.max_examples//batch_size
        logging.info(f"Evaluating on {n_batches} batches ({batch_size*n_batches} examples)")
    else:
        n_batches = -1

    checkpoint_dir = Path(args.checkpoint)
    if not (checkpoint_dir / "model.pt").exists():
        candidates = []
        for file in checkpoint_dir.iterdir():
            match = re.match("^step([0-9]+)-unsharded.*", file.name)
            if match:
                candidates.append((file, int(match.group(1))))
        if len(candidates) == 0:
            raise FileNotFoundError(f"{checkpoint_dir} is a directory but it did not "
                                    f"contain any unsharded checkpoints")
        checkpoint_dir = max(candidates, key=lambda x: x[1])[0].absolute().as_posix()
        logging.info(f"Selected {checkpoint_dir} as oldest checkpoint in {checkpoint_dir}")
    else:
        checkpoint_dir = args.checkpoint

    # eval_config = DatasetEvaluatorConfig(
    #     data=DataConfig(
    #         args.task, split=args.split, sequence_length=args.seq_len,
    #         for_inference=True, drop_last=False,
    #         shuffle=False,
    #         num_workers=2, pin_memory=True,
    #     ),
    #     max_new_tokens=args.max_new_tokens,
    #     mm_evaluator=EvaluatorConfig(
    #         n_to_log=10,
    #         num_wandb_examples=300,
    #         save_predictions="_default",
    #     ),
    #     save_to_checkpoint_dir=True,
    #     save_dir=args.save_dir,
    #     eval_name=args.eval_name,
    #     skip_if_metrics_cached=not args.overwrite,
    #     label=args.task,
    #     subset_num_batches=n_batches
    # )

    cfg = EvalConfig(
        max_crops_override=args.max_crops,
        # evaluations=[eval_config],
        load_path=checkpoint_dir,
        seed=args.seed,
        device_inf_eval_batch_size=args.device_batch_size,
        pbar=args.pbar,
        console_log_interval=10,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
        ) if args.fsdp else None,
    )

    if other_args:
        config = OmegaConf.create(cfg)
        overrides = [clean_opt(arg) for arg in other_args]
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
        cfg = cast(EvalConfig, OmegaConf.to_object(config))
    ModelEvaluator(cfg).run()


if __name__ == '__main__':
    main()