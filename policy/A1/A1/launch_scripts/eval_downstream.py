"""Evals a checkpoint on multiple tasks, run this script with 'torchrun'."""
import argparse
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from a1.config import EvalConfig, FSDPConfig, FSDPWrapStrategy, FSDPPrecision
from a1.torch_util import get_world_size
from a1.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment, )
from scripts.mm_eval import ModelEvaluator
from launch_scripts.utils import get_evaluation

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(prog="Evaluate a model on downstream tasks")
    parser.add_argument("checkpoint",
                        help="Checkpoint to evaluate, should contain a config file and unshared model file")
    parser.add_argument("tasks", nargs="+", help="Tasks to evaluate")
    parser.add_argument("--high_res", action="store_true",
                        help="User default high rese setting: max crops=36, seq len=4096 and eval_name=36crop")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--max_crops", type=int, default=None,
                        help="Override models default number of crops")
    parser.add_argument("--seed", default=6198, type=int)
    parser.add_argument("--seq_len", default=1536, type=int,
                        help="Max sequence length to use")
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save the evaluation results")
    parser.add_argument("--save_to_checkpoint_dir", action="store_true",
                        help="Save to the checkpoint directory")
    parser.add_argument("--eval_name",
                        help="Name to use as a prefix when saving results")
    parser.add_argument("--pbar", action="store_true",
                        help="Show a progress bar")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsdp", action="store_true",
                        help="Load with FSDP, can be used to avoid OOMs")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens, otherwise use task-specific default")
    args, other_args = parser.parse_known_args()

    if args.high_res:
        args.max_crops = 36
        args.seq_len = 4096
        args.eval_name = "36crop"

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    add_cached_path_clients()
    prepare_cli_environment()

    tasks = []
    for task in args.tasks:
        if task == 'A1':
            tasks += [
                "agd20k",
                "robovqa",
            ]
        if task == 'spatial':
            tasks += [
                'clevrmath',
                'superclevr',
                'trance_test_id',
            ]
        elif task == "low-res":
            # low-res validation tasks
            tasks += [
                "android_control_ll",
                "countbench_qa:huggingface",
                "pixmo_count_counting:validation"
            ]
        elif task == "high-res":
            # high-res validation tasks
            tasks += [
                "coco_2014_vqa_multi",
                "text_vqa",
                "okvqa",
                "chart_qa",
                "doc_qa",
                "info_qa",
                "science_qa_img",
                "ai2_diagram_v2_mix_transparent",
                "a_okvqa_mc",
                "a_okvqa_da",
                "mmmu_test",
                "real_world_qa_no_instruction:test",
                "math_vista_v2",
            ]
        elif task == "test-high-res":
            # high-res test tasks
            tasks = [
                "info_qa:test",  # No metrics, submit to eval server
                "doc_qa:test",  # No metrics, submit to eval server
                "chart_qa:test",
                "text_vqa",  # test server is down, so just have to use val
                "ai2_diagram_v2_mix_transparent:test",
                "math_vista_v2",  # Just use val as is common in the literature
                "real_world_qa_no_instruction:test",
                "a_okvqa_mc",
                "a_okvqa_da",
                "mmmu_test",  # standard practice is to use val
                "a_okvqa_mc:test",
                "a_okvqa_da:test",
            ]

        elif task == "test-low-res":
            # low-res test tasks
            tasks += [
                "countbench_qa:huggingface",
                "pixmo_count_counting:test",
                "android_control_hl_ll:test",
                "android_control_hl:test",
                # Do this last and low-res since it is HUGE
                "vqa_v2_test:test2015", # No metrics
            ]
        elif "," in task:
            tasks += task.split(",")   # support comma seperator just because the jax code does
        else:
            tasks.append(task)
    tasks = list({k: None for k in tasks})  # de-duplicate but keep order

    inf_evaluators = []
    for task in tasks:
        eval_config = get_evaluation(
            name=task, seq_len=args.seq_len,
            batch_size=args.device_batch_size*get_world_size(),
            max_examples=args.max_examples,
            num_workers=0,
        )
        if args.max_new_tokens:
            eval_config = replace(eval_config, max_new_tokens=args.max_new_tokens)
        inf_evaluators.append(replace(
            eval_config,
            mm_evaluator=replace(
                eval_config.mm_evaluator,
                n_to_log=4,
                num_wandb_examples=300,
                save_predictions="_default",
            ),
            save_to_checkpoint_dir=args.save_to_checkpoint_dir,
            save_dir=args.save_dir,
            eval_name=args.eval_name,
            skip_if_metrics_cached=not args.overwrite,
        ))

    checkpoint_dir = Path(args.checkpoint)
    if not (checkpoint_dir / "model.pt").exists() and args.checkpoint != "debug":
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

    cfg = EvalConfig(
        max_crops_override=args.max_crops,
        evaluations=inf_evaluators,
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


if __name__ == "__main__":
    main()
