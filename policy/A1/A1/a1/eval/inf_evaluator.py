"""Class to evaluate models based on their generation outputs"""
import dataclasses
import itertools
import logging
from collections import defaultdict
from typing import List, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchmetrics
import wandb
from tqdm import tqdm

from .evaluators import (
    HtmlTable, CountEval, TrajectoryEval, PointCountEval, PointingEval, ClockEval, VqaEval,
    SavePredictions, AndroidControlEval, MathVistaEval, PointingEval, MultiThresholdBoxEval, CoordinateAccuracyEval, ActionEval
)
from ..config import EvaluatorConfig
from ..torch_util import (
    get_global_rank,
    get_world_size,
    move_to_device,
)
from ..util import flatten_list

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InfEvaluator:
    """
    Evaluates the text outputs from a model on a task
    """
    metrics: List

    def __call__(self, predictions, example_metadata, tokenizer, device, step=None, action_tokenizer=None):
        inf_metrics = {}
        for metric in self.metrics:
            if isinstance(metric, ActionEval):
                results = metric(example_metadata, predictions, step=step, tokenizer=tokenizer, action_tokenizer=action_tokenizer)
            else:
                results = metric(example_metadata, predictions, step=step, tokenizer=tokenizer)
            assert all(k not in inf_metrics for k in results)
            inf_metrics.update(results)

        resolved_metrics = {}
        # sort so metrics are iterated on in the same order on all devices
        for k in sorted(inf_metrics):
            v = inf_metrics[k]
            if isinstance(v, torchmetrics.Metric):
                resolved_metrics[k] = v.to(device).compute().item()
            elif isinstance(v, HtmlTable):
                # Special case, we aggregate table rows from all devices to ensure we can always
                # have enough rows to show even if each device only eval-ed a few examples
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    all_rows = flatten_list([x.rows for x in all_predictions])
                    resolved_metrics[k] = wandb.Html(HtmlTable(all_rows).get_html())
                else:
                    dist.gather_object(v, None)
            else:
                raise ValueError(f"Metric {v} not understood")

        for metric in self.metrics:
            if isinstance(metric, (CountEval, PointCountEval)):
                # Counting has a macro-score that should be computed once we have
                # scores from all devices
                counting_scores = {k: resolved_metrics[k] for
                                   k in list(resolved_metrics.keys()) if k.startswith("correct_")}
                resolved_metrics["per_category_average"] = np.mean(list(counting_scores.values()))
        return resolved_metrics


def build_inf_evaluator(cfg: EvaluatorConfig, default_save_dir=None) -> InfEvaluator:
    evaluators = []
    save_predictions = cfg.save_predictions
    if save_predictions == "_default":
        if default_save_dir is None:
            logging.info(f"save_predictions is default but not default save dir set")
        save_predictions = default_save_dir
    if save_predictions:
        evaluators.append(SavePredictions(
            save_predictions,
            log_examples=cfg.n_to_log,
            save_tokens=cfg.save_tokens
        ))

    if cfg.vqa_eval:
        evaluators.append(VqaEval(cfg.vqa_eval.split(","), cfg.num_wandb_examples))
    elif cfg.clock_eval:
        evaluators.append(ClockEval(cfg.num_wandb_examples))
    elif cfg.clock_bench_eval:
        evaluators.append(ClockEval(cfg.num_wandb_examples, is_test=True))
    elif cfg.math_vista_eval:
        evaluators.append(MathVistaEval(cfg.num_wandb_examples))
    elif cfg.point_count_eval:
        evaluators.append(PointCountEval(cfg.num_wandb_examples))
    elif cfg.trajectory_eval:
        evaluators.append(TrajectoryEval(cfg.num_wandb_examples))
    elif cfg.count_eval:
        evaluators.append(CountEval(cfg.num_wandb_examples))
    elif cfg.android_eval:
        evaluators.append(AndroidControlEval(cfg.num_wandb_examples))
    elif cfg.multi_threshold_box_eval:
        evaluators.append(MultiThresholdBoxEval(cfg.num_wandb_examples))
    elif cfg.coordinate_eval:
        evaluators.append(CoordinateAccuracyEval(cfg.num_wandb_examples))
    elif cfg.action_eval:
        evaluators.append(ActionEval(cfg.num_wandb_examples))
    if cfg.pointing_eval:
        evaluators.append(PointingEval(cfg.num_wandb_examples))
    else:
        pass
    return InfEvaluator(evaluators)


@dataclasses.dataclass
class InfDatasetEvaluator:
    """Evaluates a model on a dataset"""

    dataloader: Any
    evaluator: InfEvaluator
    n_steps: int
    label: str = None
    max_new_tokens: int = 448
    console_log_interval: Optional[int] = None

    def evaluate_model(self, model, device, autocast_precision, is_distributed,
                       inference_warmup=False, pbar=False):
        eval_dataloader = self.dataloader
        eval_it = iter(eval_dataloader)
        n_steps = self.n_steps
        if n_steps is not None and n_steps < len(self.dataloader):
            eval_it = itertools.islice(eval_it, 0, n_steps)
            total_steps = n_steps
        else:
            total_steps = len(eval_dataloader)

        all_metadata = []
        predictions = defaultdict(list)
        done_init = False
        pbar = pbar and get_global_rank() == 0
        for eval_step, batch in enumerate(tqdm(eval_it, total=total_steps, ncols=100, disable=not pbar)):
            # print(f"eval_step: {eval_step}")
            if "metadata" in batch:
                batch_metadata = batch.pop("metadata")
            else:
                # Handle old-style data that used metadata/ prefix instead
                metadata = {k: batch.pop(k) for k in list(batch) if k.startswith("metadata/")}
                batch_metadata = []
                for i in range(len(batch["input_ids"])):
                    converted = {}
                    for k, v in metadata.items():
                        if isinstance(v[i], bytes):
                            converted[k] = v[i].decode("utf-8")
                        else:
                            converted[k] = v[i].tolist()
                    batch_metadata.append(converted)
            batch_inference = move_to_device(batch, device)
            
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    if inference_warmup:
                        # For reasons I don't understand doing a regular forward pass first
                        # prevents OOMs when calling generate, its annoying but we just
                        # put up with the doing an initial forward pass for now
                        model(
                            input_ids=batch_inference["input_ids"],
                            images=batch_inference.get("images"),
                            image_masks=batch_inference.get("image_masks"),
                            image_input_idx=batch_inference.get("image_input_idx"),
                        )
                        inference_warmup = False

                    olmo_gen_output = model.generate(
                        input_ids=batch_inference["input_ids"],
                        images=batch_inference.get("images"),
                        image_masks=batch_inference.get("image_masks"),
                        image_input_idx=batch_inference.get("image_input_idx"),
                        max_steps=self.max_new_tokens,
                        is_distributed=is_distributed
                    )
            pred = {
                "predictions": olmo_gen_output.token_ids[:, 0].detach().cpu().numpy(), # beam size of 1
                "prompts": batch_inference["input_ids"].detach().cpu().numpy(),
            }
            valid_ixs = [i for i, md in enumerate(batch_metadata) if md.get("valid", True)]
            all_metadata += [batch_metadata[i] for i in valid_ixs]
            for k, v in pred.items():
                for ix in valid_ixs:
                    predictions[k].append(v[ix])

            # Log to console.
            if self.console_log_interval and not pbar:
                if eval_step + 1 == n_steps or (eval_step + 1) % self.console_log_interval == 0:
                    if total_steps:
                        log.info(f"[eval_step={eval_step + 1}/{total_steps}]")
                    else:
                        log.info(f"[eval_step={eval_step + 1}/{n_steps}]")

        tokenizer = model.config.get_tokenizer()
        action_tokenizer = model.config.get_action_tokenizer()
        action_tokenizer.action_dim = model.config.action_dim
        action_tokenizer.horizon = model.config.horizon
        metrics = self.evaluator(predictions, all_metadata, tokenizer, device, action_tokenizer=action_tokenizer)
        return metrics

