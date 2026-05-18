from typing import Dict, List, Union

import torch
from torchmetrics import MeanMetric, Metric

from .inf_evaluator import build_inf_evaluator, InfDatasetEvaluator
from .loss_evaluator import LossDatasetEvaluator
from ..config import DatasetEvaluatorConfig, TrainConfig
from ..data import build_eval_dataloader

__all__ = [
    "build_evaluator",
    "build_loss_evaluators",
]

from ..torch_util import get_world_size, get_global_rank


def build_evaluator(
    train_config: TrainConfig, eval_config: DatasetEvaluatorConfig, tokenizer, device: torch.device
) -> LossDatasetEvaluator:
    eval_loader = build_eval_dataloader(
        train_config,
        eval_config.data,
        eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
    )

    def make_metric():
        return MeanMetric(nan_strategy="error").to(device)

    eval_metric: Union[Metric, Dict[str, Metric]]
    eval_metric = dict(
        Loss=make_metric(),
        Accuracy=make_metric(),
        ZLoss=make_metric()
    )
    return LossDatasetEvaluator(
        label=eval_config.label,
        eval_loader=eval_loader,
        eval_metric=eval_metric,
        subset_num_batches=eval_config.subset_num_batches or train_config.eval_subset_num_batches,
    )


def build_loss_evaluators(cfg: TrainConfig, device: torch.device) -> List[LossDatasetEvaluator]:
    evaluators = []
    tokenizer = cfg.model.get_tokenizer()
    if len(set(x.label for x in cfg.evaluators)) != len(cfg.evaluators):
        raise ValueError("Non-unique labels in evaluators")
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, tokenizer, device))
    return evaluators


def build_inf_evaluators(cfg: TrainConfig, device: torch.device) -> List[InfDatasetEvaluator]:
    evaluators = []
    tokenizer = cfg.model.get_tokenizer()
    for eval_config in cfg.inf_evaluators:
        assert eval_config.mm_evaluator is not None

        device_batch_size = eval_config.device_eval_batch_size or cfg.device_inf_eval_batch_size
        global_batch_size = device_batch_size * get_world_size()
        if eval_config.max_examples and eval_config.max_examples > 0:
            max_steps = max(eval_config.max_examples // global_batch_size, 1)
        elif eval_config.subset_num_batches:
            max_steps = eval_config.subset_num_batches
        else:
            max_steps = None

        eval_loader = build_eval_dataloader(
            cfg,
            eval_config.data,
            device_batch_size,
            max_steps=max_steps
        )
        print("*************** build_inf_evaluators: after eval_loader")

        metric_computer = build_inf_evaluator(eval_config.mm_evaluator)
        evaluators.append(InfDatasetEvaluator(
            eval_loader,
            metric_computer,
            label=eval_config.label,
            n_steps=max_steps,
            max_new_tokens=eval_config.max_new_tokens,
        ))
    return evaluators

