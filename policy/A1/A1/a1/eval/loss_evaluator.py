"""Class to build metrics for a model based on the loss"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List

import torch
from torch.utils.data import DataLoader
from torchmetrics import Metric

__all__ = ["LossDatasetEvaluator"]


@dataclass
class LossDatasetEvaluator:
    """Evaluates a model on a dataset based its on its loss or other forward-pass metrics"""

    label: str
    eval_loader: DataLoader
    eval_metric: Union[Metric, Dict[str, Metric], List[Metric]]
    subset_num_batches: Optional[int] = None

    def reset_metrics(self) -> None:
        if isinstance(self.eval_metric, Metric):
            self.eval_metric.reset()
        else:
            for metric in self.eval_metric.values():
                metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        return {f"{self.label}/{k}": v.compute().item()
                for k, v in self.eval_metric.items()}

    def update_metrics(
        self,
        batch: Dict[str, Any],
        eval_out: Dict[str, torch.Tensor],
    ) -> None:
        total_weight = eval_out["total_weight"]
        self.eval_metric["Loss"].update(eval_out["total_loss"]/total_weight, total_weight)
        self.eval_metric["Accuracy"].update(eval_out["total_accuracy"]/total_weight, total_weight)
        self.eval_metric["ZLoss"].update(eval_out["total_zloss"]/total_weight, total_weight)
