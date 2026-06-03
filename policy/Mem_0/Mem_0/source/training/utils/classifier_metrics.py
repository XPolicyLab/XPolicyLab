from collections import deque
from typing import Deque, Dict, Optional

import torch
import torch.distributed as dist


class ClassifierMetricsAggregator:
    """
    Sliding window over the most recent `window_size` sync events.
    Oldest sync blocks are popped entirely when exceeding the window budget (no partial trim).
    """

    def __init__(self, window_size: int = 10):
        self.window_size = max(1, int(window_size))
        self.window: Deque[Dict[str, float]] = deque()
        self.totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "correct": 0.0, "total": 0.0, "positive": 0.0}

    def _apply_delta(self, counts: Dict[str, float], sign: float) -> None:
        for k in self.totals:
            self.totals[k] += sign * counts.get(k, 0.0)

    def update_and_compute(self, metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        if metrics is None:
            return self.compute_window_metrics()

        new_counts = {
            "tp": metrics.get("cls_tp", 0.0),
            "fp": metrics.get("cls_fp", 0.0),
            "fn": metrics.get("cls_fn", 0.0),
            "correct": metrics.get("cls_correct", 0.0),
            "total": metrics.get("cls_total", 0.0),
            "positive": metrics.get("cls_positive", 0.0),
        }

        self.window.append(new_counts)
        self._apply_delta(new_counts, sign=1.0)
        while self.window and len(self.window) > self.window_size:
            oldest_counts = self.window.popleft()
            self._apply_delta(oldest_counts, sign=-1.0)

        return self.compute_window_metrics()

    def compute_window_metrics(self) -> Optional[Dict[str, float]]:
        if len(self.window) <= 0:
            return None

        denom = max(self.totals["total"], 1e-6)
        precision_denom = max(self.totals["tp"] + self.totals["fp"], 1e-6)
        recall_denom = max(self.totals["tp"] + self.totals["fn"], 1e-6)
        f1_denom = max(2 * self.totals["tp"] + self.totals["fp"] + self.totals["fn"], 1e-6)

        return {
            "classifier_accuracy_recent": self.totals["correct"] / denom,
            "classifier_precision_recent": self.totals["tp"] / precision_denom,
            "classifier_recall_recent": self.totals["tp"] / recall_denom,
            "classifier_f1_score_recent": (2 * self.totals["tp"]) / f1_denom,
            "classifier_prate_recent": self.totals["positive"] / denom,
            "classifier_count_recent": self.totals["total"],
            "classifier_window_size": float(min(len(self.window), self.window_size)),
            "cls_tp_recent": self.totals["tp"],
            "cls_fp_recent": self.totals["fp"],
            "cls_fn_recent": self.totals["fn"],
            "cls_correct_recent": self.totals["correct"],
            "cls_total_recent": self.totals["total"],
            "cls_positive_recent": self.totals["positive"],
        }


class ClassifierMetricsSync:
    """
    Utility to accumulate per-rank classifier metrics, synchronize at a chosen cadence,
    and optionally feed a sliding window aggregator on rank0.
    """

    COUNT_KEYS = ["cls_tp", "cls_fp", "cls_fn", "cls_correct", "cls_total", "cls_positive"]

    def __init__(
        self,
        log_interval: int,
        device: torch.device,
        is_distributed: bool,
        is_rank0: bool,
        aggregator: Optional[ClassifierMetricsAggregator] = None,
    ):
        self.log_interval = max(1, int(log_interval))
        self.device = device
        self.is_distributed = is_distributed
        self.is_rank0 = is_rank0
        self.aggregator = aggregator if is_rank0 else None

        self.pending = {k: 0.0 for k in self.COUNT_KEYS}
        self.steps_since_sync = 0

    @staticmethod
    def _derive_counts(metrics: Dict[str, float], batch_size: int) -> Dict[str, float]:
        total = float(max(batch_size, 1))
        prate = float(metrics.get("classifier_prate", 0.0))
        precision = float(metrics.get("classifier_precision", 0.0))
        recall = float(metrics.get("classifier_recall", 0.0))
        accuracy = float(metrics.get("classifier_accuracy", 0.0))

        positive = max(prate * total, 0.0)
        tp = max(recall * positive, 0.0)
        fp = max(tp * (1.0 / precision - 1.0), 0.0) if precision > 0 else 0.0
        fn = max(positive - tp, 0.0)
        correct = max(accuracy * total, 0.0)

        return {
            "cls_tp": tp,
            "cls_fp": fp,
            "cls_fn": fn,
            "cls_correct": correct,
            "cls_total": total,
            "cls_positive": positive,
        }

    def accumulate_batch(self, metrics: Dict[str, float], batch_size: int) -> None:
        if all(k in metrics for k in self.COUNT_KEYS):
            derived = {k: float(metrics[k]) for k in self.COUNT_KEYS}
        else:
            derived = self._derive_counts(metrics, batch_size)
        for k, v in derived.items():
            self.pending[k] += v
        self.steps_since_sync += 1

    def maybe_sync(self, global_step: int, force: bool = False) -> Optional[Dict[str, float]]:
        if not force and global_step % self.log_interval != 0:
            return None

        if self.steps_since_sync == 0 and not force:
            return None

        had_updates = self.steps_since_sync > 0
        counts_tensor = torch.tensor([self.pending[k] for k in self.COUNT_KEYS], device=self.device, dtype=torch.float32)
        if self.is_distributed:
            dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        reduced = {k: float(v) for k, v in zip(self.COUNT_KEYS, counts_tensor.tolist())}

        # reset pending on all ranks
        for k in self.COUNT_KEYS:
            self.pending[k] = 0.0
        self.steps_since_sync = 0

        # only rank0 updates window and exposes aggregated metrics
        window_metrics = None
        if self.aggregator is not None:
            if not had_updates and all(v == 0.0 for v in reduced.values()):
                window_metrics = self.aggregator.compute_window_metrics()
            else:
                window_metrics = self.aggregator.update_and_compute(reduced)
        if not self.is_rank0:
            return None

        return window_metrics

    def finalize_epoch(self, global_step: int) -> Dict[str, Optional[Dict[str, float]]]:
        """
        For compatibility: force sync and return window metrics.
        """
        window_metrics = self.maybe_sync(global_step, force=True)
        return {"aggregated": window_metrics, "window": window_metrics}

    def flush(self, global_step: int) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Training-end flush: force sync and return window metrics.
        """
        window_metrics = self.maybe_sync(global_step, force=True)
        return {"aggregated": window_metrics, "window": window_metrics}
