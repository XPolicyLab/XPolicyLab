"""Logging callbacks: TensorBoard, JSON log, and experiment setup."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from leap.core.callback import Callback
from leap.core.state import TrainingState
from leap.distributed.utils import is_main_process

logger = logging.getLogger(__name__)


class TrainingProgressCallback(Callback):
    """Print training progress to stdout every ``log_interval`` steps.

    Training step logs are also captured by the file handler set up in
    :class:`ExperimentSetupCallback` (``train.log`` records ALL console output).
    """
    # Must run before TensorBoard/JSON callbacks so freshly computed throughput
    # is written at the same global step, rather than one step late.
    priority: int = 40

    def __init__(self, log_interval: int = 1, eta_window_steps: int = 100) -> None:
        if eta_window_steps < 1:
            raise ValueError("eta_window_steps must be at least 1")
        self.log_interval = log_interval
        # ETA should describe the current end-to-end training rate, not be
        # permanently skewed by slow startup, cache construction, or an early
        # transient stall. Keep the most recent BATCH_END-to-BATCH_END samples;
        # each sample includes data loading, DDP synchronization, forward /
        # backward, and optimizer work.
        self.eta_window_steps = int(eta_window_steps)
        self._fit_start_time: float = 0.0
        self._last_step_end_time: float = 0.0
        self._end_to_end_step_times: Deque[float] = deque(
            maxlen=self.eta_window_steps
        )
        self._token_window_start_time: float = 0.0
        self._local_tokens_in_window: int = 0
        self._loss_sums: Dict[str, float] = defaultdict(float)
        self._loss_counts: Dict[str, int] = defaultdict(int)
        self._group_loss_sums: Dict[str, Dict[str, Dict[str, float]]] = {
            "mode": {},
            "source": {},
        }
        self._loss_weights: Optional[tuple[float, float, float]] = None
        self._profile_window: List[Dict[str, Any]] = []
        self._profile_log_file = None

    def _log(self, msg: str) -> None:
        """Log to console and train.log via root logger (rank 0 only)."""
        if not is_main_process():
            return
        logger.info(msg)

    def _gpu_mem_str(self) -> str:
        try:
            import torch
            if not torch.cuda.is_available():
                return ""
            device = torch.cuda.current_device()
            reserved = torch.cuda.memory_reserved(device) / 1024**3
        except Exception:
            return ""
        return f" gpu_mem={reserved:.1f}GiB"

    def fit_start(self, state: TrainingState) -> None:
        self._fit_start_time = time.time()
        self._last_step_end_time = self._fit_start_time
        self._end_to_end_step_times.clear()
        self._token_window_start_time = self._fit_start_time
        self._local_tokens_in_window = 0
        self._loss_sums.clear()
        self._loss_counts.clear()
        self._group_loss_sums = {"mode": {}, "source": {}}
        self._loss_weights = None
        self._profile_window.clear()
        timing_cfg = (
            state.config.get("training", {}).get("step_timing", {})
            if state.config is not None
            else {}
        ) or {}
        if is_main_process() and state.work_dir and timing_cfg.get("enabled", False):
            os.makedirs(state.work_dir, exist_ok=True)
            mode = "a" if state.global_step > 0 else "w"
            self._profile_log_file = open(
                os.path.join(state.work_dir, "timing_log.jsonl"),
                mode,
                encoding="utf-8",
            )
        total = state.max_iters
        self._log(
            f"[LEAP] Training started: max_steps={state.max_steps}, "
            f"max_epochs={state.max_epochs}, "
            f"num_batches={state.num_batches}, "
            f"total_iters={total}"
        )

    @staticmethod
    def _global_token_count(local_token_count: int) -> int:
        """Sum packed tokens across ranks without assuming equal token lengths."""
        try:
            import torch

            dist = torch.distributed
            if not dist.is_available() or not dist.is_initialized():
                return local_token_count
            backend = str(dist.get_backend()).lower()
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if backend == "nccl"
                else torch.device("cpu")
            )
            count = torch.tensor(
                local_token_count, dtype=torch.int64, device=device
            )
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            return int(count.item())
        except Exception:
            # Throughput instrumentation must never stop training. Returning the
            # local count keeps the metric useful for single-process/debug runs.
            logger.exception("Could not all-reduce packed token count; using local count")
            return local_token_count

    def _consume_token_throughput(
        self, state: TrainingState, now: float
    ) -> Optional[float]:
        """Return global packed tokens/s for this logging window, if available."""
        packed_tokens = state.extras.get("_packed_token_count")
        if packed_tokens is None:
            return None
        try:
            packed_tokens = int(packed_tokens)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid packed token count: %r", packed_tokens)
            return None
        if packed_tokens < 0:
            logger.warning("Ignoring negative packed token count: %d", packed_tokens)
            return None

        self._local_tokens_in_window += packed_tokens
        if state.global_step % self.log_interval != 0:
            return None

        if self._token_window_start_time <= 0:
            self._token_window_start_time = now
        elapsed = now - self._token_window_start_time
        global_tokens = self._global_token_count(self._local_tokens_in_window)
        self._local_tokens_in_window = 0
        self._token_window_start_time = now
        if elapsed <= 0:
            return None
        return global_tokens / elapsed

    def _accumulate_world_unified_losses(self, state: TrainingState) -> None:
        statistics = state.extras.get("_world_unified_loss_stats")
        if not statistics:
            return
        try:
            modes = list(statistics["modes"])
            sources = list(statistics["sources"])
            values = statistics["values"]
            if hasattr(values, "detach"):
                # BATCH_END runs after the optimizer, so this does not insert a
                # synchronization between forward and backward.
                values = values.detach().cpu().tolist()
            else:
                values = list(values)
        except (KeyError, TypeError, ValueError):
            logger.warning("Ignoring malformed World Unified loss statistics")
            return
        if not (len(modes) == len(sources) == len(values)) or any(
            len(row) != 6 for row in values
        ):
            logger.warning(
                "Ignoring World Unified loss statistics with invalid [B,6] shape"
            )
            return

        weights = (
            float(statistics.get("lambda_video", 1.0)),
            float(statistics.get("lambda_action", 1.0)),
            float(statistics.get("lambda_subtask", 1.0)),
        )
        if self._loss_weights is None:
            self._loss_weights = weights
        elif self._loss_weights != weights:
            logger.warning(
                "World Unified loss weights changed inside one logging window: "
                "%s -> %s",
                self._loss_weights,
                weights,
            )

        for mode, source, row in zip(modes, sources, values):
            (
                video_sum,
                video_count,
                action_sum,
                action_count,
                subtask_sum,
                subtask_count,
            ) = row
            sample_values = {
                "video_sum": float(video_sum),
                "video_count": int(video_count),
                "action_sum": float(action_sum),
                "action_count": int(action_count),
                "subtask_sum": float(subtask_sum),
                "subtask_count": int(subtask_count),
            }
            if any(
                sample_values[name] < 0
                for name in ("video_count", "action_count", "subtask_count")
            ):
                logger.warning("Ignoring negative World Unified loss counts")
                continue
            for kind, name in (
                ("mode", str(mode)),
                ("source", "<unknown>" if source is None else str(source)),
            ):
                accumulator = self._group_loss_sums[kind].setdefault(
                    name,
                    {
                        "video_sum": 0.0,
                        "video_count": 0,
                        "action_sum": 0.0,
                        "action_count": 0,
                        "subtask_sum": 0.0,
                        "subtask_count": 0,
                        "samples": 0,
                    },
                )
                for key, value in sample_values.items():
                    accumulator[key] += value
                accumulator["samples"] += 1

    def _local_loss_window(self) -> Dict[str, Any]:
        window = {
            "base": {
                key: {"sum": float(value), "count": int(self._loss_counts[key])}
                for key, value in self._loss_sums.items()
                if self._loss_counts[key] > 0
            },
            "groups": {
                kind: {
                    name: dict(values)
                    for name, values in groups.items()
                }
                for kind, groups in self._group_loss_sums.items()
            },
            "loss_weights": self._loss_weights,
        }
        self._loss_sums.clear()
        self._loss_counts.clear()
        self._group_loss_sums = {"mode": {}, "source": {}}
        self._loss_weights = None
        return window

    @staticmethod
    def _gather_loss_windows(local_window: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather one compact logging window; called only at log intervals."""

        try:
            import torch

            dist = torch.distributed
            if not dist.is_available() or not dist.is_initialized():
                return [local_window]
            windows: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
            dist.all_gather_object(windows, local_window)
            return [window for window in windows if window is not None]
        except Exception:
            logger.exception(
                "Could not gather distributed loss statistics; using local window"
            )
            return [local_window]

    @staticmethod
    def _merge_distributed_loss_windows(
        windows: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Merge rank windows into global scalar and effective-element means."""

        base: Dict[str, Dict[str, float]] = {}
        groups: Dict[str, Dict[str, Dict[str, float]]] = {
            "mode": {},
            "source": {},
        }
        loss_weights = (1.0, 1.0, 1.0)
        for window in windows:
            if window.get("loss_weights") is not None:
                loss_weights = tuple(window["loss_weights"])
            for key, values in window.get("base", {}).items():
                accumulator = base.setdefault(key, {"sum": 0.0, "count": 0})
                accumulator["sum"] += float(values["sum"])
                accumulator["count"] += int(values["count"])
            for kind in ("mode", "source"):
                for name, values in window.get("groups", {}).get(kind, {}).items():
                    accumulator = groups[kind].setdefault(
                        name,
                        {
                            "video_sum": 0.0,
                            "video_count": 0,
                            "action_sum": 0.0,
                            "action_count": 0,
                            "subtask_sum": 0.0,
                            "subtask_count": 0,
                            "samples": 0,
                        },
                    )
                    for key in accumulator:
                        accumulator[key] += values.get(key, 0)

        metrics: Dict[str, float] = {}
        for key, values in base.items():
            if values["count"] > 0:
                metrics[f"global/{key}"] = values["sum"] / values["count"]

        lambda_video, lambda_action, lambda_subtask = map(float, loss_weights)
        for kind, named_groups in groups.items():
            for name, values in named_groups.items():
                prefix = f"global/{kind}/{name}"
                total = 0.0
                has_target = False
                if values["video_count"] > 0:
                    video_loss = values["video_sum"] / values["video_count"]
                    metrics[f"{prefix}/loss_video"] = video_loss
                    total += lambda_video * video_loss
                    has_target = True
                if values["action_count"] > 0:
                    action_loss = values["action_sum"] / values["action_count"]
                    metrics[f"{prefix}/loss_action"] = action_loss
                    total += lambda_action * action_loss
                    has_target = True
                if values["subtask_count"] > 0:
                    subtask_loss = (
                        values["subtask_sum"] / values["subtask_count"]
                    )
                    metrics[f"{prefix}/loss_subtask"] = subtask_loss
                    total += lambda_subtask * subtask_loss
                    has_target = True
                if has_target:
                    metrics[f"{prefix}/loss"] = total
                metrics[f"{prefix}/samples"] = float(values["samples"])
                metrics[f"{prefix}/video_dims"] = float(values["video_count"])
                metrics[f"{prefix}/action_dims"] = float(values["action_count"])
                metrics[f"{prefix}/subtask_tokens"] = float(
                    values["subtask_count"]
                )
        return metrics

    def _consume_distributed_losses(self) -> Dict[str, float]:
        local_window = self._local_loss_window()
        return self._merge_distributed_loss_windows(
            self._gather_loss_windows(local_window)
        )

    @staticmethod
    def _summarize_profile_windows(
        windows: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        records = [record for window in windows for record in (window or [])]
        if not records:
            return {}

        steps = sorted({int(record["step"]) for record in records})
        summary: Dict[str, Any] = {
            "step_start": steps[0],
            "step_end": steps[-1],
            "rank_count": len(windows),
            "record_count": len(records),
            "timing": {},
        }

        timing_keys = (
            "data_wait_s",
            "h2d_s",
            "batch_process_s",
            "vae_gpu_s",
            "materialize_gpu_s",
            "lance_gpu_s",
            "forward_gpu_s",
            "backward_gpu_s",
            "optimizer_gpu_s",
        )
        for key in timing_keys:
            values = [float(record[key]) for record in records if key in record]
            if values:
                ordered = sorted(values)
                p95_index = max(0, int(len(ordered) * 0.95) - 1)
                summary["timing"][key] = {
                    "mean": sum(values) / len(values),
                    "p95": ordered[p95_index],
                    "max": ordered[-1],
                }

        by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_step[int(record["step"])].append(record)

        def balance(metric: str) -> Optional[Dict[str, float]]:
            triplets = []
            for step_records in by_step.values():
                values = [
                    float(record[metric])
                    for record in step_records
                    if metric in record
                ]
                if not values:
                    continue
                mean_value = sum(values) / len(values)
                triplets.append(
                    (
                        min(values),
                        mean_value,
                        max(values),
                        max(values) / mean_value if mean_value > 0 else 0.0,
                    )
                )
            if not triplets:
                return None
            return {
                "min_mean": sum(item[0] for item in triplets) / len(triplets),
                "mean": sum(item[1] for item in triplets) / len(triplets),
                "max_mean": sum(item[2] for item in triplets) / len(triplets),
                "imbalance_mean": sum(item[3] for item in triplets) / len(triplets),
                "imbalance_worst": max(item[3] for item in triplets),
            }

        summary["packed_tokens"] = balance("packed_tokens")
        summary["video_pixels"] = balance("video_pixels")

        workload_keys = (
            "samples",
            "packed_tokens",
            "video_pixels",
            "vision_condition_pixels",
            "video_target_pixels",
            "action_steps",
            "action_condition_steps",
            "action_target_steps",
            "text_chars",
        )
        for key in workload_keys:
            if key not in summary:
                summary[key] = balance(key)

        def correlation(left_key: str, right_key: str) -> Optional[float]:
            pairs = [
                (float(record[left_key]), float(record[right_key]))
                for record in records
                if left_key in record and right_key in record
            ]
            if len(pairs) < 2:
                return None
            left_mean = sum(left for left, _ in pairs) / len(pairs)
            right_mean = sum(right for _, right in pairs) / len(pairs)
            numerator = sum(
                (left - left_mean) * (right - right_mean)
                for left, right in pairs
            )
            left_var = sum((left - left_mean) ** 2 for left, _ in pairs)
            right_var = sum((right - right_mean) ** 2 for _, right in pairs)
            if left_var <= 0 or right_var <= 0:
                return None
            return numerator / (left_var * right_var) ** 0.5

        correlation_timing_keys = (
            "vae_gpu_s",
            "materialize_gpu_s",
            "lance_gpu_s",
            "forward_gpu_s",
        )
        correlations: Dict[str, Dict[str, float]] = {}
        for workload_key in workload_keys:
            values = {}
            for timing_key in correlation_timing_keys:
                value = correlation(workload_key, timing_key)
                if value is not None:
                    values[timing_key] = value
            if values:
                correlations[workload_key] = values
        if correlations:
            summary["workload_timing_correlations"] = correlations

        wait_skews = []
        for step_records in by_step.values():
            waits = [
                float(record["data_wait_s"])
                for record in step_records
                if "data_wait_s" in record
            ]
            if waits:
                wait_skews.append(max(waits) - min(waits))
        if wait_skews:
            summary["data_wait_rank_skew_s"] = {
                "mean": sum(wait_skews) / len(wait_skews),
                "max": max(wait_skews),
            }

        # Keep the full per-rank breakdown in timing_log.jsonl.  The console
        # only prints the slowest ranks, but the JSON makes it possible to tell
        # whether the same rank is repeatedly slow or the straggler follows a
        # particular source/shard.
        by_rank: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_rank[int(record.get("rank", 0))].append(record)
        rank_summaries = []
        for rank, rank_records in sorted(by_rank.items()):
            rank_summary: Dict[str, Any] = {
                "rank": rank,
                "records": len(rank_records),
                "timing": {},
                "workload": {},
            }
            for key in timing_keys:
                values = [
                    float(record[key])
                    for record in rank_records
                    if key in record
                ]
                if values:
                    rank_summary["timing"][key] = {
                        "mean": sum(values) / len(values),
                        "max": max(values),
                    }
            for key in workload_keys:
                values = [
                    float(record[key])
                    for record in rank_records
                    if key in record
                ]
                if values:
                    rank_summary["workload"][key] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }
            source_counts: Dict[str, int] = defaultdict(int)
            for record in rank_records:
                for source in record.get("sources", []):
                    source_counts[str(source)] += 1
            rank_summary["source_samples"] = dict(
                sorted(source_counts.items(), key=lambda item: item[0])
            )
            rank_summaries.append(rank_summary)
        summary["ranks"] = rank_summaries

        distinct_source_counts = []
        distinct_shape_counts = []
        for step_records in by_step.values():
            source_signatures = {
                tuple(sorted({str(source) for source in record.get("sources", [])}))
                for record in step_records
            }
            shape_signatures = {
                tuple(tuple(shape) for shape in record.get("video_shapes", []))
                for record in step_records
            }
            distinct_source_counts.append(len(source_signatures))
            distinct_shape_counts.append(len(shape_signatures))
        if distinct_source_counts:
            summary["cross_rank_alignment"] = {
                "source_signatures_mean": (
                    sum(distinct_source_counts) / len(distinct_source_counts)
                ),
                "source_signatures_max": max(distinct_source_counts),
                "shape_signatures_mean": (
                    sum(distinct_shape_counts) / len(distinct_shape_counts)
                ),
                "shape_signatures_max": max(distinct_shape_counts),
                "fully_source_aligned_steps": sum(
                    count == 1 for count in distinct_source_counts
                ),
                "steps": len(distinct_source_counts),
            }

        # This is an association, not exclusive attribution: a mixed-source
        # batch's wait is associated with every source represented in it.
        source_waits: Dict[str, List[float]] = defaultdict(list)
        source_samples: Dict[str, int] = defaultdict(int)
        for record in records:
            sources = [str(source) for source in record.get("sources", [])]
            wait = float(record.get("data_wait_s", 0.0))
            for source in set(sources):
                source_waits[source].append(wait)
            for source in sources:
                source_samples[source] += 1
        source_association = []
        for source, waits in source_waits.items():
            source_association.append(
                {
                    "source": source,
                    "batches": len(waits),
                    "samples": source_samples[source],
                    "data_wait_mean_s": sum(waits) / len(waits),
                    "data_wait_max_s": max(waits),
                }
            )
        summary["source_data_wait_association"] = sorted(
            source_association,
            key=lambda item: (item["data_wait_mean_s"], item["data_wait_max_s"]),
            reverse=True,
        )
        return summary

    def _consume_step_profiles(
        self, state: TrainingState
    ) -> Optional[Dict[str, Any]]:
        profile = state.extras.get("_step_profile")
        if profile is None:
            return None
        self._profile_window.append(dict(profile))
        if state.global_step % self.log_interval != 0:
            return None

        local_window = self._profile_window
        self._profile_window = []
        windows: List[List[Dict[str, Any]]] = [local_window]
        try:
            import torch

            dist = torch.distributed
            if dist.is_available() and dist.is_initialized():
                gathered: List[Optional[List[Dict[str, Any]]]] = [
                    None for _ in range(dist.get_world_size())
                ]
                dist.all_gather_object(gathered, local_window)
                windows = [window or [] for window in gathered]
        except Exception:
            logger.exception(
                "Could not gather distributed step profiles; using local rank"
            )

        if not is_main_process():
            return None
        summary = self._summarize_profile_windows(windows)
        if summary and self._profile_log_file is not None:
            self._profile_log_file.write(json.dumps(summary) + "\n")
            self._profile_log_file.flush()
        return summary

    def _log_profile_summary(self, summary: Dict[str, Any]) -> None:
        if not summary:
            return
        start, end = summary["step_start"], summary["step_end"]
        timing = summary.get("timing", {})

        def pair(key: str) -> str:
            values = timing.get(key)
            if not values:
                return "n/a"
            return f"{values['mean']:.3f}/{values['max']:.3f}s"

        self._log(
            f"[Profile {start}-{end}] mean/max: "
            f"data_wait={pair('data_wait_s')} h2d={pair('h2d_s')} "
            f"vae={pair('vae_gpu_s')} materialize={pair('materialize_gpu_s')} "
            f"lance={pair('lance_gpu_s')} backward={pair('backward_gpu_s')} "
            f"optimizer={pair('optimizer_gpu_s')}"
        )

        tokens = summary.get("packed_tokens")
        pixels = summary.get("video_pixels")
        wait_skew = summary.get("data_wait_rank_skew_s")
        balance_parts = []
        if tokens:
            balance_parts.append(
                "tokens/rank(min/mean/max)="
                f"{tokens['min_mean']:.0f}/{tokens['mean']:.0f}/"
                f"{tokens['max_mean']:.0f} "
                f"imbalance={tokens['imbalance_mean']:.2f}x "
                f"worst={tokens['imbalance_worst']:.2f}x"
            )
        if pixels:
            balance_parts.append(
                "video_Mpix/rank(min/mean/max)="
                f"{pixels['min_mean']/1e6:.1f}/{pixels['mean']/1e6:.1f}/"
                f"{pixels['max_mean']/1e6:.1f} "
                f"worst={pixels['imbalance_worst']:.2f}x"
            )
        if wait_skew:
            balance_parts.append(
                "data_wait_rank_skew(mean/max)="
                f"{wait_skew['mean']:.3f}/{wait_skew['max']:.3f}s"
            )
        if balance_parts:
            self._log(f"[Balance {start}-{end}] " + " | ".join(balance_parts))

        alignment = summary.get("cross_rank_alignment")
        if alignment:
            self._log(
                f"[Rank alignment {start}-{end}] distinct source "
                f"signatures(mean/max)="
                f"{alignment['source_signatures_mean']:.2f}/"
                f"{alignment['source_signatures_max']} | distinct video shapes="
                f"{alignment['shape_signatures_mean']:.2f}/"
                f"{alignment['shape_signatures_max']} | fully aligned="
                f"{alignment['fully_source_aligned_steps']}/"
                f"{alignment['steps']} steps"
            )

        rank_summaries = summary.get("ranks", [])
        ranked_waits = []
        for rank_summary in rank_summaries:
            wait = rank_summary.get("timing", {}).get("data_wait_s")
            if wait:
                ranked_waits.append(
                    (float(wait["max"]), float(wait["mean"]), rank_summary["rank"])
                )
        if ranked_waits:
            ranked_waits.sort(reverse=True)
            slowest = ", ".join(
                f"r{rank}={mean:.3f}/{maximum:.3f}s"
                for maximum, mean, rank in ranked_waits[:3]
            )
            self._log(
                f"[Rank data_wait {start}-{end}] slowest mean/max: {slowest}"
            )

        sources = summary.get("source_data_wait_association", [])[:3]
        if sources:
            source_text = ", ".join(
                f"{item['source']}={item['data_wait_mean_s']:.3f}s"
                f"(max {item['data_wait_max_s']:.3f}, n={item['samples']})"
                for item in sources
            )
            self._log(
                f"[Decode association {start}-{end}] highest batch data_wait: "
                + source_text
            )

    def batch_end(self, state: TrainingState) -> None:
        state.extras.pop("_distributed_loss_metrics", None)
        now = time.time()
        profile_summary = self._consume_step_profiles(state)
        # This is the end-to-end time for the just-completed step: it starts at
        # the previous BATCH_END, so it includes next-batch loading, DDP waits,
        # forward/backward, and the optimizer update. The same measurements
        # drive the rolling ETA below.
        step_time = now - self._last_step_end_time
        self._last_step_end_time = now
        if step_time > 0:
            self._end_to_end_step_times.append(step_time)
        tokens_per_sec = self._consume_token_throughput(state, now)

        step = state.global_step
        for k, v in state.loss_dict.items():
            val = v.item() if hasattr(v, "item") else v
            self._loss_sums[k] += float(val)
            self._loss_counts[k] += 1
        self._accumulate_world_unified_losses(state)

        if step % self.log_interval != 0:
            return

        distributed_losses = self._consume_distributed_losses()
        state.extras["_distributed_loss_metrics"] = distributed_losses

        total = state.max_iters
        if tokens_per_sec is not None:
            state.extras["_last_tokens_per_sec"] = tokens_per_sec
            state.log_scalar("train/tokens_per_sec", tokens_per_sec)

        # Format averaged loss components over the logging interval. Missing
        # components are skipped instead of filled with zero (e.g. fm_mse on
        # VQA-only batches), avoiding artificial TensorBoard/log spikes.
        loss_parts = []
        preferred_keys = (
            "loss",
            "loss_video",
            "loss_depth",
            "loss_normal",
            "loss_flow",
            "loss_action",
            "loss_lm",
            "loss_subtask",
            "loss_affordance",
            "lm_ce",
            "fm_mse",
            "act_ce_log",
            "lang_ce_log",
        )
        averaged_losses = {
            key: distributed_losses[f"global/{key}"]
            for key in state.loss_dict
            if f"global/{key}" in distributed_losses
        }
        logged_keys = set()
        for k in preferred_keys:
            if k in averaged_losses:
                loss_parts.append(f"{k}={averaged_losses[k]:.4f}")
                logged_keys.add(k)
        for k in sorted(averaged_losses):
            if k in logged_keys:
                continue
            loss_parts.append(f"{k}={averaged_losses[k]:.4f}")
        loss_str = " ".join(loss_parts)

        # Speed & ETA
        step_time_str = f"{step_time:.2f}s/step" if step_time > 0 else ""
        remaining_steps = total - step
        # Use the current rolling end-to-end window rather than the lifetime
        # average. This reacts after cache warmup, dataset changes, and stalls,
        # while a 100-step default prevents one slow video/shard from making the
        # estimate jump wildly. During warmup the available prefix is used.
        eta_window_count = len(self._end_to_end_step_times)
        rolling_step_time = (
            sum(self._end_to_end_step_times) / eta_window_count
            if eta_window_count > 0
            else 0
        )
        eta_seconds = int(remaining_steps * rolling_step_time)
        eta_h, eta_m, eta_s = eta_seconds // 3600, (eta_seconds % 3600) // 60, eta_seconds % 60
        eta_str = f"ETA {eta_h}:{eta_m:02d}:{eta_s:02d}"

        # Grad norm
        grad_str = ""
        if state.grad_norm is not None:
            gn = state.grad_norm.item() if hasattr(state.grad_norm, "item") else state.grad_norm
            grad_str = f" grad_norm={gn:.3f}"

        # GPU memory
        gpu_mem_str = self._gpu_mem_str()

        # Extra scalars from ThroughputCallback etc.
        extra_parts = []
        for k, v in state.get_scalars().items():
            if k == "train/tokens_per_sec":
                continue
            short_key = k.rsplit("/", 1)[-1]
            extra_parts.append(f"{short_key}={v:.1f}")
        extra_str = (" " + " ".join(extra_parts)) if extra_parts else ""
        token_str = (
            f" tokens/s={tokens_per_sec:.0f}"
            if tokens_per_sec is not None
            else ""
        )

        width = len(str(total))
        msg = (
            f"[Step {step:>{width}}/{total}] "
            f"{loss_str} lr={state.lr:.2e}{grad_str}{gpu_mem_str}{extra_str}"
            f" | {step_time_str}{token_str} {eta_str}"
        )
        self._log(msg)

        window_start = max(1, step - self.log_interval + 1)
        mode_prefix = "global/mode/"
        mode_losses = {
            key[len(mode_prefix) : -len("/loss")]: value
            for key, value in distributed_losses.items()
            if key.startswith(mode_prefix) and key.endswith("/loss")
        }
        if mode_losses:
            mode_order = ("forward_dynamics", "inverse_dynamics", "policy")
            names = [name for name in mode_order if name in mode_losses]
            names.extend(sorted(set(mode_losses) - set(names)))
            self._log(
                f"[Global mode loss {window_start}-{step}] "
                + " ".join(f"{name}={mode_losses[name]:.4f}" for name in names)
            )

        source_prefix = "global/source/"
        source_losses = {
            key[len(source_prefix) : -len("/loss")]: value
            for key, value in distributed_losses.items()
            if key.startswith(source_prefix) and key.endswith("/loss")
        }
        if source_losses:
            ranked_sources = sorted(
                source_losses,
                key=lambda name: (
                    -distributed_losses.get(
                        f"{source_prefix}{name}/samples", 0.0
                    ),
                    name,
                ),
            )
            shown = ranked_sources[:8]
            suffix = (
                f" (+{len(ranked_sources) - len(shown)} more in JSON/TensorBoard)"
                if len(shown) < len(ranked_sources)
                else ""
            )
            self._log(
                f"[Global source loss {window_start}-{step}] "
                + " ".join(f"{name}={source_losses[name]:.4f}" for name in shown)
                + suffix
            )
        if profile_summary is not None:
            self._log_profile_summary(profile_summary)

    def fit_end(self, state: TrainingState) -> None:
        total_time = time.time() - self._fit_start_time
        steps = state.global_step
        avg = total_time / steps if steps > 0 else 0
        self._log(
            f"[LEAP] Training finished: {steps} steps in {total_time:.1f}s "
            f"({avg:.2f}s/step)"
        )
        if self._profile_log_file is not None:
            self._profile_log_file.close()
            self._profile_log_file = None


class ExperimentSetupCallback(Callback):
    """Setup experiment: create work_dir, save metadata, print header.

    Also attaches a FileHandler to the root logger so that ALL log output
    (from any module) is captured in ``<work_dir>/train.log``.

    Runs at FIT_START with highest priority.
    """
    priority: int = -100

    def __init__(self, experiment_name: Optional[str] = None) -> None:
        self.experiment_name = experiment_name
        self._file_handler: Optional[logging.FileHandler] = None

    def fit_start(self, state: TrainingState) -> None:
        if not state.work_dir or not is_main_process():
            return
        self._setup_file_logging(state)
        self._save_metadata(state)
        self._save_config_snapshot(state)
        self._print_experiment_header(state)

    def fit_end(self, state: TrainingState) -> None:
        if self._file_handler is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def _setup_file_logging(self, state: TrainingState) -> None:
        """Attach a FileHandler to root logger so ALL log output goes to train.log."""
        os.makedirs(state.work_dir, exist_ok=True)
        log_path = os.path.join(state.work_dir, "train.log")
        self._file_handler = logging.FileHandler(log_path, mode="a")
        self._file_handler.setLevel(logging.INFO)
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(self._file_handler)

    def _collect_metadata(self, state: TrainingState) -> Dict[str, Any]:
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
        }
        # Git info
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        try:
            metadata["git_hash"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            metadata["git_branch"] = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            metadata["git_dirty"] = bool(git_status)
            if metadata["git_dirty"]:
                metadata["git_diff"] = subprocess.check_output(
                    ["git", "diff"],
                    cwd=repo_root,
                    stderr=subprocess.DEVNULL,
                ).decode()
        except (subprocess.CalledProcessError, FileNotFoundError):
            metadata["git_hash"] = "unknown"
            metadata["git_branch"] = "unknown"
            metadata["git_dirty"] = False
        # Environment
        import torch
        metadata["environment"] = {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", "N/A"),
        }
        try:
            if torch.cuda.is_available():
                metadata["environment"]["gpu"] = torch.cuda.get_device_name(0)
                metadata["environment"]["nproc"] = torch.cuda.device_count()
        except Exception:
            pass
        try:
            import transformers
            metadata["environment"]["transformers"] = transformers.__version__
        except ImportError:
            pass
        # Config snapshot
        if state.config is not None:
            from omegaconf import OmegaConf
            metadata["config"] = OmegaConf.to_container(state.config, resolve=True)
        return metadata

    def _save_metadata(self, state: TrainingState) -> None:
        metadata = self._collect_metadata(state)
        path = os.path.join(state.work_dir, "metadata.json")
        os.makedirs(state.work_dir, exist_ok=True)

        # Save git diff to a separate file to keep metadata.json readable
        git_diff = metadata.pop("git_diff", None)
        if git_diff:
            diff_path = os.path.join(state.work_dir, "git_diff.patch")
            with open(diff_path, "w") as f:
                f.write(git_diff)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _save_config_snapshot(self, state: TrainingState) -> None:
        if state.config is None:
            return
        from omegaconf import OmegaConf
        path = os.path.join(state.work_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(OmegaConf.to_yaml(state.config, resolve=True))

    def _print_experiment_header(self, state: TrainingState) -> None:
        metadata = self._collect_metadata(state)
        env = metadata.get("environment", {})

        # Config YAML
        config_str = ""
        if state.config is not None:
            from omegaconf import OmegaConf
            config_str = OmegaConf.to_yaml(state.config, resolve=True)

        header = f"""
================================================================
Experiment: {os.path.basename(state.work_dir)}
================================================================
Command:     {metadata.get('command', 'N/A')}
Git Hash:    {metadata.get('git_hash', 'N/A')}
Git Branch:  {metadata.get('git_branch', 'N/A')}
Git Dirty:   {metadata.get('git_dirty', 'N/A')}{' (diff saved to git_diff.patch)' if metadata.get('git_dirty') else ''}
----------------------------------------------------------------
Environment:
  Python:         {env.get('python', 'N/A')}
  PyTorch:        {env.get('pytorch', 'N/A')}
  CUDA:           {env.get('cuda', 'N/A')}
  GPU:            {env.get('gpu', 'N/A')}
  Transformers:   {env.get('transformers', 'N/A')}
----------------------------------------------------------------
Config:
{config_str}================================================================
"""
        print(header, flush=True)


class TensorBoardCallback(Callback):
    """Log scalars to TensorBoard (rank 0 only)."""
    priority: int = 50

    def __init__(self, log_dir: Optional[str] = None, log_interval: int = 10) -> None:
        self.log_dir = log_dir
        self.log_interval = log_interval
        self._writer = None
        self._loss_sums: Dict[str, float] = defaultdict(float)
        self._loss_counts: Dict[str, int] = defaultdict(int)

    def fit_start(self, state: TrainingState) -> None:
        if not is_main_process():
            return
        log_dir = self.log_dir or state.work_dir
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning("tensorboard not installed, TensorBoardCallback disabled")

    def batch_end(self, state: TrainingState) -> None:
        if self._writer is None:
            return
        for key, val in state.loss_dict.items():
            if hasattr(val, "item"):
                self._loss_sums[key] += float(val.item())
                self._loss_counts[key] += 1
        if state.global_step % self.log_interval != 0:
            return
        step = state.global_step
        # Log interval-averaged losses.
        for key in sorted(self._loss_sums.keys()):
            count = self._loss_counts[key]
            if count > 0:
                self._writer.add_scalar(f"train/{key}", self._loss_sums[key] / count, step)
        self._loss_sums.clear()
        self._loss_counts.clear()
        # Log LR and grad norm
        self._writer.add_scalar("train/lr", state.lr, step)
        if state.grad_norm is not None:
            gn = state.grad_norm.item() if hasattr(state.grad_norm, "item") else state.grad_norm
            self._writer.add_scalar("train/grad_norm", gn, step)
        for key, value in state.extras.get(
            "_distributed_loss_metrics", {}
        ).items():
            self._writer.add_scalar(f"train/{key}", value, step)
        # Log user-defined scalars
        for key, val in state.get_scalars().items():
            self._writer.add_scalar(key, val, step)
        state.clear_scalars()

    def fit_end(self, state: TrainingState) -> None:
        if self._writer:
            self._writer.close()


class JsonLogCallback(Callback):
    """Log training metrics to a JSON lines file."""
    priority: int = 50

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = log_interval
        self._log_file = None
        self._loss_sums: Dict[str, float] = defaultdict(float)
        self._loss_counts: Dict[str, int] = defaultdict(int)

    def fit_start(self, state: TrainingState) -> None:
        if state.work_dir and is_main_process():
            path = os.path.join(state.work_dir, "train_log.jsonl")
            self._log_file = open(path, "a")

    def batch_end(self, state: TrainingState) -> None:
        if self._log_file is None:
            return
        for key, val in state.loss_dict.items():
            if hasattr(val, "item"):
                self._loss_sums[key] += float(val.item())
                self._loss_counts[key] += 1
        if state.global_step % self.log_interval != 0:
            return
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "lr": state.lr,
        }
        for key in sorted(self._loss_sums.keys()):
            count = self._loss_counts[key]
            if count > 0:
                record[key] = self._loss_sums[key] / count
        self._loss_sums.clear()
        self._loss_counts.clear()
        if state.grad_norm is not None:
            gn = state.grad_norm
            record["grad_norm"] = gn.item() if hasattr(gn, "item") else gn
        tokens_per_sec = state.extras.get("_last_tokens_per_sec")
        if tokens_per_sec is not None:
            record["tokens_per_sec"] = float(tokens_per_sec)
        record.update(
            state.extras.get("_distributed_loss_metrics", {})
        )
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def fit_end(self, state: TrainingState) -> None:
        if self._log_file:
            self._log_file.close()


class WandbCallback(Callback):
    """Optional Weights & Biases logging callback (rank 0 only)."""
    priority: int = 50

    def __init__(self, project: str = "leap", log_interval: int = 10, **kwargs: Any) -> None:
        self.project = project
        self.log_interval = log_interval
        self.kwargs = kwargs
        self._run = None

    def fit_start(self, state: TrainingState) -> None:
        if not is_main_process():
            return
        try:
            import wandb
            from omegaconf import OmegaConf
            config = OmegaConf.to_container(state.config, resolve=True) if state.config else {}
            self._run = wandb.init(project=self.project, config=config, **self.kwargs)
        except ImportError:
            logger.warning("wandb not installed, WandbCallback disabled")

    def batch_end(self, state: TrainingState) -> None:
        if self._run is None:
            return
        if state.global_step % self.log_interval != 0:
            return
        import wandb
        log_dict = {"train/lr": state.lr, "step": state.global_step}
        for key, val in state.loss_dict.items():
            log_dict[f"train/{key}"] = val.item() if hasattr(val, "item") else val
        log_dict.update(
            {f"train/{key}": value for key, value in state.extras.get(
                "_distributed_loss_metrics", {}
            ).items()}
        )
        wandb.log(log_dict, step=state.global_step)

    def fit_end(self, state: TrainingState) -> None:
        if self._run:
            import wandb
            wandb.finish()
