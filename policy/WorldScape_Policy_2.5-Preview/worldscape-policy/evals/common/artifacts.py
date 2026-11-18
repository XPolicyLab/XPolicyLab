from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from evals.common.evaluator import EvaluationResult
from evals.common.schemas import LatencySummary


class EvaluationArtifactWriter:
    """Write a stable, tool-friendly evaluation result directory."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        config_format: str = "yaml",
        video_fps: int = 10,
    ) -> None:
        if config_format not in {"yaml", "json"}:
            raise ValueError("config_format must be 'yaml' or 'json'")
        if video_fps < 1:
            raise ValueError("video_fps must be positive")
        self.output_dir = Path(output_dir)
        self.config_format = config_format
        self.video_fps = video_fps

    def write(
        self,
        config: Mapping[str, Any] | Any,
        result: EvaluationResult,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        normalized_config = _jsonable(config)
        self._write_config(normalized_config)
        self._write_episodes(result)
        summary, per_task = _summaries(result)
        _write_json(self.output_dir / "summary.json", summary)
        self._write_per_task(per_task)
        self._write_videos(result)
        return summary

    def _write_config(self, config: Any) -> None:
        if self.config_format == "json":
            _write_json(self.output_dir / "config.json", config)
            return
        path = self.output_dir / "config.yaml"
        try:
            import yaml
        except ImportError:
            # JSON is a strict YAML subset and remains readable as config.yaml.
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        else:
            path.write_text(yaml.safe_dump(config, sort_keys=True))

    def _write_episodes(self, result: EvaluationResult) -> None:
        path = self.output_dir / "episodes.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for episode in result.episodes:
                stream.write(episode.record.to_json(include_steps=True) + "\n")

    def _write_per_task(self, rows: list[dict[str, Any]]) -> None:
        fields = (
            "task_id",
            "episodes",
            "successes",
            "success_rate",
            "completed",
            "failed",
            "mean_subgoal_completion",
            "mean_episode_ms",
            "mean_prediction_ms",
        )
        with (self.output_dir / "per_task.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_videos(self, result: EvaluationResult) -> None:
        episodes = [item for item in result.episodes if item.frames]
        if not episodes:
            return
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError(
                "Writing evaluation videos requires the optional imageio package"
            ) from exc
        directory = self.output_dir / "videos"
        directory.mkdir(exist_ok=True)
        for episode in episodes:
            frames = np.stack([_video_frame(frame) for frame in episode.frames])
            iio.imwrite(
                directory / f"{episode.spec.episode_id}.mp4",
                frames,
                fps=self.video_fps,
            )


def _summaries(
    result: EvaluationResult,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for episode in result.episodes:
        grouped[episode.spec.task.task_id].append(episode.record)
    rows = []
    for task_id in sorted(grouped):
        records = grouped[task_id]
        count = len(records)
        prediction_samples = [
            record.latency.prediction.mean_ms
            for record in records
            if record.latency.prediction.count
        ]
        subgoal_samples = [
            float(record.metrics["subgoal_completion"])
            for record in records
            if "subgoal_completion" in record.metrics
        ]
        rows.append(
            {
                "task_id": task_id,
                "episodes": count,
                "successes": sum(record.success is True for record in records),
                "success_rate": sum(record.success is True for record in records)
                / count,
                "completed": sum(record.status == "completed" for record in records),
                "failed": sum(record.status != "completed" for record in records),
                "mean_subgoal_completion": (
                    sum(subgoal_samples) / len(subgoal_samples)
                    if subgoal_samples
                    else ""
                ),
                "mean_episode_ms": sum(
                    record.latency.episode_ms for record in records
                )
                / count,
                "mean_prediction_ms": (
                    sum(prediction_samples) / len(prediction_samples)
                    if prediction_samples
                    else 0.0
                ),
            }
        )
    episode_count = len(result.episodes)
    step_records = [
        step
        for episode in result.episodes
        for step in episode.record.steps
        if step.status == "completed"
    ]
    summary = {
        "schema_version": 1,
        "artifact_schema": "worldscape-evaluation",
        "episodes": episode_count,
        "tasks": len(grouped),
        "successes": sum(
            episode.record.success is True for episode in result.episodes
        ),
        "success_rate": result.success_rate,
        "completed": sum(
            episode.record.status == "completed" for episode in result.episodes
        ),
        "failed": sum(
            episode.record.status != "completed" for episode in result.episodes
        ),
        "latency": {
            "observation": asdict(
                LatencySummary.from_values(
                    step.latency.observation_ms for step in step_records
                )
            ),
            "prediction": asdict(
                LatencySummary.from_values(
                    step.latency.prediction_ms for step in step_records
                )
            ),
            "execution": asdict(
                LatencySummary.from_values(
                    step.latency.execution_ms for step in step_records
                )
            ),
            "step": asdict(
                LatencySummary.from_values(
                    step.latency.total_ms for step in step_records
                )
            ),
            "episode": asdict(
                LatencySummary.from_values(
                    episode.record.latency.episode_ms
                    for episode in result.episodes
                )
            ),
        },
        "per_task": rows,
        "metrics": _aggregate_metrics(
            episode.record.metrics for episode in result.episodes
        ),
    }
    return summary, rows


def _aggregate_metrics(
    records: Any,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in record.items():
            grouped[str(name)].append(float(value))
    return {
        name: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        for name, values in sorted(grouped.items())
        if values
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _video_frame(value: np.ndarray) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3:
        raise ValueError(f"Video frame must have shape [H,W,C], got {frame.shape}")
    if frame.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Video frame has unsupported channels: {frame.shape}")
    if (
        np.issubdtype(frame.dtype, np.floating)
        and frame.size
        and float(frame.max()) <= 1.0
    ):
        frame = frame * 255.0
    frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return np.ascontiguousarray(frame)
