"""Compute normalization statistics and plot per-dimension data distributions."""

from collections.abc import Sequence
import dataclasses
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config

if __package__:
    from scripts import compute_norm_stats as _norm_stats
else:
    import compute_norm_stats as _norm_stats


plt.switch_backend("Agg")
_KEYS = ("state", "actions")
_NUM_PLOT_BINS = 100
_DEFAULT_OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "experimental_results/action_distribution_figure"


@dataclasses.dataclass(frozen=True)
class Histogram:
    counts: np.ndarray
    bin_edges: np.ndarray


def aggregate_histogram(counts: np.ndarray, bin_edges: np.ndarray, num_bins: int) -> Histogram:
    """Aggregate an existing histogram without revisiting the source samples."""
    counts = np.asarray(counts, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if counts.ndim != 1 or bin_edges.shape != (counts.size + 1,):
        raise ValueError("Histogram counts and bin edges have incompatible shapes.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if counts.size <= num_bins:
        return Histogram(counts=counts.copy(), bin_edges=bin_edges.copy())

    if counts.size % num_bins == 0:
        group_size = counts.size // num_bins
        aggregated_counts = counts.reshape(num_bins, group_size).sum(axis=1)
        aggregated_edges = np.concatenate((bin_edges[:-1:group_size], bin_edges[-1:]))
    else:
        aggregated_edges = np.linspace(bin_edges[0], bin_edges[-1], num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        aggregated_counts, _ = np.histogram(bin_centers, bins=aggregated_edges, weights=counts)

    return Histogram(counts=aggregated_counts, bin_edges=aggregated_edges)


class _HistogramExportMixin:
    def _expected_histogram_totals(self, vector_length: int) -> np.ndarray:
        raise NotImplementedError

    def get_histograms(self, num_bins: int = _NUM_PLOT_BINS) -> list[Histogram]:
        source_histograms = self._histograms  # type: ignore[attr-defined]
        source_bin_edges = self._bin_edges  # type: ignore[attr-defined]
        if source_histograms is None or source_bin_edges is None:
            raise ValueError("Cannot export histograms before any data has been processed.")

        active_length = next(
            (index for index, histogram in enumerate(source_histograms) if histogram is None),
            len(source_histograms),
        )
        expected_totals = self._expected_histogram_totals(active_length)
        histograms = []
        for index, (source_counts, bin_edges) in enumerate(
            zip(source_histograms[:active_length], source_bin_edges[:active_length], strict=True)
        ):
            assert source_counts is not None
            assert bin_edges is not None
            counts = np.asarray(source_counts, dtype=np.float64).copy()

            # RunningStats may lose the first approximate bin while expanding its
            # range. Repair only the plotting copy so norm-stat behavior is unchanged.
            missing_weight = float(expected_totals[index] - np.sum(counts))
            tolerance = max(1.0, abs(float(expected_totals[index]))) * 1e-10
            if missing_weight > tolerance:
                counts[0] += missing_weight

            histograms.append(aggregate_histogram(counts, bin_edges, num_bins))
        return histograms


class DistributionRunningStats(_HistogramExportMixin, normalize.RunningStats):
    def _expected_histogram_totals(self, vector_length: int) -> np.ndarray:
        return np.full(vector_length, self._count, dtype=np.float64)


class DistributionWeightedRunningStats(_HistogramExportMixin, _norm_stats.WeightedRunningStats):
    def _expected_histogram_totals(self, vector_length: int) -> np.ndarray:
        return self._weight_sum[:vector_length].copy()


def _compute_unweighted_stats(data_loader, num_batches: int) -> dict[str, DistributionRunningStats]:
    stats = {key: DistributionRunningStats() for key in _KEYS}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in _KEYS:
            stats[key].update(np.asarray(batch[key]))
    return stats


def _create_single_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, int]:
    dataset = _norm_stats.create_mixed_component_dataset(data_config, action_horizon, model_config)
    dataset_length = len(dataset)
    if dataset_length == 0:
        raise ValueError(f"Dataset '{data_config.repo_id}' is empty.")

    quota = dataset_length if max_frames is None else min(dataset_length, max_frames)
    if quota <= 0:
        raise ValueError("max_frames must allow at least one sample.")

    selected_dataset = dataset
    if quota < dataset_length:
        indices = _norm_stats.select_sample_indices(dataset_length, quota, seed, component_index=0)
        selected_dataset = torch.utils.data.Subset(dataset, indices.tolist())

    effective_batch_size = min(batch_size, quota)
    data_loader = _norm_stats.create_mixed_component_dataloader(
        selected_dataset,
        effective_batch_size,
        num_workers,
    )
    return data_loader, math.ceil(quota / effective_batch_size)


def _compute_mixed_torch_stats(
    data_config: _config.MixedDataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    seed: int = 0,
) -> dict[str, DistributionWeightedRunningStats]:
    stats = {key: DistributionWeightedRunningStats() for key in _KEYS}
    batches = _norm_stats.iter_mixed_component_batches(
        data_config,
        action_horizon,
        batch_size,
        model_config,
        num_workers,
        max_frames=max_frames,
        seed=seed,
    )
    for component_batch in batches:
        for key in _KEYS:
            stats[key].update(
                np.asarray(component_batch.batch[key]),
                sample_weight=component_batch.plan.per_sample_weight,
            )

    return stats


def plot_distribution(
    key: str,
    histograms: Sequence[Histogram],
    norm_stats: normalize.NormStats,
    output_path: pathlib.Path,
) -> None:
    if not histograms:
        raise ValueError(f"Cannot plot an empty {key} distribution.")

    num_columns = min(4, len(histograms))
    num_rows = math.ceil(len(histograms) / num_columns)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4.2 * num_columns, 3.2 * num_rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    for index, histogram in enumerate(histograms):
        axis = flat_axes[index]
        axis.stairs(histogram.counts, histogram.bin_edges, fill=True, color="#3478b8", alpha=0.75)
        axis.axvline(norm_stats.mean[index], color="#c43c35", linewidth=1.4, label="mean")
        if norm_stats.q01 is not None:
            axis.axvline(norm_stats.q01[index], color="#3d7f43", linewidth=1.1, linestyle="--", label="q01")
        if norm_stats.q99 is not None:
            axis.axvline(norm_stats.q99[index], color="#7651a8", linewidth=1.1, linestyle="--", label="q99")
        axis.set_title(f"dim {index}")
        axis.set_xlabel("Value")
        axis.set_ylabel("Weighted count")
        axis.set_xlim(histogram.bin_edges[0], histogram.bin_edges[-1])
        axis.grid(axis="y", alpha=0.2)

    for axis in flat_axes[len(histograms) :]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(labels))
    figure.suptitle(f"{key.capitalize()} distribution")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(
    config_name: str,
    max_frames: int | None = None,
    seed: int = 0,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_dir: pathlib.Path = _DEFAULT_OUTPUT_DIR,
) -> None:
    config = _config.get_config(config_name)
    effective_batch_size = config.batch_size if batch_size is None else batch_size
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    data_config = config.data.create(config.assets_dirs, config.model)
    effective_num_workers = config.num_workers if num_workers is None else num_workers
    if effective_num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    if data_config.norm_stats is None:
        print("No existing norm_stats.json found; computing statistics from the transformed dataset.")
    else:
        print("Existing norm_stats.json found; recomputing statistics from the transformed dataset.")

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = _norm_stats.create_rlds_dataloader(
            data_config,
            config.model.action_horizon,
            effective_batch_size,
            max_frames,
        )
        accumulators = _compute_unweighted_stats(data_loader, num_batches)
    elif isinstance(data_config, _config.MixedDataConfig):
        accumulators = _compute_mixed_torch_stats(
            data_config,
            config.model.action_horizon,
            effective_batch_size,
            config.model,
            effective_num_workers,
            max_frames=max_frames,
            seed=seed,
        )
    else:
        data_loader, num_batches = _create_single_torch_dataloader(
            data_config,
            config.model.action_horizon,
            effective_batch_size,
            config.model,
            effective_num_workers,
            max_frames,
            seed,
        )
        accumulators = _compute_unweighted_stats(data_loader, num_batches)

    norm_stats = {key: accumulators[key].get_statistics() for key in _KEYS}
    norm_stats_output_path = _norm_stats.resolve_output_path(config, data_config)
    print(f"Writing stats to: {norm_stats_output_path}")
    normalize.save(norm_stats_output_path, norm_stats)

    safe_config_name = config_name.replace("/", "_").replace("\\", "_")
    for key in _KEYS:
        figure_path = output_dir / f"{safe_config_name}_{key}_distribution.png"
        plot_distribution(key, accumulators[key].get_histograms(), norm_stats[key], figure_path)
        print(f"Writing {key} distribution to: {figure_path}")


if __name__ == "__main__":
    tyro.cli(main)
