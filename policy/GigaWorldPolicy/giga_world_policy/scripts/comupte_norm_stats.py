import pathlib
from typing import Any

import numpy as np
import numpydantic
import pydantic
import tyro
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from world_action_model.datasets.lerobot_dataset import LeRobotDataset


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None
    q99: numpydantic.NDArray | None = None


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1) for i in range(vector_length)]
        else:
            if vector_length != self._mean.size:
                raise ValueError('The length of new vectors does not match the initialized vector length.')
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError('Cannot compute statistics for less than 2 vectors.')

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


class TransformDataset(Dataset):
    def __init__(self, dataset, data_transforms, return_keys):
        self.dataset = dataset
        self.data_transforms = data_transforms
        self.return_keys = return_keys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        for transform in self.data_transforms:
            data = transform(data)

        result = {}
        for key in self.return_keys:
            values = np.asarray(data[key], dtype=np.float64)
            result[key] = values.reshape(-1, values.shape[-1])
        return result


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


class DeltaActions:
    """Convert absolute actions to delta (action - state) for masked dimensions."""

    def __init__(self, delta_mask: list[bool] | None):
        self.delta_mask = np.asarray(delta_mask, dtype=bool) if delta_mask is not None else None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.delta_mask is None:
            return data
        action = np.asarray(data["action"], dtype=np.float64)
        state = np.asarray(data["observation.state"], dtype=np.float64)
        if state.ndim == 2:
            state = state[0]
        d = min(len(self.delta_mask), action.shape[-1], len(state))
        mask = self.delta_mask[:d]
        if action.ndim == 2:
            for t in range(action.shape[0]):
                action[t, :d][mask] -= state[:d][mask]
        elif action.ndim == 1:
            action[:d][mask] -= state[:d][mask]
        data["action"] = action
        return data


class PadStatesAndActions:
    """Pad or truncate state and action to their respective target dimensions."""

    def __init__(self, action_dim: int, state_dim: int | None = None):
        self.action_dim = action_dim
        self.state_dim = state_dim if state_dim is not None else action_dim

    def _pad_or_truncate(self, arr: np.ndarray, target_dim: int) -> np.ndarray:
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cur = arr.shape[-1]
        if cur < target_dim:
            pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_dim - cur)]
            arr = np.pad(arr, pad_width, constant_values=0.0)
        elif cur > target_dim:
            arr = arr[..., :target_dim]
        return arr

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data["action"] = self._pad_or_truncate(
            np.asarray(data["action"], dtype=np.float64), self.action_dim
        )
        data["observation.state"] = self._pad_or_truncate(
            np.asarray(data["observation.state"], dtype=np.float64), self.state_dim
        )
        return data


def compute_norm_stats(
    data_paths: list[str],
    output_path: str | pathlib.Path,
    delta_mask: list[bool],
    sample_rate: float = 1.0,
    action_chunk: int = 50,
    action_dim: int = 32,
    state_dim: int | None = None,
    num_workers: int = 64,
) -> None:
    """Compute normalization statistics and write them to JSON.

    Loads dataset(s), applies delta action conversion and padding, accumulates
    running statistics for states and actions, and writes the results to
    ``output_path``.

    Args:
        data_paths: List of dataset paths (LeRobot v2 format) to process.
        output_path: Destination file path for the computed norm stats JSON.
        delta_mask: Per-dimension mask (True=delta, False=absolute).
        sample_rate: Fraction of samples to process, in the range [0, 1].
        action_chunk: Temporal window size for action chunks.
        action_dim: Expected action dimensionality used for padding.
        state_dim: Expected state dimensionality (defaults to action_dim).
        num_workers: Number of PyTorch DataLoader worker processes to use.
    """
    if state_dim is None:
        state_dim = action_dim

    datasets = []
    for data_path in data_paths:
        ds = LeRobotDataset(
            data_path=data_path,
            delta_info=dict(action=action_chunk),
        )
        datasets.append(ds)
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    data_transforms = [
        DeltaActions(delta_mask=delta_mask),
        PadStatesAndActions(action_dim=action_dim, state_dim=state_dim),
    ]

    keys = ['observation.state', 'action']
    stats = {key: RunningStats() for key in keys}

    num_frames = int(sample_rate * len(dataset))

    transform_dataset = TransformDataset(dataset, data_transforms, keys)
    dataloader = DataLoader(
        transform_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=num_frames):
        if batch_idx >= num_frames:
            break
        for key in keys:
            stats[key].update(batch_data[key][0].numpy())

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f'Writing stats to: {output_path}')
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_json(norm_stats))


if __name__ == '__main__':
    tyro.cli(compute_norm_stats)