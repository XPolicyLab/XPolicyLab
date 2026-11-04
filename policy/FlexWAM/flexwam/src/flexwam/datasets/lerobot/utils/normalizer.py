from typing import Literal, Dict, Annotated, Union, Any, List, Tuple, Optional
import os
import torch
import json
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import hashlib
from pathlib import Path
from git import Repo
from flexwam.utils.logging_config import get_logger

from flexwam.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[str, "format: 'const_min/const_max', where const_min and const_max give the constant range"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score", "identity"], ConstConstStr, List[str]]


class RunningQuantileStats:
    """Streaming mean/std and 1%/99% quantiles, matching OpenPI `RunningStats`.

    All non-feature dims are flattened so every vector is weighted equally by
    count. Mean/std use the population moments E[x] and E[x^2]; quantiles are
    read from 5000-bin histograms over the running min/max.
    """

    _num_quantile_bins = 5000

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None

    def update(self, batch: torch.Tensor | np.ndarray) -> None:
        if isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().float().numpy()
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim < 1:
            raise ValueError(f"Expected at least a 1D batch, got shape {batch.shape}.")
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if num_elements == 0:
            return

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._min.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
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
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)
        self._update_histograms(batch)

    def get_mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0.0, variance))
        return (
            torch.as_tensor(self._mean, dtype=torch.float32),
            torch.as_tensor(stddev, dtype=torch.float32),
        )

    def get_quantiles(self, quantiles=(0.01, 0.99)) -> List[torch.Tensor]:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                idx = min(int(idx), len(edges) - 1)
                q_values.append(edges[idx])
            results.append(torch.as_tensor(q_values, dtype=torch.float32))
        return results

    def _adjust_histograms(self):
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

class LinearNormalizer:
    def __init__(
            self, 
            shape_meta, 
            use_stepwise_action_norm,
            default_mode: NormMode, 
            exception_mode: Dict[str, Dict[str, NormMode]], 
            stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
            clip_to_unit_range: bool = False,
        ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats
        self.clip_to_unit_range = bool(clip_to_unit_range)

        for meta in shape_meta["action"]:
            key = meta["key"]
            
            if use_stepwise_action_norm:
                cur_stats = {k.removeprefix("stepwise_"): v for k, v in stats["action"][key].items() if k.startswith("stepwise_")}
            else:
                cur_stats = {k.removeprefix("global_"): v for k, v in stats["action"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "action" in exception_mode and key in exception_mode["action"]:
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode

            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats, 
                mode=cur_mode,
                clip_to_unit_range=self.clip_to_unit_range,
            )

        for meta in shape_meta["state"]:
            key = meta["key"]
            cur_stats = {k.removeprefix("global_"): v for k, v in stats["state"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "state" in exception_mode and key in exception_mode["state"]:
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode

            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats, 
                mode=cur_mode,
                clip_to_unit_range=self.clip_to_unit_range,
            )

    def get_stats(self):
        stats = {
            "action": {key: norm.get_stats() for key, norm in self.normalizers["action"].items()},
            "state": {key: norm.get_stats() for key, norm in self.normalizers["state"].items()}
        }
        return stats
                
    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                batch["action"][key] = norm.forward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])

        return batch
    
    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for key, norm in self.normalizers["action"].items():
            batch["action"][key] = norm.backward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.backward(batch["state"][key])
        
        return batch


def _mode_maps_to_unit_range(mode: str) -> bool:
    if mode in {"q01/q99", "min/max"}:
        return True
    if mode in {"z-score", "identity"}:
        return False
    if "/" in str(mode):
        return True
    return False


class SingleFieldLinearNormalizer:
    std_reg = 1e-6  # Match OpenPI z-score: (x - mean) / (std + 1e-6)
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0
    def __init__(self, stats, mode: NormMode="min/max", clip_to_unit_range: bool = False):
        self.stats = stats
        self.mode = mode
        self.clip_mask = None
        self.clip_to_unit_range = False

        if isinstance(mode, (list, tuple, ListConfig)):
            mode = list(mode)
            if "min" in stats:
                ref = stats["min"]
            elif "mean" in stats:
                ref = stats["mean"]
            else:
                raise ValueError("Dimwise normalizer requires stats containing `min` or `mean`.")
            if ref.shape[-1] != len(mode):
                raise ValueError(
                    f"Dimwise normalizer mode length {len(mode)} does not match stats dim {ref.shape[-1]}."
                )
            scale = torch.empty_like(ref)
            offset = torch.empty_like(ref)
            clip_mask = torch.zeros_like(ref, dtype=torch.bool)
            for dim_idx, dim_mode in enumerate(mode):
                dim_stats = {
                    key: value[..., dim_idx]
                    for key, value in stats.items()
                    if isinstance(value, torch.Tensor)
                }
                dim_normalizer = SingleFieldLinearNormalizer(
                    dim_stats,
                    dim_mode,
                    clip_to_unit_range=clip_to_unit_range,
                )
                scale[..., dim_idx] = dim_normalizer.scale
                offset[..., dim_idx] = dim_normalizer.offset
                clip_mask[..., dim_idx] = bool(dim_normalizer.clip_to_unit_range)
            self.scale = scale
            self.offset = offset
            self.clip_mask = clip_mask if bool(clip_mask.any().item()) else None
            return

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = - input_mean / (input_std + self.std_reg)
        elif mode == "identity":
            if "min" in stats:
                ref = stats["min"]
            elif "mean" in stats:
                ref = stats["mean"]
            else:
                raise ValueError("Identity normalizer requires stats containing `min` or `mean`.")
            scale = torch.ones_like(ref)
            offset = torch.zeros_like(ref)
        elif mode == "q01/q99":
            # Match OpenPI quantile: (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
            input_min, input_max = stats["q01"], stats["q99"]
            input_range = input_max - input_min
            scale = (self.output_max - self.output_min) / (input_range + self.std_reg)
            offset = self.output_min - scale * input_min
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            else:
                # parse const_min/const_max
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset
        self.clip_to_unit_range = bool(clip_to_unit_range) and _mode_maps_to_unit_range(str(mode))
    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.scale + self.offset
        if self.clip_mask is not None:
            clip = self.clip_mask.to(device=y.device)
            y = torch.where(clip, y.clamp(self.output_min, self.output_max), y)
        elif self.clip_to_unit_range:
            y = y.clamp(self.output_min, self.output_max)
        return y
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) / self.scale
        return x

def wait_for_dataset_stats_json(file_path: str, *, timeout_s: float = 7200.0, poll_s: float = 0.25) -> None:
    """Block until ``file_path`` contains valid JSON (multi-GPU dataset build sync)."""
    import time

    path = Path(file_path)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            try:
                with path.open("r", encoding="utf-8") as f:
                    json.load(f)
                return
            except json.JSONDecodeError:
                pass
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for dataset stats JSON: {path}")


def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_stats = convert_tensor(dataset_stats)

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

def load_dataset_stats_from_json(file_path: str, 
                                 try_convert_tensor: bool = True) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True  
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data


def search_dataset_stats_cache_json(cache_dir: str | Path, data_config: DictConfig) -> Tuple[bool, str | None]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_git_hash() -> Optional[str]:
        repo = Repo(__file__, search_parent_directories=True)
        return repo.head.commit.hexsha

    def to_plain(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def normalize_str_list(value: Any) -> List[str]:
        value = to_plain(value)
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in value if item is not None]

    def normalize_transforms(value: Any) -> Any:
        value = to_plain(value)
        if isinstance(value, dict):
            return [value]
        return value

    def normalize_dataset_dirs(cfg: DictConfig) -> Any:
        dataset_cfg = cfg.get("dataset")
        if dataset_cfg is None:
            return None
        embodiment_datasets = dataset_cfg.get("embodiment_datasets")
        if embodiment_datasets is not None:
            emb_dirs: Dict[str, List[str]] = {}
            for emb, emb_cfg in embodiment_datasets.items():
                dataset_groups = emb_cfg.get("dataset_groups")
                if dataset_groups is None:
                    emb_dirs[emb] = []
                    continue
                dirs: List[str] = []
                for group in dataset_groups:
                    group_dirs = group.get("dataset_dirs")
                    if group_dirs is None:
                        continue
                    dirs.extend(normalize_str_list(group_dirs))
                emb_dirs[emb] = sorted(dirs)
            return emb_dirs

        dataset_dirs = dataset_cfg.get("dataset_dirs")
        return sorted(normalize_str_list(dataset_dirs))

    def normalize_action_state_transforms(cfg: DictConfig) -> Any:
        processor_cfg = cfg.get("processor")
        if processor_cfg is None:
            return None
        embodiment_processors = processor_cfg.get("embodiment_processors")
        if embodiment_processors is not None:
            emb_transforms: Dict[str, Any] = {}
            for emb, emb_cfg in embodiment_processors.items():
                transforms = emb_cfg.get("action_state_transforms")
                emb_transforms[emb] = normalize_transforms(transforms)
            return emb_transforms

        transforms = processor_cfg.get("action_state_transforms")
        return normalize_transforms(transforms)

    signature = {
        "action_size": data_config.dataset.action_size, 
        "dataset_dirs": normalize_dataset_dirs(data_config),
        "action_state_transforms": normalize_action_state_transforms(data_config),
    }
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    dataset_hash = hashlib.sha256(signature_json.encode("utf-8"), usedforsecurity=False).hexdigest()

    git_hash = get_git_hash()
    precise_name = f"dataset_stats_{dataset_hash}_{git_hash}.json"
    precise = cache_dir / precise_name
    if precise.exists():
        logger.info(f"Found dataset stats cache with precisely matching dataset and git hash: {precise_name}.")
        return True, str(precise)
    
    candidates = sorted(cache_dir.glob(f"dataset_stats_{dataset_hash}_*.json"))
    if not candidates:
        logger.info(f"No dataset stats cache found for dataset hash {dataset_hash}")
        return False, str(precise) # return precise cache path for saving cache

    picked = candidates[0]
    prefix = f"dataset_stats_{dataset_hash}_"
    picked_git_hash = picked.name[len(prefix):-5]
    assert picked_git_hash != git_hash
    logger.warning(f"Found substitute dataset stats cache {picked.name} which mismatch current git hash {git_hash}.")
    return True, str(picked)
