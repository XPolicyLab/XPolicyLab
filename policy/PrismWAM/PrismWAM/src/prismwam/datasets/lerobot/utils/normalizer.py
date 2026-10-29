from collections import defaultdict
import json
from typing import Any, Dict

import numpy as np
import torch

from prismwam.utils.pytorch_utils import dict_apply


class LinearNormalizer:
    """Per-key min/max normalization of the ``action`` and ``state`` groups."""

    def __init__(
        self,
        shape_meta,
        stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    ):
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats
        for group in ("action", "state"):
            for meta in shape_meta[group]:
                key = meta["key"]
                cur_stats = {
                    k.removeprefix("global_"): v
                    for k, v in stats[group][key].items()
                    if k.startswith("global_")
                }
                self.normalizers[group][key] = SingleFieldLinearNormalizer(cur_stats)

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


class SingleFieldLinearNormalizer:
    """Map ``[min, max]`` to ``[-1, 1]``; constant dimensions are centred."""

    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0

    def __init__(self, stats):
        self.stats = stats
        input_min, input_max = stats["min"], stats["max"]
        input_range = input_max - input_min
        ignore_dim = input_range < self.range_tol
        input_range[ignore_dim] = self.output_max - self.output_min
        scale = (self.output_max - self.output_min) / input_range
        offset = self.output_min - scale * input_min
        offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[
            ignore_dim
        ]
        self.scale = scale
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return x

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) / self.scale
        return x


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

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)


def load_dataset_stats_from_json(
    file_path: str, try_convert_tensor: bool = True
) -> Dict[str, Any]:

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

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data
