from typing import Literal, Dict, Annotated, Union, Any, List, Tuple, Optional
import torch
import json
import numpy as np
from awomo.awomo.utils.logging_config import get_logger

from awomo.awomo.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[str, "format: 'const_min/const_max', where const_min and const_max give the constant range"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score"], ConstConstStr]

class LinearNormalizer:
    def __init__(
            self,
            shape_meta,
            use_stepwise_action_norm,
            default_mode: NormMode,
            exception_mode: Dict[str, Dict[str, NormMode]],
            stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]]
        ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats

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
            )


    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                batch["action"][key] = norm.forward(batch["action"][key])
            self._zero_masked_dims(batch, "action", "action_dim_is_pad")

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])
        self._zero_masked_dims(batch, "state", "state_dim_is_pad")

        return batch


    @staticmethod
    def _zero_masked_dims(batch: Dict[str, Any], field: Literal["action", "state"], mask_key: str) -> None:
        mask = batch.get(mask_key)
        if mask is None:
            return
        if len(batch[field]) != 1:
            return
        key = next(iter(batch[field]))
        x = batch[field][key]
        mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
        if mask.ndim != 1 or mask.shape[0] != x.shape[-1]:
            raise ValueError(
                f"`{mask_key}` must be 1D with length {x.shape[-1]}, got shape {tuple(mask.shape)}"
            )
        if bool(mask.any().item()):
            x = x.clone()
            x[..., mask] = 0.0
            batch[field][key] = x


class SingleFieldLinearNormalizer:
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0
    def __init__(self, stats, mode: NormMode="min/max"):
        self.stats = stats
        self.mode = mode

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = - input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            else:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return x
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) / self.scale
        return x


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
