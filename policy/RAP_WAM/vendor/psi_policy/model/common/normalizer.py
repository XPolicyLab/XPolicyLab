from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import zarr

from psi_policy.common.pytorch_util import dict_apply
from psi_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                if key not in self.params_dict:
                    result[key] = value
                    continue
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result

        if "_default" not in self.params_dict:
            raise RuntimeError("Not initialized")
        params = self.params_dict["_default"]
        return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]):
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]):
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = {}
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            return dict_apply(input_stats, self.normalize)

        result = {}
        for key, group in input_stats.items():
            this_dict = {}
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            return x.flatten()

        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(dict_apply(input_stats_dict, to_tensor)),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_from_input_stats(
        cls,
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        *,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        params_dict = _params_from_input_stats(
            input_stats_dict=input_stats_dict,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]):
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]):
        return _normalize(x, self.params_dict, forward=False)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]):
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_stats_dict = {
        "min": input_min,
        "max": input_max,
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
    }
    return _params_from_input_stats(
        input_stats_dict=input_stats_dict,
        dtype=dtype,
        mode=mode,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
        fit_offset=fit_offset,
    )


def _params_from_input_stats(
    *,
    input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    assert mode in ["limits", "gaussian"]
    assert output_max > output_min

    required_stats = ("min", "max", "mean", "std")
    missing_stats = [name for name in required_stats if name not in input_stats_dict]
    if missing_stats:
        raise KeyError(f"input_stats_dict missing keys: {', '.join(missing_stats)}")

    stats = {
        name: _to_flat_tensor(input_stats_dict[name], dtype=dtype)
        for name in required_stats
    }
    reference_shape = stats["min"].shape
    if any(value.shape != reference_shape for value in stats.values()):
        raise ValueError("all input stats must share the same flattened shape")

    input_min = stats["min"]
    input_max = stats["max"]
    input_mean = stats["mean"]
    input_std = stats["std"]

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(stats),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _to_flat_tensor(x, *, dtype=torch.float32):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().flatten()
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x


def _normalize(x, params, forward=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    return x.reshape(src_shape)
