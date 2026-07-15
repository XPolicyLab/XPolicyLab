# Copyright (C) 2026 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import importlib.util
import warnings
from os.path import join as osp

import torch

from src.models import XR1


class _DictConfig(dict):
    """Minimal dict wrapper supporting attribute access for nested config."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, dict) and not isinstance(val, _DictConfig):
            val = _DictConfig(val)
            self[key] = val
        return val

    def __setattr__(self, key, value):
        self[key] = value


def _to_dict_config(obj):
    if isinstance(obj, dict):
        return _DictConfig({k: _to_dict_config(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dict_config(v) for v in obj)
    return obj


def load_cfg(path):
    spec = importlib.util.spec_from_file_location("_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = _DictConfig()
    for key in dir(module):
        if key.startswith("_"):
            continue
        val = getattr(module, key)
        if isinstance(val, dict):
            cfg[key] = _to_dict_config(val)
    return cfg


def format_state_dict(state_dict, prefix):
    new_state_dict = {}
    prefix_len = len(prefix)
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            new_state_dict[new_key] = value
    return new_state_dict


def build_model(cfg, model_path):
    model_cfg = dict(cfg.model.params.model)
    model_cfg.pop("type", None)
    model = XR1(**model_cfg).to(torch.bfloat16)
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = format_state_dict(state_dict, "model.")
    model.eval()
    msg = model.load_state_dict(state_dict, assign=True)
    print(msg)
    return model


def extract_dataset_length(cfg):
    dataset_action_lengths = {}
    for source_name, source_config in cfg.data.params.train_datasets.sources.items():
        for transform in source_config.get('transforms', []):
            if transform.get('type') == 'LoadData':
                tmp_dataset_length = transform.get("dataset_action_length", 30)
                dataset_action_lengths[source_name] = tmp_dataset_length
    return dataset_action_lengths


def extract_all_normalize_params(cfg):
    results = {}
    for source_name, source_config in cfg.data.params.train_datasets.sources.items():
        action_norm = {"mode": None}
        state_norm = {"mode": None}
        for transform in source_config.get('transforms', []):
            if transform.get('type') != 'Normalize':
                continue
            mode = transform.get('mode', 'gaussian')
            data_flow = transform.get('data_flow', {})

            if mode == 'gaussian':
                raw_p1, raw_p2 = transform.get('mean', {}), transform.get('std', {})
                p1_key, p2_key = 'mean', 'std'
            else:
                raw_p1, raw_p2 = transform.get('q01', {}), transform.get('q99', {})
                p1_key, p2_key = 'q01', 'q99'

            p1, p2 = {}, {}
            for src, tgt in data_flow.items():
                if src in raw_p1:
                    p1[tgt] = raw_p1[src]
                if src in raw_p2:
                    p2[tgt] = raw_p2[src]

            is_action = any(tgt.startswith('action_') for tgt in data_flow.values())
            is_state = any(tgt.startswith('proprio_') for tgt in data_flow.values())

            norm_info = {"mode": mode, p1_key: p1, p2_key: p2}
            if is_action:
                action_norm = norm_info
            if is_state:
                state_norm = norm_info

        results[source_name] = {"action": action_norm, "state": state_norm}
    return results


def _get_predefined_dim(composition):
    dim = -1
    for item in composition.values():
        if isinstance(item, (list, tuple)):
            if isinstance(item[1], (list, tuple)):
                dim = max(dim, item[1][-1])
            else:
                dim = max(dim, item[-1])
    return dim


def _compose_params_to_tensor(raw_params, composition, temporal_len, dim, device):
    tensor = torch.zeros((temporal_len, dim), device=device)
    for src, tgt in composition.items():
        if not isinstance(tgt, (list, tuple)):
            continue
        if isinstance(tgt[1], (list, tuple)):
            s_start, s_end = tgt[0]
            t_start, t_end = tgt[1]
            if src in raw_params:
                param = torch.tensor(raw_params[src], device=device)[:temporal_len, s_start:s_end]
                clamped_end = min(t_end, t_start + param.shape[-1])
                if clamped_end != t_end:
                    warnings.warn(
                        f"[deploy] norm param '{src}' slot [{t_start},{t_end}] wider than "
                        f"param width {param.shape[-1]}; filling [{t_start},{clamped_end}] "
                        f"and leaving [{clamped_end},{t_end}] as zero."
                    )
                tensor[:temporal_len, t_start:clamped_end] = param[:, :clamped_end - t_start]
        else:
            start, end = tgt
            if src in raw_params:
                param = torch.tensor(raw_params[src], device=device)[:temporal_len]
                clamped_end = min(end, start + param.shape[-1])
                if clamped_end != end:
                    warnings.warn(
                        f"[deploy] norm param '{src}' slot [{start},{end}] wider than "
                        f"param width {param.shape[-1]}; filling [{start},{clamped_end}] "
                        f"and leaving [{clamped_end},{end}] as zero."
                    )
                tensor[:temporal_len, start:clamped_end] = param[:, :clamped_end - start]
    return tensor


def helper(args):
    cfg = load_cfg(osp(args.model, "config.py"))

    device = "cuda:0"
    model_path = osp(args.model, "model_states.pt")
    model = build_model(cfg, model_path).to(device)

    source_transforms = list(cfg.data.params.train_datasets.sources.values())[0].transforms
    dataset_transforms = getattr(cfg.data.params.train_datasets, 'transforms', [])

    action_composition = None
    state_composition = None
    for t in list(source_transforms) + list(dataset_transforms):
        if not isinstance(t, dict):
            continue
        if t.get("type") == "ComposeAction" and action_composition is None:
            action_composition = list(t["reversed_data_flow"].values())[-1]
        elif t.get("type") == "ComposeState" and state_composition is None:
            state_composition = list(t["reversed_data_flow"].values())[-1]

    action_dim = _get_predefined_dim(action_composition) if action_composition else 0
    state_dim = _get_predefined_dim(state_composition) if state_composition else 0

    dataset_action_lengths = extract_dataset_length(cfg)
    all_norm_params = extract_all_normalize_params(cfg)

    action_norms = {}
    state_norms = {}

    for source_name, norm_info in all_norm_params.items():
        dal = dataset_action_lengths.get(source_name, 10)

        a = norm_info["action"]
        if a["mode"] == "gaussian" and action_composition:
            action_norms[source_name] = {
                "mode": "gaussian",
                "mean": _compose_params_to_tensor(a["mean"], action_composition, dal, action_dim, device),
                "std": _compose_params_to_tensor(a["std"], action_composition, dal, action_dim, device) + 1e-6,
            }
        elif a["mode"] == "quantile" and action_composition:
            action_norms[source_name] = {
                "mode": "quantile",
                "q01": _compose_params_to_tensor(a["q01"], action_composition, dal, action_dim, device),
                "q99": _compose_params_to_tensor(a["q99"], action_composition, dal, action_dim, device) + 1e-6,
            }
        else:
            action_norms[source_name] = {"mode": None}

        s = norm_info["state"]
        obs_length = getattr(cfg.data.params, 'obs_length', 1)
        if s["mode"] == "gaussian" and state_composition:
            state_norms[source_name] = {
                "mode": "gaussian",
                "mean": _compose_params_to_tensor(s["mean"], state_composition, obs_length, state_dim, device),
                "std": _compose_params_to_tensor(s["std"], state_composition, obs_length, state_dim, device) + 1e-6,
            }
        elif s["mode"] == "quantile" and state_composition:
            state_norms[source_name] = {
                "mode": "quantile",
                "q01": _compose_params_to_tensor(s["q01"], state_composition, obs_length, state_dim, device),
                "q99": _compose_params_to_tensor(s["q99"], state_composition, obs_length, state_dim, device) + 1e-6,
            }
        else:
            state_norms[source_name] = {"mode": None}

    return model, action_norms, state_norms, action_composition, device
