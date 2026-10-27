# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic diffusion training lifecycle, state management and native checkpoint recovery."""

from __future__ import annotations
import getpass
import json
import logging
import os
import re
import socket
from datetime import datetime
from pathlib import Path
import torch
from awomo.awomo.training.shared.torch_patches import apply_torch_patches
apply_torch_patches()
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.processing_utils import ProcessorMixin
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
except ImportError:
    from transformers.tokenization_utils import PreTrainedTokenizerBase
from awomo.awomo.training.config.loader import ConfigNode, config_to_yaml_str
from awomo.awomo.training.distributed.mesh_utils import get_flat_mesh
from awomo.awomo.training.optim.scheduler import OptimizerParamScheduler
from awomo.awomo.training.garbage_collection import GarbageCollection
from awomo.awomo.training.rng import StatefulRNG
from awomo.awomo.training.step_scheduler import StepScheduler

def has_load_restore_state(object):
    return all((callable(getattr(object, attr, None)) for attr in ('load_state_dict', 'state_dict')))

def is_dataloader(object):
    return isinstance(object, StatefulDataLoader) and has_load_restore_state(object)

def is_tokenizer(object):
    return isinstance(object, (PreTrainedTokenizerBase, ProcessorMixin))

def is_lr_scheduler(object):
    return isinstance(object, OptimizerParamScheduler) or (isinstance(object, list) and all((isinstance(item, OptimizerParamScheduler) for item in object)) and (len(object) > 0))

def is_optimizer(object):
    return isinstance(object, Optimizer) or (isinstance(object, list) and len(object) > 0 and all((isinstance(item, Optimizer) for item in object)))

def is_distributed_stateful(object):
    return bool(getattr(object, 'use_distributed_checkpointing', False)) and has_load_restore_state(object)

def is_model(object):
    return isinstance(object, nn.Module) or (isinstance(object, list) and len(object) > 0 and all((isinstance(item, nn.Module) for item in object)))

def _list_existing_checkpoints(ckpt_root: Path) -> list[Path]:
    if not ckpt_root.exists():
        return []
    return list(ckpt_root.glob('*step_*'))

def _resolve_restore_from_to_ckpt_dir(checkpoint_dir: str, restore_from: str) -> str | None:
    if restore_from.upper() == 'LATEST':
        return _find_latest_checkpoint(checkpoint_dir)
    if os.path.sep not in restore_from and (not os.path.isabs(restore_from)):
        return os.path.join(checkpoint_dir, restore_from)
    return restore_from

def _format_missing_checkpoint_dir_error(checkpoint_dir: str, restore_from: str, resolved_ckpt_dir: str) -> str:
    error_msg = [f"\n{'=' * 80}", 'ERROR: Checkpoint directory does not exist', f"{'=' * 80}", f"Specified: checkpoint.restore_from: '{restore_from}'", f'Resolved to: {resolved_ckpt_dir}', '', 'Please check:', '  1. The checkpoint directory exists', f"  2. The path is correct (restore_from: '{restore_from}')", f'  3. Available checkpoints in {checkpoint_dir}:']
    ckpt_root = Path(checkpoint_dir)
    available_ckpts = _list_existing_checkpoints(ckpt_root)
    if available_ckpts:
        error_msg += [f"       {', '.join([p.name for p in available_ckpts[:5]])}"]
        if len(available_ckpts) > 5:
            error_msg += [f'       ... and {len(available_ckpts) - 5} more']
    else:
        error_msg += ['       (no checkpoints found)'] if ckpt_root.exists() else ['       (checkpoint_dir does not exist)']
    error_msg += [f"{'=' * 80}"]
    return '\n'.join(error_msg)

def _is_rank_0() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def _dist_barrier(group=None) -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group=group)

def _broadcast_bool_from_rank0(value: bool, group=None) -> bool:
    if not torch.distributed.is_initialized():
        return bool(value)
    values = [bool(value)]
    torch.distributed.broadcast_object_list(values, src=0, group=group)
    return bool(values[0])

def _find_latest_checkpoint(checkpoint_dir):
    root = Path(checkpoint_dir)
    if not root.exists():
        return
    latest_link = os.path.join(os.fspath(root), 'LATEST')
    resolved = None
    if os.path.islink(latest_link):
        try:
            resolved = os.readlink(latest_link)
        except OSError:
            pass
    elif os.path.isfile(latest_link + '.txt'):
        try:
            with open(latest_link + '.txt', 'r') as f:
                resolved = f.read().strip()
        except OSError:
            pass
    if resolved:
        if not os.path.isabs(resolved):
            resolved = os.path.abspath(os.path.join(os.fspath(root), resolved))
        if os.path.isdir(resolved):
            return resolved
    checkpoint_files = list(root.glob('*step_*'))
    if not checkpoint_files:
        return

    def _step_num(path: Path):
        m = re.search('step_(\\d+)$', path.stem)
        return int(m.group(1)) if m else -1
    latest = max(checkpoint_files, key=_step_num)
    if _step_num(latest) == -1:
        return
    return latest

def _extract_model_signature(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}
    model_cfg = cfg.get('model', {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    keys = ('pretrained_model_name_or_path', 'architectures', 'model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'vocab_size', 'tie_word_embeddings', 'rope_theta', 'rope_scaling')
    sig = {k: model_cfg.get(k, None) for k in keys}
    peft_cfg = cfg.get('peft', None)
    sig['_has_peft'] = peft_cfg is not None and isinstance(peft_cfg, dict)
    return sig

def _normalize_signature_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f == int(f) else f
        except (ValueError, TypeError):
            return v
    if isinstance(v, dict):
        return {k: _normalize_signature_value(x) for (k, x) in v.items()}
    if isinstance(v, list):
        return [_normalize_signature_value(x) for x in v]
    if hasattr(v, '_orig_value'):
        return _normalize_signature_value(getattr(v, '_orig_value'))
    return v

def _signatures_match(cur_sig: dict, ckpt_sig: dict) -> bool:
    if set(cur_sig.keys()) != set(ckpt_sig.keys()):
        return False
    for k in cur_sig:
        cur_v = _normalize_signature_value(cur_sig[k])
        ckpt_v = _normalize_signature_value(ckpt_sig[k])
        if cur_v != ckpt_v:
            return False
    return True

def _is_checkpoint_model_config_compatible(current_cfg, ckpt_dir: str) -> tuple[bool, str]:
    config_path = os.path.join(os.fspath(ckpt_dir), 'config.yaml')
    if not os.path.exists(config_path):
        return (True, 'checkpoint has no config.yaml (cannot validate)')
    try:
        with open(config_path, 'r') as f:
            ckpt_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        return (True, f'failed to read checkpoint config.yaml (cannot validate): {e}')
    try:
        if hasattr(current_cfg, 'raw_config'):
            cur_cfg = current_cfg.raw_config
        elif hasattr(current_cfg, 'to_dict'):
            cur_cfg = current_cfg.to_dict()
        else:
            cur_cfg = dict(current_cfg)
    except Exception:
        cur_cfg = {}
    ckpt_sig = _extract_model_signature(ckpt_cfg)
    cur_sig = _extract_model_signature(cur_cfg)
    if not ckpt_sig or not cur_sig:
        return (True, 'could not extract model signature (cannot validate)')
    if not _signatures_match(cur_sig, ckpt_sig):
        return (False, f'model signature mismatch (checkpoint={ckpt_sig}, current={cur_sig})')
    return (True, 'model signature matches')
import hashlib
import math
import time
from math import ceil
import torch.distributed.checkpoint as dcp
import wandb
from awomo.awomo.training.transformer_utils.mfu import get_device_flops
from awomo.awomo.training.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from awomo.awomo.training.checkpoint.checkpointing import Checkpointer, CheckpointingConfig, _is_safetensors_checkpoint
from awomo.awomo.training.loggers.log_utils import setup_logging
from awomo.awomo.training.loggers.wandb_utils import suppress_wandb_log_messages
from awomo.awomo.models.backbones.schedulers.scheduler_continuous import FlowMatchScheduler
from awomo.awomo.training.ema import EMAManager
from awomo.awomo.training.sampler import ResumableBatchSampler
from awomo.awomo.training.gradient_utils import clip_grad_norm
from awomo.awomo.training.lr import build_lr_scheduler
from awomo.awomo.training.builders import build_distributed, build_wandb
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformers.trainer_pt_utils import get_parameter_names
LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LocalResponseNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm, torch.nn.RMSNorm]

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def _load_state_dict_cpu(path: str):
    if path.endswith('.safetensors'):
        return load_file(path, device='cpu')
    return torch.load(path, map_location='cpu', weights_only=True)

def _parameter_local_hash_record(parameter: torch.nn.Parameter) -> tuple[str, tuple[int, ...], tuple[int, ...], bool, str]:
    tensor = parameter.detach()
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    tensor = tensor.detach().contiguous()
    digest = hashlib.sha256()
    bytes_tensor = tensor.view(torch.uint8)
    if bytes_tensor.device.type != 'cpu':
        bytes_tensor = bytes_tensor.cpu()
    digest.update(bytes_tensor.numpy().tobytes())
    return (str(parameter.dtype), tuple((int(dim) for dim in parameter.shape)), tuple((int(dim) for dim in tensor.shape)), bool(parameter.requires_grad), digest.hexdigest())

class Trainer:

    def _save_checkpoint_artifacts(self, checkpoint_path: str) -> None:
        pass

    def __setattr__(self, key, value):
        if key == '__state_tracked':
            raise ValueError('cannot set __state_tracked')
        if '__state_tracked' not in self.__dict__:
            self.__dict__['__state_tracked'] = set()
        if '_best_val_loss' not in self.__dict__:
            self.__dict__['_best_val_loss'] = float('inf')
        should_track = is_model(value) or has_load_restore_state(value) or is_tokenizer(value) or is_lr_scheduler(value) or is_optimizer(value) or isinstance(value, ConfigNode) or is_dataloader(value)
        if should_track and (not any((substr in key.lower() for substr in ('val', 'eval', 'test', 'loss')))):
            assert key not in self.__dict__['__state_tracked']
            self.__dict__['__state_tracked'].add(key)
        super().__setattr__(key, value)

    def untrack_state(self, *keys: str) -> None:
        tracked = self.__dict__.get('__state_tracked')
        if tracked is None:
            return
        for key in keys:
            tracked.discard(key)

    def _save_tracked_checkpoint(self, epoch: int, step: int, train_loss: float, val_loss: dict[str, float] | None=None, best_metric_key: str='default'):
        if self.checkpointer is None or not self.checkpointer.config.enabled:
            return
        self.checkpointer.async_wait()
        prev_pending = getattr(self, '_last_pending_checkpoint_dir', None)
        is_dist_initialized = torch.distributed.is_initialized()
        is_rank_0 = not is_dist_initialized or torch.distributed.get_rank() == 0
        if prev_pending is not None:
            if is_rank_0:
                self._update_latest_symlink(prev_pending)
            setattr(self, '_last_pending_checkpoint_dir', None)
            if is_dist_initialized:
                torch.distributed.barrier()
        prev_best_pending = getattr(self, '_last_pending_best_checkpoint_info', None)
        if prev_best_pending is not None:
            if is_rank_0 and prev_best_pending.get('val') is not None:
                self._update_best_symlink(prev_best_pending['path'], float(prev_best_pending['val']))
            setattr(self, '_last_pending_best_checkpoint_info', None)
            if is_dist_initialized:
                torch.distributed.barrier()
        path = self.checkpointer.config.checkpoint_dir
        path = os.path.join(path, f'epoch_{epoch}_step_{step}')
        best_val_metric = val_loss[next(iter(val_loss.keys())) if len(val_loss) == 1 else best_metric_key] if val_loss else None
        complete_marker = os.path.join(path, '.complete')
        checkpoint_group = getattr(getattr(self.checkpointer, '_model_ctx', None), 'process_group', None)
        skip_checkpoint = os.path.exists(complete_marker) if is_rank_0 else False
        skip_checkpoint = _broadcast_bool_from_rank0(skip_checkpoint, group=checkpoint_group)
        if skip_checkpoint:
            if is_rank_0:
                print(f'Checkpoint {path} is complete, skipping save.', flush=True)
            return
        if is_rank_0:
            os.makedirs(path, exist_ok=True)
        if is_rank_0:
            print(f'Saving checkpoint to {path}', flush=True)
        if is_rank_0:

            def to_item(x):
                if isinstance(x, torch.Tensor):
                    return x.item()
                return x
            loss_dict = {'train_loss': train_loss}
            if val_loss:
                key = next(iter(val_loss.keys()))
                loss_dict['val_loss'] = val_loss.pop(key) if len(val_loss) == 1 else loss_dict.update(val_loss)
            with open(os.path.join(path, 'losses.json'), 'w') as f:
                try:
                    json.dump({k: to_item(v) for (k, v) in loss_dict.items()}, f)
                except:
                    pass
        (model, optimizer, scheduler, tokenizer, config) = (None, None, None, None, None)
        dp_rank_states = []
        for key in sorted(self.__dict__['__state_tracked']):
            if is_model(getattr(self, key)):
                if key == 'teacher_model':
                    continue
                model = getattr(self, key)
            elif is_optimizer(getattr(self, key)):
                optimizer = getattr(self, key)
            elif isinstance(getattr(self, key), ConfigNode):
                config = getattr(self, key)
            elif is_lr_scheduler(getattr(self, key)):
                scheduler = getattr(self, key)
            elif is_tokenizer(getattr(self, key)):
                tokenizer = getattr(self, key)
            elif is_dataloader(getattr(self, key)) or isinstance(getattr(self, key), StatefulRNG):
                dp_rank_states.append((getattr(self, key), key))
            elif is_distributed_stateful(getattr(self, key)):
                self.checkpointer.save_distributed_state(getattr(self, key), key, path)
            elif is_rank_0:
                torch.save(getattr(self, key).state_dict(), os.path.join(path, f'{key}.pt'))
        if isinstance(model, list) and len(model) > 1:
            self.checkpointer.save_model(model, path, peft_config=getattr(self, 'peft_config', None), tokenizer=tokenizer)
        else:
            unwrapped_model = model[0] if isinstance(model, list) else model
            if isinstance(unwrapped_model, DistributedDataParallel):
                unwrapped_model = unwrapped_model.module
            if hasattr(unwrapped_model, 'save_pretrained') and hasattr(unwrapped_model.save_pretrained, '__func__'):
                from awomo.awomo.training.model_utils.hf_checkpointing_mixin import HFCheckpointingMixin
                if isinstance(unwrapped_model, HFCheckpointingMixin):
                    unwrapped_model.save_pretrained(save_directory=path, checkpointer=self.checkpointer, tokenizer=tokenizer, peft_config=getattr(self, 'peft_config', None))
                else:
                    self.checkpointer.save_model(model=unwrapped_model, weights_path=path, peft_config=getattr(self, 'peft_config', None), tokenizer=tokenizer)
            else:
                self.checkpointer.save_model(model=unwrapped_model, weights_path=path, peft_config=getattr(self, 'peft_config', None), tokenizer=tokenizer)
        for (state_obj, state_key) in dp_rank_states:
            self.checkpointer.save_on_dp_ranks(state_obj, state_key, path)
        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        for opt in optimizers:
            if hasattr(opt, 'synchronize_for_checkpoint'):
                opt.synchronize_for_checkpoint()
        self.checkpointer.save_optimizer(optimizer, model, path, scheduler)
        from awomo.awomo.training.checkpoint.checkpointing import save_config
        save_config(config.raw_config, path)
        artifact_save_error = None
        if is_rank_0:
            try:
                self._save_checkpoint_artifacts(path)
            except Exception as error:
                artifact_save_error = error
        artifact_save_succeeded = _broadcast_bool_from_rank0(artifact_save_error is None, group=checkpoint_group)
        if not artifact_save_succeeded:
            if artifact_save_error is not None:
                raise artifact_save_error
            raise RuntimeError('Rank 0 failed to save recipe-specific checkpoint artifacts.')
        if is_dist_initialized:
            torch.distributed.barrier(group=checkpoint_group)
        if is_rank_0:
            with open(complete_marker, 'w') as f:
                f.write('complete\n')
        if is_dist_initialized:
            torch.distributed.barrier(group=checkpoint_group)
        if getattr(self.checkpointer.config, 'is_async', False):
            setattr(self, '_last_pending_checkpoint_dir', path)
            if best_val_metric is not None:
                setattr(self, '_last_pending_best_checkpoint_info', {'path': path, 'val': float(best_val_metric)})
        else:
            if is_rank_0:
                self._update_latest_symlink(path)
                if best_val_metric is not None:
                    self._update_best_symlink(path, float(best_val_metric))
            if is_dist_initialized:
                torch.distributed.barrier()

    def _update_checkpoint_symlink(self, link_name: str, target_dir: str) -> None:
        ckpt_root = self.checkpointer.config.checkpoint_dir
        link_path = os.path.join(ckpt_root, link_name)
        if os.path.lexists(link_path):
            os.remove(link_path)
        ckpt_root_abs = os.path.abspath(ckpt_root)
        target_abs = os.path.abspath(target_dir)
        relative_target = os.path.relpath(target_abs, start=ckpt_root_abs)
        try:
            os.symlink(relative_target, link_path)
        except OSError:
            with open(f'{link_path}.txt', 'w') as f:
                f.write(relative_target)

    def _update_latest_symlink(self, target_dir: str) -> None:
        self._update_checkpoint_symlink('LATEST', target_dir)

    def _update_best_symlink(self, target_dir: str, val_loss: float) -> None:
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._update_checkpoint_symlink('LOWEST_VAL', target_dir)
            logging.info(f'Updated LOWEST_VAL checkpoint symlink to {os.path.basename(target_dir)} (val_loss={val_loss:.4f})')

    def _validate_checkpoint_dir_exists(self, ckpt_dir: str, restore_from: str, is_rank_0: bool) -> None:
        if os.path.exists(ckpt_dir):
            return
        if is_rank_0:
            error_msg = _format_missing_checkpoint_dir_error(checkpoint_dir=self.checkpointer.config.checkpoint_dir, restore_from=restore_from, resolved_ckpt_dir=ckpt_dir)
        else:
            error_msg = f'Checkpoint directory does not exist: {ckpt_dir}'
        _dist_barrier()
        raise FileNotFoundError(error_msg)

    def _load_checkpoint_tracked_state(self, ckpt_dir: str):
        (model, optimizer, scheduler) = (None, None, None)
        for key in sorted(self.__dict__['__state_tracked']):
            obj = getattr(self, key)
            if is_model(obj):
                if key == 'teacher_model':
                    continue
                model = obj
            elif is_optimizer(obj):
                optimizer = obj
            elif is_lr_scheduler(obj):
                scheduler = obj
            elif is_dataloader(obj) or isinstance(obj, StatefulRNG):
                self.checkpointer.load_on_dp_ranks(obj, key, ckpt_dir)
            elif is_distributed_stateful(obj):
                self.checkpointer.load_distributed_state(obj, key, ckpt_dir)
            elif is_tokenizer(obj) or isinstance(obj, ConfigNode):
                continue
            else:
                obj.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{key}.pt'), weights_only=False))
        return (model, optimizer, scheduler)

    def load_checkpoint(self, restore_from: str | None=None):
        if self.checkpointer is None or not self.checkpointer.config.enabled:
            if _is_rank_0() and restore_from is not None:
                print('Enable checkpointing to resume from a checkpoint, skipping...', flush=True)
            return
        is_rank_0 = _is_rank_0()
        if restore_from:
            ckpt_dir = _resolve_restore_from_to_ckpt_dir(self.checkpointer.config.checkpoint_dir, restore_from)
            if ckpt_dir is None:
                if is_rank_0:
                    logging.warning(f"restore_from='LATEST' specified but no checkpoint found in {self.checkpointer.config.checkpoint_dir}. Starting fresh.")
                return
            self._validate_checkpoint_dir_exists(ckpt_dir, restore_from=restore_from, is_rank_0=is_rank_0)
        else:
            ckpt_dir = _find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
            if ckpt_dir is None:
                return
            ckpt_dir = str(ckpt_dir)
        cfg = getattr(self, 'cfg', None)
        if cfg is not None:
            (ok, reason) = _is_checkpoint_model_config_compatible(cfg, ckpt_dir)
            if not ok:
                if not restore_from:
                    if is_rank_0:
                        logging.warning(f'Auto-detected checkpoint at {ckpt_dir} is incompatible with current model configuration: {reason}. Skipping restore.')
                    return
                elif is_rank_0:
                    logging.warning(f'Checkpoint at {ckpt_dir} may be incompatible with current model configuration: {reason}. Proceeding with restore anyway.')
        if is_rank_0:
            print(f'Loading checkpoint from {ckpt_dir}', flush=True)
        (model, optimizer, scheduler) = self._load_checkpoint_tracked_state(ckpt_dir)
        self.checkpointer.load_model(model, os.path.join(ckpt_dir, 'model'))
        self.checkpointer.load_optimizer(optimizer, model, ckpt_dir, scheduler)

    def _log_experiment_details(self):
        if not getattr(self, 'dist_env', None) or not getattr(self.dist_env, 'is_main', False):
            return
        details = {'Timestamp': datetime.now().isoformat(timespec='seconds'), 'User': getpass.getuser(), 'Host': socket.gethostname(), 'World size': getattr(self.dist_env, 'world_size', None), 'Backend': getattr(getattr(self, 'cfg', {}), 'get', lambda *_: None)('dist_env.backend', 'nccl'), 'Recipe': self.__class__.__name__, 'Model name': getattr(getattr(self, 'cfg', None), 'model', None) and getattr(self.cfg.model, 'pretrained_model_name_or_path', None)}
        try:
            details_yaml = yaml.safe_dump(details, sort_keys=False, default_flow_style=False).strip()
            for line in ('Experiment_details:\n' + details_yaml).splitlines():
                logging.info(line)
        except Exception:
            logging.info(f'Experiment details: {details}')
        try:
            cfg_obj = getattr(self, 'cfg', None)
            cfg_yaml = config_to_yaml_str(cfg_obj, use_orig_values=True)
            if cfg_yaml:
                print(cfg_yaml, flush=True)
        except Exception:
            logging.info('Recipe config: <unavailable>')

    def _log_library_versions(self):
        if not getattr(self, 'dist_env', None) or not getattr(self.dist_env, 'is_main', False):
            return
        try:
            import awomo as awomo_package
            package_path = Path(getattr(awomo_package, '__file__', '<unknown>')).resolve().as_posix()
        except Exception:
            awomo_package = None
            package_path = '<unknown>'
        try:
            import transformers as hf_transformers
            tfm_path = Path(getattr(hf_transformers, '__file__', '<unknown>')).resolve().as_posix()
        except Exception:
            tfm_path = '<unknown>'
        libs = {'awomo': {'version': getattr(awomo_package, '__version__', None), 'import_path': package_path}, 'transformers': {'version': getattr(hf_transformers, '__version__', None), 'import_path': tfm_path}, 'torch': {'version': torch.__version__, 'cuda': getattr(torch.version, 'cuda', None)}}
        logging.info('Library versions:')
        for (key, value) in libs.items():
            if 'cuda' in value:
                logging.info(f"- {key}: {value['version']} CUDA {value['cuda']}")
            else:
                logging.info(f"- {key}: {value['version']} ({value['import_path']})")

    def _log_model_and_optimizer_details(self, model: nn.Module | list[nn.Module] | None=None, optimizer: Optimizer | list[Optimizer] | None=None, lr_scheduler: OptimizerParamScheduler | list[OptimizerParamScheduler] | None=None):
        if not isinstance(model, list):
            model = [model]
        for (i, m) in enumerate(model):
            if m is None:
                logging.info(f'Model Part {i}: <unavailable>')
                continue
            model_str = str(m)
            model_lines = model_str.splitlines()
            logging.info(f'Model Part {i}:')
            for line in model_lines[:40]:
                logging.info(line)
            if len(model_lines) > 40:
                logging.info('...')
        if optimizer:
            if not isinstance(optimizer, list):
                optimizer = [optimizer]
            for opt in optimizer:
                for line in ('Optimizer:\n' + str(opt)).splitlines():
                    logging.info(line)
        else:
            logging.info('Optimizer: <unavailable>')
        if lr_scheduler:
            if not isinstance(lr_scheduler, list):
                lr_scheduler = [lr_scheduler]
            for sched in lr_scheduler:
                for line in ('LR scheduler:\n' + str(sched)).splitlines():
                    logging.info(line)
        else:
            logging.info('LR scheduler: <unavailable>')

    def _log_step_scheduler_details(self, step_scheduler: StepScheduler):
        attrs = {'Gradient accumulation steps': step_scheduler.grad_acc_steps, 'Checkpoint every steps': step_scheduler.ckpt_every_steps, 'Garbage collect every steps': getattr(step_scheduler, 'gc_every_steps', None), 'Current Epoch': step_scheduler.epoch, 'Number of epochs': step_scheduler.num_epochs, 'Validation every steps': step_scheduler.val_every_steps, 'Max train steps': step_scheduler.max_steps}
        logging.info('Step scheduler:')
        for (k, v) in attrs.items():
            logging.info(f'- {k}: {v}')

    def _setup_garbage_collection(self, step_scheduler: StepScheduler | None=None) -> None:
        """Initialize manual garbage collection based on step scheduler config."""
        if step_scheduler is None:
            step_scheduler = getattr(self, 'step_scheduler', None)
        gc_every_steps = getattr(step_scheduler, 'gc_every_steps', None)
        if gc_every_steps is None:
            self.garbage_collector = None
            return
        self.garbage_collector = GarbageCollection(gc_every_steps=gc_every_steps)

    def _maybe_collect_garbage(self) -> None:
        step_scheduler = getattr(self, 'step_scheduler', None)
        garbage_collector = getattr(self, 'garbage_collector', None)
        if step_scheduler is None or garbage_collector is None:
            return
        garbage_collector.run(step_scheduler.step)

    def _get_dp_group(self, include_cp: bool=False):
        if not self.device_mesh:
            return None
        if include_cp and self.device_mesh['cp'].size() > 1:
            return get_flat_mesh(self.device_mesh, 'dp_cp').get_group()
        return get_flat_mesh(self.device_mesh, 'dp').get_group()

    def _get_dp_group_size(self, include_cp: bool=False):
        dp_group = self._get_dp_group(include_cp=include_cp)
        if dp_group is None:
            if dist.is_initialized():
                return dist.get_world_size()
            return 1
        return dp_group.size()

    def _get_cp_group_size(self):
        if not self.device_mesh or self.device_mesh['cp'].size() == 1:
            return 1
        return self.device_mesh['cp'].size()

    def _get_dp_rank(self, include_cp: bool=False):
        if not self.device_mesh:
            if dist.is_initialized():
                return dist.get_rank()
            return 0
        if include_cp and self.device_mesh['cp'].size() > 1:
            return get_flat_mesh(self.device_mesh, 'dp_cp').get_local_rank()
        return get_flat_mesh(self.device_mesh, 'dp').get_local_rank()

    def _get_tp_rank(self):
        if not self.device_mesh or self.device_mesh['tp'].size() == 1:
            return 0
        return self.device_mesh.get_local_rank('tp')

    def _get_pp_rank(self):
        if not self.device_mesh or 'pp' not in self.device_mesh.mesh_dim_names or self.device_mesh['pp'].size() == 1:
            return 0
        return self.device_mesh.get_local_rank('pp')

    def _dp_allreduce(self, tensor, op=dist.ReduceOp.SUM, include_cp: bool=False):
        dp_group = self._get_dp_group(include_cp=include_cp)
        if dp_group is not None:
            tensor = tensor.cuda()
            dist.all_reduce(tensor, op=op, group=dp_group)
            tensor = tensor.cpu()
        return tensor

    def _make_progress_bar(self):
        if not _is_rank_0():
            return None
        from tqdm import tqdm
        return tqdm(total=getattr(self.step_scheduler, 'max_steps', None), initial=getattr(self.step_scheduler, 'step', 0), desc='Training', unit='step', dynamic_ncols=True)

    def _update_progress_bar(self, pbar, metrics: dict) -> None:
        if pbar is None:
            return
        postfix = {}
        for loss_key in ('loss', 'Loss/Train_Total'):
            if loss_key in metrics:
                postfix['loss'] = f'{metrics[loss_key]:.4f}'
                break
        for lr_key in ('lr', 'Train/lr'):
            if lr_key in metrics:
                postfix['lr'] = f'{metrics[lr_key]:.2e}'
                break
        for tps_key in ('tps', 'Train/tps'):
            if tps_key in metrics:
                postfix['tps'] = f'{metrics[tps_key]:.0f}'
                break
        pbar.set_postfix(**postfix)
        pbar.update(1)
    image_encoder_cls = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.ema = None
        self._activation_checkpointing_block_count = 0
        self._broadcast_inference_weights_from_rank0 = False

    @property
    def _model_cfg(self):
        raise NotImplementedError('Subclasses must implement _model_cfg')

    def _setup_tracking(self) -> None:
        if not self.dist_env.is_main:
            return
        if 'wandb' not in self.cfg:
            return
        suppress_wandb_log_messages()
        run = build_wandb(self.cfg)
        if run is not None:
            logging.info('View run at {}'.format(run.url))

    def _after_optimizer_setup(self) -> None:
        pass

    def _clip_gradients(self) -> float:
        return float(clip_grad_norm(max_grad_norm=self.clip_grad_max_norm, model_parts=[self.model], device_mesh=self.device_mesh, foreach=True))

    def _checkpoint_model_state_dict_keys(self) -> list[str] | None:
        logging.info(f'[DEBUG][rank {self.local_rank}] collecting setup-time state_dict keys')
        model_state_dict_keys = list(self.model.state_dict().keys())
        logging.info(f'[DEBUG][rank {self.local_rank}] setup-time state_dict key collection done, num_keys={len(model_state_dict_keys)}')
        return model_state_dict_keys

    def _checkpoint_create_process_groups(self) -> bool:
        return True

    def _before_step_group(self) -> None:
        pass

    def _get_step_flops(self, batch_group) -> float:
        return self._flops_per_micro_batch * len(batch_group)

    def _extend_train_log_dict(self, log_dict: dict[str, float], group_loss_mean: float, grad_norm: float, step_time: float, mfu: float) -> None:
        pass

    def _on_epoch_log(self) -> None:
        pass

    def _on_training_finish(self) -> None:
        pass

    def _restore_resumable_sampler_position(self, raw_steps_per_epoch: int) -> None:
        optimizer_epoch = int(self.step_scheduler.epoch)
        optimizer_step = int(self.step_scheduler.step)
        optimizer_steps_per_epoch = ceil(raw_steps_per_epoch / self.step_scheduler.grad_acc_steps)
        completed_steps_in_epoch = optimizer_step - optimizer_epoch * optimizer_steps_per_epoch
        assert 0 <= completed_steps_in_epoch <= optimizer_steps_per_epoch
        completed_microbatches_in_epoch = min(completed_steps_in_epoch * self.step_scheduler.grad_acc_steps, raw_steps_per_epoch)
        data_step = optimizer_epoch * raw_steps_per_epoch + completed_microbatches_in_epoch
        self.sampler.set_data_step(data_step)

    def _restore_step_and_data_state(self, restored_ckpt_dir: str, restore_data_state: bool, raw_steps_per_epoch: int) -> None:
        step_scheduler_state = torch.load(os.path.join(restored_ckpt_dir, 'step_scheduler.pt'), weights_only=False)
        self.step_scheduler.load_state_dict(step_scheduler_state)
        if not restore_data_state:
            self.step_scheduler.epoch = 0
            if isinstance(self.sampler, ResumableBatchSampler):
                self.sampler.set_data_step(0)
            logging.info('[INFO] Restored global step %d and reset data epoch and sampler position', self.step_scheduler.step)
            return
        if isinstance(self.sampler, ResumableBatchSampler):
            self.sampler.load_state_dict(torch.load(os.path.join(restored_ckpt_dir, 'sampler.pt'), weights_only=False))
        if self.sampler is not None:
            self.sampler.set_epoch(self.step_scheduler.epoch)
        loader_restored = False
        if is_dataloader(self.dataloader):
            if os.path.isdir(os.path.join(restored_ckpt_dir, 'dataloader')):
                self.checkpointer.load_on_dp_ranks(self.dataloader, 'dataloader', restored_ckpt_dir)
                loader_restored = True
            else:
                logging.warning('Checkpoint has no DataLoader snapshots: restoring window positions only; worker augmentation and history sampling will not match uninterrupted training.')
        if isinstance(self.sampler, ResumableBatchSampler) and not loader_restored:
            self._restore_resumable_sampler_position(raw_steps_per_epoch)

    def _run_train_batch_group(self, batch_group, current_step: int, before_clip_callback=None) -> tuple[dict[str, float], float, float]:
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)
        step_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sums = {'total_loss': 0.0, 'dynamics_loss': 0.0, 'action_loss': 0.0}
        for (micro_idx, micro_batch) in enumerate(batch_group):
            loss_dict = self.train_step(micro_batch, step=current_step, micro_idx=micro_idx)
            loss = loss_dict['total_loss'] / len(batch_group)
            loss.backward()
            for loss_name in loss_sums:
                loss_sums[loss_name] += float(loss_dict[loss_name].item())
        if before_clip_callback is not None:
            before_clip_callback()
        grad_norm = self._clip_gradients()
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)
        if self.lr_scheduler is not None:
            self.lr_scheduler[0].step(1)
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)
        step_time = time.perf_counter() - step_start
        loss_log_scalars = self._average_log_scalars_across_ranks({loss_name: loss_sum / len(batch_group) for (loss_name, loss_sum) in loss_sums.items()})
        return (loss_log_scalars, grad_norm, step_time)

    def _average_log_scalars_across_ranks(self, scalars: dict[str, float]) -> dict[str, float]:
        if not dist.is_initialized():
            return scalars
        world_size = dist.get_world_size()
        if world_size == 1:
            return scalars
        keys = list(scalars)
        values = torch.tensor([scalars[key] for key in keys], dtype=torch.float64, device=self.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
        return dict(zip(keys, values.cpu().tolist()))

    def _aggregate_step_metrics_across_ranks(self, step_flops: float, step_time: float) -> tuple[float, float]:
        if not dist.is_initialized():
            return (step_flops, step_time)
        world_size = dist.get_world_size()
        if world_size == 1:
            return (step_flops, step_time)
        local_metrics = torch.tensor((step_flops, step_time), dtype=torch.float64, device=self.device)
        gathered_metrics = [torch.empty_like(local_metrics) for _ in range(world_size)]
        dist.all_gather(gathered_metrics, local_metrics)
        metrics = torch.stack(gathered_metrics)
        return (float(metrics[:, 0].mean().item()), float(metrics[:, 1].max().item()))

    def _get_model_parallel_size_for_mfu(self) -> int:
        if self.device_mesh is None:
            return 1
        parallel_size = 1
        for mesh_dim_name in ('tp', 'cp'):
            if mesh_dim_name in self.device_mesh.mesh_dim_names:
                parallel_size *= self.device_mesh[mesh_dim_name].size()
        return parallel_size

    def _set_peak_flops_for_mfu(self) -> str:
        device_name = torch.cuda.get_device_name(self.device)
        peak_tflops = get_device_flops(unit='T', device_name=device_name)
        if not math.isfinite(peak_tflops):
            raise ValueError(f'Unsupported device for training MFU: {device_name}')
        self._peak_flops_per_gpu = peak_tflops * 1000000000000.0
        return device_name

    def setup(self):
        self.dist_env = build_distributed(self.cfg.get('dist_env', {}))
        self.peft_config = None
        setup_logging()
        model_cfg_inner = self._model_cfg
        self.loss_lambda_video = 1.0
        self.loss_lambda_action = float(model_cfg_inner.get('loss_lambda_action', 1.0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        assert torch.cuda.is_available(), 'Video diffusion training requires CUDA.'
        self.device = torch.device('cuda', self.local_rank)
        self.local_world_size = max(self.local_world_size, 1)
        self._setup_tracking()
        self.seed = self.cfg.get('seed', 42)
        self.rng = StatefulRNG(seed=self.seed, ranked=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.learning_rate = self.cfg.get('optim.learning_rate', 0.0001)
        self.clip_grad_max_norm = float(self.cfg.get('optim.clip_grad', 1.0))
        logging.info('[INFO] Video diffusion training recipe setup')
        logging.info(f'[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}')
        model_cfg = self.cfg.get('model', {})
        self.model_id = model_cfg.get('pretrained_model_name_or_path', None)
        self.noise_mode = model_cfg_inner.noise_mode
        self.video_beta_alpha = model_cfg_inner.video_beta_alpha
        self.high_noise_beta_alpha = model_cfg_inner.high_noise_beta_alpha
        self.num_frame_per_block = model_cfg_inner.num_frame_per_block
        self.concat_first_frame_latent = model_cfg_inner.concat_first_frame_latent
        self.tiled_vae = model_cfg_inner.tiled_vae
        self.tile_size_height = model_cfg_inner.tile_size_height
        self.tile_size_width = model_cfg_inner.tile_size_width
        self.tile_stride_height = model_cfg_inner.tile_stride_height
        self.tile_stride_width = model_cfg_inner.tile_stride_width
        self.target_video_height = model_cfg_inner.get('target_video_height', None)
        self.target_video_width = model_cfg_inner.get('target_video_width', None)
        self.cfg_scale = float(model_cfg_inner.get('cfg_scale', 5.0))
        if self.noise_mode == 'decouple_video_action_noise':
            self.video_beta_dist = torch.distributions.Beta(self.video_beta_alpha, 1.0)
        elif self.noise_mode == 'high_noise_emphasis':
            self.high_noise_beta_dist = torch.distributions.Beta(self.high_noise_beta_alpha, 1.0)
        logging.info(f'[DEBUG][rank {self.local_rank}] Starting _build_models...')
        self._build_models(model_cfg, model_cfg_inner)
        dit_cfg = model_cfg_inner.get('dit', {})
        self.model_action_dim = dit_cfg.action_dim
        self.model_max_state_dim = dit_cfg.max_state_dim
        checkpoint_cfg = self.cfg.get('checkpoint', None)
        restored_ckpt_dir = None
        restore_from = checkpoint_cfg.get('restore_from', None)
        warm_start_from = checkpoint_cfg.get('warm_start_from', None)
        assert not (restore_from and warm_start_from), 'Cannot set both restore_from and warm_start_from'
        _weights_src = restore_from or warm_start_from
        if _weights_src is not None:
            _ckpt_dir = _resolve_restore_from_to_ckpt_dir(checkpoint_cfg.get('checkpoint_dir'), _weights_src)
            assert _ckpt_dir is not None
            if restore_from is not None:
                restored_ckpt_dir = _ckpt_dir
                logging.info(f'[INFO] Resuming training from {_ckpt_dir}')
            else:
                logging.info(f'[INFO] Warm-starting model weights from {_ckpt_dir} (training state reset)')
            state_dict = self.model.state_dict()
            _model_dir = os.path.join(_ckpt_dir, 'model')
            _storage_reader = None
            if _is_safetensors_checkpoint(_model_dir):
                _storage_reader = _HuggingFaceStorageReader(path=_model_dir)
            dcp.load(state_dict, checkpoint_id=_model_dir, storage_reader=_storage_reader)
            self.model.load_state_dict(state_dict)
        logging.info(f'[DEBUG][rank {self.local_rank}] _build_models done, starting _setup_fsdp...')
        fsdp_cfg = self.cfg.get('fsdp', None)
        self._setup_fsdp(fsdp_cfg)
        assert checkpoint_cfg is not None
        self._assert_hsdp_replicated_parameter_hashes_match('after FSDP before optimizer')
        ema_cfg = self.cfg.get('ema', None)
        if ema_cfg is not None:
            self.ema = EMAManager(self.model, decay=float(ema_cfg.decay))
            logging.info('[INFO] EMA enabled with decay=%s, params=%d', ema_cfg.decay, len(self.ema))
        self._build_optimizer()
        self._after_optimizer_setup()
        scheduler_cfg = model_cfg_inner.scheduler
        num_train_timesteps = scheduler_cfg.num_train_timesteps
        self.scheduler = FlowMatchScheduler(extra_one_step=True, num_inference_steps=num_train_timesteps, num_train_timesteps=num_train_timesteps, shift=scheduler_cfg.shift, sigma_min=0.0)
        self.scheduler.set_timesteps(num_inference_steps=num_train_timesteps, training=True)
        logging.info(f'[DEBUG][rank {self.local_rank}] scheduler setup done')
        dataloader_cfg = self.cfg.get('data.dataloader')
        self.local_batch_size = self.cfg.step_scheduler.local_batch_size
        self.global_batch_size = self.cfg.step_scheduler.global_batch_size
        self.num_epochs = self.cfg.step_scheduler.num_epochs
        self.log_every = self.cfg.get('step_scheduler.log_every', 5)
        logging.info(f'[DEBUG][rank {self.local_rank}] starting dataloader setup')
        (self.dataloader, self.sampler) = dataloader_cfg.instantiate(dp_rank=self._get_dp_rank(), dp_world_size=self._get_dp_group_size(), batch_size=self.local_batch_size)
        logging.info(f'[DEBUG][rank {self.local_rank}] dataloader setup done, local_batches={len(self.dataloader)}')
        if checkpoint_cfg is not None and checkpoint_cfg.get('enabled', False):
            model_state_dict_keys = self._checkpoint_model_state_dict_keys()
            self.checkpoint_config = CheckpointingConfig(enabled=checkpoint_cfg.get('enabled'), checkpoint_dir=checkpoint_cfg.get('checkpoint_dir'), model_cache_dir=checkpoint_cfg.get('model_cache_dir', checkpoint_cfg.get('checkpoint_dir')), model_save_format=checkpoint_cfg.get('model_save_format', 'torch_save'), model_repo_id=self.model_id or '', save_consolidated=checkpoint_cfg.get('save_consolidated', False), is_peft=self.lora_enabled, model_state_dict_keys=model_state_dict_keys)
            self.restore_from = checkpoint_cfg.get('restore_from', None)
            self.checkpointer = Checkpointer(config=self.checkpoint_config, dp_rank=self._get_dp_rank(include_cp=True), tp_rank=self._get_tp_rank(), pp_rank=self._get_pp_rank(), moe_mesh=None, create_process_groups=self._checkpoint_create_process_groups())
            logging.info(f'[DEBUG][rank {self.local_rank}] checkpoint setup done')
        else:
            self.checkpoint_config = None
            self.checkpointer = None
            self.restore_from = None
            logging.info(f'[DEBUG][rank {self.local_rank}] checkpoint setup skipped')
        self.dp_size = self.world_size
        raw_steps_per_epoch = len(self.dataloader)
        grad_acc_steps = max(1, self.global_batch_size // max(1, self.local_batch_size * self.dp_size))
        self.steps_per_epoch = ceil(raw_steps_per_epoch / grad_acc_steps)
        max_steps = self.cfg.step_scheduler.max_steps
        assert max_steps is not None and max_steps > 0, 'step_scheduler.max_steps must be provided and greater than 0'
        total_steps = max_steps
        lr_scheduler_cfg = self.cfg.get('lr_scheduler', None)
        if lr_scheduler_cfg is not None:
            lr_scheduler = build_lr_scheduler(lr_scheduler_cfg, self.optimizer, total_steps)
            assert lr_scheduler is not None, 'lr_scheduler config must build a scheduler.'
            self.lr_scheduler = [lr_scheduler]
        else:
            self.lr_scheduler = None
        self.global_step = 0
        self.start_epoch = 0
        self.step_scheduler = StepScheduler(global_batch_size=self.global_batch_size, local_batch_size=self.local_batch_size, dp_size=int(self.dp_size), ckpt_every_steps=self.cfg.step_scheduler.ckpt_every_steps, save_checkpoint_every_epoch=self.cfg.get('step_scheduler.save_checkpoint_every_epoch', True), dataloader=self.dataloader, val_every_steps=None, start_step=0, start_epoch=0, num_epochs=int(self.num_epochs), max_steps=max_steps)
        if restored_ckpt_dir is not None:
            self.checkpointer.load_on_dp_ranks(self.rng, 'rng', restored_ckpt_dir)
            restore_data_state = checkpoint_cfg.get('restore_data_state', True)
            self._restore_step_and_data_state(restored_ckpt_dir, restore_data_state, raw_steps_per_epoch)
            if self.ema is not None and ema_cfg.restore_from_checkpoint:
                self.checkpointer.load_distributed_state(self.ema, 'ema', restored_ckpt_dir)
                logging.info(f"[INFO] Restored EMA state from {os.path.join(restored_ckpt_dir, 'ema')}")
            restore_optimizer = checkpoint_cfg.get('restore_optimizer', True)
            restore_lr_scheduler = self.cfg.get('lr_scheduler.restore_from_checkpoint', True)
            if restore_optimizer:
                if restore_lr_scheduler:
                    self.checkpointer.load_optimizer(self.optimizer, self.model, restored_ckpt_dir, self.lr_scheduler)
                else:
                    self.checkpointer.load_optimizer(self.optimizer, self.model, restored_ckpt_dir)
                optim_cfg = self.cfg.optim.optimizer
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] = optim_cfg.weight_decay
                    param_group['betas'] = tuple(optim_cfg.betas)
                if not restore_lr_scheduler:
                    assert self.lr_scheduler is not None
                    self.lr_scheduler[0].step(0)
                logging.info(f"[INFO] Restored optimizer state from {os.path.join(restored_ckpt_dir, 'optim')}")
            else:
                if restore_lr_scheduler:
                    for scheduler in self.lr_scheduler:
                        scheduler.step(int(self.step_scheduler.step))
                logging.info('[INFO] Skipped optimizer state restore; optimizer starts from empty state')
            self._assert_hsdp_replicated_parameter_hashes_match('after pre-FSDP checkpoint restore')
        if isinstance(self.sampler, ResumableBatchSampler):
            assert self.sampler.global_batch_size == self.local_batch_size * self.dp_size
            assert self.sampler.local_batch_size == self.local_batch_size
        if is_main_process() and self.checkpoint_config is not None:
            os.makedirs(self.checkpoint_config.checkpoint_dir, exist_ok=True)
        if dist.is_initialized():
            logging.info(f'[DEBUG][rank {self.local_rank}] entering setup barrier')
            dist.barrier()
            logging.info(f'[DEBUG][rank {self.local_rank}] setup barrier done')
        self._compute_theoretical_flops_per_step()
        logging.info('[INFO] Training recipe setup complete')

    def _assert_hsdp_replicated_parameter_hashes_match(self, reason: str) -> None:
        if not dist.is_initialized():
            return
        device_mesh = self.device_mesh
        if device_mesh is None:
            return
        mesh_dim_names = tuple(device_mesh.mesh_dim_names)
        assert 'dp_replicate' in mesh_dim_names, mesh_dim_names
        if device_mesh['dp_replicate'].size() <= 1:
            return
        coordinate = device_mesh.get_coordinate()
        assert coordinate is not None
        records = {}
        for (name, parameter) in self.model.named_parameters():
            records[name] = _parameter_local_hash_record(parameter)
        payload = {'rank': dist.get_rank(), 'coordinate': tuple((int(dim) for dim in coordinate)), 'records': records}
        replicate_group = device_mesh.get_group('dp_replicate')
        group_size = dist.get_world_size(group=replicate_group)
        gathered_payloads = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_payloads, payload, group=replicate_group)
        mismatches = []
        reference = gathered_payloads[0]
        reference_records = reference['records']
        reference_names = set(reference_records)
        for other in gathered_payloads[1:]:
            other_records = other['records']
            other_names = set(other_records)
            for name in sorted(reference_names ^ other_names):
                mismatches.append(f"param={name} rank {reference['rank']} != rank {other['rank']}")
                if len(mismatches) >= 8:
                    break
            if len(mismatches) >= 8:
                break
            for name in sorted(reference_names & other_names):
                if reference_records[name] != other_records[name]:
                    mismatches.append(f"param={name} rank {reference['rank']} != rank {other['rank']}")
                if len(mismatches) >= 8:
                    break
            if len(mismatches) >= 8:
                break
        if mismatches:
            joined = '\n'.join(mismatches)
            raise RuntimeError(f'HSDP replicated parameter hash mismatch detected {reason}. Refusing to continue because DCP checkpointing would save a mixed model.\n{joined}')
        if is_main_process():
            logging.info('[INFO] HSDP replicated parameter hash check passed %s, params=%d', reason, len(records))

    def save_checkpoint(self, epoch: int, step: int, train_loss: float, val_loss: dict[str, float] | None=None, best_metric_key: str='default'):
        self._assert_hsdp_replicated_parameter_hashes_match(f'before checkpoint save epoch={epoch} step={step}')
        return self._save_tracked_checkpoint(epoch=epoch, step=step, train_loss=train_loss, val_loss=val_loss, best_metric_key=best_metric_key)

    def _distributed_barrier(self):
        if not dist.is_initialized():
            return
        if dist.get_backend() == 'nccl':
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def _broadcast_module_state(self, module: torch.nn.Module) -> None:
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)

    def save_hf_checkpoint(self, epoch: int, step: int, output_dir: str=None):
        logging.warning(f'[HF Export] save_hf_checkpoint called on {type(self).__name__} which does not implement it — skipping.')

    def _load_model_weights_only(self, restore_from: str, checkpoint_cfg, *, no_dist: bool=False) -> None:
        ckpt_dir = restore_from
        model_path = os.path.join(ckpt_dir, 'model')
        if not os.path.exists(model_path):
            model_path = ckpt_dir
        if os.path.exists(model_path):
            logging.info(f'[INFO] Loading model weights from {model_path}')
            if no_dist:
                state_dict = self.model.state_dict()
                storage_reader = None
                if _is_safetensors_checkpoint(model_path):
                    storage_reader = _HuggingFaceStorageReader(path=model_path)
                dcp.load(state_dict, checkpoint_id=model_path, storage_reader=storage_reader, no_dist=True)
                self.model.load_state_dict(state_dict)
            else:
                from awomo.awomo.training.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
                checkpoint_config = CheckpointingConfig(enabled=True, checkpoint_dir=checkpoint_cfg.get('checkpoint_dir'), model_cache_dir=checkpoint_cfg.get('model_cache_dir', checkpoint_cfg.get('checkpoint_dir')), model_save_format=checkpoint_cfg.get('model_save_format', 'torch_save'), model_repo_id=self.model_id or '', save_consolidated=checkpoint_cfg.get('save_consolidated', False), is_peft=self.lora_enabled, model_state_dict_keys=None)
                checkpointer = Checkpointer(config=checkpoint_config, dp_rank=self._get_dp_rank(include_cp=True), tp_rank=self._get_tp_rank(), pp_rank=self._get_pp_rank(), moe_mesh=None, create_process_groups=self._checkpoint_create_process_groups())
                checkpointer.load_model(self.model, model_path, is_init_step=True)
            logging.info('[INFO] Model weights loaded successfully')
        else:
            logging.warning(f'[WARN] Model checkpoint path {model_path} does not exist, skipping weight load')

    def _build_optimizer(self):
        named_trainable_params = [(name, param) for (name, param) in self.model.named_parameters() if param.requires_grad]
        if not named_trainable_params:
            raise RuntimeError('No trainable parameters found!')
        optim_cfg = self.cfg.optim.optimizer
        weight_decay = optim_cfg.weight_decay
        betas = optim_cfg.betas
        exclude_bias_and_norm = optim_cfg.get('exclude_bias_and_norm_from_weight_decay', False)
        if exclude_bias_and_norm:
            decay_parameter_names = set(get_parameter_names(self.model, LAYERNORM_LAYERS))
            decay_parameter_names = {name for name in decay_parameter_names if 'bias' not in name}
            decay_params = [param for (name, param) in named_trainable_params if name in decay_parameter_names]
            no_decay_params = [param for (name, param) in named_trainable_params if name not in decay_parameter_names]
            optimizer_parameters = [{'params': decay_params, 'weight_decay': weight_decay, 'wd_mult': 1.0}, {'params': no_decay_params, 'weight_decay': 0.0, 'wd_mult': 0.0}]
        else:
            optimizer_parameters = [param for (_, param) in named_trainable_params]
        logging.info(f'[INFO] Optimizer betas: {tuple(betas)}')
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.learning_rate, weight_decay=weight_decay, betas=tuple(betas))
        num_trainable = sum((param.numel() for (_, param) in named_trainable_params))
        logging.info(f'[INFO] Optimizer: {len(self.optimizer.param_groups)} param groups, {num_trainable:,} total elements')

    def run_train_validation_loop(self):
        logging.info('[INFO] Starting training')
        logging.info(f'[INFO] Global batch: {self.global_batch_size}, Local batch: {self.local_batch_size}')
        logging.info(f'[INFO] Noise mode: {self.noise_mode}')
        global_step = int(self.step_scheduler.step)
        for epoch in self.step_scheduler.epochs:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            if is_main_process():
                max_steps = self.step_scheduler.max_steps
                self.step_scheduler.dataloader = tqdm(self.dataloader, desc='Training', total=max_steps * self.step_scheduler.grad_acc_steps, initial=self.step_scheduler.step * self.step_scheduler.grad_acc_steps)
            else:
                self.step_scheduler.dataloader = self.dataloader
            epoch_loss = 0.0
            num_steps = 0
            for batch_group in self.step_scheduler:
                current_step = int(self.step_scheduler.step)
                self._before_step_group()
                (loss_log_scalars, grad_norm, step_time) = self._run_train_batch_group(batch_group, current_step)
                group_loss_mean = loss_log_scalars['total_loss']
                epoch_loss += group_loss_mean
                num_steps += 1
                global_step = int(self.step_scheduler.step)
                should_log = self.log_every and self.log_every > 0 and (global_step % self.log_every == 0)
                if should_log:
                    step_flops = self._get_step_flops(batch_group)
                    (step_flops, step_time) = self._aggregate_step_metrics_across_ranks(step_flops, step_time)
                if should_log and is_main_process():
                    avg_loss = epoch_loss / num_steps
                    mfu = step_flops / (step_time * self._peak_flops_per_gpu) * 100
                    mem_alloc = torch.cuda.memory_allocated() / 1024 ** 3
                    mem_reserved = torch.cuda.memory_reserved() / 1024 ** 3
                    mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                    mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3
                    log_dict = {'train/avg_loss': avg_loss, 'train/lr': self.optimizer.param_groups[0]['lr'], 'train/grad_norm': grad_norm, 'train/step_time': step_time, 'train/mfu': mfu, 'epoch': epoch, 'global_step': global_step}
                    for (metric_name, metric_value) in loss_log_scalars.items():
                        log_dict[f'train/{metric_name}'] = metric_value
                    self._extend_train_log_dict(log_dict, group_loss_mean, grad_norm, step_time, mfu)
                    if wandb.run is not None:
                        wandb.log(log_dict, step=global_step)
                    logging.info(f"[Step {global_step}] loss={group_loss_mean:.4f} dyn={loss_log_scalars['dynamics_loss']:.4f} act={loss_log_scalars['action_loss']:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e} gn={grad_norm:.2f} t={step_time:.1f}s mfu={mfu:.2f}% mem={mem_alloc:.1f}/{mem_reserved:.1f}/{mem_peak:.1f}/{mem_total:.1f}GB (alloc/rsv/peak/total)")
                if self.step_scheduler.is_ckpt_step and self.checkpointer is not None:
                    self.save_checkpoint(epoch, global_step, epoch_loss / num_steps)
                    hf_export_cfg = self.cfg.get('checkpoint.hf_export', None)
                    if hf_export_cfg is not None and hf_export_cfg.get('enabled', True):
                        self.save_hf_checkpoint(epoch, global_step)
            if num_steps == 0:
                logging.info(f'[INFO] Epoch {epoch + 1} skipped')
                continue
            avg_loss = epoch_loss / num_steps
            logging.info(f'[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}')
            if is_main_process():
                self._on_epoch_log()
                if wandb.run is not None:
                    wandb.log({'epoch/avg_loss': avg_loss, 'epoch/num': epoch + 1}, step=global_step)
        if is_main_process():
            logging.info(f'[INFO] Training complete at step {global_step}')
            self._on_training_finish()
            if wandb.run is not None:
                wandb.finish()
