"""Tensor-only model weights and separate full-training-state contracts."""
from __future__ import annotations

import copy
import re
import torch

CHECKPOINT_MODULES = (
    'mot', 'proprio_encoder', 'dino_feature_norm', 'dino_embedder',
    'dino_proj_out', 'pt_patch_embedding', 'pt_head', 'multisource_conditioning',
)


def checkpoint_tensors(model):
    """Return plain flat tensors, omitting state_dict's auxiliary attributes."""
    tensors = {}
    for module_name in CHECKPOINT_MODULES:
        for local_name, tensor in getattr(model, module_name).state_dict().items():
            if not torch.is_tensor(tensor):
                raise TypeError(f'Non-tensor model state: {module_name}.{local_name}')
            tensors[f'{module_name}.{local_name}'] = tensor
    return tensors


def validate_tensor_checkpoint(payload, expected=None):
    """Validate everything before a loader copies any model parameters."""
    if type(payload) is not dict or not payload:
        raise TypeError('Weights must be a nonempty plain dict[str, Tensor]')
    for key, tensor in payload.items():
        if type(key) is not str or not any(key.startswith(name + '.') for name in CHECKPOINT_MODULES):
            raise ValueError(f'Invalid model tensor key: {key!r}')
        if type(tensor) is not torch.Tensor:
            raise TypeError(f'Checkpoint values must be plain torch.Tensor: {key}')
        if tensor.__dict__:
            raise TypeError(f'Additional tensor attributes are not allowed: {key}')
        if tensor.device.type != 'cpu' or tensor.layout != torch.strided:
            raise TypeError(f'Checkpoint values must be dense CPU tensors: {key}')
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError(f'Unsupported model weight dtype for {key}: {tensor.dtype}')
    if expected is not None:
        missing = set(expected) - set(payload)
        unexpected = set(payload) - set(expected)
        if missing or unexpected:
            raise ValueError(f'Checkpoint keys differ: missing={sorted(missing)[:8]}, '
                             f'unexpected={sorted(unexpected)[:8]}')
        for key, tensor in payload.items():
            reference = expected[key]
            shape = tuple(reference.shape) if torch.is_tensor(reference) else tuple(reference)
            if tuple(tensor.shape) != shape:
                raise ValueError(f'Checkpoint shape differs for {key}: '
                                 f'{tuple(tensor.shape)} != {shape}')
    return payload


def read_tensor_checkpoint(path, expected=None):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return validate_tensor_checkpoint(payload, expected)


def load_model_tensors(model, path):
    expected = checkpoint_tensors(model)
    payload = read_tensor_checkpoint(path, expected)
    for module_name in CHECKPOINT_MODULES:
        prefix = module_name + '.'
        state = {key[len(prefix):]: tensor for key, tensor in payload.items()
                 if key.startswith(prefix)}
        getattr(model, module_name).load_state_dict(state, strict=True)
    return payload


def save_model_tensors(model, path):
    # Copy each visible tensor into its own exact-size CPU allocation. Saving a
    # view directly can serialize unrelated values from its underlying storage.
    payload = {key: tensor.detach().to(device='cpu', memory_format=torch.contiguous_format, copy=True)
               for key, tensor in checkpoint_tensors(model).items()}
    validate_tensor_checkpoint(payload)
    for key, tensor in payload.items():
        if (not tensor.is_contiguous() or tensor.storage_offset() != 0
                or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
            raise ValueError(f'Weight storage is not compact and independent: {key}')
    torch.save(payload, path)


def public_contract(value):
    """Canonicalize the separate full-state/data contract, never weight files."""
    if isinstance(value, dict):
        return {public_contract(k): public_contract(v) for k, v in value.items()
                if k not in {'root', 'sidecar_root', 'text_cache', 'source_path',
                             'output_dir', 'hostname', 'username', 'host', 'ip'}}
    if isinstance(value, (list, tuple)):
        return [public_contract(x) for x in value]
    if isinstance(value, str):
        if value.startswith('/') or re.search(r'(?:\d{1,3}\.){3}\d{1,3}', value):
            return 'LOCAL_RESOURCE'
    return value


def portable_manifest(value):
    result = copy.deepcopy(value)
    result['root'] = 'DATA_ROOT'
    result['source']['text_cache'] = 'TEXT_CACHE_ROOT'
    result['source']['sidecar_root'] = 'SIDECAR_ROOT'
    return result
