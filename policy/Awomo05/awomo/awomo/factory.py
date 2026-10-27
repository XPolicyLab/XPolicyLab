from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from awomo.awomo.utils.logging_config import get_logger

logger = get_logger(__name__)


def _to_dict(value, name: str) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def create_imagewam_flux2_klein_actionpatch_vl(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    vl_model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    vl_qwen_layers=(9, 18, 27),
    vl_max_prompt_len: int = 768,
    vl_lora=None,
    vl_attn_implementation: str = "sdpa",
    vl_ar=None,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    target_views=None,
    view_time_stride: float = 1.0,
    history_max_slots: int | None = None,
    history_pool_grid=(4, 4),
    history_vae_micro_batch: int = 16,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from awomo.awomo.model_vl import ImageWAMActionPatchVL, VLPromptEncoder

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    ar_cfg = dict(_to_dict(vl_ar, "vl_ar"))
    for legacy in ("enabled", "boundary_threshold"):
        if legacy in ar_cfg:
            logger.warning("vl_ar.%s is a v1 option and is ignored in VL-v2.", legacy)
            ar_cfg.pop(legacy)

    model = ImageWAMActionPatchVL.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=None,
        qwen_context_len=int(vl_max_prompt_len),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=False,
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
        target_views=(None if target_views is None else [int(v) for v in target_views]),
        view_time_stride=float(view_time_stride),
        history_max_slots=(None if history_max_slots is None else int(history_max_slots)),
        history_pool_grid=tuple(int(v) for v in history_pool_grid),
        history_vae_micro_batch=int(history_vae_micro_batch),
    )
    enc = VLPromptEncoder(
        vl_model_path,
        qwen_layers=tuple(int(l) for l in vl_qwen_layers),
        summary_layers=tuple(int(l) for l in ar_cfg.get("summary_layers", (18, 27, 36))),
        max_prompt_len=int(vl_max_prompt_len),
        device=device,
        torch_dtype=model_dtype,
        lora=_to_dict(vl_lora, "vl_lora"),
        attn_implementation=str(vl_attn_implementation),
        image_size=tuple(ar_cfg.get("image_size", (448, 448))),
        history_image_size=tuple(ar_cfg.get("history_image_size", (448, 448))),
        history_pool=tuple(ar_cfg.get("history_pool", (4, 4))),
        history_num_slots=int(ar_cfg.get("history_num_slots", 20)),
        history_period_s=float(ar_cfg.get("history_period_s", 1.0)),
        history_fps=float(ar_cfg.get("history_fps", 25.0)),
        max_new_tokens=int(ar_cfg.get("max_new_tokens", 24)),
        decode_every=int(ar_cfg.get("decode_every", 4)),
    )
    model.attach_vl_encoder(enc)
    return model


# Published checkpoint bundle.
# Copyright (c) 2026 Awomo-WAM Team. SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import re
from pathlib import Path

def validate_checkpoint_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or set(payload) != {'dit', 'dit_config', 'qwen_lora'}:
        raise ValueError('model.pt must contain exactly dit, dit_config and qwen_lora')
    if not all((isinstance(payload[name], dict) for name in payload)):
        raise ValueError('Checkpoint model, architecture and adapter entries must be dictionaries')
    if not payload['dit'] or not payload['dit_config'] or (not payload['qwen_lora']):
        raise ValueError('Checkpoint entries must not be empty')

def canonical_lora(name: str) -> str:
    match = re.search('language_model\\.(layers\\.\\d+\\..+)\\.lora_([AB])(?:\\.default)?\\.weight$', name)
    if not match:
        raise ValueError(f'Unsupported VLM adapter key: {name}')
    return match[1] + '.lora_' + match[2]

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_checkpoint_config(directory: str | Path, *, verify_checksums: bool=True,
                           allow_base_only: bool=False) -> dict:
    directory = Path(directory)
    base_only = allow_base_only and not (directory / 'awomo_config.json').exists()
    required = {'model.pt'} if base_only else {'model.pt', 'awomo_config.json', 'dataset_stats.json'}
    checksums = {}
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        (digest, name) = line.split(maxsplit=1)
        if not re.fullmatch('[0-9a-f]{64}', digest) or name != Path(name).name or name in checksums:
            raise ValueError('Invalid checkpoint checksum manifest')
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or indirect checkpoint file: {name}')
        if verify_checksums and file_sha256(path) != digest:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
        checksums[name] = digest
    if not required.issubset(checksums):
        raise ValueError('Incomplete prepared checkpoint')
    if base_only:
        if set(checksums) != {'model.pt'}:
            raise ValueError('Pretrain SHA256SUMS must contain exactly model.pt')
        return {'artifact_kind': 'base', 'vl_text_layout': 'prefix_v2'}
    config = json.loads((directory / 'awomo_config.json').read_text())
    if config.get('model_type') != 'awomo-0.5i-robodojo-sft' or config.get('format_version') != 1:
        raise ValueError('Unsupported prepared checkpoint format')
    if config.get('artifact_kind') not in {'base', 'sft'} or config.get('parameter_dtype') != 'float32':
        raise ValueError('Expected FP32 pretrain or SFT weights')
    return config

def load_checkpoint(model, vl_encoder, directory: str | Path, *, verify_checksums: bool=True,
                    allow_base_only: bool=False) -> dict:
    directory = Path(directory)
    config = read_checkpoint_config(directory, verify_checksums=verify_checksums, allow_base_only=allow_base_only)
    payload = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    validate_checkpoint_payload(payload)
    if config['vl_text_layout'] != vl_encoder.text_layout:
        raise ValueError('VLM text-layout mismatch')
    saved = payload['dit']
    current = model.state_dict()
    if set(saved) != set(current):
        raise ValueError('Prepared model keys do not match the constructed model')
    for (name, tensor) in saved.items():
        if tensor.dtype != torch.float32 or tensor.shape != current[name].shape or (not torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid prepared model tensor: {name}')
        if current[name].dtype != torch.float32:
            raise ValueError(f'Construct FP32 master parameters before loading: {name}')
    model.load_state_dict(saved, strict=True)
    parameters = {name: p for (name, p) in vl_encoder.named_parameters() if 'lora_' in name}
    mapped = {canonical_lora(name): name for name in parameters}
    adapters = payload['qwen_lora']
    if len(mapped) != len(parameters) or set(mapped) != set(adapters):
        raise ValueError('Prepared VLM adapter keys do not match the constructed encoder')
    with torch.no_grad():
        for (canonical, name) in mapped.items():
            (value, parameter) = (adapters[canonical], parameters[name])
            if value.dtype != parameter.dtype or value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f'Invalid prepared adapter shape/dtype: {canonical}')
            if not torch.isfinite(value).all():
                raise ValueError(f'Non-finite prepared adapter: {canonical}')
            parameter.copy_(value.to(parameter.device))
    return {'path': str(directory), 'model_keys': len(saved), 'lora_keys': len(adapters), 'artifact_kind': config['artifact_kind']}
