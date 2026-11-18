"""Frozen, checkpoint-local VLM features with a separately verified source identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import torch
from torch import nn

from .protocol import AutoPlanningFeatures

_METADATA_FILES = ("manifest.json", "encoder_config.json")
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value, label):
    if not isinstance(value, str) or _HEX_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("VLM bundle paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts) or path.as_posix() != value or value == ".":
        raise ValueError(f"Unsafe VLM bundle path: {value!r}")
    return path


def _safe_asset(root, relative, *, directory=False):
    parts = _relative_path(relative).parts
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"VLM bundle must not contain symbolic links: {relative}")
    if directory:
        if not path.is_dir():
            raise ValueError(f"Missing VLM bundle directory: {relative}")
    elif not path.is_file():
        raise ValueError(f"Missing VLM bundle file: {relative}")
    return path


def resolve_vlm_assets_dir(checkpoint_dir, assets_dir):
    """Resolve bundled relative paths from their checkpoint, never from cwd.

    Absolute paths remain usable for explicitly selected resources. Relative
    paths are confined to the checkpoint and cannot traverse symlink parents.
    """
    value = str(assets_dir)
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("VLM assets must be a real directory")
        return path.resolve()
    if checkpoint_dir is None:
        raise ValueError("Relative VLM assets_dir requires checkpoint_dir")
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    return _safe_asset(root, value, directory=True)


def _root_path(root):
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("VLM assets must be a real directory")
    return path.resolve()


def bundle_identity(root):
    """Identity of the portable manifest and feature configuration bytes."""
    root = _root_path(root)
    digest = hashlib.sha256()
    for name in _METADATA_FILES:
        digest.update(name.encode())
        digest.update(_safe_asset(root, name).read_bytes())
    return digest.hexdigest()


def _read_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"VLM metadata must contain an object: {path.name}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundled_assets(root, *, expected_identity=None,
                            feature_source_identity=None, full_hashes=False,
                            weight_storage="separate_assets"):
    """Validate bundle metadata, safe paths and sizes; optionally hash all files.

    Native checkpoint validation already hashes the outer artifact inventory.
    The optional full pass is useful for independent bundle export validation.
    The source identity certifies which precomputed features this bundle uses;
    it is distinct from the identity of the renamed portable metadata.
    """
    root = _root_path(root)
    identity = bundle_identity(root)
    if expected_identity is not None:
        if identity != _digest(expected_identity, "bundle_identity"):
            raise ValueError("VLM bundle identity differs from checkpoint")
    manifest = _read_object(_safe_asset(root, "manifest.json"))
    config = _read_object(_safe_asset(root, "encoder_config.json"))
    source_identity = _digest(manifest.get("feature_source_identity"), "feature_source_identity")
    if feature_source_identity is not None and source_identity != _digest(feature_source_identity, "feature_source_identity"):
        raise ValueError("VLM feature source identity differs from checkpoint")
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("VLM bundle manifest must contain nonempty files")
    for relative, record in records.items():
        path = _safe_asset(root, relative)
        if not isinstance(record, dict) or type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ValueError(f"Invalid VLM file size record: {relative}")
        checksum = _digest(record.get("sha256"), f"VLM checksum for {relative}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Incomplete VLM asset: {relative}")
        if full_hashes and _sha256(path) != checksum:
            raise ValueError(f"VLM checksum mismatch: {relative}")
    if config.get("weight_storage", "separate_assets") != weight_storage:
        raise ValueError("VLM weight storage differs from the configured encoder")
    base = config.get("base_directory")
    _safe_asset(root, base, directory=True)
    required = ["encoder_config.json", f"{base}/config.json"]
    if weight_storage == "separate_assets":
        required += [config.get("lora_file"), f"{base}/model.safetensors.index.json"]
    elif weight_storage == "native_model_shards":
        if config.get("lora_file") is not None:
            raise ValueError("Native VLM must not reference a separate adapter weight file")
        if any(name.endswith((".safetensors", ".bin", "model.safetensors.index.json")) for name in records):
            raise ValueError("Native VLM assets must contain only processor and architecture metadata")
    else:
        raise ValueError("Unknown VLM weight storage")
    for name in required:
        _relative_path(name)
        if name not in records:
            raise ValueError(f"VLM manifest must declare {name!r}")
    if weight_storage == "separate_assets":
        index = _read_object(_safe_asset(root, f"{base}/model.safetensors.index.json"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("VLM model index must contain a nonempty weight_map")
        for shard in weight_map.values():
            path = _relative_path(shard)
            if len(path.parts) != 1 or not shard.endswith(".safetensors") or f"{base}/{shard}" not in records:
                raise ValueError("VLM model index contains an undeclared or unsafe shard")
    if config.get("text_layout") != "prefix_v2" or config.get("teacher_input_color_order") != "bgr":
        raise ValueError("VLM bundle requires prefix_v2 features and BGR inputs")
    parameters = config.get("encoder")
    expected = dict(qwen_layers=[9, 18, 27], summary_layers=[18, 27, 36],
                    history_num_slots=20, history_period_s=1.0, history_fps=25.0,
                    history_pool=[4, 4])
    if not isinstance(parameters, dict) or any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError("VLM bundle differs from the trained feature/history contract")
    if not parameters.get("lora", {}).get("enabled"):
        raise ValueError("VLM bundle requires the trained adapter")
    return dict(root=root, config=config, manifest=manifest,
                bundle_identity=identity, feature_source_identity=source_identity)


class CachedVLMFeatureEncoder(nn.Module):
    """Frozen VLM kept outside the policy state and loaded only for inference."""
    output_dim = 7680

    def __init__(self, *, assets_dir, bundle_identity, feature_source_identity,
                 teacher_color_order="bgr", attn_implementation="sdpa", cache_only=False):
        super().__init__()
        if teacher_color_order != "bgr" or attn_implementation != "sdpa":
            raise ValueError("VLM feature contract requires BGR and SDPA")
        checked = validate_bundled_assets(assets_dir, expected_identity=bundle_identity,
                                          feature_source_identity=feature_source_identity)
        self.assets_dir = str(checked["root"])
        self.bundle_identity = checked["bundle_identity"]
        self.feature_source_identity = checked["feature_source_identity"]
        self.cache_only = cache_only
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        object.__setattr__(self, "_teacher", None)

    def train(self, mode=True):
        super().train(False)
        return self

    def encode_cached(self, features, mask):
        if (features.ndim != 3 or features.shape[-1] != self.output_dim
                or mask is None or mask.dtype != torch.bool or mask.shape != features.shape[:2]
                or not torch.isfinite(features).all()):
            raise ValueError("Cached VLM features must be finite [B,L,7680] with bool[B,L] mask")
        return AutoPlanningFeatures(perception_features=features.detach(),
                                    direct_condition_mask=mask, semantic_supervision=False)

    def encode_planning(self, **kwargs):
        if self.cache_only:
            raise RuntimeError("Cached VLM training requires precomputed features; online fallback is disabled")
        if self._teacher is None:
            from .bundled_features import BundledVLMFeatureEncoder
            teacher = BundledVLMFeatureEncoder(
                assets_dir=self.assets_dir, bundle_identity=self.bundle_identity,
                feature_source_identity=self.feature_source_identity,
                device=self._device_anchor.device,
                teacher_color_order="bgr", attn_implementation="sdpa")
            object.__setattr__(self, "_teacher", teacher)
        self._teacher.to(device=self._device_anchor.device)
        return self._teacher.encode_planning(**kwargs)
