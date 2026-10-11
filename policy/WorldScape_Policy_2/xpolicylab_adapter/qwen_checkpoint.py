"""Validate the packaged Qwen component without external adapter utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

BUNDLE_MANIFEST = "interactive_qwen.json"
QWEN_DIRECTORY = "interactive_qwen"
NATIVE_MANIFEST = "checkpoint_manifest.json"


def _json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError(f"Unsafe bundle path: {value!r}")
    return value


def _regular(root, relative):
    path = root / _relative(relative)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Missing or non-local bundle file: {path}")
    return path


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _weight_files(root):
    index = root / "model.safetensors.index.json"
    single = root / "model.safetensors"
    if index.exists() and single.exists():
        raise ValueError(f"Both single and sharded weights found: {root}")
    if single.is_file():
        return [single.name]
    values = _json(index).get("weight_map")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"Invalid weight index: {index}")
    names = set()
    for name in values.values():
        _relative(name)
        if "/" in name or not name.endswith(".safetensors") or name == single.name:
            raise ValueError(f"Invalid weight shard name: {name}")
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
        names.add(name)
    return [index.name, *sorted(names)]


def _qwen_files(root):
    if _json(root / "config.json").get("model_type") != "qwen3_vl":
        raise ValueError("Expected a Qwen3-VL HF model directory")
    for name in ("tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    if not any((root / name).is_file() for name in ("chat_template.json", "chat_template.jinja")):
        if not _json(root / "tokenizer_config.json").get("chat_template"):
            raise ValueError("Qwen processor is missing its chat template")
    assets = {p.name for p in root.iterdir() if p.is_file()
              and p.suffix in {".json", ".txt", ".model", ".jinja", ".tiktoken"}}
    return sorted(assets | set(_weight_files(root)))


def bundled_qwen_path(checkpoint, *, verify_hashes=True):
    """Return the local bundled component, or None for legacy checkpoints.

    A present but broken bundle is an error, never an external-path fallback.
    Native policy verification remains the native loader's responsibility.
    """
    root = Path(checkpoint)
    manifest_path = root / BUNDLE_MANIFEST
    if not manifest_path.exists():
        if (root / QWEN_DIRECTORY).exists():
            raise ValueError("Incomplete Interactive Qwen bundle: missing manifest")
        return None
    _regular(root, BUNDLE_MANIFEST)
    manifest = _json(manifest_path)
    if manifest.get("format_version") != 1 or manifest.get("component") != QWEN_DIRECTORY:
        raise ValueError("Unsupported Interactive Qwen bundle format")
    if manifest.get("policy_manifest_sha256") != _sha256(_regular(root, NATIVE_MANIFEST)):
        raise ValueError("Interactive Qwen bundle does not match the policy manifest")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Interactive Qwen bundle has no file records")
    seen = set()
    component = root / QWEN_DIRECTORY
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid Interactive Qwen file record")
        name = _relative(record.get("path"))
        if name in seen or "/" in name:
            raise ValueError(f"Invalid or duplicate Qwen file: {name}")
        seen.add(name)
        path = _regular(root, f"{QWEN_DIRECTORY}/{name}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"Qwen file size mismatch: {name}")
        if verify_hashes and _sha256(path) != record.get("sha256"):
            raise ValueError(f"Qwen file checksum mismatch: {name}")
    if seen != set(_qwen_files(component)):
        raise ValueError("Interactive Qwen file inventory mismatch")
    return component
