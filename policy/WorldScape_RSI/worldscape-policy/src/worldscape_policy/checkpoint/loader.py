from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from omegaconf import OmegaConf

from worldscape_policy.checkpoint.format import (
    NATIVE_ARTIFACT_SCHEMA_VERSION,
    REQUIRED_NATIVE_ARTIFACTS,
)
from worldscape_policy.checkpoint.transforms import (
    TRANSFORM_BUNDLE_FILENAME,
    CheckpointTransformArtifact,
)
from worldscape_policy.checkpoint.validation import ConversionResult
from worldscape_policy.checkpoint.weights_io import (
    DEFAULT_MAX_SHARD_SIZE,
    MODEL_FILENAME,
    MODEL_INDEX_FILENAME,
    checkpoint_weight_files,
    checkpoint_weights_sha256,
    checkpoint_weights_sha256_from_records,
    load_checkpoint_state_dict as load_safetensors_state_dict,
    save_checkpoint_state_dict,
)
from worldscape_policy.model_config import GenerationConfig, ModelConfig


def source_checkpoint_fingerprint(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash the complete, ordered checkpoint payload, including all shards."""

    root = Path(path)
    files = _checkpoint_source_files(root)
    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for file_path in files:
        relative = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        digest = _sha256_file(file_path)
        size = file_path.stat().st_size
        records.append({"path": relative, "size_bytes": size, "sha256": digest})
        encoded_path = relative.encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(8, "big"))
        aggregate.update(encoded_path)
        aggregate.update(size.to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest))
    return f"sha256:{aggregate.hexdigest()}", records


def _checkpoint_source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    index_path = path / MODEL_INDEX_FILENAME
    if index_path.exists() or (path / MODEL_FILENAME).exists():
        files = checkpoint_weight_files(path)
    else:
        files = [
            candidate
            for name in ("model.safetensors", "pytorch_model.bin")
            if (candidate := path / name).exists()
        ]
    if not files:
        raise FileNotFoundError(f"No supported checkpoint weights under {path}")
    missing = [item for item in files if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint shard does not exist: {missing[0]}")
    return sorted(files, key=lambda item: item.name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_state_dict(path: str | Path) -> dict[str, Any]:
    """Load a single-file or sharded Hugging Face checkpoint."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".safetensors":
        return load_safetensors_state_dict(checkpoint_path)
    if checkpoint_path.is_file():
        return _load_file(checkpoint_path)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    index_path = checkpoint_path / MODEL_INDEX_FILENAME
    if index_path.exists() or (checkpoint_path / MODEL_FILENAME).exists():
        return load_safetensors_state_dict(checkpoint_path)

    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = checkpoint_path / filename
        if candidate.exists():
            return _load_file(candidate)
    raise FileNotFoundError(
        f"No model.safetensors(.index.json) or pytorch_model.bin under {checkpoint_path}"
    )


def save_native_checkpoint(
    result: ConversionResult,
    output_dir: str | Path,
    *,
    model_variant: str,
    source_checkpoint_hash: str,
    source_files: list[dict[str, Any]],
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
    normalization: dict[str, Any],
    transform_bundle: CheckpointTransformArtifact,
    provenance: dict[str, Any],
    tokenizer_source: str | Path,
    external_dependencies: list[dict[str, Any]] | None = None,
    git_commit: str | None = None,
    max_shard_size: int | str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    """Save and validate a self-describing native checkpoint bundle."""

    ModelConfig.from_dict(model_config)
    GenerationConfig.from_dict(generation_config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "checkpoint_manifest.json"
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in result.state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    if len(tensors) != len(result.state_dict):
        raise TypeError("Native checkpoint serialization only accepts torch.Tensor values")
    save_checkpoint_state_dict(
        tensors, destination, max_shard_size=max_shard_size
    )
    tokenizer_dependency = _write_tokenizer(tokenizer_source, destination)
    _write_yaml(destination / "model_config.yaml", model_config)
    _write_yaml(destination / "generation_config.yaml", generation_config)
    (destination / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WorldScapePolicy"],
                "model_type": "worldscape_policy",
                "worldscape_policy_model_config": model_config,
                "generation_config": generation_config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (destination / "normalization.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n"
    )
    transform_bundle.write(destination / TRANSFORM_BUNDLE_FILENAME)
    provenance_record = {
        "schema_version": NATIVE_ARTIFACT_SCHEMA_VERSION,
        **provenance,
        "source_checkpoint": {
            "sha256": source_checkpoint_hash,
            "files": source_files,
        },
        "key_mapping_version": result.key_mapping_version,
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance_record, indent=2, sort_keys=True) + "\n"
    )
    manifest = result.manifest(
        model_variant=model_variant,
        source_checkpoint_hash=source_checkpoint_hash,
        source_files=source_files,
        git_commit=git_commit,
    )
    manifest.update(
        {
            "artifact_schema_version": NATIVE_ARTIFACT_SCHEMA_VERSION,
            "model_sha256": checkpoint_weights_sha256(destination),
            "dependencies": {
                "python": ["torch", "safetensors", "hydra-core", "omegaconf"],
                "components": _component_targets(model_config),
                "external_assets": [
                    tokenizer_dependency,
                    *(external_dependencies or []),
                ],
            },
        }
    )
    artifact_paths = _discover_native_artifacts(destination)
    manifest["artifacts"] = artifact_paths
    manifest["artifact_checksums"] = [
        _artifact_record(destination, path) for path in artifact_paths
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
        + "\n"
    )
    validate_native_checkpoint_artifacts(destination)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a mapping")
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(value), resolve=True))


def _component_targets(config: dict[str, Any]) -> list[str]:
    targets: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            target = value.get("_target_")
            if isinstance(target, str) and target:
                targets.add(target)
            native_target = value.get("target")
            if (
                isinstance(native_target, str)
                and native_target
                and isinstance(value.get("parameters"), dict)
            ):
                targets.add(native_target)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    return sorted(targets)


def _write_tokenizer(source: str | Path, destination: Path) -> dict[str, Any]:
    value = str(source)
    local = Path(value).expanduser()
    if local.exists():
        target = destination / "tokenizer"
        if local.is_dir():
            _copy_tokenizer_assets(local, target)
        elif local.is_file():
            target.mkdir()
            shutil.copy2(local, target / local.name)
        else:
            raise ValueError(f"Unsupported tokenizer asset: {local}")
        return {
            "name": "tokenizer",
            "kind": "bundled",
            "reference": "tokenizer",
            "required": True,
        }
    reference = {
        "schema_version": NATIVE_ARTIFACT_SCHEMA_VERSION,
        "kind": "huggingface",
        "identifier": value,
    }
    (destination / "tokenizer_reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n"
    )
    return {
        "name": "tokenizer",
        "kind": "huggingface",
        "reference": value,
        "required": True,
    }


_TOKENIZER_ASSET_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


def _copy_tokenizer_assets(source: Path, destination: Path) -> None:
    """Copy tokenizer vocabulary/config files without language-model weights."""

    assets = [
        path
        for path in source.iterdir()
        if path.is_file() and path.name in _TOKENIZER_ASSET_NAMES
    ]
    if not assets:
        raise FileNotFoundError(
            f"Tokenizer directory contains no supported tokenizer assets: {source}"
        )
    destination.mkdir()
    for path in assets:
        shutil.copy2(path, destination / path.name)


def validate_native_checkpoint_artifacts(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Validate the complete native bundle before parsing behavior artifacts."""

    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Native checkpoint directory does not exist: {directory}")
    manifest_path = directory / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Native checkpoint is missing required artifact(s): checkpoint_manifest.json"
        )
    manifest = _read_json_object(manifest_path)
    if manifest.get("format_version") != "2":
        raise ValueError(
            "Invalid native checkpoint artifacts: unsupported format_version "
            f"{manifest.get('format_version')!r}; expected '2'"
        )

    missing = [
        name
        for name in REQUIRED_NATIVE_ARTIFACTS
        if name != "checkpoint_manifest.json" and not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Native checkpoint is missing required artifact(s): " + ", ".join(missing)
        )
    failures = _validate_artifact_inventory(directory, manifest)
    if failures:
        raise ValueError("Invalid native checkpoint artifacts: " + "; ".join(failures))

    tokenizer_reference_path = directory / "tokenizer_reference.json"
    provenance = _read_json_object(directory / "provenance.json")
    normalization = _read_json_object(directory / "normalization.json")
    try:
        transform_bundle = CheckpointTransformArtifact.read(
            directory / TRANSFORM_BUNDLE_FILENAME
        )
    except (OSError, ValueError, TypeError) as error:
        transform_bundle = None
        failures = [f"{TRANSFORM_BUNDLE_FILENAME} is invalid: {error}"]
    else:
        failures = []
    model_config = OmegaConf.to_container(
        OmegaConf.load(directory / "model_config.yaml"), resolve=True
    )
    generation_config = OmegaConf.to_container(
        OmegaConf.load(directory / "generation_config.yaml"), resolve=True
    )
    if tokenizer_reference_path.is_file():
        tokenizer_reference = _read_json_object(tokenizer_reference_path)
        if (
            tokenizer_reference.get("schema_version")
            != NATIVE_ARTIFACT_SCHEMA_VERSION
            or tokenizer_reference.get("kind") != "huggingface"
            or not isinstance(tokenizer_reference.get("identifier"), str)
            or not tokenizer_reference["identifier"]
        ):
            failures.append("tokenizer_reference.json has an invalid schema")
    for key in (
        "format_version",
        "artifact_schema_version",
        "model_variant",
        "wam_plugin",
        "source_checkpoint_hash",
        "source_files",
        "key_mapping_version",
        "loaded_keys",
        "missing_keys",
        "unexpected_keys",
        "groups",
        "model_sha256",
        "artifacts",
        "artifact_checksums",
        "dependencies",
    ):
        if key not in manifest:
            failures.append(f"manifest missing {key!r}")
    if manifest.get("artifact_schema_version") != NATIVE_ARTIFACT_SCHEMA_VERSION:
        failures.append("manifest has unsupported artifact_schema_version")
    dependencies = manifest.get("dependencies")
    if (
        not isinstance(dependencies, dict)
        or not isinstance(dependencies.get("python"), list)
        or not isinstance(dependencies.get("components"), list)
        or not isinstance(dependencies.get("external_assets"), list)
    ):
        failures.append(
            "manifest dependencies must declare python, components, and external_assets"
        )
    else:
        for index, dependency in enumerate(dependencies["external_assets"]):
            if not isinstance(dependency, dict) or not all(
                key in dependency
                for key in ("name", "kind", "reference", "required")
            ):
                failures.append(
                    f"manifest external_assets[{index}] has an invalid schema"
                )
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        failures.append("manifest source_files must be a non-empty list")
    else:
        for index, record in enumerate(source_files):
            if not isinstance(record, dict) or not all(
                key in record for key in ("path", "size_bytes", "sha256")
            ):
                failures.append(f"manifest source_files[{index}] has an invalid schema")
    for key in ("loaded_keys", "missing_keys", "unexpected_keys"):
        if not isinstance(manifest.get(key), list):
            failures.append(f"manifest {key} must be a list")
    if provenance.get("schema_version") != NATIVE_ARTIFACT_SCHEMA_VERSION:
        failures.append("provenance has unsupported schema_version")
    provenance_source = provenance.get("source_checkpoint")
    if not isinstance(provenance_source, dict) or provenance_source.get(
        "sha256"
    ) != manifest.get("source_checkpoint_hash"):
        failures.append("provenance and manifest source hashes differ")
    if normalization.get("schema_version") != NATIVE_ARTIFACT_SCHEMA_VERSION:
        failures.append("normalization has unsupported schema_version")
    visual = normalization.get("visual")
    if not isinstance(visual, dict) or visual.get("input_range") not in {
        "uint8",
        "zero_one",
        "minus_one_one",
    }:
        failures.append("normalization.visual.input_range is invalid")
    try:
        ModelConfig.from_dict(model_config)
    except (TypeError, ValueError) as error:
        failures.append(f"model_config.yaml is invalid: {error}")
    try:
        GenerationConfig.from_dict(generation_config)
    except (TypeError, ValueError) as error:
        failures.append(f"generation_config.yaml is invalid: {error}")
    try:
        actual_model_hash = checkpoint_weights_sha256_from_records(
            directory,
            manifest.get("artifact_checksums", []),
        )
    except (FileNotFoundError, ValueError) as error:
        failures.append(f"model weights are invalid: {error}")
    else:
        if manifest.get("model_sha256") != actual_model_hash:
            failures.append("model_sha256 does not match checkpoint weights")
    if transform_bundle is not None and not transform_bundle.embodiments:
        failures.append("transform bundle has no embodiments")
    if failures:
        raise ValueError("Invalid native checkpoint artifacts: " + "; ".join(failures))
    return manifest


def _artifact_record(directory: Path, relative: str) -> dict[str, Any]:
    path = directory / relative
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _safe_artifact_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or value != path.as_posix()
    ):
        return None
    return path.as_posix()


def _discover_native_artifacts(directory: Path) -> list[str]:
    paths = {
        "provenance.json",
        "model_config.yaml",
        "generation_config.yaml",
        "normalization.json",
        TRANSFORM_BUNDLE_FILENAME,
    }
    single_model = directory / "model.safetensors"
    index_path = directory / "model.safetensors.index.json"
    if single_model.is_file() and index_path.exists():
        raise ValueError("Native checkpoint cannot contain both single and sharded models")
    if single_model.is_file():
        paths.add("model.safetensors")
    elif index_path.is_file():
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has an invalid weight_map")
        paths.add("model.safetensors.index.json")
        for shard in weight_map.values():
            safe = _safe_artifact_path(shard)
            if safe is None or "/" in safe or not safe.endswith(".safetensors"):
                raise ValueError("model index contains an unsafe shard path")
            paths.add(safe)
    else:
        raise FileNotFoundError("Native checkpoint requires model weights")

    tokenizer_dir = directory / "tokenizer"
    tokenizer_reference = directory / "tokenizer_reference.json"
    if tokenizer_dir.exists() and tokenizer_reference.exists():
        raise ValueError("Native checkpoint must use exactly one tokenizer source")
    if tokenizer_dir.is_dir():
        tokenizer_files = []
        for path in tokenizer_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError("Tokenizer bundle must not contain symbolic links")
            if path.is_file():
                tokenizer_files.append(path.relative_to(directory).as_posix())
        if not tokenizer_files:
            raise ValueError("Bundled tokenizer directory is empty")
        paths.update(tokenizer_files)
    elif tokenizer_reference.is_file():
        paths.add("tokenizer_reference.json")
    else:
        raise FileNotFoundError(
            "Native checkpoint requires tokenizer/ or tokenizer_reference.json"
        )

    # Additional model shards or behavior-defining configs are never silently ignored.
    paths.update(path.name for path in directory.glob("*.safetensors") if path.is_file())
    if (directory / "config.json").exists():
        paths.add("config.json")
    return sorted(paths)


def _validate_artifact_inventory(
    directory: Path, manifest: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    records = manifest.get("artifact_checksums")
    declared = manifest.get("artifacts")
    if not isinstance(records, list) or not isinstance(declared, list):
        return ["manifest artifacts and artifact_checksums must be lists"]
    record_paths: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            failures.append(f"artifact_checksums[{index}] has an invalid schema")
            continue
        relative = _safe_artifact_path(record["path"])
        if relative is None:
            failures.append(f"artifact_checksums[{index}] has an unsafe path")
            continue
        record_paths.append(relative)
        path = directory / relative
        if not path.is_file() or path.is_symlink():
            failures.append(f"artifact {relative!r} is missing or not a regular file")
            continue
        if record["size_bytes"] != path.stat().st_size:
            failures.append(f"artifact {relative!r} size does not match manifest")
        actual = _sha256_file(path)
        if record["sha256"] != actual:
            failures.append(f"artifact {relative!r} checksum does not match manifest")
    if len(record_paths) != len(set(record_paths)):
        failures.append("manifest contains duplicate artifact paths")
    safe_declared = [_safe_artifact_path(path) for path in declared]
    if None in safe_declared:
        failures.append("manifest artifacts contains an unsafe path")
    elif sorted(safe_declared) != sorted(record_paths):
        failures.append("manifest artifacts and checksum paths differ")
    if failures:
        return failures
    try:
        expected = _discover_native_artifacts(directory)
    except (FileNotFoundError, ValueError) as error:
        failures.append(str(error))
    else:
        if sorted(record_paths) != expected:
            missing = sorted(set(expected) - set(record_paths))
            extra = sorted(set(record_paths) - set(expected))
            if missing:
                failures.append(f"manifest is missing expected artifacts: {missing}")
            if extra:
                failures.append(f"manifest has extra expected artifacts: {extra}")
    return failures


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def load_native_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Fail closed while loading a complete WorldScape native checkpoint."""

    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Native checkpoint directory does not exist: {directory}")
    manifest = validate_native_checkpoint_artifacts(directory)

    state_dict = load_checkpoint_state_dict(directory)
    target = model.state_dict()
    source_keys = set(state_dict)
    target_keys = set(target)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    shape_mismatches = sorted(
        key
        for key in source_keys & target_keys
        if tuple(state_dict[key].shape) != tuple(target[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append(f"{len(missing)} missing key(s)")
        if unexpected:
            details.append(f"{len(unexpected)} unexpected key(s)")
        if shape_mismatches:
            details.append(f"{len(shape_mismatches)} shape mismatch(es)")
        raise ValueError("Native checkpoint validation failed: " + ", ".join(details))
    model.load_state_dict(state_dict, strict=True)
    return manifest


def _load_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint {path} did not contain a state dict")
    return dict(loaded)
