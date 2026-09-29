#!/usr/bin/env python3
"""Convert a training checkpoint into a releasable CogWAM inference artifact.

Training writes a raw ``state_dict`` (``torch.save(accelerator.get_state_dict(model))``)
into ``<run>/checkpoints/steps_<N>_pytorch_model.pt``. That file is a fine resume
target but a poor release format: it is a pickle, it carries no integrity metadata,
and it sits next to four other 7.8 GiB siblings.

This tool turns one such file into an artifact directory::

    <artifact>/
      model.safetensors        the same tensors, same key names, BF16
      checkpoint_keys.json     key -> [dtype, shape] inventory, for fast validation
      dataset_statistics.json  action/state normalisation stats (copied verbatim)
      artifact_manifest.json   sha256 over everything above
      inference_config.yaml    optional, written when --config is given

Key names are preserved byte-for-byte. They are derived from the framework's
``nn.Module`` attribute names, never from the class name, so renaming the
framework class does not touch a single key. See README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

MODEL_WEIGHTS_FILENAME = "model.safetensors"
KEY_INVENTORY_FILENAME = "checkpoint_keys.json"
DATASET_STATISTICS_FILENAME = "dataset_statistics.json"
MANIFEST_FILENAME = "artifact_manifest.json"
INFERENCE_CONFIG_FILENAME = "inference_config.yaml"

MANIFEST_VERSION = "cogwam_inference_artifact_v1"

# The four top-level state-dict prefixes, and the module attribute each comes from.
# A checkpoint carrying anything else is not a CogWAM checkpoint.
EXPECTED_PREFIXES: dict[str, str] = {
    "action_model": "CogWAM.action_model (causal DINO mixture-of-transformers)",
    "qwen_vl_interface": "CogWAM.qwen_vl_interface (RynnBrain1.1 / Qwen3.5 backbone)",
    "world_plan_queries": "CogWAM.world_plan_queries (learned query bank)",
    "action_plan_queries": "CogWAM.action_plan_queries (learned query bank)",
}

# Identity today. It exists so that a future rename of one of the four module
# attributes above has exactly one place to land, instead of being scattered
# across the loader, the exporter and every published artifact.
LEGACY_PREFIX_REMAP: dict[str, str] = {}

# The DINO teacher is attached with object.__setattr__ and its normalisation
# buffers are registered with persistent=False, so it never enters state_dict().
# Anything that looks like DINO weights means the model tree changed shape.
FORBIDDEN_PREFIXES = ("dino", "_dino")

_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_state_dict(source: Path) -> dict[str, torch.Tensor]:
    if source.suffix == ".safetensors":
        return safetensors_load(str(source))
    if source.suffix == ".pt":
        # weights_only rejects arbitrary pickles; mmap keeps the 7.8 GiB read lazy.
        state = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, dict):
            raise SystemExit(f"{source}: expected a state_dict, got {type(state).__name__}")
        return state
    raise SystemExit(f"{source}: unsupported suffix (want .pt or .safetensors)")


def apply_prefix_remap(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not LEGACY_PREFIX_REMAP:
        return state
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        head, _, tail = key.partition(".")
        new_head = LEGACY_PREFIX_REMAP.get(head, head)
        remapped[f"{new_head}.{tail}" if tail else new_head] = tensor
    return remapped


def validate_keys(state: dict[str, torch.Tensor]) -> dict[str, int]:
    counts = Counter(key.split(".", 1)[0] for key in state)
    unexpected = sorted(set(counts) - set(EXPECTED_PREFIXES))
    if unexpected:
        raise SystemExit(f"Unexpected top-level prefixes: {unexpected}")
    missing = sorted(set(EXPECTED_PREFIXES) - set(counts))
    if missing:
        raise SystemExit(f"Missing top-level prefixes: {missing}")
    forbidden = sorted(k for k in state if k.startswith(FORBIDDEN_PREFIXES))
    if forbidden:
        raise SystemExit(
            f"Checkpoint carries {len(forbidden)} DINO-like tensors (e.g. {forbidden[:3]}). "
            "The DINO teacher must stay outside the module tree."
        )
    return dict(counts)


def describe_geometry(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Recover the few shapes a reader needs to know before loading."""

    def shape_of(key: str) -> list[int] | None:
        tensor = state.get(key)
        return list(tensor.shape) if tensor is not None else None

    action_queries = shape_of("action_plan_queries.embedding")
    world_queries = shape_of("world_plan_queries.embedding")
    lm_head = shape_of("qwen_vl_interface.model.lm_head.weight")
    world_input = shape_of("action_model.world_input.weight")
    action_input = shape_of("action_model.action_input.weight")
    layer_ids = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("action_model.layers.") and key.split(".")[2].isdigit()
    }
    return {
        "action_horizon": action_queries[1] if action_queries else None,
        "num_world_queries": world_queries[1] if world_queries else None,
        "planner_dim": action_queries[2] if action_queries else None,
        "vocab_size": lm_head[0] if lm_head else None,
        "world_hidden_size": world_input[0] if world_input else None,
        "dino_embed_dim": world_input[1] if world_input else None,
        "action_hidden_size": action_input[0] if action_input else None,
        "action_dim": action_input[1] if action_input else None,
        "num_mot_layers": len(layer_ids),
    }


def build_key_inventory(state: dict[str, torch.Tensor]) -> dict[str, list[Any]]:
    return {key: [str(tensor.dtype).removeprefix("torch."), list(tensor.shape)] for key, tensor in sorted(state.items())}


def unshare_storage(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """safetensors refuses tensors that alias the same storage.

    RynnBrain1.1 sets ``tie_word_embeddings: true``, so ``lm_head.weight`` and
    ``embed_tokens.weight`` may or may not alias depending on how the trainer
    gathered the state dict. Clone whichever duplicates we find rather than
    dropping a key -- a released artifact must stay loadable with strict=True.
    """
    seen: dict[tuple[int, int], str] = {}
    cloned: list[str] = []
    result: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        tensor = tensor.contiguous()
        fingerprint = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if fingerprint[0] != 0 and fingerprint in seen:
            tensor = tensor.clone()
            cloned.append(key)
        else:
            seen[fingerprint] = key
        result[key] = tensor
    return result, cloned


def find_dataset_statistics(source: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"--dataset-statistics {explicit} does not exist")
        return explicit
    # <run>/checkpoints/steps_N_pytorch_model.pt -> <run>/dataset_statistics.json
    for parent in list(source.parents)[:4]:
        candidate = parent / DATASET_STATISTICS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="training checkpoint (.pt or .safetensors)")
    parser.add_argument("--output-dir", required=True, type=Path, help="artifact directory to create")
    parser.add_argument("--dataset-statistics", type=Path, default=None, help="override stats auto-discovery")
    parser.add_argument("--config", type=Path, default=None, help="training config to copy as inference_config.yaml")
    parser.add_argument("--run-id", default=None, help="run identifier recorded in the manifest")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing artifact directory")
    args = parser.parse_args()

    source: Path = args.source
    if not source.is_file():
        raise SystemExit(f"--source {source} does not exist")
    out: Path = args.output_dir
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists (pass --overwrite to replace)")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[convert] reading {source} ({source.stat().st_size / 2**30:.2f} GiB)", flush=True)
    state = apply_prefix_remap(load_state_dict(source))
    counts = validate_keys(state)
    geometry = describe_geometry(state)
    print(f"[convert] {len(state)} tensors; prefixes {counts}", flush=True)
    print(f"[convert] geometry {geometry}", flush=True)

    state, cloned = unshare_storage(state)
    if cloned:
        print(f"[convert] un-shared {len(cloned)} aliased tensors: {cloned}", flush=True)

    inventory = build_key_inventory(state)
    total_numel = sum(tensor.numel() for tensor in state.values())
    dtypes = sorted({str(tensor.dtype).removeprefix("torch.") for tensor in state.values()})

    weights_path = out / MODEL_WEIGHTS_FILENAME
    print(f"[convert] writing {weights_path}", flush=True)
    safetensors_save(
        state,
        str(weights_path),
        metadata={"format": "pt", "artifact_version": MANIFEST_VERSION},
    )
    del state

    (out / KEY_INVENTORY_FILENAME).write_text(json.dumps(inventory, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    files: dict[str, dict[str, Any]] = {}
    stats_path = find_dataset_statistics(source, args.dataset_statistics)
    if stats_path is not None:
        shutil.copyfile(stats_path, out / DATASET_STATISTICS_FILENAME)
        print(f"[convert] copied dataset statistics from {stats_path}", flush=True)
    else:
        print("[convert][WARN] no dataset_statistics.json found; the policy server cannot un-normalise actions")

    if args.config is not None:
        if not args.config.is_file():
            raise SystemExit(f"--config {args.config} does not exist")
        shutil.copyfile(args.config, out / INFERENCE_CONFIG_FILENAME)

    for name in (MODEL_WEIGHTS_FILENAME, KEY_INVENTORY_FILENAME, DATASET_STATISTICS_FILENAME, INFERENCE_CONFIG_FILENAME):
        path = out / name
        if path.is_file():
            # safetensors honours the process umask, which on many clusters is
            # 0077. A release artifact nobody else can read is not a release.
            path.chmod(0o644)
            files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            print(f"[convert] sha256 {name} = {files[name]['sha256']}", flush=True)

    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "model": {
            "framework": "CogWAM",
            "checkpoint_kind": "inference_model_state",
            "weight_format": "safetensors",
            "run_id": args.run_id,
            "source_checkpoint": source.name,
            "tensor_count": len(inventory),
            "total_numel": total_numel,
            "dtypes": dtypes,
            "prefix_counts": counts,
            "geometry": geometry,
            # Covers names/shapes/dtypes only -- two runs of the same recipe share
            # it. Use files["model.safetensors"].sha256 to tell weights apart.
            "key_inventory_sha256": sha256_json(inventory),
        },
        "backbones": {
            "note": "Backbone weights are NOT redistributed here; see the Installation section of README.md.",
            "vlm": {"model_id": "RynnBrain1.1-2B", "env": "COGWAM_BASE_VLM", "license": "apache-2.0"},
            "dino": {
                "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "env": "COGWAM_DINO_MODEL",
                "license": "dinov3-license (gated)",
            },
        },
        "files": files,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[convert] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
