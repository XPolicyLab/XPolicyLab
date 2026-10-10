#!/usr/bin/env python3
"""Download and validate the released StarVLA RoboDojo checkpoints.

The Hugging Face repositories are training-run directories rather than
``transformers.from_pretrained`` directories.  StarVLA therefore needs the
checkpoint, both YAML files, and ``dataset_statistics.json`` to remain in the
same run-directory layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


POLICY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "hf_robodojo_checkpoints.json"
VARIANT_ALIASES = {
    "oft": "oft",
    "qwen_oft": "oft",
    "qwenoft": "oft",
    "groot": "groot",
    "gr00t": "groot",
    "qwen_groot": "groot",
    "qwengroot": "groot",
    "pi": "pi_v3",
    "pi_v3": "pi_v3",
    "piv3": "pi_v3",
    "qwenpi_v3": "pi_v3",
}


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}: {manifest.get('schema_version')!r}")
    return manifest


def resolve_variant(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    try:
        return VARIANT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown StarVLA RoboDojo variant {value!r}; choose 'oft', 'groot', or 'pi_v3'."
        ) from exc


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _validate_stat_vector(stats: dict[str, Any], modality: str, name: str, dim: int) -> None:
    vector = _nested(stats, "arx_x5", modality, name)
    if not isinstance(vector, list) or len(vector) != dim:
        raise ValueError(
            f"dataset_statistics.json arx_x5.{modality}.{name} must have {dim} values; "
            f"got {None if vector is None else len(vector)}"
        )


def validate_checkpoint_dir(
    run_dir: Path,
    spec: dict[str, Any],
    runtime: dict[str, Any],
    *,
    verify_weight_hash: bool = True,
) -> Path:
    """Validate files and the inference contract, then return the checkpoint path."""

    run_dir = run_dir.expanduser().absolute()
    checkpoint_path = run_dir / spec["checkpoint_path"]
    required = [
        checkpoint_path,
        run_dir / "config.yaml",
        run_dir / "config.full.yaml",
        run_dir / "dataset_statistics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete StarVLA checkpoint directory; missing: " + ", ".join(missing))

    expected_size = int(spec["checkpoint_size"])
    actual_size = checkpoint_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Checkpoint size mismatch for {checkpoint_path}: expected {expected_size}, got {actual_size}. "
            "Remove the partial file and download it again."
        )

    with (run_dir / "config.yaml").open("r", encoding="utf-8") as stream:
        compact_config = yaml.safe_load(stream) or {}
    with (run_dir / "config.full.yaml").open("r", encoding="utf-8") as stream:
        full_config = yaml.safe_load(stream) or {}
    with (run_dir / "dataset_statistics.json").open("r", encoding="utf-8") as stream:
        stats = json.load(stream)

    framework = _nested(compact_config, "framework", "name")
    if framework != spec["framework"]:
        raise ValueError(f"Expected framework {spec['framework']!r}, got {framework!r} in config.yaml")

    base_vlm = _nested(compact_config, "framework", "qwenvl", "base_vlm")
    if base_vlm != runtime["base_vlm"]:
        raise ValueError(f"Expected base_vlm={runtime['base_vlm']!r}, got {base_vlm!r}")

    action_dim = _nested(compact_config, "framework", "action_model", "action_dim")
    action_horizon = _nested(compact_config, "framework", "action_model", "action_horizon")
    if int(action_dim or -1) != int(runtime["action_dim"]):
        raise ValueError(f"Expected action_dim={runtime['action_dim']}, got {action_dim!r}")
    if int(action_horizon or -1) != int(runtime["action_horizon"]):
        raise ValueError(f"Expected action_horizon={runtime['action_horizon']}, got {action_horizon!r}")

    include_state = _nested(full_config, "datasets", "vla_data", "include_state")
    if include_state is not True:
        raise ValueError(f"Released RoboDojo checkpoint must use include_state=true, got {include_state!r}")

    data_mix = _nested(compact_config, "datasets", "vla_data", "data_mix")
    if data_mix not in {"robodojo_v21_all_h50_q99", "robodojo_arx_x5_h50_q99"}:
        raise ValueError(f"Unexpected RoboDojo data_mix in config.yaml: {data_mix!r}")

    runtime_requirements = spec.get("runtime_requirements", {})
    if runtime_requirements.get("pi_v3_forward") == "canonical_interleaved":
        interleaved = _nested(
            compact_config,
            "framework",
            "action_model",
            "diffusion_model_cfg",
            "interleave_self_attention",
        )
        if framework != "QwenPI_v3" or interleaved is not True:
            raise ValueError(
                "canonical_interleaved is only valid for a QwenPI_v3 checkpoint "
                "with interleave_self_attention=true"
            )

    dim = int(runtime["action_dim"])
    for modality in ("state", "action"):
        for name in ("q01", "q99"):
            _validate_stat_vector(stats, modality, name, dim)

    sidecar_hashes = {
        "config.yaml": spec["config_sha256"],
        "config.full.yaml": spec["config_full_sha256"],
        "dataset_statistics.json": spec["dataset_statistics_sha256"],
    }
    for name, expected_hash in sidecar_hashes.items():
        actual_hash = sha256_file(run_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{name} SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
            )

    if verify_weight_hash:
        weight_sha = sha256_file(checkpoint_path)
        if weight_sha != spec["checkpoint_sha256"]:
            raise ValueError(
                f"Checkpoint SHA256 mismatch: expected {spec['checkpoint_sha256']}, got {weight_sha}"
            )

    return checkpoint_path


def download_checkpoint(run_dir: Path, spec: dict[str, Any], *, local_files_only: bool = False) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Run this tool inside the StarVLA policy environment."
        ) from exc

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        local_dir=run_dir,
        allow_patterns=[
            spec["checkpoint_path"],
            "config.yaml",
            "config.full.yaml",
            "dataset_statistics.json",
            "README.md",
        ],
        local_files_only=local_files_only,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="Released variant: oft, groot, or pi_v3")
    parser.add_argument("--output-dir", type=Path, required=True, help="Materialized HF run directory")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true", help="Do not access Hugging Face")
    parser.add_argument("--local-files-only", action="store_true", help="Use only the local HF cache")
    parser.add_argument(
        "--skip-weight-hash",
        action="store_true",
        help="Check the 10 GB file size but skip its SHA256 (not recommended for first use)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    variant = resolve_variant(args.variant)
    spec = manifest["variants"][variant]

    if not args.verify_only:
        print(
            f"[starVLA HF] downloading {spec['repo_id']}@{spec['revision']} to {args.output_dir}",
            file=sys.stderr,
        )
        download_checkpoint(args.output_dir, spec, local_files_only=args.local_files_only)

    checkpoint_path = validate_checkpoint_dir(
        args.output_dir,
        spec,
        manifest["runtime"],
        verify_weight_hash=not args.skip_weight_hash,
    )
    print(
        f"[starVLA HF] verified variant={variant}, framework={spec['framework']}, "
        f"image_color_order={manifest['runtime']['image_color_order']}, include_state=true, "
        f"normalization={manifest['runtime']['normalization_key']}",
        file=sys.stderr,
    )
    print(checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
