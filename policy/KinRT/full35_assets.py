"""Download and verify the fixed RoboDojo Full-35, 60,000-step delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

import numpy as np


HF_REPO = "Gleez/kinrt-robodojo-full35-a800-60k"
HF_REVISION = "9460d07a9c7677ef3c72ece08df1f34eba7e45c7"
DATASET_ID = "RoboDojo_lerobot_v30_video"
CONFIG_NAME = "kinrt_full_robodojo"
CHECKPOINT_PREFIX = "checkpoints/60000/"
SOURCE_PREFIX = "full35_full_k4_b256_s0_60k/60000/"
LABELS_NAME = "router_labels_k4_full35.npy"
LABELS_SHA256 = "3a7f5265006a32a892dada84c38130a424a38cc4ab185a5e5ea31ad215e64cd7"
NORM_SHA256 = "51d09fc068d7af5fce5fb87a95d3e72960c060b986dd2733cac144d5f008e661"
CHECKPOINT_NORM_SHA256 = "b3d1307c5f8b1c0334fe8377cb5aef422067b22204fb1529fb5c8289a98d2c68"
CHECKPOINT_MANIFEST = "full35_full_k4_b256_s0_60k_step60000.sha256"
MANIFEST_DIGESTS = {
    "delivery/delivery_files.sha256": "b0b3be3aaefbde8de6629898a95b9d4013a8de092e2fdbc712dd7325425770fa",
    "delivery/" + CHECKPOINT_MANIFEST: "b645ed1d082ed55e8a918206a6354dff60642ce9256828e4dcd8831d1ef3f607",
    "delivery/DELIVERY.md": "8210620c00d034e5cd9c514db5c4bd36597ef21bad241a49be63743954eac7f7",
}
DATASET_DIGESTS = {
    "info.json": "92c53f4b509aa70a55c70433c3e8a4b3038e25cf98059ca0fc0fdcfbf6318231",
    "tasks.parquet": "39e934f1eea211fc471be6a32cfd920fd5bc5a8c06a01c819cfc7e68b6990f65",
    "stats.json": "3a406ee1fe0cee18a4d8561baba40b1ca340a9c43ff1e022a79aa42d9fae4038",
}
CHECKPOINT_NORM = CHECKPOINT_PREFIX + "assets/" + DATASET_ID + "/norm_stats.json"
COUNTS = [213759, 453930, 587902, 604011]
TOTAL_FRAMES = sum(COUNTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes artifact directory: {relative!r}")
    return destination


def read_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            raise ValueError(f"Invalid checksum manifest: {path}")
        digest, name = match.groups()
        if name in entries:
            raise ValueError(f"Duplicate checksum path: {name}")
        safe_path(path.parent, name)
        entries[name] = digest
    return entries


def checkpoint_entries(root: Path) -> dict[str, str]:
    manifest = root / "delivery" / CHECKPOINT_MANIFEST
    require_hash(manifest, MANIFEST_DIGESTS["delivery/" + CHECKPOINT_MANIFEST])
    entries = {}
    for name, digest in read_manifest(manifest).items():
        if not name.startswith(SOURCE_PREFIX):
            raise ValueError(f"Unexpected source checkpoint path: {name}")
        entries[CHECKPOINT_PREFIX + name[len(SOURCE_PREFIX) :]] = digest
    if len(entries) != 1057:
        raise ValueError("Expected 1057 entries in the final checkpoint manifest.")
    return entries


def selected_checkpoint_file(name: str, *, assets_only: bool, include_training_state: bool) -> bool:
    if assets_only:
        return name == CHECKPOINT_NORM
    return include_training_state or not name.startswith(CHECKPOINT_PREFIX + "train_state/")


def validate_labels(path: Path) -> dict:
    require_hash(path, LABELS_SHA256)
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    if labels.dtype != np.dtype("int32") or labels.shape != (TOTAL_FRAMES,):
        raise ValueError(f"Expected int32 router labels with shape ({TOTAL_FRAMES},).")
    if np.any(labels < 0) or np.any(labels >= 4):
        raise ValueError("Router labels must be in the range 0..3.")
    counts = np.bincount(labels, minlength=4).tolist()
    if counts != COUNTS:
        raise ValueError(f"Unexpected router class counts: {counts}")
    return {"frames": TOTAL_FRAMES, "router_classes": 4, "counts": counts}


def validate_norm(path: Path) -> dict:
    if not path.is_file() or sha256(path) not in {NORM_SHA256, CHECKPOINT_NORM_SHA256}:
        raise ValueError(f"Expected the normalization statistics used in the delivered run: {path}")
    norms = json.loads(path.read_text(encoding="utf-8"))["norm_stats"]
    for group in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = np.asarray(norms[group][field])
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid normalization field: {group}.{field}")
        if np.any(np.asarray(norms[group]["std"]) <= 0):
            raise ValueError(f"Nonpositive normalization standard deviation: {group}")
    return norms


def verify(root: Path, *, assets_only: bool = False, include_training_state: bool = False) -> dict:
    for name, digest in MANIFEST_DIGESTS.items():
        require_hash(safe_path(root, name), digest)
    delivery = read_manifest(root / "delivery/delivery_files.sha256")
    for name, digest in delivery.items():
        require_hash(safe_path(root / "delivery", name), digest)
    selected = {
        name: digest
        for name, digest in checkpoint_entries(root).items()
        if selected_checkpoint_file(name, assets_only=assets_only, include_training_state=include_training_state)
    }
    for name, digest in selected.items():
        require_hash(safe_path(root, name), digest)
    label_info = validate_labels(root / "delivery" / LABELS_NAME)
    delivered_norm = validate_norm(root / "delivery/norm_stats.json")
    if validate_norm(root / CHECKPOINT_NORM) != delivered_norm:
        raise ValueError("Checkpoint normalization differs from the training delivery.")
    for name, digest in DATASET_DIGESTS.items():
        require_hash(root / "delivery/dataset_meta" / name, digest)
    return {
        "checkpoint_step": 60000,
        "train_config_name": CONFIG_NAME,
        "normalization_key": DATASET_ID,
        "checkpoint_files_verified": len(selected),
        "delivery_files_verified": len(delivery),
        "assets_only": assets_only,
        "training_state_included": include_training_state and not assets_only,
        "router_labels": label_info,
        "inference_executed": False,
    }


def validate_dataset(root: Path) -> None:
    for name, digest in DATASET_DIGESTS.items():
        require_hash(root / "meta" / name, digest)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if (info["total_tasks"], info["total_episodes"], info["total_frames"]) != (35, 3500, TOTAL_FRAMES):
        raise ValueError("The original Full-35 dataset must contain 35 tasks, 3500 episodes and 1859602 frames.")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("The delivered training dataset uses LeRobot v3.0.")
    for directory, pattern in (("data", "*.parquet"), ("videos", "*.mp4"), ("meta/episodes", "*.parquet")):
        if next((root / directory).rglob(pattern), None) is None:
            raise FileNotFoundError(f"Original Full-35 dataset content is missing under {root / directory}")


def validate_training(dataset_root: Path, labels: Path, norm_stats: Path) -> dict:
    validate_dataset(dataset_root)
    label_info = validate_labels(labels)
    validate_norm(norm_stats)
    return {
        "dataset": DATASET_ID,
        "router_labels": label_info,
        "metadata_matches_delivery": True,
        "frame_order_verified": False,
        "requirement": "Use the original unmodified Full-35 dataset in its original global frame order.",
    }


def copy_matching(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise FileExistsError(f"Refusing to overwrite a different training artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def prepare_training(root: Path, dataset_root: Path, openpi_root: Path) -> dict:
    verify(root, assets_only=True)
    validate_dataset(dataset_root)
    labels = dataset_root / "meta/router_labels_k4_full35/router_labels.npy"
    norms = openpi_root / "assets" / CONFIG_NAME / DATASET_ID / "norm_stats.json"
    copy_matching(root / "delivery" / LABELS_NAME, labels)
    if norms.is_file() and sha256(norms) == CHECKPOINT_NORM_SHA256:
        validate_norm(norms)
    else:
        copy_matching(root / "delivery/norm_stats.json", norms)
    result = validate_training(dataset_root, labels, norms)
    result.update({"labels_path": str(labels.resolve()), "normalization_path": str(norms.resolve())})
    return result


def download(args: argparse.Namespace) -> dict:
    from huggingface_hub import snapshot_download

    patterns = ["delivery/*", "delivery/dataset_meta/*", CHECKPOINT_NORM]
    if not args.assets_only:
        patterns.extend(
            [
                CHECKPOINT_PREFIX + "params/*",
                CHECKPOINT_PREFIX + "assets/*",
                CHECKPOINT_PREFIX + "_CHECKPOINT_METADATA",
                CHECKPOINT_PREFIX + "commit_success.txt",
            ]
        )
        if args.include_training_state:
            patterns.append(CHECKPOINT_PREFIX + "train_state/*")
    # The standard HF_TOKEN or saved Hugging Face login supplies private-repo access.
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        repo_type="model",
        local_dir=args.destination,
        allow_patterns=patterns,
    )
    return verify(args.destination, assets_only=args.assets_only, include_training_state=args.include_training_state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download_parser = commands.add_parser("download")
    download_parser.add_argument("--destination", type=Path, required=True)
    download_parser.add_argument("--repo-id", default=HF_REPO)
    download_parser.add_argument("--revision", default=HF_REVISION)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--artifact-root", type=Path, required=True)
    for subparser in (download_parser, verify_parser):
        mode = subparser.add_mutually_exclusive_group()
        mode.add_argument("--assets-only", action="store_true")
        mode.add_argument("--include-training-state", action="store_true")
    prepare = commands.add_parser("prepare-training")
    prepare.add_argument("--artifact-root", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--openpi-root", type=Path, required=True)
    training = commands.add_parser("validate-training")
    training.add_argument("--dataset-root", type=Path, required=True)
    training.add_argument("--labels", type=Path, required=True)
    training.add_argument("--norm-stats", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "download":
            result = download(args)
        elif args.command == "verify":
            result = verify(
                args.artifact_root, assets_only=args.assets_only, include_training_state=args.include_training_state
            )
        elif args.command == "prepare-training":
            result = prepare_training(args.artifact_root, args.dataset_root, args.openpi_root)
        else:
            result = validate_training(args.dataset_root, args.labels, args.norm_stats)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"[KinRT] {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
