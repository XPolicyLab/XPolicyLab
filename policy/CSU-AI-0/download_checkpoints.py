from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "checkpoint_sources.json"
sys.path.insert(0, str(HERE))
from verify_checkpoints import (  # noqa: E402
    build_report,
    load_manifest,
    resolve_root,
    write_report,
)


def _safe_join(root: pathlib.Path, relative: str) -> pathlib.Path:
    parts = pathlib.PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts:
        raise ValueError(f"unsafe relative path: {relative!r}")
    candidate = root.joinpath(*parts.parts)
    candidate.resolve().relative_to(root.resolve())
    return candidate


def _download_huggingface(spec: dict[str, Any], work: pathlib.Path) -> pathlib.Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required; run install.sh or install it in "
            "CSU_DOWNLOAD_PYTHON"
        ) from error

    source_subdir = spec["source_subdir"].rstrip("/")
    local_dir = work / "snapshot"
    allow_patterns = [
        f"{source_subdir}/{record['source_path']}" for record in spec["files"]
    ]
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec.get("repo_type", "model"),
        revision=spec["revision"],
        allow_patterns=allow_patterns,
        local_dir=local_dir,
    )
    return local_dir / pathlib.PurePosixPath(source_subdir)


def _download_modelscope(spec: dict[str, Any], work: pathlib.Path) -> pathlib.Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "modelscope is required; run install.sh or install it in "
            "CSU_DOWNLOAD_PYTHON"
        ) from error

    downloaded = snapshot_download(
        spec["repo_id"],
        revision=spec["revision"],
        cache_dir=str(work / "cache"),
    )
    root = pathlib.Path(downloaded)
    source_subdir = spec.get("source_subdir", "")
    return _safe_join(root, source_subdir) if source_subdir else root


def _link_or_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve(strict=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _normalize_expert(
    source_root: pathlib.Path,
    destination_root: pathlib.Path,
    spec: dict[str, Any],
) -> None:
    for record in spec["files"]:
        source = _safe_join(source_root, record["source_path"])
        destination = _safe_join(destination_root, record["path"])
        if not source.is_file():
            raise RuntimeError(f"downloaded source file is missing: {source}")
        actual_size = source.stat().st_size
        if actual_size != int(record["size"]):
            raise RuntimeError(
                f"downloaded source size mismatch for {source}: "
                f"{actual_size} != {record['size']}"
            )
        _link_or_copy(source, destination)


def _print_plan(manifest: dict[str, Any], root: pathlib.Path) -> None:
    payload = {
        "bundle_root": str(root),
        "mode": "dry-run",
        "sources": {
            name: {
                "provider": spec["provider"],
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "public_url": spec["public_url"],
                "destination": str(root / spec["destination_relpath"]),
                "files": len(spec["files"]),
                "bytes": sum(int(item["size"]) for item in spec["files"]),
            }
            for name, spec in manifest["sources"].items()
        },
        "final_verification": "all files: exact path + size + SHA256",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the three public expert sources independently, normalize them "
            "under one bundle root, then run one complete SHA256 verification."
        )
    )
    parser.add_argument("--root", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-invalid", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST_PATH)
    root = resolve_root(args.root, manifest)
    root.parent.mkdir(parents=True, exist_ok=True)
    report_path = root.with_name(root.name + ".verification.json")

    if args.dry_run:
        _print_plan(manifest, root)
        return 0

    if root.exists():
        existing = build_report(root, MANIFEST_PATH, check_sha256=True)
        if existing["status"] == "PASS":
            existing["install_status"] = "already-present-and-valid"
            write_report(report_path, existing)
            print(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if not args.replace_invalid:
            existing["install_status"] = "refused-to-overwrite-invalid-existing-bundle"
            write_report(report_path, existing)
            print(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True))
            print(
                "[CSU-AI-0][ERROR] Existing bundle is invalid. Inspect the report, "
                "then rerun with CSU_REPLACE_INVALID_CHECKPOINTS=1 to preserve it "
                "as a timestamped backup and install a verified replacement.",
                file=sys.stderr,
            )
            return 2

    temporary_root = pathlib.Path(
        tempfile.mkdtemp(prefix=".CSU-AI-0-v1.download-", dir=root.parent)
    )
    staged_bundle = temporary_root / "bundle"
    staged_bundle.mkdir()
    backup: pathlib.Path | None = None
    try:
        for name in ("xiaomi", "spatial_forcing", "hunyuan"):
            spec = manifest["sources"][name]
            work = temporary_root / "sources" / name
            work.mkdir(parents=True)
            print(
                f"[CSU-AI-0] downloading {name} from "
                f"{spec['provider']}:{spec['repo_id']}@{spec['revision']}",
                flush=True,
            )
            if spec["provider"] == "huggingface":
                source_root = _download_huggingface(spec, work)
            elif spec["provider"] == "modelscope":
                source_root = _download_modelscope(spec, work)
            else:
                raise RuntimeError(f"unsupported provider: {spec['provider']}")
            destination_root = _safe_join(
                staged_bundle, spec["destination_relpath"]
            )
            destination_root.mkdir(parents=True)
            _normalize_expert(source_root, destination_root, spec)

        print("[CSU-AI-0] running one complete bundle verification", flush=True)
        report = build_report(staged_bundle, MANIFEST_PATH, check_sha256=True)
        if report["status"] != "PASS":
            report["install_status"] = "staged-bundle-verification-failed"
            write_report(report_path, report)
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 1

        if root.exists():
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = root.with_name(root.name + f".invalid-backup-{stamp}")
            os.replace(root, backup)
        try:
            os.replace(staged_bundle, root)
        except BaseException:
            if backup is not None and backup.exists() and not root.exists():
                os.replace(backup, root)
            raise
        report["bundle_root"] = str(root)
        for name, spec in manifest["sources"].items():
            report["experts"][name]["path"] = str(
                _safe_join(root, spec["destination_relpath"])
            )
        report["install_status"] = "downloaded-normalized-verified-installed"
        report["validated_before_atomic_install"] = True
        if backup is not None:
            report["replaced_bundle_backup"] = str(backup)
        write_report(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
