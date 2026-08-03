from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "checkpoint_sources.json"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str, label: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def load_manifest(path: pathlib.Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint_sources.json schema_version")
    sources = payload.get("sources")
    if set(sources or {}) != {"xiaomi", "spatial_forcing", "hunyuan"}:
        raise ValueError("checkpoint manifest must contain exactly three experts")
    return payload


def resolve_root(
    root: pathlib.Path | None, manifest: dict[str, Any]
) -> pathlib.Path:
    configured = root or pathlib.Path(
        os.environ.get("CSU_CHECKPOINT_ROOT", manifest["bundle"]["root"])
    )
    configured = configured.expanduser()
    if not configured.is_absolute():
        configured = HERE / configured
    return configured.resolve()


def build_report(
    root: pathlib.Path,
    manifest_path: pathlib.Path = DEFAULT_MANIFEST,
    *,
    check_sha256: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    root = root.expanduser().resolve()
    errors: list[str] = []
    experts: dict[str, Any] = {}
    expected_bundle_files: set[str] = set()

    if not root.is_dir():
        errors.append(f"missing bundle directory: {root}")

    for expert, spec in manifest["sources"].items():
        destination = _safe_relative(spec["destination_relpath"], "destination_relpath")
        expert_root = root.joinpath(*destination.parts)
        expected_files: set[str] = set()
        checked = 0
        checked_bytes = 0

        for record in spec["files"]:
            relative = _safe_relative(record["path"], f"{expert} file path")
            relative_text = relative.as_posix()
            if relative_text in expected_files:
                errors.append(f"{expert}: duplicate manifest path {relative_text}")
                continue
            expected_files.add(relative_text)
            bundle_relative = (destination / relative).as_posix()
            expected_bundle_files.add(bundle_relative)
            target = expert_root.joinpath(*relative.parts)

            if target.is_symlink():
                errors.append(f"{expert}: symlink is not allowed: {target}")
                continue
            if not target.is_file():
                errors.append(f"{expert}: missing file {target}")
                continue
            actual_size = target.stat().st_size
            expected_size = int(record["size"])
            if actual_size != expected_size:
                errors.append(
                    f"{expert}: size mismatch {relative_text}: "
                    f"{actual_size} != {expected_size}"
                )
                continue
            if check_sha256:
                actual_sha = _sha256(target)
                if actual_sha != record["sha256"]:
                    errors.append(
                        f"{expert}: SHA256 mismatch {relative_text}: "
                        f"{actual_sha} != {record['sha256']}"
                    )
                    continue
            checked += 1
            checked_bytes += actual_size

        actual_files: set[str] = set()
        if expert_root.is_dir():
            for path in expert_root.rglob("*"):
                if path.is_file() or path.is_symlink():
                    actual_files.add(path.relative_to(expert_root).as_posix())
        for extra in sorted(actual_files - expected_files):
            errors.append(f"{expert}: unexpected file {expert_root / extra}")

        experts[expert] = {
            "path": str(expert_root),
            "provider": spec["provider"],
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "expected_files": len(expected_files),
            "checked_files": checked,
            "checked_bytes": checked_bytes,
        }

    actual_bundle_files: set[str] = set()
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file() or path.is_symlink():
                actual_bundle_files.add(path.relative_to(root).as_posix())
    for extra in sorted(actual_bundle_files - expected_bundle_files):
        errors.append(f"bundle: unexpected file {root / extra}")

    return {
        "schema_version": 1,
        "bundle_root": str(root),
        "manifest": str(manifest_path.resolve()),
        "verification": "size+sha256" if check_sha256 else "size-only",
        "experts": experts,
        "expected_files": len(expected_bundle_files),
        "checked_files": sum(item["checked_files"] for item in experts.values()),
        "checked_bytes": sum(item["checked_bytes"] for item in experts.values()),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every normalized CSU-AI-0 expert checkpoint file."
    )
    parser.add_argument("--root", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="diagnostic only; the downloader and release audit use full SHA256",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    root = resolve_root(args.root, manifest)
    report = build_report(
        root,
        args.manifest,
        check_sha256=not args.size_only,
    )
    if args.report:
        write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
