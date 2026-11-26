"""Install the pinned ICL data artifacts without relying on author workspaces."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

try:
    from .preflight import relative_artifact_path, sha256_file
except ImportError:
    from preflight import relative_artifact_path, sha256_file

POLICY_DIR = Path(__file__).resolve().parent


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"ICL metadata needs a fixed SHA-256: {field}")
    return value


def load_icl_metadata(policy_dir: Path = POLICY_DIR) -> tuple[dict, dict]:
    """Verify committed metadata before opening any data archive or URL."""
    submission = json.loads((policy_dir / "submission.json").read_text())
    pins = submission.get("icl")
    if not isinstance(pins, dict):
        raise ValueError("submission.json is missing the ICL resource pins")
    documents = {}
    for name in ("catalog", "artifacts"):
        relative = relative_artifact_path(pins.get(f"{name}_path"), f"icl.{name}_path")
        path = policy_dir / relative
        expected = _digest(pins.get(f"{name}_sha256"), f"icl.{name}_sha256")
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"Committed ICL {name} is missing or changed")
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError(f"Unsupported ICL {name} schema")
        documents[name] = document
    catalog, artifacts = documents["catalog"], documents["artifacts"]
    if catalog.get("source_commit") != submission["source"]["commit"]:
        raise ValueError("ICL catalog and RoboRSI source commits differ")
    upstream = artifacts.get("upstream", {})
    if upstream.get("commit") != catalog["source_commit"] or upstream.get(
        "inventory_sha256"
    ) != catalog.get("source_inventory_sha256"):
        raise ValueError("ICL artifacts and source inventory identities differ")
    bundles, records, tasks = (
        catalog.get("bundles"),
        artifacts.get("artifacts"),
        catalog.get("tasks"),
    )
    if (
        not isinstance(bundles, dict)
        or not bundles
        or not isinstance(records, dict)
        or set(records) != set(bundles)
        or not isinstance(tasks, dict)
        or not tasks
    ):
        raise ValueError("ICL bundle catalog/artifact mapping is incomplete")
    if any(
        not isinstance(task, str) or bundle not in bundles
        for task, bundle in tasks.items()
    ):
        raise ValueError("ICL task mapping refers to an unavailable bundle")
    for identifier, record in records.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or not isinstance(
            record, dict
        ):
            raise ValueError("Invalid ICL bundle identifier or artifact record")
        _digest(record.get("archive_sha256"), f"{identifier}.archive_sha256")
        _digest(record.get("manifest_sha256"), f"{identifier}.manifest_sha256")
        relative_artifact_path(record.get("archive_path"), f"{identifier}.archive_path")
        if type(record.get("archive_size")) is not int or record["archive_size"] <= 0:
            raise ValueError(f"ICL archive size is missing: {identifier}")
        expected = bundles[identifier]
        if (
            record.get("payload_files") != expected.get("file_count")
            or record.get("payload_bytes") != expected.get("size")
            or not isinstance(expected.get("source_files"), dict)
            or len(expected["source_files"]) != expected["file_count"]
        ):
            raise ValueError(f"ICL payload inventory differs: {identifier}")
    return catalog, artifacts


def _installed_manifest(
    package: Path, identifier: str, expected: dict, record: dict
) -> None:
    manifest = package / "manifest.json"
    if package.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"ICL package is missing: {identifier}; run bash install.sh")
    if sha256_file(manifest) != record["manifest_sha256"]:
        raise ValueError(f"ICL package manifest changed: {identifier}")
    document = json.loads(manifest.read_text())
    if (
        document.get("schema_version") != 1
        or document.get("package_type") != "rpent.icl.context"
        or document.get("package_id") != identifier
        or document.get("files") != expected["source_files"]
        or not (package / "payload").is_dir()
    ):
        raise ValueError(
            f"ICL package does not match the source inventory: {identifier}"
        )
    if any(
        document.get(field, [])
        for field in ("skills", "skill_paths", "memory", "initial_context")
    ):
        raise ValueError(
            "Historical ICL instructions must not be automatically preloaded"
        )


def validate_icl_install(policy_dir: Path = POLICY_DIR, *, full: bool = False) -> dict:
    """Check all manifests; optionally hash every payload with the upstream loader."""
    catalog, artifacts = load_icl_metadata(policy_dir)
    for identifier, record in artifacts["artifacts"].items():
        package = policy_dir / "runtime/icl/packages" / identifier
        _installed_manifest(package, identifier, catalog["bundles"][identifier], record)
        if full:
            from roborsi.components.icl.package import ContextPackage

            ContextPackage.load(package)
    return {
        "bundles": len(catalog["bundles"]),
        "mapped_tasks": len(catalog["tasks"]),
        "check": "all payload hashes"
        if full
        else "pinned package manifests; selected payload verified per episode",
    }


def _archive_path(policy_dir: Path, identifier: str, record: dict) -> Path:
    root = os.environ.get("ROBORSI_ICL_ARCHIVE_DIR")
    if root:
        return Path(root).expanduser().resolve() / f"{identifier}.tar.gz"
    return policy_dir / relative_artifact_path(record["archive_path"], "ICL archive")


def _get_archive(path: Path, identifier: str, record: dict) -> None:
    if not path.is_file():
        url = record.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(
                f"ICL archive {identifier} is missing and its fixed download URL is not published. "
                "Supply the pinned archives via assets/icl/ or ROBORSI_ICL_ARCHIVE_DIR; "
                "the author's local workspace is not an installation source."
            )
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "ICL download URLs must use HTTPS without embedded credentials"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".icl-download-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                with urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, stream)
                stream.flush()
                if (
                    temporary.stat().st_size != record["archive_size"]
                    or sha256_file(temporary) != record["archive_sha256"]
                ):
                    raise ValueError(
                        f"Downloaded ICL archive identity mismatch: {identifier}"
                    )
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    if (
        path.is_symlink()
        or path.stat().st_size != record["archive_size"]
        or sha256_file(path) != record["archive_sha256"]
    ):
        raise ValueError(f"ICL archive identity mismatch: {identifier}")


def install_icl(policy_dir: Path = POLICY_DIR) -> dict:
    """Install every complete pinned experience tree, preserving existing data."""
    from roborsi.components.icl.package import ContextPackage

    catalog, artifacts = load_icl_metadata(policy_dir)
    destination = policy_dir / "runtime/icl/packages"
    destination.mkdir(parents=True, exist_ok=True)
    for identifier, record in artifacts["artifacts"].items():
        target = destination / identifier
        expected = catalog["bundles"][identifier]
        if target.exists() or target.is_symlink():
            _installed_manifest(target, identifier, expected, record)
            ContextPackage.load(target)
            continue
        archive = _archive_path(policy_dir, identifier, record)
        _get_archive(archive, identifier, record)
        allowed = {
            "manifest.json",
            *(f"payload/{name}" for name in expected["source_files"]),
        }
        directories = {"payload"}
        for name in allowed:
            directories.update(
                str(parent)
                for parent in PurePosixPath(name).parents
                if str(parent) != "."
            )
        with tempfile.TemporaryDirectory(
            prefix=f".{identifier}-", dir=destination
        ) as temporary:
            staging = Path(temporary) / "package"
            staging.mkdir()
            with tarfile.open(archive, mode="r:gz") as package:
                members = package.getmembers()
                files = []
                for member in members:
                    path = PurePosixPath(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or path.as_posix() != member.name
                    ):
                        raise ValueError("ICL archive contains an unsafe path")
                    if member.isfile() and member.name in allowed:
                        files.append(member.name)
                    elif not (member.isdir() and member.name in directories):
                        raise ValueError(
                            "ICL archive contains an unexpected entry or link"
                        )
                if len(files) != len(set(files)) or set(files) != allowed:
                    raise ValueError(
                        "ICL archive file set differs from its pinned inventory"
                    )
                package.extractall(staging, filter="data")
            _installed_manifest(staging, identifier, expected, record)
            ContextPackage.load(staging)
            staging.replace(target)
    result = validate_icl_install(policy_dir)
    result["payload_integrity"] = "every file verified during installation"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "check"))
    parser.add_argument(
        "--full", action="store_true", help="hash every installed payload file"
    )
    args = parser.parse_args()
    try:
        result = (
            install_icl()
            if args.command == "install"
            else validate_icl_install(full=args.full)
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ICL resource preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
