"""Preserve original LeRobot metadata separately from its source overlay.

The historical RECORD describes the original distribution before source replacement.
Executable source identity is checked by pi05_runtime's independent overlay manifest.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import tempfile
from pathlib import Path

try:
    from .pi05_assets import Pi05CheckpointError, _sha256
except ImportError:
    from pi05_assets import Pi05CheckpointError, _sha256


METADATA_MANIFEST_SHA256 = (
    "0acf1e99148532f0d10c41daac386ac88c6474abec75ea5601c86e9e53409614"
)
_FILES = {"INSTALLER", "METADATA", "RECORD", "REQUESTED", "WHEEL"}
_DIRECTORY = "lerobot-0.1.0.dist-info"


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _plain_file(path):
    return not path.is_symlink() and path.is_file()


def _check_files(directory, files):
    if directory.is_symlink() or not directory.is_dir():
        raise Pi05CheckpointError(
            "pi0.5 LeRobot metadata directory is missing or linked"
        )
    if {p.name for p in directory.iterdir()} != set(files):
        raise Pi05CheckpointError("pi0.5 LeRobot metadata file inventory differs")
    for name, record in files.items():
        path = directory / name
        if (
            not _plain_file(path)
            or path.stat().st_size != record["size"]
            or _sha256(path) != record["sha256"]
        ):
            raise Pi05CheckpointError(
                f"pi0.5 original LeRobot metadata differs: {name}"
            )


def validate_lerobot_metadata(policy_dir):
    """Authenticate the committed original metadata and complete 157-package versions."""
    policy = Path(policy_dir).resolve()
    root = policy / "pi05/lerobot-metadata"
    path = root / "manifest.json"
    if (
        (policy / "pi05").is_symlink()
        or root.is_symlink()
        or not _plain_file(path)
        or _sha256(path) != METADATA_MANIFEST_SHA256
    ):
        raise Pi05CheckpointError("pi0.5 original metadata manifest changed")
    metadata = json.loads(path.read_text())
    if (
        metadata["distribution"] != "lerobot"
        or metadata["version"] != "0.1.0"
        or metadata["directory"] != _DIRECTORY
        or set(metadata["files"]) != _FILES
        or len(metadata["runtime_distributions"]) != 157
    ):
        raise Pi05CheckpointError("Unsupported pi0.5 original metadata identity")
    _check_files(
        root,
        {
            **metadata["files"],
            "manifest.json": {
                "size": path.stat().st_size,
                "sha256": METADATA_MANIFEST_SHA256,
            },
        },
    )
    return metadata


def expected_runtime_distributions(policy_dir):
    """Require exactly156 unchanged artifact pins plus original LeRobot metadata."""
    policy = Path(policy_dir).resolve()
    metadata = validate_lerobot_metadata(policy)
    lock = policy / "pi05/requirements.lock"
    if not _plain_file(lock):
        raise Pi05CheckpointError("pi0.5 runtime hash lock is missing or linked")
    pins = {}
    for name, version in re.findall(
        r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock.read_text(), re.MULTILINE
    ):
        name = _canonical(name)
        if name in pins:
            raise Pi05CheckpointError("pi0.5 runtime hash lock has duplicate pins")
        pins[name] = version
    expected = metadata["runtime_distributions"]
    if len(pins) != 156 or {**pins, "lerobot": "0.1.0"} != expected:
        raise Pi05CheckpointError("pi0.5 runtime hash lock differs from 157-package identity")
    return expected


def install_lerobot_metadata(policy_dir, site):
    """Copy only five original dist-info files; never install LeRobot source."""
    metadata = validate_lerobot_metadata(policy_dir)
    site = Path(site)
    if site.is_symlink() or not site.is_dir():
        raise Pi05CheckpointError("pi0.5 metadata destination is missing or linked")
    destination = site / _DIRECTORY
    if destination.exists() or destination.is_symlink():
        _check_files(destination, metadata["files"])
        return
    staging = Path(tempfile.mkdtemp(prefix=".pi05-metadata-", dir=site))
    try:
        for name in metadata["files"]:
            shutil.copyfile(
                Path(policy_dir) / "pi05/lerobot-metadata" / name, staging / name
            )
            (staging / name).chmod(0o644)
        _check_files(staging, metadata["files"])
        staging.chmod(0o755)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def check_runtime_distributions(policy_dir, site):
    """Check original metadata and the full distribution set without importing VLA."""
    import hashlib

    expected = expected_runtime_distributions(policy_dir)
    metadata = validate_lerobot_metadata(policy_dir)
    site = Path(site)
    _check_files(site / _DIRECTORY, metadata["files"])
    actual = {}
    duplicates = []
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = distribution.metadata.get("Name")
        if not name:
            raise Pi05CheckpointError("pi0.5 installed distribution has no name")
        name = _canonical(name)
        if name in actual:
            duplicates.append(name)
        actual[name] = distribution.version
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches = {
        name: {"expected": version, "actual": actual[name]}
        for name, version in expected.items()
        if name in actual and actual[name] != version
    }
    if missing or extra or mismatches or duplicates:
        raise Pi05CheckpointError(
            "pi0.5 157-package distribution inventory differs: "
            + json.dumps(
                {
                    "missing": missing,
                    "extra": extra,
                    "versions": mismatches,
                    "duplicates": duplicates,
                },
                sort_keys=True,
            )
        )
    versions = dict(sorted(actual.items()))
    return {
        "count": len(actual),
        "versions": versions,
        "versions_sha256": hashlib.sha256(
            json.dumps(versions, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "lerobot_metadata_manifest_sha256": METADATA_MANIFEST_SHA256,
        "historical_RECORD": metadata["historical_RECORD"]["note"],
    }
