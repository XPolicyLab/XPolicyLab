"""Keep B157 provenance separate from the local inference distribution set.

The old LeRobot metadata describes a replaced PyPI source tree and training extras.
Keep it as evidence, without installing its conflicting dependency declarations.
Executable source identity is checked by pi05_runtime's independent overlay manifest.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

try:
    from .pi05_assets import Pi05CheckpointError, _sha256
except ImportError:
    from pi05_assets import Pi05CheckpointError, _sha256


METADATA_MANIFEST_SHA256 = (
    "ef63c367dcca34d570166f07b57553c4e694f9dbf9be618fb53cddaf495a19a5"
)
_FILES = {"INSTALLER", "METADATA", "RECORD", "REQUESTED", "WHEEL"}
_DIRECTORY = "lerobot-0.1.0.dist-info"
# Public checkpoint/tokenizer fetching uses the adapter's urllib installers.
# Local inference needs neither GCS/OAuth clients nor stale PyPI LeRobot metadata.
# Every retained distribution remains at its authenticated B version.
EXCLUDED_RUNTIME_DISTRIBUTIONS = frozenset(
    {
        "gcsfs",
        "google-api-core",
        "google-auth",
        "google-auth-oauthlib",
        "google-cloud-core",
        "google-cloud-storage",
        "google-crc32c",
        "google-resumable-media",
        "googleapis-common-protos",
        "lerobot",
        "oauthlib",
        "proto-plus",
        "pyasn1",
        "pyasn1-modules",
        "requests-oauthlib",
    }
)


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
    """Authenticate the committed original metadata and complete B157 versions."""
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
    """Require the explicit inference subset with unchanged B reference versions."""
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
    reference = metadata["runtime_distributions"]
    if not EXCLUDED_RUNTIME_DISTRIBUTIONS <= reference.keys():
        raise Pi05CheckpointError("pi0.5 runtime exclusions differ from B157 identity")
    expected = {
        name: version
        for name, version in reference.items()
        if name not in EXCLUDED_RUNTIME_DISTRIBUTIONS
    }
    if pins != expected:
        raise Pi05CheckpointError(
            "pi0.5 runtime hash lock differs from B157 inference subset"
        )
    return expected


def check_runtime_distributions(policy_dir, site):
    """Check historical provenance and the inference set without importing VLA."""
    import hashlib

    expected = expected_runtime_distributions(policy_dir)
    metadata = validate_lerobot_metadata(policy_dir)
    site = Path(site)
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
            "pi0.5 inference distribution inventory differs: "
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
        "reference_distribution_count": len(metadata["runtime_distributions"]),
        "excluded_distributions": {
            name: metadata["runtime_distributions"][name]
            for name in sorted(EXCLUDED_RUNTIME_DISTRIBUTIONS)
        },
        "lerobot_metadata_installed": False,
    }
