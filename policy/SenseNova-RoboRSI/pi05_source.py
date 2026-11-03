"""Verify the shipped pi0.5 service and dependency source snapshot."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

try:
    from .pi05_assets import Pi05CheckpointError, _sha256, _target
except ImportError:
    from pi05_assets import Pi05CheckpointError, _sha256, _target

POLICY_DIR = Path(__file__).resolve().parent
PI05_SOURCE_MANIFEST_SHA256 = (
    "2dfd45f0b61835c57321adeb72948c055fe56eab18adbba5af2208a9181f2d5b"
)


def load_openpi_client(policy_dir: Path = POLICY_DIR):
    """Expose the authenticated NumPy/msgpack codec to the Agent environment.

    The VLA interpreter is separate, so its imports do not make this original
    client available to the Agent. No JAX/OpenPI model code is added here.
    """
    identity = validate_pi05_source(policy_dir)
    client_root = Path(identity["openpi_client_src"]).resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "openpi_client" or name.startswith("openpi_client."):
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(
                client_root
            ):
                raise Pi05CheckpointError(
                    "Agent already imported a different OpenPI client source"
                )
    if str(client_root) not in sys.path:
        sys.path.insert(0, str(client_root))
    codec = importlib.import_module("openpi_client.msgpack_numpy")
    if Path(codec.__file__).resolve() != client_root / "openpi_client/msgpack_numpy.py":
        raise Pi05CheckpointError("Agent OpenPI codec source changed")
    return codec


def validate_pi05_source(policy_dir: Path = POLICY_DIR) -> dict:
    """Check the pinned service snapshot and documented portability patches."""
    manifest = policy_dir / "vendor/pi05-source.json"
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or _sha256(manifest) != PI05_SOURCE_MANIFEST_SHA256
    ):
        raise Pi05CheckpointError("Committed pi0.5 source manifest changed")
    metadata = json.loads(manifest.read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("source_root") != "vendor/pi05"
    ):
        raise Pi05CheckpointError("Unsupported pi0.5 source manifest")
    root = policy_dir / "vendor/pi05"
    if root.is_symlink() or not root.is_dir():
        raise Pi05CheckpointError("pi0.5 source directory is missing or linked")
    records = metadata.get("files")
    if not isinstance(records, dict) or len(records) != metadata.get("file_count"):
        raise Pi05CheckpointError("Incomplete pi0.5 source inventory")
    if sum(record["size"] for record in records.values()) != metadata.get(
        "payload_bytes"
    ):
        raise Pi05CheckpointError("pi0.5 source size inventory changed")
    actual = set()
    for path in root.rglob("*"):
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise Pi05CheckpointError("pi0.5 source paths must not be symlinks")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(records):
        raise Pi05CheckpointError("pi0.5 source file inventory differs")
    for relative, record in records.items():
        path = _target(root, relative)
        if path.stat().st_size != record["size"] or _sha256(path) != record["sha256"]:
            raise Pi05CheckpointError(f"pi0.5 source identity mismatch: {relative}")
    return {
        "verified_files": len(records),
        "verified_bytes": metadata["payload_bytes"],
        "verification": "every source file size and SHA-256",
        "source_root": str(root.resolve()),
        "openpi_src": str(root / "openpi/src"),
        "openpi_client_src": str(root / "openpi/packages/openpi-client/src"),
        "server_package_root": str(root),
        "source_overrides": str(root / "site-packages"),
        "xpolicylab_source_commit": metadata["openpi"]["xpolicylab_commit"],
        "openpi_upstream_commit_recorded": False,
    }


def main() -> int:
    try:
        receipt = validate_pi05_source()
    except (OSError, ValueError) as error:
        print(f"pi0.5 source preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
