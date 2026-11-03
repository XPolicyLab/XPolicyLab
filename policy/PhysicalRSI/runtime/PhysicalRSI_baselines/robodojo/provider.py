"""Register an external policy provider without copying its weights."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, file_digest


SCHEMA = "physicalrsi.robodojo.policy-provider/v1"


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def describe_provider(root: str | Path, *, policy: str, checkpoint: str | Path | None = None) -> dict:
    """Describe a local provider and pin its adapter files and optional checkpoint."""
    if policy not in {"pi05", "pi05-sparse-memory"}:
        raise ValueError("Unsupported provider policy: " + policy)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Provider directory does not exist: " + str(root))
    required = [root / "model.py"]
    for name in ("process_data.sh", "train.sh", "eval.sh", "deploy.yml"):
        candidate = root / name
        if candidate.exists():
            required.append(candidate)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Provider files missing: " + ", ".join(missing))
    files = {str(path): file_digest(path) for path in required}
    checkpoint_path = None
    checkpoint_files = {}
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise ValueError("Checkpoint directory does not exist: " + str(checkpoint_path))
        for path in sorted(checkpoint_path.rglob("*")):
            if path.is_file():
                checkpoint_files[str(path)] = file_digest(path)
        if not checkpoint_files:
            raise ValueError("Checkpoint directory is empty: " + str(checkpoint_path))
        files.update(checkpoint_files)
    return {
        "schema": SCHEMA,
        "policy": policy,
        "provider_root": str(root),
        "provider_revision": _git_revision(root),
        "checkpoint_root": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_files": checkpoint_files,
        "files": files,
        "weights": "external; not copied into this release",
    }


def register(root: str | Path, output: str | Path, *, policy: str, checkpoint: str | Path | None = None) -> dict:
    manifest = describe_provider(root, policy=policy, checkpoint=checkpoint)
    atomic_json(Path(output).expanduser().resolve(), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=("pi05", "pi05-sparse-memory"))
    parser.add_argument("provider")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.provider, args.output, policy=args.policy, checkpoint=args.checkpoint), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
