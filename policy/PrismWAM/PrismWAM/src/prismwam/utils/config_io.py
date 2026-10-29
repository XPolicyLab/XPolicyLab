"""Atomic configuration publication for concurrent distributed rank startup."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from omegaconf import OmegaConf


def atomic_save_config(payload, path: str | Path) -> None:
    """Publish a whole YAML document without truncating a reader's input file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            OmegaConf.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_training_config(
    payload,
    output_dir: str | Path,
    *,
    source_config_path: str | Path | None = None,
    rank: int = 0,
) -> Path:
    """Keep a launcher's input immutable and publish resolved YAML on rank zero.

    Some shared filesystems invalidate readers when an open file is replaced.
    Atomic rename alone cannot protect a configuration still being read by
    other ranks, so use a different output when the launcher reads config.yaml.
    """
    destination = Path(output_dir) / "config.yaml"
    if source_config_path is not None:
        if Path(source_config_path).resolve() == destination.resolve():
            destination = destination.with_name("config.resolved.yaml")
    if rank == 0:
        atomic_save_config(payload, destination)
    return destination
