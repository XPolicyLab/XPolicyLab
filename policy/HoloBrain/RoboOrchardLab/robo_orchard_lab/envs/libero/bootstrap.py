# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import importlib.util
import os
from pathlib import Path

import yaml

LIBERO_CONFIG_ENV_VAR = "LIBERO_CONFIG_PATH"
LIBERO_CONFIG_FILE_NAME = "config.yaml"

__all__ = [
    "build_libero_default_path_config",
    "ensure_libero_config",
    "get_libero_config_file",
    "get_libero_config_root",
    "resolve_libero_benchmark_root",
]


def resolve_libero_benchmark_root(
    benchmark_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the installed LIBERO benchmark root.

    Args:
        benchmark_root (str | os.PathLike[str] | None, optional): Optional
            explicit LIBERO benchmark root override. If omitted, infer the
            installed path from the `libero.libero` module.

    Returns:
        Path: The resolved LIBERO benchmark root directory.
    """
    if benchmark_root is not None:
        return Path(benchmark_root).expanduser().resolve()

    try:
        spec = importlib.util.find_spec("libero.libero")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "libero.libero is not installed; cannot prepare LIBERO config."
        ) from exc
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "libero.libero is not installed; cannot prepare LIBERO config."
        )

    return Path(spec.origin).resolve().parent


def get_libero_config_root(
    config_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the directory that stores the LIBERO user config.

    Args:
        config_root (str | os.PathLike[str] | None, optional): Optional
            explicit config root override. If omitted, honor
            `LIBERO_CONFIG_PATH` and fall back to `~/.libero`.

    Returns:
        Path: The resolved LIBERO config directory.
    """
    if config_root is not None:
        return Path(config_root).expanduser().resolve()

    env_config_root = os.environ.get(LIBERO_CONFIG_ENV_VAR)
    if env_config_root is not None:
        return Path(env_config_root).expanduser().resolve()

    return (Path.home() / ".libero").resolve()


def get_libero_config_file(
    config_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the full LIBERO config file path.

    Args:
        config_root (str | os.PathLike[str] | None, optional): Optional
            explicit config root override.

    Returns:
        Path: The resolved config file path.
    """
    return get_libero_config_root(config_root) / LIBERO_CONFIG_FILE_NAME


def build_libero_default_path_config(
    benchmark_root: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Build the default LIBERO path configuration.

    Args:
        benchmark_root (str | os.PathLike[str] | None, optional): Optional
            explicit LIBERO benchmark root override.

    Returns:
        dict[str, str]: The default LIBERO path configuration.
    """
    benchmark_root_path = resolve_libero_benchmark_root(benchmark_root)
    return {
        "benchmark_root": str(benchmark_root_path),
        "bddl_files": str((benchmark_root_path / "bddl_files").resolve()),
        "init_states": str((benchmark_root_path / "init_files").resolve()),
        "datasets": str((benchmark_root_path / "../datasets").resolve()),
        "assets": str((benchmark_root_path / "assets").resolve()),
    }


def ensure_libero_config(
    config_root: str | os.PathLike[str] | None = None,
    benchmark_root: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Ensure the LIBERO config file exists with non-interactive defaults.

    Args:
        config_root (str | os.PathLike[str] | None, optional): Optional
            explicit config root override.
        benchmark_root (str | os.PathLike[str] | None, optional): Optional
            explicit LIBERO benchmark root override.
        overwrite (bool, optional): Whether to overwrite an existing config.
            Default is False.

    Returns:
        Path: The LIBERO config file path.
    """
    config_file = get_libero_config_file(config_root)
    if config_file.exists() and not overwrite:
        return config_file

    config_file.parent.mkdir(parents=True, exist_ok=True)
    config = build_libero_default_path_config(benchmark_root)
    with open(config_file, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False)

    return config_file


def main() -> None:
    """Ensure the default LIBERO config exists and print its path."""
    config_file = ensure_libero_config()
    print(config_file)


if __name__ == "__main__":
    main()
