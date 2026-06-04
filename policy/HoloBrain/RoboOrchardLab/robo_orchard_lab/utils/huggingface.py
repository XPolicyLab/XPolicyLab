# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import os
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import unquote, urlparse

import fsspec
from accelerate import Accelerator
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile

from robo_orchard_lab.utils.env import set_env

__all__ = [
    "get_accelerate_project_last_checkpoint_id",
    "accelerator_load_state",
    "AcceleratorState",
    "download_hf_resource",
    "auto_add_repo_type",
    "resolve_hf_compatible_path",
]


def get_accelerate_project_last_checkpoint_id(project_dir: str) -> int:
    """Helper function to get last checkpoint id.

    Retrieves the ID of the last checkpoint in the specified project directory.

    This function specifically handles checkpoints saved using the
    `Accelerator.save_state` method from the Hugging Face `accelerate`
    library, which follows an automatic checkpoint naming convention.
    It searches the specified `project_dir/checkpoints` directory,
    extracts numerical IDs from folder names, and returns the highest ID,
    representing the most recent checkpoint.

    Args:
        project_dir (str): Path to the project directory containing the
            `checkpoints` folder. This directory should contain only
            checkpoints saved by `Accelerator.save_state`.

    Returns:
        int: The ID of the last (most recent) checkpoint found in the
            project directory. Returns `-1` if the `checkpoints` directory
            does not exist or is empty.

    Raises:
        ValueError: If no valid checkpoint IDs are found in the `checkpoints`
            directory.

    Example:
        >>> get_accelerate_project_last_checkpoint_id("/path/to/project")
        42

    Note:
        This function assumes that all entries in the `checkpoints` directory
        follow the automatic checkpoint naming pattern used by
        `Accelerator.save_state`. Checkpoints not saved with
        `Accelerator.save_state` may cause this function to fail.
    """
    input_dir = os.path.join(project_dir, "checkpoints")

    if not os.path.exists(input_dir):
        return -1

    iter_ids = []
    for folder_i in os.listdir(input_dir):
        iter_ids.append(
            int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder_i)[0])
        )

    iter_ids.sort()

    return iter_ids[-1]


def accelerator_load_state(
    accelerator: Accelerator,
    input_dir: str,
    cache_dir: str | None = None,
    safe_serialization: bool = True,
    **kwargs,
) -> None:
    """Load the state of the accelerator from a checkpoint.

    This function extends the functionality of `accelerator.load_state` to
    support loading checkpoints from remote filesystems (e.g., S3, GCS).

    It first checks if the `input_dir` is a local path or a remote path.
    If it's a local path, it directly calls `accelerator.load_state`. If it's
    a remote path, it synchronizes the checkpoint files to a local cache
    directory before loading the state.

    Args:
        accelerator (Accelerator): The `Accelerator` instance to load the
            state into.
        input_dir (str): The path to the checkpoint directory or file.
            This can be a local path or a remote path (e.g., S3, GCS).
        cache_dir (str | None): The local directory to cache the checkpoint
            files. This is required if `input_dir` is a remote path.
        safe_serialization (bool): Whether to use safe serialization when
            loading the state. This is used when input_dir is a remote
            path. The names of checkpoint files depend on whether
            `safe_serialization` is set to `True` or `False`. Users should
            ensure that the checkpoint files in the remote directory are
            compatible with the specified `safe_serialization` option.
        **kwargs: Additional arguments passed to `accelerator.load_state`.
    """

    def get_fs_protocol(path: str) -> str:
        """Get the filesystem protocol from a path."""
        path_splits = path.split("://")
        if len(path_splits) == 1:
            protocol = "file"
        else:
            protocol = path_splits[0]
        return protocol

    def sync_remote_checkpoints(
        accelerator: Accelerator,
        remote_dir: str,
        cache_dir: str,
        safe_serialization: bool = True,
    ) -> None:
        """Sync remote checkpoints to local cache."""
        if not accelerator.is_local_main_process:
            raise RuntimeError(
                "sync_remote_checkpoints should only be called "
                "on the main process."
            )
        pj_config = accelerator.project_configuration
        old_v = pj_config.automatic_checkpoint_naming
        # disable automatic checkpoint naming to use given checkpoint
        # directory!
        pj_config.automatic_checkpoint_naming = False
        accelerator.save_state(
            cache_dir, safe_serialization=safe_serialization
        )
        file_names = list(os.listdir(cache_dir))
        for file_name in file_names:
            try:
                with (
                    fsspec.open(
                        os.path.join(remote_dir, file_name), "rb"
                    ) as remote_file,
                    open(
                        os.path.join(cache_dir, file_name), "wb"
                    ) as local_file,
                ):
                    # chunk read and write with 32MB
                    while True:
                        data = remote_file.read(1024 * 1024 * 32)  # type: ignore
                        if not data:
                            break
                        local_file.write(data)
            except FileNotFoundError:
                warnings.warn(
                    f"File {file_name} not found in {cache_dir}. Skipping."
                )
        pj_config.automatic_checkpoint_naming = old_v

    input_dir_fs_protocol = get_fs_protocol(input_dir)
    if input_dir_fs_protocol == "file":
        if not os.path.exists(input_dir):
            raise ValueError(
                f"Checkpoint directory {input_dir} does not exist."
            )
        return accelerator.load_state(input_dir, **kwargs)
    else:
        if cache_dir is None:
            raise ValueError(
                "cache_dir should be specified when input_dir is "
                "not a local path."
            )
        if not os.path.exists(cache_dir):
            raise ValueError(f"Cache directory {cache_dir} does not exist.")

        if accelerator.is_local_main_process:
            sync_remote_checkpoints(
                accelerator,
                input_dir,
                cache_dir,
                safe_serialization=safe_serialization,
            )
        accelerator.wait_for_everyone()
        accelerator.load_state(cache_dir, **kwargs)


@dataclass
class AcceleratorState:
    """A data class for storing the state of the Accelerator.

    This class implements the `state_dict` and `load_state_dict` methods to
    save and load the state of the Accelerator. Any dataclass that is used by
    Accelerator.load_state should inherit from this class.
    """

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the training progress as a dictionary.

        Returns:
            dict: A dictionary containing the current epoch, step, and
                global step IDs.
        """
        return asdict(self)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Loads the state of the training progress from a dictionary.

        Args:
            state (dict): A dictionary containing the state to be loaded.
        """
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Key {key} not found in TrainerProgressState.")


VALID_REPO_TYPES = {"model", "dataset", "space"}


def resolve_hf_compatible_path(path: str, repo_type: str = "model") -> str:
    """Resolve the repository's ``hf://`` URI form into a local resource.

    Non-``hf://`` inputs are returned unchanged so callers can layer their own
    path semantics, such as ``abspath`` or local-existence checks, on top.

    Args:
        path (str): Local path, standard Hugging Face identifier, or
            repository-specific ``hf://`` URI.
        repo_type (str, optional): Default repo type to inject for bare
            ``hf://`` URIs that omit it. Default is ``"model"``.

    Returns:
        str: Local downloaded path for ``hf://`` URIs, otherwise the original
            input path.
    """

    if path.startswith("hf://"):
        return download_hf_resource(auto_add_repo_type(path, repo_type))
    return path


def download_hf_resource(url: str) -> str:
    """Downloads a resource (repo, dir, or file) from the Hugging Face Hub using a custom HF URI scheme.

    The URI scheme is defined as:
    ``hf://[<token>@]<repo_type>/<repo_id>[/<path>][@<revision>]``

    - <token>@: (Optional) User token

    - repo_type: (Optional) 'model', 'dataset', or 'space'

    - repo_id: The repository ID (e.g., 'gpt2' or 'meta-llama/Llama-2-7b-chat-hf')

    - /<path>: (Optional) Download a single file or a specific subdirectory

    - @<revision>: (Optional) A git revision

    Args:
        url (str): The HF URI string.

    Returns:
        str: The local directory path (for repo/dir) or file path.

    Raises:
        ValueError: If the URL format is invalid or missing required components.
    """  # noqa: E501
    if not url.startswith("hf://"):
        raise ValueError("URL must start with hf://")

    parsed_uri = urlparse(url)

    # Extract Token (Priority: Userinfo > Query Param)
    token = parsed_uri.username
    if token:
        token = unquote(token)

    # Handle revision at the end of the path (e.g., .../file.txt@v1.0)
    raw_path = parsed_uri.path
    revision = None
    if "@" in raw_path:
        raw_path, revision = raw_path.rsplit("@", 1)

    # Path reconstruction
    netloc = parsed_uri.netloc

    # remove token part
    if "@" in netloc:
        host_part = netloc.rsplit("@", 1)[-1]
    else:
        host_part = netloc

    path_part = raw_path.lstrip("/")
    full_path_str = (
        f"{host_part}/{path_part}"
        if host_part and path_part
        else (host_part or path_part)
    )
    parts = [p for p in full_path_str.split("/") if p]

    if not parts:
        raise ValueError(f"Invalid URI: Empty path in '{url}'")

    if parts[0] not in VALID_REPO_TYPES:
        raise ValueError(f"Invalid repo type {parts[0]}")
    repo_type = parts[0]
    segments = parts[1:]

    if len(segments) < 2:
        raise ValueError(
            f"Invalid repo_id structure. Expected 'username/repo_name', "
            f"but got: {'/'.join(segments)}"
        )

    repo_id = f"{segments[0]}/{segments[1]}"
    path_inside_repo = "/".join(segments[2:]) if len(segments) > 2 else None

    # download
    with set_env(HF_HUB_DISABLE_PROGRESS_BARS="1"):
        if not path_inside_repo:
            return snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
            )

        api = HfApi(token=token)
        try:
            paths_info = api.get_paths_info(
                repo_id=repo_id,
                repo_type=repo_type,
                paths=[path_inside_repo],
                revision=revision,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to resolve path '{path_inside_repo}' in repo: {e}"
            )

        if not paths_info:
            raise ValueError(
                f"Path '{path_inside_repo}' does not exist in {repo_id}"
            )

        if isinstance(paths_info[0], RepoFile):
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=path_inside_repo,
                revision=revision,
                token=token,
            )
        else:
            allow_pattern = f"{path_inside_repo.rstrip('/')}/*"
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                allow_patterns=[allow_pattern],
            )
            return os.path.join(path, path_inside_repo)


def auto_add_repo_type(url: str, repo_type: str = "model") -> str:
    """Automatically add repo type to url."""

    if not url.startswith("hf://"):
        raise ValueError("URL must start with hf://")

    if repo_type not in VALID_REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}")

    parsed_uri = urlparse(url)

    url_repo_type = parsed_uri.netloc.split("@")[-1]

    if url_repo_type in VALID_REPO_TYPES:
        if url_repo_type != repo_type:
            raise ValueError(
                f"url already has repo type {url_repo_type} "
                f"but not matched required repo type {repo_type}"
            )

        return url

    if parsed_uri.username:
        token, organize = parsed_uri.netloc.split("@")
        netloc = f"{token}@{repo_type}/{organize}/"
    else:
        netloc = f"{repo_type}/{parsed_uri.netloc}"

    return f"{parsed_uri.scheme}://{netloc}/{parsed_uri.path}"
