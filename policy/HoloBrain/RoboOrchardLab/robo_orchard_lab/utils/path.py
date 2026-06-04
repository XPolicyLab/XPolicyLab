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
from contextlib import contextmanager

__all__ = ["DirectoryNotEmptyError", "is_empty_directory", "in_cwd", "abspath"]


class DirectoryNotEmptyError(Exception):
    """Exception raised when a directory is not empty as expected."""

    pass


def is_empty_directory(directory: str) -> bool:
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a directory!")
    try:
        next(os.scandir(directory))
        return False
    except StopIteration:
        return True


@contextmanager
def in_cwd(destination: str):
    """Context manager to temporarily change the current working directory.

    This provides a safe way to perform operations within a specific directory.
    It changes the directory to the given path upon entering the 'with'
    block and guarantees that the original directory is restored upon exiting,
    even if an exception occurs.

    Args:
        destination (str): The path of the directory to change into.

    Yields:
        str: The changed directory

    Example:
        >>> import os
        >>> if not os.path.exists("my_temp_dir"):
        ...     os.makedirs("my_temp_dir")
        >>> original_dir = os.getcwd()
        >>> print(f"Starting in: {original_dir}")
        Starting in: /path/to/current
        >>> with in_cwd("my_temp_dir"):
        ...     print(f"Inside 'with' block: {os.getcwd()}")
        Inside 'with' block: /path/to/current/my_temp_dir
        >>> print(f"Back in: {os.getcwd()}")
        Back in: /path/to/current
        >>> os.getcwd() == original_dir
        True
    """
    try:
        original_path = os.getcwd()
        os.chdir(destination)
        yield destination
    finally:
        os.chdir(original_path)


def abspath(path: str) -> str:
    """Get absolute path."""
    return os.path.abspath(os.path.expanduser(path))
