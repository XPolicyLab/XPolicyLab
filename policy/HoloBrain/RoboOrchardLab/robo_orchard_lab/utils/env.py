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

import contextlib
import os

__all__ = ["set_env", "get_robo_orchard_home"]


def get_robo_orchard_home() -> str:
    """Get the RoboOrchard home directory from environment or default location.

    Returns:
        str: The path to the RoboOrchard home directory.
    """
    return os.environ.get(
        "ROBO_ORCHARD_HOME",
        os.path.expanduser(os.path.join("~", ".robo_orchard")),
    )


@contextlib.contextmanager
def set_env(*remove, **update):
    """Temporarily sets, updates, or removes environment variables.

    This context manager allows for modifying environment variables only within
    the scope of a `with` statement. Upon exiting the `with` block,
    the original state of all affected environment variables is restored.

    Environment variables specified in the `update` keyword arguments are
    set to the provided values. Environment variables named in the `remove`
    positional arguments are unset.

    If a variable name appears in both `update` and `remove`, it will be
    updated first and then immediately removed, meaning it will be unset
    during the execution of the `with` block. Upon exit, its original
    value (or absence) before entering the context will be restored.

    All values provided for updates should be strings, as environment
    variables can only store strings. While `os.environ.update` might
    attempt to convert non-string values, it's best practice to
    provide strings.

    This implementation is inspired by a solution discussed on Stack Overflow:
    https://stackoverflow.com/questions/2059482/what-is-a-good-way-to-handle-temporary-changes-to-os-environ

    Args:
        *remove (str): Variable positional arguments representing the names
            of environment variables to be temporarily removed (unset).
        **update (str): Arbitrary keyword arguments where keys are environment
            variable names (str) and values are the string values to
            set them to temporarily.

    Yields:
        None: This context manager does not yield a specific value.

    Example:
        >>> os.environ["MY_VAR"] = "initial_value"
        >>> os.environ["TO_DELETE"] = "delete_me"
        >>>
        >>> with set_env(
        ...     "TO_DELETE",
        ...     "NON_EXISTENT_TO_DELETE",
        ...     MY_VAR="new_value",
        ...     NEW_VAR="created",
        ... ):
        ...     print(f"MY_VAR: {os.environ.get('MY_VAR')}")
        ...     print(f"NEW_VAR: {os.environ.get('NEW_VAR')}")
        ...     print(f"TO_DELETE: {os.environ.get('TO_DELETE')}")
        ...     print(
        ...         f"NON_EXISTENT_TO_DELETE: {os.environ.get('NON_EXISTENT_TO_DELETE')}"
        ...     )
        MY_VAR: new_value
        NEW_VAR: created
        TO_DELETE: None
        NON_EXISTENT_TO_DELETE: None
        >>>
        >>> print(f"MY_VAR after: {os.environ.get('MY_VAR')}")
        MY_VAR after: initial_value
        >>> print(f"NEW_VAR after: {os.environ.get('NEW_VAR')}")
        NEW_VAR after: None
        >>> print(f"TO_DELETE after: {os.environ.get('TO_DELETE')}")
        TO_DELETE after: delete_me
        >>>
        >>> # Cleanup example variables
        >>> del os.environ["MY_VAR"]
        >>> del os.environ["TO_DELETE"]
    """  # noqa: E501
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]
