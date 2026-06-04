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

import warnings


def warn_deprecated_package(
    package_name: str, message: str, *, stacklevel: int = 2
) -> None:
    """Emit a package-level deprecation warning.

    Args:
        package_name (str): Deprecated package name. The value is accepted to
            keep call sites self-documenting even though the warning message is
            fully supplied by ``message``.
        message (str): Deprecation message shown to callers.
        stacklevel (int, optional): Warning stacklevel used to point at the
            importing caller. Default is 2.
    """
    del package_name
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
