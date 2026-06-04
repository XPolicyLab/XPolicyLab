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

import logging

import deprecated

from robo_orchard_lab.distributed.utils import get_dist_info

__all__ = ["basic_config"]


@deprecated.deprecated(
    version="0.2.0",
    reason="Use `robo_orchard_core.utils.logging module` instead.",
)
def basic_config(
    format: str = "%rank %(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",  # noqa
    datefmt: str = "%m/%d/%Y %H:%M:%S",
    **kwargs,
):
    """Configures the logging system for distributed training environments.

    This function extends Python's `logging.basicConfig` to include
    distributed rank information (`%rank`) in the log format. It replaces
    the `%rank` placeholder with the rank and world size
    from `get_dist_info()`.

    Args:
        format (str, optional): The log message format. Default value is
            "%(rank)s %(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s".
            - ``%rank``: A placeholder replaced with "Rank[<rank>/<world_size>]".
            - Other placeholders include:

              - ``%(asctime)s``: Timestamp of the log message.
              - ``%(levelname)s``: Log severity level (e.g., DEBUG, INFO).
              - ``%(name)s``: Name of the logger.
              - ``%(lineno)d``: Line number where the log was issued.
        datefmt (str, optional): Date and time format for log messages.
            Default: "%m/%d/%Y %H:%M:%S".
        **kwargs: Additional arguments passed to `logging.basicConfig`, such as
            `level`.

    Raises:
        TypeError: If `format` or `datefmt` is not a string.
        ValueError: If `get_dist_info()` fails to return valid rank/world_size.

    Examples:
        Basic Usage:
            >>> from robo_orchard_lab.logging import basic_config
            >>> import logging
            >>>
            >>> basic_config(level=logging.DEBUG)
            >>> logger = logging.getLogger("example_logger")
            >>> logger.info("This is an info message.")
            # Output might look like (rank/world_size and timestamp will vary):
            # Rank[0/1] 05/22/2025 18:40:00 INFO example_logger:X This is
            # an info message.

        Custom Format:
            >>> basic_config(
            ...     format="%(asctime)s - %(rank)s - %(levelname)s - %(message)s",
            ...     level=logging.INFO,
            ...     datefmt="%Y-%m-%d %H:%M:%S",
            ... )
            >>> logger = logging.getLogger("custom_logger")
            >>> logger.warning("This is a warning message.")
            # Output might look like:
            # 2025-05-22 18:40:00 - Rank[0/1] - WARNING - This is a
            # warning message.

        Non-Distributed Environment (assuming get_dist_info returns 0, 1):
            >>> basic_config()
            >>> logger = logging.getLogger("single_logger")
            >>> logger.debug("Debug message in single-node environment.")
            # Output might look like:
            # Rank[0/1] 05/22/2025 18:40:00 DEBUG single_logger:Y
            # Debug message in single-node environment.

    """  # noqa: E501

    if "%rank" in format:
        dist_info = get_dist_info()
        format = format.replace(
            "%rank", "Rank[{}/{}]".format(dist_info.rank, dist_info.world_size)
        )

    logging.basicConfig(format=format, datefmt=datefmt, **kwargs)
