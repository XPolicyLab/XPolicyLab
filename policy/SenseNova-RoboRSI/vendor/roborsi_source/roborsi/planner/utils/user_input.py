# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Queue input shared by planner integrations."""

from __future__ import annotations

import queue

QUIT_TOKENS = frozenset({"/quit", "/exit", "/q"})


def next_user_line(input_queue: "queue.Queue[str | None]") -> str | None:
    """Block for the next actionable user line from an interactive input queue.

    Returns the trimmed line, or ``None`` when the session should end (the queue
    yielded ``None`` or a quit token such as ``/quit``). Empty lines are skipped.
    This is a blocking call; async callers should wrap it with
    :func:`asyncio.to_thread`.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line:
            return line
