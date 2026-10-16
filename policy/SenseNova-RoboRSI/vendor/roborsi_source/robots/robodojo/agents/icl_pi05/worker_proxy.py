# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Original worker's adapter-shaped transport; no simulator or policy here."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

from policylab.api import InvalidAction


class Adapter:
    """Relay the original private backend operations to the Runner bridge."""

    def __init__(self, environment_dir, options):
        self.socket = os.environ["ROBORSI_ICL_PRIVATE_SOCKET"]
        self.timeout = float(os.environ["ROBORSI_ICL_REQUEST_SECONDS"])

    def _call(self, **request):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket)
            connection.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
            with connection.makefile("rb") as stream:
                line = stream.readline()
        if not line:
            raise RuntimeError(
                "Original ICL private transport closed; do not repeat action"
            )
        reply = json.loads(line)
        if "invalid_action" in reply:
            raise InvalidAction(reply["invalid_action"])
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def reset(self, episode, record_dir: Path, observation_dir: Path):
        self.run = record_dir.parents[2]
        return self._call(
            op="reset",
            episode=episode,
            record_dir=str(record_dir),
            observation_dir=str(observation_dir),
        )

    def vla_propose(self, subgoal):
        return self._call(op="vla_propose", subgoal=subgoal)

    def step(self, action):
        return self._call(op="step", action=action)

    def finish(self):
        # Original no-reset requests are a protocol failure even if native
        # finish detects success. Retain the caller's recorded operation.
        requests = self.run / "requests.jsonl"
        last = (
            json.loads(requests.read_text().splitlines()[-1])
            if requests.exists()
            else {}
        )
        reason = (
            "reset_requested"
            if last.get("request", {}).get("op") == "reset"
            else "original_icl_finish"
        )
        return self._call(op="finish", original_reason=reason)

    def close(self):
        self._call(op="close")
