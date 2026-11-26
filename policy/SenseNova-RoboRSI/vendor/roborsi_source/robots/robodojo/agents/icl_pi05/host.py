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

"""Run the unchanged ICL controller in its own private Python process."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from roborsi.planner.codex_cli import CodexCliExternalLauncher
from roborsi.planner.codex_cli_config import CodexCliConfig

from .bwrap_transport import BubblewrapArgumentTransport, model_base_relocation


def main():
    """Inject the backend explicitly; the original controller owns all policy."""
    request = json.loads(Path(sys.argv[1]).read_text())
    cfg = json.loads(Path(request["config"]).read_text())
    source = Path(request["source"])
    sys.path[:0] = [str(source / "code"), str(source)]
    import hybrid_host
    import local_runtime
    from policylab import codex
    from policylab.runner import run_experiment

    cfg = local_runtime.resolve_runtime_config(cfg, expected_model=cfg.get("model") or None)
    hybrid_host.install()
    local_runtime.BWRAP = Path(request["bubblewrap"])
    codex.command = local_runtime.command
    codex.sandbox_command = local_runtime.sandbox_command
    # Scope argument transport to this original launcher only. The shared
    # subprocess module and native worker's subprocess behavior stay intact.
    codex.subprocess = SimpleNamespace(
        PIPE=codex.subprocess.PIPE,
        Popen=BubblewrapArgumentTransport(
            codex.subprocess.Popen,
            local_runtime.BWRAP,
            request.get("execution_identity", {}).get("bubblewrap_sha256"),
            base_relocation=model_base_relocation(
                Path(cfg["agent_python"]), request.get("execution_identity", {})
            ),
        ),
    )
    pin = CodexCliConfig.from_dict(request["cli_identity"])
    launcher = CodexCliExternalLauncher(
        launch=codex.launch,
        identity_config=pin,
        resolve_executable=lambda cfg: codex.native_binary(cfg["codex_binary"]),
    )
    native_config_sha = hashlib.sha256(Path(request["config"]).read_bytes()).hexdigest()
    # Replace only the private simulator transport in this controller process.
    # The Benchmark service keeps the true original adapter and interpreter.
    cfg["benchmark_python"] = request.get("controller_python", sys.executable)
    cfg["adapter"] = str(Path(__file__).with_name("worker_proxy.py")) + ":Adapter"

    def launch(cfg, run, private_home, segment, prompt, session=None):
        # This sidecar binds a source run to its concrete framework attempt;
        # it is private metadata and never part of the policy prompt.
        sidecar = run / "roborsi-controller.json"
        binding = {
            "framework_episode_id": request["framework_episode_id"],
            "native_config_sha256": native_config_sha,
            "source_run": str(run),
            "case_dir": request.get("case_dir"),
            "attempt_dir": request.get("attempt_dir"),
        }
        if sidecar.exists() and json.loads(sidecar.read_text()) != binding:
            raise ValueError("Original controller attempt binding changed")
        sidecar.write_text(json.dumps(binding, indent=2))
        return launcher.launch(cfg, run, private_home, segment, prompt, session)

    run, summary = run_experiment(cfg, request["runs"], launch=launch)
    if not (run / "roborsi-controller.json").exists():
        (run / "roborsi-controller.json").write_text(
            json.dumps(
                {
                    "framework_episode_id": request["framework_episode_id"],
                    "native_config_sha256": native_config_sha,
                    "source_run": str(run),
                    "case_dir": request.get("case_dir"),
                    "attempt_dir": request.get("attempt_dir"),
                },
                indent=2,
            )
        )
    Path(request["receipt"]).write_text(
        json.dumps({"run": str(run), "summary": summary}, indent=2)
    )
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
