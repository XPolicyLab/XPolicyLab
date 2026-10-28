"""OEH_Review policy server: OpenWAM-Alpha proposals reviewed by an LLM through the OEH runtime.

The runtime is vendored under ``runtime/`` (``runtime/MANIFEST.json`` names its source commit). ``ckpt_name`` selects
a method configuration from ``configs/<ckpt_name>.template.json``; ``setup_eval_policy_server.sh`` freezes it for
the task at hand into ``state/<ckpt_name>/<task_name>.frozen.json`` (``tools/freeze_config.py``), which this server
loads. The frozen file carries the checkpoint, source and kernel identities; the OEH runtime refuses to start on a
mismatch.

Episodes: XPolicyLab's clients run several episodes against one policy server. The OEH runtime holds one episode
(its kernel, its evidence, its model conversation); ``deploy.py`` closes it at the end of every episode, and the next
``reset`` starts a fresh runtime on the same loaded OpenWAM backend, so nothing of an episode is carried into the
next one.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import threading
import time

POLICY_DIR = Path(__file__).resolve().parent
RUNTIME = POLICY_DIR / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from oeh_policy.model import Model as _OEHModel  # noqa: E402


def frozen_config(ckpt_name: str | None, task_name: str | None) -> Path:
    """The frozen configuration ``setup_eval_policy_server.sh`` prepared for this launch."""
    if not ckpt_name or not (POLICY_DIR / "configs" / f"{ckpt_name}.template.json").is_file():
        known = sorted(p.name[:-len(".template.json")] for p in (POLICY_DIR / "configs").glob("*.template.json"))
        raise RuntimeError(f"ckpt_name must name a method configuration in configs/: {known}, got {ckpt_name!r}")
    if not task_name:
        raise RuntimeError("task_name is required: the frozen configuration carries the task's native step limit")
    path = POLICY_DIR / "state" / ckpt_name / f"{task_name}.frozen.json"
    if not path.is_file():
        raise RuntimeError(f"no frozen configuration at {path}; run tools/freeze_config.py --ckpt {ckpt_name} "
                           f"--task {task_name} (setup_eval_policy_server.sh does this before starting the server)")
    return path


class Model(_OEHModel):
    def __init__(self, model_cfg):
        path = frozen_config(model_cfg.get("ckpt_name"), model_cfg.get("task_name"))
        os.environ["OEH_POLICY_CONFIG"] = str(path)
        os.environ.pop("OEH_EVAL_INTERNAL_CONFIG", None)  # the campaign evaluator's route; a submission uses the file
        self._episodes = 0
        self._run_label = time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
        os.environ.pop("ROBODOJO_RUN_ID", None)  # per-episode attempt ids are set below
        super().__init__(model_cfg)
        frozen = json.loads(path.read_text(encoding="utf-8"))
        print(f"[OEH_Review] configuration {path.name}: mode={frozen['mode']} native_step_limit="
              f"{frozen['native_step_limit']} output={frozen['output']}", flush=True)

    def _rearm(self) -> None:
        """A new episode on the same backend: fresh loop, fresh runtime, nothing carried over."""
        self.runtime = None
        self.used = False
        self.reset_response = None
        self.teacher_start = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="oeh-policy-runtime", daemon=True)
        self.thread.start()

    def reset(self):
        if self.used and self.loop.is_closed():
            # deploy.py closed the previous episode (close_episode stops the runtime and its loop).
            self._rearm()
        if not self.used:
            self._episodes += 1
            self.attempt_id = f"{self._run_label}-ep{self._episodes:03d}"
        return super().reset()
