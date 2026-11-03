"""OEH_Review episode loop, run inside the RoboDojo evaluation client.

XPolicyLab's official OpenWAM loop (``policy/OpenWAM/deploy.py``) drives the environment unchanged; the OEH native
gateway (``runtime/oeh_policy/deploy.py``) sits between that loop and the task environment, so every native action
is the one the OEH runtime admitted, and the policy's observations, decisions and receipts are recorded in an
evidence sidecar under ``runs/`` (``OEH_REVIEW_RUNS_ROOT``). It diverges from ``demo_policy/deploy.py`` only in that
wrapping; the action chunks still come from the policy server.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

POLICY_DIR = Path(__file__).resolve().parent
RUNTIME = POLICY_DIR / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

_EPISODES = 0
_RUN_LABEL = os.environ.get("OEH_REVIEW_RUN_LABEL") or time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
# Which observation bytes the sidecar keeps (runtime/oeh_eval/direct_bridge.parse_retention): every array, the frames
# the model read, the first and last observation, the head camera of every 5th observation. "all" keeps every frame.
_DEFAULT_RETENTION = json.dumps({"retention": "read-endpoints-head", "head_every": 5})


def _prepare_episode() -> Path:
    """One fresh sidecar directory per episode; the gateway refuses to reuse one."""
    global _EPISODES
    _EPISODES += 1
    root = Path(os.environ.get("OEH_REVIEW_RUNS_ROOT") or POLICY_DIR / "runs").resolve()
    episode_dir = root / _RUN_LABEL / f"ep{_EPISODES:03d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    os.environ["OEH_EVAL_EVENTS"] = str(episode_dir / "policy-events.jsonl")
    os.environ.setdefault("OEH_EVAL_OBSERVATION_RETENTION", _DEFAULT_RETENTION)
    print(f"[OEH_Review] episode {_EPISODES}: evidence in {episode_dir}", flush=True)
    return episode_dir


def eval_one_episode(TASK_ENV, model_client):
    _prepare_episode()
    from oeh_policy.deploy import eval_one_episode as oeh_episode
    oeh_episode(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise RuntimeError("OEH_Review serves one environment per policy server (deploy.yml: eval_batch: false)")
