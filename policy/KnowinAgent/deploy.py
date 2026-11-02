"""KnowinAgent on the official RoboDojo evaluator.

The RoboDojo client calls ``eval_one_episode(TASK_ENV, model_client)`` once per episode after resetting the
layout. KnowinAgent runs one test episode in this process: a Codex task agent (and its local agent) drive the robot
through KnowinAgent's program and tool interface, which reads the official observation (``TASK_ENV.get_obs()``) and
acts through ``TASK_ENV.take_action``. Frozen knowledge is optional and explicitly configured; camera calibration
comes from rig constants. Nothing reads the evaluator's reward, success flag, scene or layout. When the agent
stops before the evaluator ended the episode, the arms hold still until it does, so the evaluator scores the
tiers reached. The policy server's model is a no-op (``model.py``); ``model_client`` is not used.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

if __package__:
    from ._paths import ADAPTER_ROOT as HERE, CONFIG, KNOWLEDGE, PACKAGE_ROOT
else:  # train.sh exposes this adapter for the core's top-level deploy import.
    from _paths import ADAPTER_ROOT as HERE, CONFIG, KNOWLEDGE, PACKAGE_ROOT


def _harness_on_path() -> None:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))


def knowledge_dir(task: str) -> Path | None:
    """Use only an explicit frozen directory; preview episodes otherwise start empty."""
    override = os.environ.get("KNOWIN_AGENT_KNOWLEDGE_DIR")
    if not override:
        return None
    _harness_on_path()
    from knowin_agent.workspace.freeze import verify

    path = Path(override).expanduser().resolve()
    verify(path)
    return path


def episode_config(task: str, *, use_environment_knowledge: bool = True) -> dict:
    """The test-episode config (``test_config.json``) for ``task``, with machine-specific values from the env."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["backend"]["task"] = task
    source = knowledge_dir(task) if use_environment_knowledge else None
    if source is not None:
        config["knowledge_from"] = str(source)
    switches = config.setdefault("switches", {})
    switches["agent.model"] = os.environ.get("KNOWIN_AGENT_MODEL", switches.get("agent.model"))
    switches["agent.effort"] = os.environ.get("KNOWIN_AGENT_EFFORT", switches.get("agent.effort"))
    agent = config["agent"]
    agent["binary"] = os.environ.get("KNOWIN_AGENT_CODEX_BIN", agent.get("binary", "codex"))
    agent["python"] = os.environ.get("KNOWIN_AGENT_PYTHON", sys.executable)
    agent["pythonpath"] = str(PACKAGE_ROOT)
    if os.environ.get("KNOWIN_AGENT_CODEX_HOME"):
        agent["codex_home_source"] = os.environ["KNOWIN_AGENT_CODEX_HOME"]
    if os.environ.get("KNOWIN_AGENT_AGENT_ISOLATION"):
        agent["isolation"] = json.loads(Path(os.environ["KNOWIN_AGENT_AGENT_ISOLATION"]).read_text(encoding="utf-8"))
    if os.environ.get("KNOWIN_AGENT_AGENT_SANDBOX"):
        agent["sandbox"] = os.environ["KNOWIN_AGENT_AGENT_SANDBOX"]
    if os.environ.get("KNOWIN_AGENT_VIDEO_STRIDE"):
        config["video_stride"] = int(os.environ["KNOWIN_AGENT_VIDEO_STRIDE"])
    return config


def _trace_dir(task_env: Any, task: str) -> Path:
    root = os.environ.get("KNOWIN_AGENT_TRACE_DIR") or os.path.join(getattr(task_env, "save_dir", "."), "knowin_agent")
    return Path(root) / f"{task}-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"


def _debug_episode(task_env: Any, model_client: Any, task: str) -> None:
    """Validate assembly, then exercise the official transport with joint holds only.

    The mock backend is used solely to assemble the runtime and frozen workspace. The
    debug client's observations/actions go through XPolicyLab's actual model server;
    no task agent, simulator, or perception service is started.
    """
    from knowin_agent.assembly import assemble
    from knowin_agent.backends.mock import demo_world
    from knowin_agent.workspace.freeze import verify

    config = copy.deepcopy(episode_config(task))
    if config.get("knowledge_from"):
        verify(Path(config["knowledge_from"]))
    config["perception"] = {"depth": "none"}
    config["video_stride"] = 0
    # Use a unique directory outside the immutable policy assets even when the official
    # TestEnv has no save_dir. Keep the assembly receipt available for inspection.
    trace_root = os.environ.get("KNOWIN_AGENT_TRACE_DIR")
    if trace_root:
        Path(trace_root).mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"knowin-agent-debug-{task}-", dir=trace_root))
    backend = demo_world()
    try:
        kit = assemble(config, run_dir, backend=backend)
        if kit.workspace is None or kit.program is None:
            raise RuntimeError("debug assembly did not create the program workspace")
        actions = 0
        while not task_env.is_episode_end():
            # Send raw OR encoded colors untouched: the official server owns decoding.
            model_client.call(func_name="update_obs", obs=task_env.get_obs())
            action = model_client.call(func_name="get_action")
            task_env.take_action(action)
            actions += 1
        receipt = {"mode": "debug", "task": task, "actions": actions, "model_requests": 0,
                   "perception_requests": 0, "scored": False, "workspace": str(kit.workspace.root)}
        (run_dir / "debug.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"[KnowinAgent][DEBUG] {task}: {actions} hold actions; model/perception requests=0; "
              f"not scored; receipt={run_dir / 'debug.json'}", flush=True)
    finally:
        backend.close()


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    mode = os.environ.get("EVAL_ENV_TYPE", "sim") or "sim"
    if mode != "debug" and os.environ.get("KNOWIN_AGENT_DRY_RUN", "0") == "1":
        raise RuntimeError("KNOWIN_AGENT_DRY_RUN=1 is only permitted with EVAL_ENV_TYPE=debug")
    _harness_on_path()
    if mode == "debug":
        task = getattr(TASK_ENV, "task_name", None) or getattr(TASK_ENV, "deploy_cfg", {}).get("task_name")
        if not task:
            raise ValueError("debug environment has no task_name")
        _debug_episode(TASK_ENV, model_client, str(task))
        return
    del model_client  # Simulation: the existing client-side agent owns all actions.
    from knowin_agent.backends.robodojo.official import OfficialRoboDojoBackend
    from knowin_agent.bridge.host import run_hosted_episode

    task = str(getattr(TASK_ENV, "task_name"))
    backend = OfficialRoboDojoBackend.attach(TASK_ENV, task)
    run_dir = _trace_dir(TASK_ENV, task)
    try:
        result = run_hosted_episode(episode_config(task), run_dir, backend=backend)
        print(f"[KnowinAgent] {task}: agent stopped ({result.get('stop_reason')}), "
              f"{result.get('executions')} programs", flush=True)
    except Exception:  # noqa: BLE001 - the evaluator still scores what the robot reached
        print(f"[KnowinAgent] {task}: episode error\n{traceback.format_exc()}", flush=True)
    finally:
        held = backend.hold_until_end()
        if held:
            print(f"[KnowinAgent] {task}: held still for the remaining {held} actions", flush=True)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    if int(getattr(TASK_ENV, "num_envs", 1) or 1) > 1:
        raise RuntimeError("KnowinAgent runs one agent per environment; set eval_batch: false in deploy.yml")
    eval_one_episode(TASK_ENV, model_client)
