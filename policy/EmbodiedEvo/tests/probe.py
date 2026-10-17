"""No-LLM diagnostic of the real persistent Python/feedback bridge."""

import traceback
from pathlib import Path

from XPolicyLab.policy.EmbodiedEvo.runtime.xpolicylab.bridge import BridgeTask
from XPolicyLab.policy.EmbodiedEvo.runtime.config import config_from_dict
from XPolicyLab.policy.EmbodiedEvo.runtime.files import write_json
from XPolicyLab.policy.EmbodiedEvo.runtime.execution.session import Session
from XPolicyLab.policy.EmbodiedEvo.runtime.workspace import create_workspace


def run_probe(connection, spec, observation, step_limit, episode):
    root = Path(spec["run_dir"])
    run = root / "episodes" / f"episode_{episode:04d}"
    try:
        cfg = config_from_dict(spec["task_config"])
        create_workspace(run, Path(spec.get("inputs_dir", root / "inputs")),
                         Path(spec.get("harness_dir", root / "frozen_harness")))
        env = BridgeTask(connection, observation, step_limit, phase="test")
        session = Session(env, run, phase="test", seed=0, budget=cfg.test.budget,
                          seconds=None, execution_seconds=cfg.test.submission_timeout_sec,
                          fps=cfg.feedback_fps)
        write_json(run / "runtime.json", {"status": "running", "diagnostic": True})
        assert not env._exec_globals, "Python globals leaked between episodes"

        def execute(code, *, ok=True):
            response = session.handle({"op": "exec", "files": [{"path": "probe.py", "code": code}]})
            if bool(response["ok"]) != ok:
                raise AssertionError(response)
            return response

        execute("marker = 7\ninitial_instruction = get_instruction()\n"
                "a = {k:v for k,v in get_observation()['state'].items() if k.endswith('joint_state')}\n"
                "step(a)")
        execute("assert marker == 7\nassert get_instruction() == initial_instruction\n"
                "a = {k:v for k,v in get_observation()['state'].items() "
                "if (k.endswith('ee_pose') and not k.endswith('delta_ee_pose')) "
                "or k.endswith('ee_joint_state')}\nstep(a)")
        before = env.control_steps
        denied = execute("reset()", ok=False)
        assert "PermissionError" in denied["stderr"] and env.control_steps == before
        session.handle({"op": "finish"})
        write_json(run / "runtime.json", {
            "status": "finished", "diagnostic": True, "model_called": False,
            "native_steps": env.control_steps, "executions": session.used,
            "test_reset_denied": True, "persistent_globals": True, "score": None,
        })
        connection.send({"kind": "finished"})
    except BaseException:
        error = traceback.format_exc()
        write_json(run / "runtime.json", {"status": "failed", "diagnostic": True, "error": error})
        connection.send({"kind": "error", "error": error})
    finally:
        connection.close()
