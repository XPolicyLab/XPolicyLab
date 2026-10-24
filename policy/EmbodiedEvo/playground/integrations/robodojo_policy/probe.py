"""Deterministic interface diagnostic, explicitly without a model or score."""

import traceback
from pathlib import Path

from orchestrator.config import config_from_dict
from orchestrator.files import write_json
from orchestrator.session import Session
from orchestrator.session_runner import create_workspace
from .bridge import BridgeTask


def run_probe(connection, spec, observation, step_limit, episode):
    root = Path(spec["run_dir"])
    run = root / "episodes" / f"episode_{episode:04d}"
    cfg = config_from_dict(spec["task_config"])
    phase = spec["phase"]
    try:
        create_workspace(cfg, run, root / "inputs", phase=phase,
                         harness=root / "frozen_harness" if phase == "test" else None)
        env = BridgeTask(connection, observation, step_limit, phase=phase)
        limits = cfg.playground if phase == "playground" else cfg.test
        session = Session(env, run, phase=phase, seed=0, budget=limits.budget,
                          seconds=None, execution_seconds=limits.submission_timeout_sec,
                          fps=cfg.feedback_fps)

        def execute(code):
            response = session.handle({"op": "exec", "files": [{"path": "probe.py", "code": code}]})
            if not response["ok"]:
                raise RuntimeError(response.get("stderr", "Probe execution failed"))
            return response

        assert not env._exec_globals
        execute("marker = 7\ninitial_instruction = get_instruction()\n"
                "a = {k:v for k,v in get_observation()['state'].items() if k.endswith('joint_state')}\n"
                "step(a)")
        if phase == "playground":
            from simulators.primitives import rgb_images

            for _ in range(spec.get("probe_reset_cycles", 1)):
                used = session.used
                execute("reset()\nassert marker == 7\nassert get_instruction() == initial_instruction")
                assert session.used == used + 1
                assert session.status()["native_steps_remaining"] == step_limit
                frames = rgb_images(env.public_observation())
                assert len(frames) == 3
                assert all(frame.shape == (480, 640, 3) and frame.dtype.name == "uint8"
                           and float(frame.std()) > 1 for frame in frames.values()), "Invalid reset RGB"
            (run / "workspace/lessons/diagnostic.md").write_text(
                "Interface probe only; no task-solving knowledge was learned.\n")
        else:
            try:
                session.handle({"op": "reset"})
            except PermissionError:
                pass
            else:
                raise AssertionError("Test accepted reset")
            assert (run / "workspace/lessons/diagnostic.md").is_file()
        execute("a = {k:v for k,v in get_observation()['state'].items() "
                "if k.endswith('ee_pose') or k.endswith('ee_joint_state')}\nstep(a)")
        session.handle({"op": "finish"})
        write_json(run / "runtime.json", {
            "status": "finished", "diagnostic": True, "model_called": False,
            "phase": phase, "native_steps": env.control_steps,
            "executions": session.used, "success_score": None,
        })
        connection.send({"kind": "finished", "success": env.success})
    except BaseException:
        error = traceback.format_exc()
        write_json(run / "runtime.json", {"status": "failed", "error": error, "diagnostic": True})
        connection.send({"kind": "error", "error": error})
    finally:
        connection.close()
