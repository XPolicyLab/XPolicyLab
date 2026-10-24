"""Host-only real RoboDojo check, using only the configured Playground scene."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--env-task", default="stack_bowls")
    parser.add_argument("--check-action-limit", action="store_true",
                        help="Exhaust native actions and verify Playground reset budgets")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from orchestrator.config import load_task
    from orchestrator.files import write_json
    from orchestrator.resources import sim_env
    from orchestrator.session import Session
    from orchestrator.session_runner import create_workspace, prepare_inputs
    from simulators.backend import build

    os.environ.update(sim_env(args.gpu, "robodojo"))
    run = args.run_dir.resolve()
    if not args.worker:
        if run.exists():
            raise FileExistsError("Smoke requires a fresh --run-dir")
        # Kit can exit(0) during close() even after a Python exception. The
        # parent verifies the explicit result, not just the native exit code.
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.smoke_robodojo",
                "--worker",
                "--gpu",
                args.gpu,
                "--env-task",
                args.env_task,
                "--run-dir",
                str(run),
                *(["--check-action-limit"] if args.check_action_limit else []),
            ],
            check=False,
        )
        path = run / "smoke.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        return 0 if process.returncode == 0 and result.get("passed") is True else 1
    run.mkdir(parents=True, exist_ok=False)
    cfg = load_task("RoboDojo", env_task=args.env_task)
    prepare_inputs(cfg, run)
    episode = run / "smoke"
    create_workspace(cfg, episode, run / "inputs", phase="playground")
    env = None
    result = {
        "passed": False,
        "kind": "host-only-smoke",
        "env_task": args.env_task,
        "python": sys.executable,
        "python_prefix": sys.prefix,
    }
    try:
        env = build(cfg, seed=cfg.playground.seed)
        import curobo
        import isaaclab

        result.update(isaaclab_source=isaaclab.__file__, curobo_source=curobo.__file__)
        session = Session(
            env,
            episode,
            phase="playground",
            seed=cfg.playground.seed,
            budget=10,
            seconds=None,
            execution_seconds=1200,
        )
        assert not env.success, "Initial upstream viability flag must not become success"
        assert env.official_action_limit
        assert session.status()["native_steps_remaining"] == env.native_step_limit
        from simulators.primitives import rgb_images

        frames = rgb_images(env.public_observation())
        assert len(frames) == 3
        for image in frames.values():
            assert image.shape == (480, 640, 3) and image.dtype.name == "uint8"
            assert float(image.std()) > 1, "Blank RGB camera"
        result["cameras"] = {k: list(v.shape) for k, v in frames.items()}

        def submit(code):
            response = session.handle(
                {"op": "exec", "files": [{"path": "submission/solution.py", "code": code}]}
            )
            assert response["ok"], response.get("stderr", response)
            return response

        submit("remember = 17\nprint(get_instruction())\n")
        response = submit(
            "assert remember == 17\n"
            "s = get_observation()['state']\n"
            "action = {k: v for k, v in s.items() if k.endswith('joint_state')}\n"
            "for i in range(5):\n"
            "    transition = step(action)\n"
            "assert len(transition) == 5\n"
        )
        assert response["native_steps_used"] == 5
        assert response["native_steps_remaining"] == env.native_step_limit - 5
        assert response["videos"]
        submit(
            "before_reset = get_observation()['instruction']\nreset()\n"
            "assert remember == 17\n"
            "assert get_instruction() == before_reset\n"
        )
        assert env.control_steps == 5
        assert session.status()["native_steps_remaining"] == env.native_step_limit
        assert session.status()["executions_used"] == 3
        # Exercise the same reset prohibition as formal Test without selecting Test layouts.
        env.execution_phase = "test"
        transition = env.step("restart = reset\nrestart()")
        assert not transition[-1]["ok"] and "PermissionError" in transition[-1]["stderr"]
        env.execution_phase = "playground"
        submit(
            "s = get_observation()['state']\n"
            "action = {k: v for k, v in s.items() if k.endswith('ee_pose') "
            "or k.endswith('ee_joint_state')}\n"
            "transition = step(action)\nassert len(transition) == 5\n"
        )
        assert env.control_steps == 6
        if args.check_action_limit:
            response = submit(
                "s = get_observation()['state']\n"
                "action = {k: v for k, v in s.items() if k.endswith('joint_state')}\n"
                f"for i in range({env.native_step_limit + 1}):\n"
                "    step(action)\n"
            )
            assert response["truncated"], "Expected native action cutoff on the diagnostic hold"
            assert not response["success"]
            assert response["native_steps_remaining"] == 0
            assert env.native.take_action_cnt[0] == env.native_step_limit
            before = session.status()
            response = submit("step(action)")
            assert response["native_steps_used"] == before["native_steps_used"]
            started = session.started
            response = session.handle({"op": "reset"})
            assert response["ok"], response
            assert not response["truncated"]
            assert response["native_steps_remaining"] == env.native_step_limit
            assert response["executions_used"] == before["executions_used"] + 2
            assert session.started == started
            response = submit("step(action)")
            assert response["native_steps_used"] == before["native_steps_used"] + 1
            result["action_limit_checked"] = True
        session.handle({"op": "finish"})
        result.update(
            passed=True,
            native_steps=env.control_steps,
            native_action_limit=env.native_step_limit,
            success=env.success,
            reset_budget_preserved=True,
            reset_renews_native_action_budget=True,
            test_reset_rejected=True,
            persistent_variables=True,
        )
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        write_json(run / "smoke.json", result)
        if env is not None:
            env.close()
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
