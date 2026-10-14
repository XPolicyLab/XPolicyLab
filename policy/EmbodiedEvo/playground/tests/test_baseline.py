"""RoboDojo direct Test baseline: clean episodes and no transferred knowledge."""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_robodojo import task as robodojo_task

from orchestrator import launch_agent, replay, session_runner
from orchestrator.__main__ import main
from orchestrator.config import REPO, config_dict, config_from_dict, load_task
from orchestrator.files import hashes, test_seeds, write_json
from orchestrator.resources import GpuSlots
from orchestrator.session import Session
from scripts import smoke_session


class BaselineTests(unittest.TestCase):
    def setUp(self):
        recipe_switch = patch.dict(os.environ, {"EMBODIEDEVO_USE_RECIPE": "0"})
        recipe_switch.start()
        self.addCleanup(recipe_switch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cwd = patch("simulators.robodojo.robodojo_cwd", contextlib.nullcontext)
        cwd.start()
        self.addCleanup(cwd.stop)
        cfg = load_task("tasks_baseline/RoboDojo")
        self.cfg = replace(cfg, test=replace(cfg.test, scenes=2))

    def episode(self, cfg, scene, experiment, *, phase, harness=None, **kwargs):
        self.assertEqual(phase, "test")
        self.assertIsNone(harness)
        workspace = session_runner.create_workspace(
            cfg, scene, experiment / "inputs", phase=phase, harness=harness
        )
        self.assertEqual(
            {item.name for item in workspace.iterdir()},
            {"instruction.md", "primitives", "scripts", "submission", "observations", "scratch"},
        )
        self.assertEqual(
            (workspace / "instruction.md").read_bytes(),
            (cfg.workspace / "instruction.md").read_bytes(),
        )
        for component in ("submission", "scratch", "observations"):
            self.assertEqual(hashes(workspace / component), hashes(cfg.workspace / component))
        (workspace / "submission/solution.py").write_text("scene_only = 99\n")
        (workspace / "scratch/note.md").write_text(scene.name)
        (workspace / "observations/scene-only.json").write_text("{}")
        result = {"valid": True, "task_completed": True}
        write_json(scene / "result.json", result)
        return result

    def test_robodojo_baseline_pins_one_instruction_and_roundtrips(self):
        definitions = [REPO / "tasks_baseline/RoboDojo/task.toml"]
        self.assertTrue(definitions)
        for definition in definitions:
            with self.subTest(task=definition.parent.name):
                cfg = load_task(definition.parent, env_task="another_native_task")
                self.assertTrue(cfg.baseline)
                self.assertIsNone(cfg.playground)
                self.assertEqual(cfg.simulator.env_task, "another_native_task")
                self.assertEqual(config_from_dict(config_dict(cfg)), cfg)
                self.assertTrue(cfg.expose_success_feedback)
                session_runner.validate(cfg)
                run = self.root / definition.parent.name
                session_runner.prepare_inputs(cfg, run)
                self.assertTrue((run / "inputs/instruction.md").is_file())
                for name in ("skills", "lessons", "playground.md", "test.md"):
                    self.assertFalse((run / "inputs" / name).exists())
        regular = config_dict(load_task("RoboDojo"))
        self.assertNotIn("baseline", regular)
        self.assertNotIn("feedback_fps", regular["test"])
        self.assertEqual(config_dict(config_from_dict(regular)), regular)

    def test_direct_test_starts_clean_scenes_and_resumes_without_replaying(self):
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent):
                run = self.root / ("parallel-direct" if concurrent else "serial-direct")
                options = {"test_gpus": [0], "gpu_slots": 2} if concurrent else {}
                with (
                    patch.object(session_runner, "run_episode", side_effect=self.episode)
                    as execute,
                    patch("simulators.robodojo_layouts.available_scenes", return_value=[0, 1]),
                    patch.object(
                        session_runner, "GpuSlots",
                        side_effect=lambda gpus, count: GpuSlots(
                            gpus, count, directory=self.root / "locks"
                        ),
                    ),
                ):
                    self.assertEqual(session_runner.test(self.cfg, run, **options), 0)
                self.assertEqual(execute.call_count, 2)
                self.assertFalse((run / "playground").exists())
                self.assertFalse((run / "frozen_harness").exists())
                plan = (run / "test/plan.json").read_bytes()
                self.assertEqual(set(json.loads(plan)["seeds"]), {0, 1})
                with (
                    patch.object(
                        session_runner, "test_seeds", side_effect=AssertionError("resampled")
                    ),
                    patch.object(
                        session_runner, "run_episode", side_effect=AssertionError("replayed")
                    ),
                    patch.object(
                        session_runner, "GpuSlots",
                        side_effect=lambda gpus, count: GpuSlots(
                            gpus, count, directory=self.root / "locks"
                        ),
                    ),
                ):
                    self.assertEqual(session_runner.test(self.cfg, run, **options), 0)
                self.assertEqual((run / "test/plan.json").read_bytes(), plan)
                report = json.loads((run / "test.json").read_text())
                self.assertEqual(report["success_rate"], 1.0)
                self.assertEqual(report["scene_indices"], [0, 1])

    def test_resume_launches_only_unstarted_scenes_and_checks_pinned_inputs(self):
        run = self.root / "resume"
        with patch.object(session_runner, "test_seeds", return_value=[0, 1]):
            state, seeds = session_runner.prepare_test(self.cfg, run)
        scene = run / "test/scene_0000"
        write_json(scene / "identity.json", {
            "scene_index": 0, "seed": seeds[0], "definition": state["definition"],
        })
        self.episode(self.cfg, scene, run, phase="test")
        with (
            patch.object(session_runner, "test_seeds", side_effect=AssertionError("resampled")),
            patch.object(session_runner, "run_episode", side_effect=self.episode) as execute,
        ):
            self.assertEqual(session_runner.test(self.cfg, run), 0)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["seed"], 1)
        (run / "inputs/instruction.md").write_text("changed instruction")
        with self.assertRaisesRegex(ValueError, "Pinned workspace source changed"):
            session_runner.test(self.cfg, run)

    def test_baseline_rejects_playground_and_inherited_knowledge(self):
        with self.assertRaisesRegex(ValueError, "no Playground"):
            session_runner.explore(self.cfg, self.root / "explore")
        self.assertFalse((self.root / "explore").exists())
        for phase, harness, submission in (
            ("playground", None, None),
            ("test", self.root / "harness", None),
            ("test", None, self.root / "old-solution.py"),
        ):
            with (
                self.subTest(phase=phase, harness=harness, submission=submission),
                self.assertRaisesRegex(ValueError, "without inherited"),
            ):
                session_runner.create_workspace(
                    self.cfg, self.root / "rejected", self.cfg.workspace,
                    phase=phase, harness=harness, submission=submission,
                )
        with patch.object(session_runner, "test_seeds", return_value=[0, 1]):
            session_runner.prepare_test(self.cfg, self.root / "contaminated")
        (self.root / "contaminated/frozen_harness").mkdir()
        with self.assertRaisesRegex(ValueError, "must not inherit"):
            session_runner.prepare_test(self.cfg, self.root / "contaminated")

    def test_baseline_rejects_legacy_config_and_workspace_templates(self):
        cfg = replace(self.cfg, root=self.root / "source")
        shutil.copytree(self.cfg.root, cfg.root)
        task = cfg.root / "task.toml"
        task.write_text(task.read_text() + "\n[playground]\nseed = 0\n")
        with self.assertRaisesRegex(ValueError, r"remove \[playground\]"):
            load_task(cfg.root)
        for name in ("skills", "lessons", "playground.md", "test.md"):
            with self.subTest(name=name):
                path = cfg.workspace / name
                path.touch()
                with self.assertRaisesRegex(ValueError, "Baseline workspace cannot contain"):
                    session_runner.prepare_inputs(cfg, self.root / "rejected")
                path.unlink()

    def test_baseline_samples_all_layouts_including_zero(self):
        with patch("simulators.robodojo_layouts.available_scenes", return_value=[0, 1]):
            self.assertEqual(set(test_seeds(self.cfg)), {0, 1})
            mixed = test_seeds(self.cfg, settings=["standard", "random"])
            self.assertEqual(len(mixed), 2)
            self.assertTrue(set(mixed) <= {0, 1})
        with (
            patch("simulators.robodojo_layouts.available_scenes", return_value=[0]),
            self.assertRaisesRegex(ValueError, "available Test layouts"),
        ):
            test_seeds(self.cfg)

    def test_paired_random_plans_log_variants_and_never_resample_on_resume(self):
        settings = ["standard"] * 3 + ["random"] * 2
        shared = self.root / "shared/stack_bowls.json"
        cfgs = [replace(self.cfg, agent=replace(self.cfg.agent, model=model, reasoning_effort=effort),
                        test=replace(self.cfg.test, scenes=5, excluded_seed=0, shared_plan=str(shared)))
                for model, effort in [("gpt-6-astra", "xhigh"), ("gpt-6-astra", "medium"),
                                      ("gpt-5.6-sol", "xhigh"), ("gpt-5.6-luna", "xhigh")]]
        banks = {"stack_bowls": [0, 7, 8, 9], "stack_bowls_random": [0, 7]}

        def prepare(i):
            return session_runner.prepare_test(cfgs[i], self.root / str(i), settings=settings)[1]

        with patch("simulators.robodojo_layouts.available_scenes", side_effect=banks.__getitem__) as available, \
                patch("orchestrator.files.random.SystemRandom.sample", side_effect=lambda values, count: values[-count:]):
            with ThreadPoolExecutor(max_workers=4) as pool:
                selected = list(pool.map(prepare, range(4)))
        self.assertEqual(available.call_count, 2)  # One draw for each native variant, across all four groups.
        self.assertEqual(selected, [[7, 8, 9, 0, 7]] * 4)
        before = shared.read_bytes()
        for i, cfg in enumerate(cfgs):
            local = self.root / str(i) / "test/plan.json"
            plan = json.loads(local.read_text())
            self.assertEqual(plan["scenes"][3], {"scene_index": 3, "env_task": "stack_bowls_random",
                "variant": "random", "seed": 0, "bank": 0, "layout_ordinal": 0})
            self.assertEqual(plan["shared_plan_source"], str(shared))
            self.assertFalse((self.root / str(i) / "inputs" / shared.name).exists())
            with patch.object(session_runner, "shared_test_seeds", side_effect=AssertionError("resampled")):
                self.assertEqual(prepare(i), selected[i])
            plan["seeds"][1] = plan["seeds"][0]
            write_json(local, plan)
            with self.assertRaisesRegex(ValueError, "Test plan changed"):
                prepare(i)
        self.assertEqual(before, shared.read_bytes())

    def test_client_only_submits_current_episode_code_and_has_no_reset_command(self):
        source = self.cfg.workspace / "scripts/env.py"
        spec = importlib.util.spec_from_file_location("baseline_client", source)
        client = importlib.util.module_from_spec(spec)
        with patch("sys.dont_write_bytecode", True):
            spec.loader.exec_module(client)
        workspace = self.root / "client"
        (workspace / "submission").mkdir(parents=True)
        (workspace / "submission/solution.py").write_text("advance()\n")
        (workspace / "submission/helpers.py").write_text("def advance():\n    step({})\n")
        files = client.source_snapshot(
            workspace, "submission/solution.py", ["submission/helpers.py"]
        )
        self.assertEqual(
            [item["path"] for item in files],
            ["submission/helpers.py", "submission/solution.py"],
        )
        (workspace / "skills").mkdir()
        (workspace / "skills/old.py").write_text("step({})")
        for path in ("skills/old.py", str(source)):
            with self.assertRaises(ValueError):
                client.source_snapshot(workspace, path, [])
        (workspace / "submission/linked.py").symlink_to(workspace / "skills/old.py")
        with self.assertRaises(ValueError):
            client.source_snapshot(workspace, "submission/linked.py", [])
        with (
            patch.object(client, "request") as request,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exit_code,
        ):
            client.main(["reset"])
        self.assertEqual(exit_code.exception.code, 2)
        request.assert_not_called()

    def test_docker_mounts_have_no_knowledge_directories(self):
        with (
            patch.object(launch_agent, "require_image"),
            patch.object(launch_agent, "ensure_network"),
            patch.object(launch_agent, "stage_home", return_value=self.root / "home"),
        ):
            _, argv = session_runner.agent_command(
                self.cfg, self.root, self.root / "socket", "test"
            )
        mounts = [item for item in argv if item.startswith("type=bind,")]
        self.assertFalse(any("/skills" in item or "/lessons" in item for item in mounts))
        for component in ("instruction.md", "primitives", "scripts", "observations"):
            self.assertTrue(any(f"dst=/workspace/{component},readonly" in m for m in mounts))

    def test_cli_test_and_run_skip_explore(self):
        for command in ("run", "test"):
            with (
                self.subTest(command=command),
                patch.object(session_runner, "explore") as explore,
                patch.object(session_runner, "test", return_value=0) as evaluate,
            ):
                self.assertEqual(main([
                    command, "--task", "tasks_baseline/RoboDojo",
                    "--env-task", "some_native_task", "--run-dir", str(self.root / command),
                ]), 0)
                explore.assert_not_called()
                self.assertTrue(evaluate.call_args.args[0].baseline)
                self.assertEqual(evaluate.call_args.args[0].simulator.env_task, "some_native_task")
        with (
            contextlib.redirect_stderr(io.StringIO()),
            patch.object(session_runner, "explore") as explore,
            self.assertRaises(SystemExit) as exit_code,
        ):
            main(["explore", "--task", "tasks_baseline/RoboDojo", "--run-dir", str(self.root)])
        self.assertEqual(exit_code.exception.code, 2)
        explore.assert_not_called()


    def test_robodojo_baseline_session_keeps_state_but_cannot_reset(self):
        run = self.root / "session"
        session_runner.prepare_inputs(self.cfg, run)
        workspace = session_runner.create_workspace(
            self.cfg, run / "scene", run / "inputs", phase="test"
        )
        env = robodojo_task()
        obj = Session(
            env, run / "scene", phase="test", seed=0,
            budget=100, seconds=None, execution_seconds=2,
        )

        def submit(code):
            with patch("orchestrator.session.write_videos", return_value={}):
                return obj.handle({
                    "op": "exec",
                    "files": [{"path": "submission/solution.py", "code": code}],
                })

        first = submit(
            "remember = 17\n"
            "state = get_observation()['state']\n"
            "action = {k: v for k, v in state.items() if k.endswith('joint_state')}\n"
            "step(action)"
        )
        self.assertTrue(first["ok"], first)
        registrations = env.native.reward_registrations
        with self.assertRaisesRegex(PermissionError, "forbidden in Test"):
            obj.handle({"op": "reset"})
        self.assertEqual(obj.used, 1)
        denied = submit("restart = reset\nrestart()")
        self.assertFalse(denied["ok"])
        self.assertIn("PermissionError", denied["stderr"])
        self.assertEqual(env.native.reward_registrations, registrations)
        continued = submit("assert remember == 17\nstep(action)")
        self.assertTrue(continued["ok"], continued)
        self.assertEqual(continued["native_steps_used"], 2)
        self.assertEqual(continued["executions_remaining"], 97)
        self.assertFalse((workspace / "skills").exists())
        self.assertFalse((workspace / "lessons").exists())
        self.assertFalse((workspace / "ledger.jsonl").exists())
        self.assertTrue((run / "scene/ledger.jsonl").is_file())
        self.assertTrue(obj.handle({"op": "finish"})["closed"])

    def test_smoke_baseline_starts_test_with_an_explicit_diagnostic_scene(self):
        for host_client in (False, True):
            with self.subTest(host_client=host_client):
                run = self.root / ("host-smoke" if host_client else "docker-smoke")
                with (
                    patch.object(smoke_session, "run_episode", return_value={"valid": True})
                    as execute,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    rc = smoke_session.main([
                        "--task", "tasks_baseline/RoboDojo", "--run-dir", str(run),
                        "--gpu", "4", "--seed", "0",
                        *(["--host-client"] if host_client else []),
                    ])
                self.assertEqual(rc, 0)
                self.assertEqual(execute.call_args.kwargs["phase"], "test")
                self.assertEqual(execute.call_args.kwargs["seed"], 0)
                self.assertIsNone(execute.call_args.args[0].playground)
                report = json.loads((run / "smoke.json").read_text())
                self.assertTrue(report["baseline"])
                self.assertFalse(report["scored"])
                self.assertFalse((run / "test/plan.json").exists())

    def test_smoke_rejects_baseline_playground_and_missing_diagnostic_scene(self):
        for options in ([], ["--seed", "-1"], ["--seed", "0", "--phase", "playground"]):
            with (
                self.subTest(options=options),
                patch.object(smoke_session, "run_episode") as execute,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as exit_code,
            ):
                smoke_session.main([
                    "--task", "tasks_baseline/RoboDojo", "--gpu", "4",
                    "--run-dir", str(self.root / "rejected-smoke"), *options,
                ])
            self.assertEqual(exit_code.exception.code, 2)
            execute.assert_not_called()
            self.assertFalse((self.root / "rejected-smoke").exists())






    def test_archival_replay_uses_baseline_feedback_without_playground_config(self):
        run = self.root / "replay"
        source = run / "workspace/observations/000001/code.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "state = get_observation()['state']\n"
            "step({k: v for k, v in state.items() if k.endswith('joint_state')})\n"
        )
        replay.assemble_history(run, phase="test", seed=0)
        env = robodojo_task()
        env.native.take_action = lambda action: env.native.end_flag.__setitem__(0, True)
        with (
            patch("simulators.backend.build", return_value=env),
            patch.object(env, "close"),
            patch.object(replay, "TraceVideos") as videos,
        ):
            videos.return_value.paths = {}
            videos.return_value.append.return_value = {}
            result = replay.replay_history(self.cfg, run)
        self.assertTrue(result["complete"])
        self.assertTrue(result["success"])
        self.assertEqual(videos.call_args.args[1], 25)


if __name__ == "__main__":
    unittest.main()
