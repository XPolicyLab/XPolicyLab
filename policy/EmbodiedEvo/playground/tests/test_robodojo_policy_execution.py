"""Shared Debug/campaign execution contracts; no Docker, GPU or model calls."""

import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from integrations.robodojo_policy import bootstrap, debug, execution, worker
from orchestrator import session_runner
from orchestrator.config import REPO, config_dict, load_task
from orchestrator.files import hashes, write_json
from tests.fixtures import observation


class SharedExecutionTests(unittest.TestCase):
    def test_retired_collision_cleanup_is_task_scoped_and_excludes_baseline(self):
        cfg = load_task("RoboDojo")
        profile = execution.read(self.profile)
        profile["disable_retired_collision_tasks"] = [cfg.simulator.env_task]
        write_json(self.profile, profile)
        with patch.object(execution, "apptainer_command", side_effect=self.command) as command, \
                patch.object(execution.subprocess, "Popen", side_effect=self.launch), \
                patch.object(execution, "stop_process"):
            self.assertEqual(session_runner.explore(cfg, self.root / "selected", gpu="0"), 0)
            self.assertIn("--disable-retired-collisions", command.call_args.args[0])
            baseline_cfg = load_task(REPO / "tasks_baseline/RoboDojo")
            baseline = self.root / "baseline"
            session_runner.prepare_test(baseline_cfg, baseline, create_plan=False)
            with self.assertRaisesRegex(ValueError, "only supported in Playground"):
                session_runner.run_episode(baseline_cfg, baseline / "test/scene_0000", baseline,
                    phase="test", seed=1_000_003, seconds=None, gpu="0")
            profile["disable_retired_collision_tasks"] = ["different_task"]
            write_json(self.profile, profile)
            self.assertEqual(session_runner.explore(cfg, self.root / "unselected", gpu="0"), 0)
            self.assertNotIn("--disable-retired-collisions", command.call_args.args[0])

    def test_rigid_reuse_is_task_scoped_and_rejected_in_baseline(self):
        cfg = load_task("RoboDojo")
        profile = execution.read(self.profile)
        profile["reuse_reset_rigid_bodies_tasks"] = [cfg.simulator.env_task]
        write_json(self.profile, profile)
        with patch.object(execution, "apptainer_command", side_effect=self.command) as command, \
                patch.object(execution.subprocess, "Popen", side_effect=self.launch), \
                patch.object(execution, "stop_process"):
            self.assertEqual(session_runner.explore(cfg, self.root / "selected", gpu="0"), 0)
            self.assertIn("--reuse-reset-rigid-bodies", command.call_args.args[0])
            baseline_cfg = load_task(REPO / "tasks_baseline/RoboDojo")
            baseline = self.root / "baseline"
            session_runner.prepare_test(baseline_cfg, baseline, create_plan=False)
            with self.assertRaisesRegex(ValueError, "only supported in Playground"):
                session_runner.run_episode(baseline_cfg, baseline / "test/scene_0000", baseline,
                    phase="test", seed=1_000_003, seconds=None, gpu="0")
            profile["reuse_reset_rigid_bodies_tasks"] = ["different_task"]
            write_json(self.profile, profile)
            self.assertEqual(session_runner.explore(cfg, self.root / "unselected", gpu="0"), 0)
            self.assertNotIn("--reuse-reset-rigid-bodies", command.call_args.args[0])

    def test_collision_capacity_preserves_default_and_solver_settings(self):
        from simulators.robodojo_startup import configure_collision_stack
        physx = SimpleNamespace(gpu_collision_stack_size=100000000, solver_type=1,
                                gpu_max_rigid_patch_count=163840)
        self.assertEqual(configure_collision_stack(physx)["effective_bytes"], 100000000)
        record = configure_collision_stack(physx, 268435456)
        self.assertEqual(record, {"original_bytes": 100000000, "requested_bytes": 268435456,
                                  "effective_bytes": 268435456})
        self.assertEqual(vars(physx), {"gpu_collision_stack_size": 268435456,
                                     "solver_type": 1, "gpu_max_rigid_patch_count": 163840})
        for invalid in (True, -1, 1.5, 100000000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                configure_collision_stack(physx, invalid)
        self.assertEqual(physx.gpu_collision_stack_size, 268435456)

    def test_capacity_override_is_scoped_to_configured_task(self):
        cfg = load_task("RoboDojo")
        profile = execution.read(self.profile)
        profile["collision_stack_bytes_by_task"] = {cfg.simulator.env_task: 268435456}
        write_json(self.profile, profile)
        with patch.object(execution, "apptainer_command", side_effect=self.command) as command, \
                patch.object(execution.subprocess, "Popen", side_effect=self.launch), \
                patch.object(execution, "stop_process"):
            self.assertEqual(session_runner.explore(cfg, self.root / "selected", gpu="0"), 0)
            argv = command.call_args.args[0]
            self.assertEqual(argv[argv.index("--collision-stack-bytes") + 1], "268435456")
            profile["collision_stack_bytes_by_task"] = {"different_task": 268435456}
            write_json(self.profile, profile)
            self.assertEqual(session_runner.explore(cfg, self.root / "unselected", gpu="0"), 0)
            self.assertNotIn("--collision-stack-bytes", command.call_args.args[0])

    def test_patch_capacity_preserves_default_and_solver_settings(self):
        from simulators.robodojo_startup import configure_rigid_patch_count
        physx = SimpleNamespace(gpu_max_rigid_patch_count=163840, solver_type=1,
                                gpu_collision_stack_size=100000000)
        self.assertEqual(configure_rigid_patch_count(physx)["effective_count"], 163840)
        record = configure_rigid_patch_count(physx, 327680)
        self.assertEqual(record, {"original_count": 163840, "requested_count": 327680,
                                  "effective_count": 327680})
        self.assertEqual(vars(physx), {"gpu_max_rigid_patch_count": 327680,
                                     "solver_type": 1, "gpu_collision_stack_size": 100000000})
        for invalid in (True, -1, 1.5, 163840):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                configure_rigid_patch_count(physx, invalid)
        self.assertEqual(physx.gpu_max_rigid_patch_count, 327680)

    def test_patch_capacity_override_is_scoped_to_configured_task(self):
        cfg = load_task("RoboDojo")
        profile = execution.read(self.profile)
        profile["rigid_patch_count_by_task"] = {cfg.simulator.env_task: 327680}
        write_json(self.profile, profile)
        with patch.object(execution, "apptainer_command", side_effect=self.command) as command, \
                patch.object(execution.subprocess, "Popen", side_effect=self.launch), \
                patch.object(execution, "stop_process"):
            self.assertEqual(session_runner.explore(cfg, self.root / "selected", gpu="0"), 0)
            argv = command.call_args.args[0]
            self.assertEqual(argv[argv.index("--rigid-patch-count") + 1], "327680")
            profile["rigid_patch_count_by_task"] = {"different_task": 327680}
            write_json(self.profile, profile)
            self.assertEqual(session_runner.explore(cfg, self.root / "unselected", gpu="0"), 0)
            self.assertNotIn("--rigid-patch-count", command.call_args.args[0])

    def test_capture_survives_exception_and_temp_cleanup(self):
        for scenario in ('native_error','launch_error'):
            with self.subTest(scenario=scenario):
                bridge_path=None
                def command(argv,profile,bridge,sif):
                    nonlocal bridge_path
                    bridge_path=bridge
                    log=bridge/'kit/logs/Kit/Isaac-Sim/5.1/failure.log'
                    log.parent.mkdir(parents=True);log.write_text('exact native failure evidence\n')
                    return self.command(argv,profile,bridge,sif)
                def launch(argv,**kwargs):
                    run=Path(argv[argv.index('--run-dir')+1])
                    if 'server' in argv:
                        write_json(run/'policy-server.json',{'status':'ready'})
                        return Mock(pid=123,poll=Mock(return_value=None))
                    if scenario=='launch_error':raise RuntimeError('simulated launch error')
                    (run/'container-ready').touch()
                    write_json(run/'client-error.json',{'error':'native physics failed'})
                    return Mock(pid=125,returncode=None,poll=Mock(return_value=None),stderr=io.BytesIO())
                task=self.root/scenario
                with patch.object(execution,'apptainer_command',side_effect=command), \
                     patch.object(execution.subprocess,'Popen',side_effect=launch), \
                     patch.object(execution,'stop_process'), self.assertRaises(RuntimeError):
                    session_runner.explore(load_task('RoboDojo'),task,gpu='0')
                self.assertFalse(bridge_path.exists())
                self.assertEqual((task/'playground/native-kit-tail.log').read_text(),'exact native failure evidence\n')
                self.assertFalse((task/'playground/result.json').exists())

    def test_official_factory_waits_for_renderer_before_planner_construction(self):
        events = []
        app, config, state = object(), object(), object()
        def factory(received_config, received_app, **kwargs):
            self.assertEqual(events, [app])
            self.assertIs(received_config, config)
            self.assertIs(received_app, app)
            self.assertEqual(kwargs, {"resume_state": state})
            return "native environment"
        with patch.dict(os.environ, {"EMBODIEDEVO_ROBODOJO_RENDERER_BARRIER": "1"}), \
                patch("simulators.robodojo_startup.wait_for_renderer", side_effect=events.append):
            self.assertEqual(bootstrap.create_native_env(factory, config, app, resume_state=state),
                             "native environment")

    def test_failed_renderer_prevents_native_environment_construction(self):
        factory = Mock()
        with patch.dict(os.environ, {"EMBODIEDEVO_ROBODOJO_RENDERER_BARRIER": "1"}), \
                patch("simulators.robodojo_startup.wait_for_renderer", side_effect=TimeoutError), \
                self.assertRaises(TimeoutError):
            bootstrap.create_native_env(factory, object(), object())
        factory.assert_not_called()

    def setUp(self):
        recipe_switch = patch.dict(os.environ, {"EMBODIEDEVO_USE_RECIPE": "0"})
        recipe_switch.start()
        self.addCleanup(recipe_switch.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sif = self.root / "sim.sif"
        self.sif.touch()
        self.profile = self.root / "runtime.json"
        write_json(self.profile, {"sif": str(self.sif), "material_mirror": str(self.root)})
        self.environment = patch.dict(os.environ, {
            "EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE": str(self.profile), "SLURM_JOB_ID": "123",
            "SLURM_JOB_GPUS": "0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.bad_report = False
        self.seen = []

    def launch(self, argv, **_):
        run = Path(argv[argv.index("--run-dir") + 1])
        if "server" in argv:
            write_json(run / "policy-server.json", {"status": "ready"})
            return Mock(pid=123, poll=Mock(return_value=None))
        spec = execution.read(run / "adapter.json")
        connection = Mock()

        def agent(cfg, episode, *_):
            workspace = episode / "workspace"
            self.seen.append((cfg.baseline, spec["phase"], workspace))
            self.assertIn("# Your exact budgets for this task", (workspace / "instruction.md").read_text())
            if spec["phase"] == "playground":
                (workspace / "skills/learned.py").write_text("learned = True\n")
                (workspace / "scratch/not-inherited.txt").write_text("private exploration")
            elif cfg.baseline:
                self.assertFalse((workspace / "skills").exists())
                self.assertFalse((workspace / "lessons").exists())
                source = (cfg.workspace / "instruction.md").read_text().split("# Your exact budgets")[0]
                self.assertTrue((workspace / "instruction.md").read_text().startswith(source))
            else:
                self.assertTrue((workspace / "skills/learned.py").exists())
                self.assertFalse((workspace / "scratch/not-inherited.txt").exists())
            return "test-agent", ["fake-agent"]

        process = Mock(pid=124, returncode=0, poll=Mock(return_value=0))
        with patch.object(worker, "agent_command", side_effect=agent), \
                patch.object(worker.subprocess, "Popen", return_value=process), \
                patch.object(worker.subprocess, "run"), patch.object(worker.time, "sleep"):
            worker.run_worker(connection, spec, observation(), 2, 0)
        message = connection.send.call_args.args[0]
        self.assertEqual(message["kind"], "finished", message)
        if spec["phase"] == "test":
            write_json(run / "official/eval_result/test/_result.json", {
                "eval_time": 1, "score": 25.0, "details": {"0": {
                    "layout_id": spec["layout_ordinal"] + int(self.bad_report),
                    "success": False, "score": 0.25,
                }},
            })
        return Mock(pid=125, returncode=0, poll=Mock(return_value=0), stderr=io.BytesIO())

    def command(self, argv, profile, bridge, sif):
        return ["apptainer", str(sif), *argv]

    def test_campaign_freeze_test_and_baseline_share_debug_executor(self):
        self.assertIs(debug.episode, execution.episode)
        cfg = load_task("RoboDojo")
        task = self.root / "adapted"
        baseline_cfg = load_task(REPO / "tasks_baseline/RoboDojo")
        baseline = self.root / "baseline"
        with patch.object(execution, "apptainer_command", side_effect=self.command), \
                patch.object(execution.subprocess, "Popen", side_effect=self.launch), \
                patch.object(execution, "stop_process"):
            self.assertEqual(session_runner.explore(cfg, task, gpu="0"), 0)
            frozen = hashes(task / "frozen_harness")
            self.assertEqual(execution.read(task / "protocol.json")["harness"], frozen)
            session_runner.prepare_test(baseline_cfg, baseline, create_plan=False)
            for selected, source, harness in [(cfg, task, task / "frozen_harness"),
                                               (baseline_cfg, baseline, None)]:
                result = session_runner.run_episode(selected, source / "test/scene_0000", source,
                    phase="test", seed=1_000_003, seconds=None, gpu="0", harness=harness)
                self.assertTrue(result["valid"])
                self.assertEqual(result["score"], 25.0)
                self.assertFalse(result["task_completed"])
                spec = execution.read(source / "test/scene_0000/adapter.json")
                self.assertEqual((spec["seed"], spec["layout_ordinal"]), (1, 3))
                self.assertTrue((source / "test/scene_0000/workspace/instruction.md").is_file())
            self.assertEqual(hashes(task / "frozen_harness"), frozen)
            self.assertFalse(list(baseline.rglob("frozen_harness")))
            self.assertEqual([item[:2] for item in self.seen],
                             [(False, "playground"), (False, "test"), (True, "test")])
            self.bad_report = True
            with self.assertRaisesRegex(ValueError, "wrong scene identity"):
                session_runner.run_episode(baseline_cfg, baseline / "test/scene_0001", baseline,
                    phase="test", seed=1_000_004, seconds=None, gpu="0")


    def test_selected_gpu_overrides_multigpu_slurm_visibility(self):
        profile = {
            "image": "sha256:" + "a" * 64,
            "python": "/runtime/bin/python",
            "source_root": "/workspace/RoboDojo",
            "cache_dir": str(self.root / "cache"),
        }
        (self.root / "vendor/robodojo").mkdir(parents=True)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}), \
                patch("orchestrator.robodojo_image.REPO", self.root):
            command = execution.apptainer_command([
                "-m", "integrations.robodojo_policy.bootstrap", "client",
                "--run-dir", str(self.root / "scene"), "--gpu", "3",
            ], profile, self.root / "bridge", self.sif)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)
        self.assertIn(str(self.sif), command)



if __name__ == "__main__":
    unittest.main()
