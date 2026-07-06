import importlib
import os
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("websockets", types.SimpleNamespace(connect=None))
sys.modules.setdefault(
    "msgpack_numpy",
    types.SimpleNamespace(encode=lambda obj: obj, decode=lambda obj: obj),
)


class EvalEnvTypeRegressionTests(unittest.TestCase):
    def test_env_client_server_exports_error_normalizer(self):
        from station.daemon import normalize_execution_error

        self.assertTrue(callable(normalize_execution_error))

    def test_default_runner_uses_sim_runner_for_sim(self):
        runner = importlib.import_module("station.env_client.runner")

        selected = runner._default_env_trial_runner("sim")

        self.assertIs(selected, runner.run_sim_trial)

    def test_default_runner_rejects_unknown_eval_env_type(self):
        runner = importlib.import_module("station.env_client.runner")

        with self.assertRaises(runner.TrialRunnerError) as ctx:
            runner._default_env_trial_runner("bogus")

        self.assertIn("bogus", str(ctx.exception))

    def test_default_runner_uses_real_runner_for_real_world(self):
        runner = importlib.import_module("station.env_client.runner")

        selected = runner._default_env_trial_runner("real_world")

        self.assertIs(selected, runner.run_real_trial)

    def test_run_sim_trial_fails_on_missing_deploy_fields(self):
        runner = importlib.import_module("station.env_client.runner")

        result = runner.run_sim_trial({"eval_batch": False})

        self.assertEqual(result["status"], "failed")
        self.assertIn(result["error"]["code"], ("missing_eval_policy_script", "missing_sim_deploy_cfg"))

    @staticmethod
    def _run_sim_trial_with_fakes(runner, sim_result, deploy_cfg=None, captured=None):
        if captured is None:
            captured = {}

        class FakeProc:
            def __init__(self, cmd, cwd=None, **kwargs):
                captured["cmd"] = cmd
                captured["cwd"] = cwd
                captured["kwargs"] = kwargs
                self.pid = 12345

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

        original_popen = runner.subprocess.Popen
        original_isfile = runner.os.path.isfile
        original_load = runner._load_sim_result
        runner.subprocess.Popen = FakeProc
        runner.os.path.isfile = lambda _p: True
        runner._load_sim_result = lambda _root, _run_id: sim_result
        try:
            return runner.run_sim_trial(
                deploy_cfg
                or {
                    "task_name": "stack_bowls",
                    "env_cfg_type": "arx_x5",
                    "policy_name": "X_VLA",
                    "host": "localhost",
                    "port": 6000,
                    "bench_name": "RoboDojo",
                    "eval_batch": False,
                    "root_dir": "/tmp/robodojo-root",
                }
            )
        finally:
            runner.subprocess.Popen = original_popen
            runner.os.path.isfile = original_isfile
            runner._load_sim_result = original_load

    def test_run_sim_trial_shells_out_to_eval_policy(self):
        runner = importlib.import_module("station.env_client.runner")

        captured = {}
        result = self._run_sim_trial_with_fakes(
            runner, {"eval_time": 3, "success_rate": 0.5}, captured=captured
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["eval_env_type"], "sim")
        self.assertEqual(result["sim_result"]["eval_time"], 3)
        cmd = captured["cmd"]
        self.assertEqual(cmd[0], "bash")
        self.assertIn("--task_name", cmd)
        self.assertIn("stack_bowls", cmd)
        self.assertIn("--bench_name", cmd)
        self.assertIn("RoboDojo", cmd)
        # Whole process group must be killable and _result.json discoverable.
        self.assertTrue(captured["kwargs"].get("start_new_session"))
        self.assertIn("ROBODOJO_RUN_ID", captured["kwargs"].get("env", {}))

    def test_run_sim_trial_requires_result_json(self):
        """Exit code 0 without a valid _result.json must not count as success."""
        runner = importlib.import_module("station.env_client.runner")

        result = self._run_sim_trial_with_fakes(runner, None)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "missing_sim_result")

    def test_run_sim_trial_wraps_command_in_conda_env(self):
        runner = importlib.import_module("station.env_client.runner")

        captured = {}
        os.environ["XPOLICYLAB_SIM_CONDA_ENV"] = "SimBench"
        try:
            self._run_sim_trial_with_fakes(
                runner, {"eval_time": 1}, captured=captured
            )
        finally:
            os.environ.pop("XPOLICYLAB_SIM_CONDA_ENV", None)

        cmd = captured["cmd"]
        self.assertEqual(cmd[:2], ["bash", "-c"])
        self.assertIn("conda activate SimBench", cmd[2])
        self.assertIn("eval_policy.sh", cmd[2])

    def test_task_level_bench_name_reaches_deploy_cfg(self):
        from station.env_client.api import (
            EnvClientBaselineConfig,
            dispatch_trial_to_deploy_cfg,
        )
        from station.schemas import DispatchPayload

        dispatch = DispatchPayload.model_validate(
            {
                "task_id": "stack_bowls",
                "model_name": "X_VLA",
                "policy_server_url": "ws://localhost:6000",
                "evaluation_plan": {
                    "task": {
                        "id": "stack_bowls",
                        "name": "stack_bowls",
                        "env_cfg_type": "arx_x5",
                        "bench_name": "RoboDojo",
                    },
                    "trials": [{"action_case_id": "case-1", "trial_index": 1}],
                },
            }
        )
        trial = dispatch.evaluation_plan.trials[0]
        trial_run = {
            "trial_id": "trial-1",
            "action_case_id": trial.action_case_id,
            "trial_index": trial.trial_index,
            "case_meta": trial.model_dump(exclude_none=True),
        }

        deploy_cfg = dispatch_trial_to_deploy_cfg(
            dispatch,
            trial_run,
            EnvClientBaselineConfig(),
            evaluation_id="eval-1",
        )

        self.assertEqual(deploy_cfg["bench_name"], "RoboDojo")

    def test_legacy_dataset_name_maps_to_bench_name(self):
        """Older control planes still dispatch the pre-rename dataset_name key."""
        from station.env_client.api import (
            EnvClientBaselineConfig,
            dispatch_trial_to_deploy_cfg,
        )
        from station.schemas import DispatchPayload

        dispatch = DispatchPayload.model_validate(
            {
                "task_id": "stack_bowls",
                "model_name": "X_VLA",
                "policy_server_url": "ws://localhost:6000",
                "dataset_name": "RoboDojo",
                "evaluation_plan": {
                    "task": {
                        "id": "stack_bowls",
                        "name": "stack_bowls",
                        "env_cfg_type": "arx_x5",
                    },
                    "trials": [{"action_case_id": "case-1", "trial_index": 1}],
                },
            }
        )
        trial = dispatch.evaluation_plan.trials[0]
        trial_run = {
            "trial_id": "trial-1",
            "action_case_id": trial.action_case_id,
            "trial_index": trial.trial_index,
            "case_meta": trial.model_dump(exclude_none=True),
        }

        deploy_cfg = dispatch_trial_to_deploy_cfg(
            dispatch,
            trial_run,
            EnvClientBaselineConfig(),
            evaluation_id="eval-1",
        )

        self.assertEqual(deploy_cfg["bench_name"], "RoboDojo")

    def test_legacy_dataset_name_in_case_meta_maps_to_bench_name(self):
        from station.env_client.api import (
            EnvClientBaselineConfig,
            TrialRunRequest,
            trial_request_to_deploy_cfg,
        )

        request = TrialRunRequest(
            evaluation_id="eval-1",
            trial_id="trial-1",
            action_case_id="case-1",
            policy_server_url="ws://localhost:6000",
            case_meta={"dataset_name": "RoboDojo", "task_name": "stack_bowls"},
        )

        deploy_cfg = trial_request_to_deploy_cfg(request, EnvClientBaselineConfig())

        self.assertEqual(deploy_cfg["bench_name"], "RoboDojo")

    def test_reset_deploy_cfg_includes_repeat_index(self):
        """TestEnv/RealEnv read deploy_cfg["repeat_index"] unconditionally."""
        from station.env_client.api import EnvClientBaselineConfig
        from station.env_client.runner import baseline_to_reset_deploy_cfg

        deploy_cfg = baseline_to_reset_deploy_cfg(
            EnvClientBaselineConfig(eval_env_type="debug", host="localhost", port=6000)
        )

        self.assertIn("repeat_index", deploy_cfg)
        self.assertIn("policy_server_url", deploy_cfg)

    def test_trial_loop_dispatches_batch_eval(self):
        runner = importlib.import_module("station.env_client.runner")
        calls = []

        class FakeEnv:
            episode_step = 3

            def reset(self):
                pass

            def eval_one_episode(self):
                calls.append("single")

            def eval_one_episode_batch(self):
                calls.append("batch")

            def finish_episode(self):
                pass

        runner._run_trial_loop(
            FakeEnv(), stop_check=lambda: False, eval_batch=True, max_episodes=1
        )
        self.assertEqual(calls, ["batch"])

        calls.clear()
        runner._run_trial_loop(
            FakeEnv(), stop_check=lambda: False, eval_batch=False, max_episodes=1
        )
        self.assertEqual(calls, ["single"])

    def test_failed_runner_construction_releases_trial_registration(self):
        """A failed runner construction must not wedge the daemon after /start."""
        from station.daemon import state as daemon_state
        from station.daemon.state import EnvClientServerConfig, EnvClientServerState
        from station.env_client.api import EnvClientBaselineConfig
        from station.env_client.runner import TrialRunnerError

        state = EnvClientServerState(
            baseline=EnvClientBaselineConfig(eval_env_type="sim"),
            config=EnvClientServerConfig(
                artifact_root=Path("/tmp/robodojo-test-artifacts")
            ),
        )

        def _boom(*_args, **_kwargs):
            raise TrialRunnerError("runner construction failed")

        original = daemon_state.make_dispatch_trial_runner
        daemon_state.make_dispatch_trial_runner = _boom
        try:
            with self.assertRaises(TrialRunnerError):
                state.trial_runner_with_stop("eval-1", 1)
        finally:
            daemon_state.make_dispatch_trial_runner = original

        self.assertFalse(state.trial_control.has_active_trials())


if __name__ == "__main__":
    unittest.main()
