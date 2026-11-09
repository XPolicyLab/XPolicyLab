"""Offline tests for rerun placement/preflight; no model, simulator, GPU or auth."""

import argparse
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wall_oss_rerun", ROOT / "scripts/wall_oss_rerun.py")
rerun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rerun)


class RerunTests(unittest.TestCase):
    def test_full_skills_plan_contains_all_native_pairs_without_reusing_results(self):
        tasks = rerun.full_tasks(ROOT / "third_party/RoboDojo")
        self.assertEqual(len(tasks), 54)
        self.assertEqual(len(set(tasks)), 54)
        self.assertEqual(sum(task.endswith("_random") for task in tasks), 12)
        self.assertTrue(rerun.PROTECTED_TASKS.issubset(set(tasks)))
        for task in ("push_T", "arrange_largest_number", "pack_objects_into_box"):
            self.assertIn(task, tasks)
            self.assertIn(task + "_random", tasks)

    def test_allowlist_and_exact_plan(self):
        self.assertEqual(rerun.parse_gpus("1,2,3,4,5,7"), ["1", "2", "3", "4", "5", "7"])
        self.assertEqual(rerun.parse_gpus("1,2,3,4,7"), ["1", "2", "3", "4", "7"])
        self.assertEqual(rerun.parse_gpus("7"), ["7"])
        self.assertEqual(rerun.parse_gpus("5"), ["5"])
        for csv in ("0", "6", "1,1", "", "1, 2", "GPU-123"):
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                rerun.parse_gpus(csv)
        self.assertEqual(len(rerun.TASKS), 20)
        self.assertEqual(len(set(rerun.TASKS)), 20)
        self.assertFalse(rerun.PROTECTED_TASKS.intersection(rerun.TASKS))
        for task in ("make_toast", "pack_objects_into_box"):
            self.assertIn(task, rerun.TASKS)
            self.assertIn(task + "_random", rerun.TASKS)
        for task in ("pick_from_conveyor_by_image", "play_tic_tac_toe", "arrange_largest_number_random"):
            self.assertIn(task, rerun.TASKS)

    def test_gpu_queries_are_per_allowed_card_and_refuse_busy_or_unhealthy(self):
        calls = []
        def query(argv, **kwargs):
            calls.append(argv)
            gpu = argv[2]
            self.assertIn(gpu, rerun.ISAAC_GPUS)
            if gpu == "4":
                raise subprocess.CalledProcessError(1, argv)
            if "--query-compute-apps=pid" in argv:
                return "123\n" if gpu == "2" else ""
            return f"{gpu}, GPU-test-{gpu}, {120 if gpu == '7' else 5}\n"
        with mock.patch("sys.stdout", new=io.StringIO()):
            problems = rerun.gpu_problems(rerun.parse_gpus("1,2,3,4,7"), query)
        self.assertEqual(len(problems), 3)
        self.assertIn("GPU2 busy", problems[0])
        self.assertIn("GPU4 health query failed", problems[1])
        self.assertIn("GPU7 busy", problems[2])
        self.assertEqual(len(calls), 9)  # unhealthy query is never retried

    def test_launcher_claims_include_multigpu_and_positional_client(self):
        for argv, expected in ((["bash", "/repo/smoke_all_tasks.sh", "--env-gpu-ids", "1,2,7"], {"1", "2", "7"}),
                               (["bash", "/repo/eval_policy.sh", "--device_id", "4"], {"4"}),
                               (["python", "src/eval_client/main.py", "--device_id", "3"], {"3"}),
                               (["bash", "setup_eval_env_client.sh", "RoboDojo", "task", "ckpt", "arx_x5",
                                 "ee", "0", "2", "env"], {"2"}),
                               (["bash", "-c", "print smoke_all_tasks.sh --env-gpu 1"], set())):
            self.assertEqual(rerun.claimed_gpus(argv), expected)

    def test_check_is_read_only_and_refuses_busy_without_creating_out(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            auth = base / "auth"
            auth.touch()
            args = argparse.Namespace(package=base / "package", robodojo=base / "dojo", out=base / "out",
                                      auth=auth, gpus="1,2,3,4,7")
            before = list(base.rglob("*"))
            with mock.patch.object(rerun, "validate_package", return_value={"lock": {"agent_version": "1.11.0"},
                                                                          "identity": "test"}), \
                 mock.patch.object(rerun.socket, "create_connection"), \
                 mock.patch.object(rerun, "gpu_problems", return_value=["GPU1 busy"]), \
                 mock.patch.object(rerun, "process_problems", return_value=[]), \
                 mock.patch("sys.stdout", new=io.StringIO()):
                with self.assertRaisesRegex(ValueError, "GPU1 busy"):
                    rerun.check(args)
                with mock.patch.object(rerun, "gpu_problems", return_value=[]):
                    rerun.check(args)
            self.assertEqual(list(base.rglob("*")), before)
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
