import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

SPEC = importlib.util.spec_from_file_location("robodojo_policy_eval", ROOT / "scripts" / "robodojo_policy_eval.py")
policy_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy_eval)

from harness_policy.runtime.report import load_episodes  # noqa: E402


def write_run(results, task, policy, run, details, label="baseline", videos=True):
    """A RoboDojo eval_result run folder as src/eval_client/eval_env.py writes it."""
    folder = policy_eval.run_folder(results, task, policy, "arx_x5", 0, label, "joint") / run
    folder.mkdir(parents=True)
    episodes = {str(index): {"layout_id": layout, "success": success, "score": score}
                for index, (layout, success, score) in enumerate(details)}
    (folder / "_result.json").write_text(json.dumps({"success_rate": 0.0, "eval_time": len(details),
                                                    "score": 0.0, "details": episodes}))
    for index, (_layout, success, _score) in enumerate(details):
        for camera in policy_eval.CAMERAS:
            if videos:
                tag = "success" if success else "fail"
                (folder / f"episode_{index:07d}_{camera}_{tag}.mp4").write_bytes(b"mp4")
    return folder


class ConvertTests(unittest.TestCase):
    def test_newest_run_per_task_becomes_harness_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, out = Path(tmp) / "eval_result", Path(tmp) / "out"
            write_run(results, "stack_bowls", "G05", "2026-10-06_00-16-32", [(0, True, 1.0)])
            write_run(results, "stack_bowls", "G05", "2026-10-06_05-42-23", [(0, False, 0.25), (2, True, 1.0)])
            write_run(results, "push_T", "G05", "2026-10-06_06-10-00", [(1, False, 0.0)])
            count = policy_eval.convert(results, "G05", "arx_x5", 0, "baseline", "joint", "2026-10-06_05-00-00",
                                        out, "G0.5")
            self.assertEqual(count, 3)
            # The older stack_bowls run (before --since) is not used; layouts become seeds.
            names = sorted(str(p.relative_to(out)) for p in out.glob("*/seed-*"))
            self.assertEqual(names, ["push_T/seed-1", "stack_bowls/seed-0", "stack_bowls/seed-2"])
            first = json.loads((out / "stack_bowls" / "seed-0" / "summary.json").read_text())
            self.assertEqual((first["success"], first["process_score"], first["model_calls"]), (False, 25.0, 0))
            self.assertEqual(sorted(p.name for p in (out / "stack_bowls" / "seed-2" / "videos").iterdir()),
                             [f"episode-{camera}.mp4" for camera in sorted(policy_eval.CAMERAS)])
            manifest = json.loads((out / "stack_bowls" / "seed-2" / "manifest.json").read_text())
            self.assertEqual(manifest["config"]["runtime"]["label"], "G0.5 (RoboDojo eval client)")
            events = [json.loads(line) for line in (out / "push_T" / "seed-1" / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["observation"]["state"]["layout_source"], "eval")
            # The harness report reads them like any episode.
            self.assertEqual(len(load_episodes([out])), 3)

    def test_other_policies_labels_and_missing_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, out = Path(tmp) / "eval_result", Path(tmp) / "out"
            write_run(results, "general_pickup", "Pi_05", "2026-10-07_10-00-00", [(0, True, 1.0)], label="pi05-run",
                      videos=False)
            write_run(results, "general_pickup", "G05", "2026-10-07_10-00-00", [(1, True, 1.0)])
            count = policy_eval.convert(results, "Pi_05", "arx_x5", 0, "pi05-run", "joint", "", out, "π0.5")
            self.assertEqual(count, 1)
            episode = out / "general_pickup" / "seed-0"
            self.assertTrue((episode / "summary.json").exists())
            self.assertEqual(list((episode / "videos").iterdir()), [])
            self.assertFalse((out / "general_pickup" / "seed-1").exists())  # another policy's run

    def test_a_rerun_replaces_the_older_runs_episodes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, out = Path(tmp) / "eval_result", Path(tmp) / "out"
            write_run(results, "stack_bowls", "G05", "2026-10-06_05-00-00", [(0, True, 1.0), (1, True, 1.0), (2, True, 1.0)])
            policy_eval.convert(results, "G05", "arx_x5", 0, "baseline", "joint", "", out, "G0.5")
            harness = out / "stack_bowls" / "seed-9"
            harness.mkdir()
            (harness / "summary.json").write_text(json.dumps({"task": "stack_bowls", "seed": 9}))  # not ours
            write_run(results, "stack_bowls", "G05", "2026-10-06_06-00-00", [(0, False, 0.0), (3, True, 1.0)])
            policy_eval.convert(results, "G05", "arx_x5", 0, "baseline", "joint", "", out, "G0.5")
            self.assertEqual(sorted(p.name for p in (out / "stack_bowls").iterdir()), ["seed-0", "seed-3", "seed-9"])
            summary = json.loads((out / "stack_bowls" / "seed-0" / "summary.json").read_text())
            self.assertEqual((summary["success"], summary["status"]), (False, "episode_ended"))

    def test_other_evaluations_in_the_output_are_never_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, out = Path(tmp) / "eval_result", Path(tmp) / "out"
            write_run(results, "push_T", "Pi_05", "2026-10-06_05-00-00", [(0, True, 1.0), (4, True, 1.0)], label="pi")
            policy_eval.convert(results, "Pi_05", "arx_x5", 0, "pi", "joint", "", out, "π0.5")
            harness = out / "push_T" / "seed-1"
            harness.mkdir()
            (harness / "summary.json").write_text(json.dumps({"task": "push_T", "seed": 1, "status": "finished"}))
            write_run(results, "push_T", "G05", "2026-10-06_06-00-00", [(0, False, 0.0), (1, False, 0.0), (2, True, 1.0)])
            count = policy_eval.convert(results, "G05", "arx_x5", 0, "baseline", "joint", "", out, "G0.5")
            # Only layout 2 is free: π0.5's seed-0 and seed-4 and the harness's seed-1 are kept as they were.
            self.assertEqual(count, 1)
            self.assertEqual(sorted(p.name for p in (out / "push_T").iterdir()), ["seed-0", "seed-1", "seed-2", "seed-4"])
            self.assertTrue(json.loads((out / "push_T" / "seed-0" / "summary.json").read_text())["success"])
            self.assertEqual(json.loads((harness / "summary.json").read_text())["status"], "finished")

    def test_tasks_without_results_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "eval_result"
            folder = policy_eval.run_folder(results, "insert_key", "G05", "arx_x5", 0, "baseline", "joint")
            (folder / "2026-10-06_07-00-00").mkdir(parents=True)  # stopped before its first batch
            self.assertEqual(policy_eval.convert(results, "G05", "arx_x5", 0, "baseline", "joint", "",
                                                 Path(tmp) / "out", "G0.5"), 0)


class RunTests(unittest.TestCase):
    def test_client_command_is_robodojos_official_client(self):
        args = argparse.Namespace(policy="G05", host="127.0.0.1", port=19005, eval_num="2", gpu=3,
                                  env_cfg="arx_x5", seed=0, label="baseline", action_type="joint")
        command = policy_eval.client_command(Path("/rd"), "stack_bowls", args)
        self.assertEqual(command[:3], ["bash", "/rd/scripts/robodojo.sh", "client"])
        options = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(options, {"--task": "stack_bowls", "--policy-name": "G05", "--policy-host": "127.0.0.1",
                                   "--policy-port": "19005", "--eval-num": "2", "--env-gpu": "3",
                                   "--env-cfg": "arx_x5", "--seed": "0", "--ckpt": "baseline",
                                   "--action-type": "joint"})

    def test_task_lists(self):
        self.assertEqual(len(policy_eval.expand_tasks("all")), 42)
        self.assertEqual(policy_eval.expand_tasks("Memory,push_T,push_T")[-1], "push_T")
        both = policy_eval.expand_tasks("all,random")
        self.assertEqual(len(both), 54)
        self.assertIn("stack_bowls_random", both)


if __name__ == "__main__":
    unittest.main()
