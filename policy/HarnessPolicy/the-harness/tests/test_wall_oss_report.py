"""Fake results only: seed 0, whole-task fallback, no max or simulation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benches.robodojo.wall_oss_report import (
    RunResult, read_harness_summary, read_official, read_result, select_seed0, _rules)


def run(count, score=0.2, *, success=False, errors=()):
    return RunResult(tuple((success, score) for _ in range(count)), tuple(errors))


def records(**tasks):
    return {task: {0: result} for task, result in tasks.items()}


def fake_file(root, task="press_by_number", policy="WallX", seed=0, count=50,
              score=0.2, checkpoint="fixed", mode="ee", timestamp="2026-11-01_00-00-00"):
    path = Path(root) / task / policy / "arx_x5" / f"{seed}_ckpt_name={checkpoint},action_type={mode}" / timestamp / "_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reverse insertion order checks upstream layout sorting, not dictionary order.
    path.write_text(json.dumps({"details": {str(i): {"success": False, "score": score}
                                           for i in reversed(range(count))}}))
    return path


class SelectionTests(unittest.TestCase):
    def test_complete_harness_is_kept_even_when_base_score_is_higher(self):
        harness = records(press_by_number=run(50, 0.1))
        base = records(press_by_number=run(50, 0.9, success=True))
        row = select_seed0(harness, base, tasks=["press_by_number"])["tasks"][0]
        self.assertEqual((row["source"], row["episodes"], row["success_rate"]),
                         ("harness+wall", 50, 0.))
        self.assertAlmostEqual(row["score"], 10.)
        self.assertEqual(row["fallback_reasons"], [])

    def test_incomplete_high_scoring_harness_falls_back_to_lower_base(self):
        row = select_seed0(records(press_by_number=run(49, 1.0, success=True)),
                           records(press_by_number=run(50, 0.1)), tasks=["press_by_number"])["tasks"][0]
        self.assertEqual(row["source"], "WallX")
        self.assertAlmostEqual(row["score"], 10.)
        self.assertIn("49/50", row["fallback_reasons"][0])

    def test_run_error_falls_back_even_with_all_episodes_present(self):
        row = select_seed0(records(solve_equation=run(50, 1., errors=["worker exited 2"])),
                           records(solve_equation=run(50, 0.2)), tasks=["solve_equation"])["tasks"][0]
        self.assertEqual(row["source"], "WallX")
        self.assertAlmostEqual(row["score"], 20.)

    def test_failed_episodes_stay_in_denominator_not_run_errors(self):
        mixed = RunResult(((True, 1.),) * 10 + ((False, 0.),) * 40)
        report = select_seed0(records(general_pickup=mixed), {}, tasks=["general_pickup"])
        self.assertEqual((report["tasks"][0]["source"], report["success_rate"], report["score"]),
                         ("harness+wall", 20., 20.))

    def test_generalization_requires_25_plus_25_and_uses_first_layouts(self):
        harness = records(arrange_largest_number=RunResult(((True, 0.2),) * 25 + ((False, 1.),) * 25),
                          arrange_largest_number_random=run(25, 0.4))
        row = select_seed0(harness, {}, tasks=["arrange_largest_number"])["tasks"][0]
        self.assertEqual((row["episodes"], row["success_rate"]), (50, 50.))
        self.assertAlmostEqual(row["score"], 30.)

    def test_missing_random_falls_back_whole_task_without_cross_source_splicing(self):
        harness = records(stack_bowls=run(50, 1.0))
        base = records(stack_bowls=run(25, 0.1), stack_bowls_random=run(25, 0.3))
        row = select_seed0(harness, base, tasks=["stack_bowls"])["tasks"][0]
        self.assertEqual(row["source"], "WallX")
        self.assertAlmostEqual(row["score"], 20.)

    def test_one_short_or_error_generalization_half_replaces_both_halves(self):
        base = records(stack_bowls=run(25, 0.1), stack_bowls_random=run(25, 0.1))
        for random in (run(24, 1.), run(25, 1., errors=["error"])):
            harness = records(stack_bowls=run(25, 1.), stack_bowls_random=random)
            row = select_seed0(harness, base, tasks=["stack_bowls"])["tasks"][0]
            self.assertEqual(row["source"], "WallX")
            self.assertAlmostEqual(row["score"], 10.)

    def test_only_seed_zero_is_considered(self):
        harness = {"press_by_number": {1: run(50, 1.), 2: run(50, 1.)}}
        base = {"press_by_number": {0: run(50, 0.3), 1: run(50, 0.9)}}
        report = select_seed0(harness, base, tasks=["press_by_number"])
        self.assertEqual(report["seed"], 0)
        self.assertEqual(report["tasks"][0]["seed"], 0)
        self.assertAlmostEqual(report["score"], 30.)
        with self.assertRaisesRegex(ValueError, "fallback unavailable"):
            select_seed0({}, {"press_by_number": {1: run(50)}}, tasks=["press_by_number"])

    def test_overall_is_the_mean_of_dimension_means(self):
        # Two Memory tasks at 0 and one Open task at 100. A flat task mean is
        # 33.3; the official overall is (0 + 100) / 2 = 50.
        report = select_seed0(records(
            cover_blocks=run(50, 0.0), swap_blocks=run(50, 0.0),
            align_blocks=run(50, 1.0),
        ), {}, tasks=["cover_blocks", "swap_blocks", "align_blocks"])
        self.assertAlmostEqual(report["dimensions"]["Memory"]["score"], 0.)
        self.assertAlmostEqual(report["dimensions"]["Open"]["score"], 100.)
        self.assertAlmostEqual(report["score"], 50.)

    def test_missing_or_error_base_is_explicit_error_not_denominator_shrink(self):
        for base in ({}, records(press_by_number=run(49)),
                     records(press_by_number=run(50, errors=["base error"]))):
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, "fallback unavailable"):
                select_seed0({}, base, tasks=["press_by_number"])

    def test_full_report_keeps_all_42_tasks_in_denominator(self):
        base = {}
        for task in _rules.ALL_TASKS:
            base[task] = {0: run(50)}
            if task in _rules.GENERALIZATION_TASKS:
                base[task + "_random"] = {0: run(25)}
        report = select_seed0({}, base)
        self.assertEqual(report["task_count"], 42)
        self.assertTrue(all(row["episodes"] == 50 and row["seed"] == 0 for row in report["tasks"]))


class FileInputTests(unittest.TestCase):
    def test_official_uses_only_one_checkpoint_ee_seed_zero_latest_not_best(self):
        with tempfile.TemporaryDirectory() as root:
            fake_file(root, score=1., timestamp="2026-10-31_00-00-00")
            latest = fake_file(root, score=0.1)
            fake_file(root, score=1., checkpoint="other")
            fake_file(root, score=1., mode="joint")
            fake_file(root, score=1., seed=1)
            selected = read_official(root, "fixed")
        self.assertEqual(selected["press_by_number"][0].path, str(latest))
        self.assertEqual(set(selected["press_by_number"]), {0})
        self.assertEqual(selected["press_by_number"][0].entries, ((False, 0.1),) * 50)

    def test_runner_error_with_full_results_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            path = fake_file(root, policy="HarnessPolicyWall", score=1.)
            summary = Path(root) / "summary.json"
            summary.write_text(json.dumps({"results": [{"task": "press_by_number", "status": "FAIL",
                                                        "exit_code": "2", "result_path": str(path)}]}))
            runs = read_harness_summary(summary)
        self.assertEqual(len(runs["press_by_number"][0].entries), 50)
        self.assertTrue(runs["press_by_number"][0].errors)

    def test_old_joint_results_cannot_count_as_corrected_ee_ablation(self):
        with tempfile.TemporaryDirectory() as root:
            path = fake_file(root, policy="HarnessPolicyWall", mode="joint", score=1.)
            summary = Path(root) / "summary.json"
            summary.write_text(json.dumps({"results": [{"task": "press_by_number", "status": "PASS",
                                                        "exit_code": "0", "result_path": str(path)}]}))
            runs = read_harness_summary(summary)
        self.assertTrue(runs["press_by_number"][0].errors)
        self.assertEqual(runs["press_by_number"][0].entries, ())

    def test_bad_json_invalid_records_and_run_errors_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            path = fake_file(root)
            for data in ("{", '{"details": {"0": {"success": false, "score": null}}}',
                         '{"error": "RPC failed", "details": {}}'):
                path.write_text(data)
                result = read_result(path)
                self.assertTrue(result.errors)
            path.write_text(json.dumps({"details": {"2": {"success": True, "score": 1.},
                                                     "0": {"success": False, "score": 0.}}}))
            result = read_result(path)
        self.assertEqual(result.entries, ((False, 0.), (True, 1.)))
        self.assertEqual(result.errors, ())

    def test_seed_one_summary_row_does_not_replace_seed_zero_even_if_joint(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [fake_file(root, policy="HarnessPolicyWall"),
                     fake_file(root, policy="HarnessPolicyWall", seed=1, mode="joint")]
            summary = Path(root) / "summary.json"
            summary.write_text(json.dumps({"results": [
                {"task": "press_by_number", "status": "PASS", "exit_code": "0", "result_path": str(path)}
                for path in paths]}))
            result = read_harness_summary(summary)["press_by_number"][0]
        self.assertEqual(result.path, str(paths[0]))
        self.assertFalse(result.errors)

    def test_failed_runner_with_no_result_path_remains_a_fallback_reason(self):
        with tempfile.TemporaryDirectory() as root:
            summary = Path(root) / "summary.json"
            summary.write_text(json.dumps({"results": [{"task": "press_by_number", "status": "FAIL",
                                                        "exit_code": "1", "result_path": None}]}))
            runs = read_harness_summary(summary)
        row = select_seed0(runs, records(press_by_number=run(50)), tasks=["press_by_number"])["tasks"][0]
        self.assertEqual(row["source"], "WallX")
        self.assertTrue(any("runner FAIL" in reason for reason in row["fallback_reasons"]))


if __name__ == "__main__":
    unittest.main()
