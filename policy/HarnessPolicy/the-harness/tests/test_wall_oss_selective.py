"""Offline selective runner and scoreboard contracts; mocks are NOT eval scores."""
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wall_oss_selective as selective
import wall_oss_scoreboard as scoreboard


class SelectiveTests(unittest.TestCase):
    def test_tier1_and_optional_preservation_plan(self):
        dojo = ROOT / "third_party/RoboDojo"
        self.assertEqual(len(selective.task_plan(dojo)), 15)
        self.assertEqual(len(selective.task_plan(dojo, afix=True)), 20)
        self.assertEqual(len(selective.task_plan(dojo, tier2=True)), 17)
        self.assertNotIn("plug_in_charger", selective.task_plan(dojo, afix=True, tier2=True))

    def test_retest_requires_both_gen_halves_and_no_p2(self):
        dojo = ROOT / "third_party/RoboDojo"
        for tasks in ("push_T", "push_T_random", "solve_equation,solve_equation", "insert_key", "fasten_screws", ""):
            if not tasks:
                continue
            with self.assertRaises(ValueError):
                selective.task_plan(dojo, tasks=tasks)
        self.assertEqual(selective.task_plan(dojo, tasks="push_T,push_T_random"), ("push_T", "push_T_random"))
        self.assertEqual(selective.task_plan(dojo, tasks="fasten_screws", afix=True), ("fasten_screws",))

    def test_r2_floor_preserving_recommended_subset(self):
        tasks = ("swap_blocks", "general_pickup", "play_Xylophone", "align_blocks", "push_T", "push_T_random",
                 "pack_objects_into_box", "pack_objects_into_box_random", "solve_equation")
        self.assertEqual(selective.task_plan(ROOT / "third_party/RoboDojo", tasks=",".join(tasks)), tasks)
        for deferred in ("press_by_number", "arrange_largest_number", "arrange_largest_number_random",
                         "classify_objects_by_language"):
            self.assertNotIn(deferred, tasks)

    def test_cli_default_excludes_gpu7(self):
        with mock.patch.object(selective, "preflight") as preflight:
            self.assertEqual(selective.main(["--package", "package", "--robodojo", "dojo", "--out", "out",
                                             "--auth", "auth", "--lock-root", "locks"]), 0)
        self.assertEqual(preflight.call_args.args[0].gpus, "1,2,3,4")

    def test_check_busy_is_read_only_waiting_and_run_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            auth = root / "auth"
            auth.touch()
            args = argparse.Namespace(package=root / "package", robodojo=ROOT / "third_party/RoboDojo",
                out=root / "out", auth=auth, lock_root=root, tasks=None, afix=False, tier2=False,
                gpus="1,2,3,4", require_idle=False, locks_held=False)
            agent = {"lock": {"agent_version": "1.14.0"}, "identity": "fixture", "sources": {"agent": {"skills": {"enable": ["complementary"]}}}}
            before = list(root.rglob("*"))
            with mock.patch.object(selective, "validate_package", return_value=agent) as validate, mock.patch.object(selective.socket, "create_connection"), mock.patch.object(selective, "gpu_problems", return_value=["GPU busy"]), mock.patch.object(selective, "process_problems", return_value=[]), mock.patch("sys.stdout", new=io.StringIO()) as output:
                self.assertEqual(selective.preflight(args)[2], [])
                validate.assert_called_once_with(args.package, expected_version="1.14.0")
                self.assertIn("waiting for GPU", output.getvalue())
                args.require_idle = True
                with self.assertRaises(ValueError):
                    selective.preflight(args)
            self.assertEqual(before, list(root.rglob("*")))

    def test_no_lock_inode_no_create(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(selective.lock_available("7", root))
            self.assertEqual(list(root.iterdir()), [])


class ScoreboardTests(unittest.TestCase):
    def plan(self, root, tasks, version="1.14.0", policy="HarnessPolicyWallCodexSkills55R2", **overrides):
        plan = {"policy": policy, "agent_version": version, "identity": f"fixture-{version}",
                "seed": 0, "eval_num": "native", "tasks": tasks, **overrides}
        (root / "rerun-plan.json").write_text(json.dumps(plan))

    def result(self, root, task, scores, policy="HarnessPolicyWallCodexSkills55", timestamp="20261108"):
        path = root / "eval_result/RoboDojo" / task / policy / "arx_x5" / f"0_ckpt_name={policy},action_type=ee" / timestamp / "_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"details": {str(i): {"score": value, "success": False} for i, value in enumerate(scores)}}))
        return path

    def test_five_dimension_mean_not_flat_42(self):
        dims = scoreboard.dimensions(ROOT / "third_party/RoboDojo")
        rows = {t: {"value": 100 if dim == "Memory" else 0} for dim, tasks in dims.items() for t in tasks}
        self.assertEqual(scoreboard.means(rows, dims, "value")[1], 20)

    def test_gen_pair_and_failed_cases_stay_in_native_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {"push_T": self.result(root, "push_T", [1]*10),
                     "push_T_random": self.result(root, "push_T_random", [0]*25)}
            result = scoreboard.whole_task(paths, "push_T", {"push_T"}, conservative=True)
            self.assertEqual(result["score"], 20)
            self.assertEqual(result["n"], 35)
            self.assertFalse(result["complete"])
            self.assertIsNone(scoreboard.whole_task({"push_T": paths["push_T"]}, "push_T", {"push_T"}))

    def test_latest_timestamp_not_best_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.result(root, "solve_equation", [1]*50, timestamp="001")
            latest = self.result(root, "solve_equation", [0]*50, timestamp="002")
            paths = scoreboard.scan(root, ["solve_equation"], policy="HarnessPolicyWallCodexSkills55")
            self.assertEqual(paths["solve_equation"], latest)
            self.assertEqual(scoreboard.whole_task(paths, "solve_equation", set())["score"], 0)

    def test_invalid_score_or_duplicate_normalized_layout_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "_result.json"
            for details in ({"0": {"score": 2}}, {"0": {"score": float("nan")}},
                            {"0": {"score": 1}, "00": {"score": 0}}):
                path.write_text(json.dumps({"details": details}))
                entries, errors = scoreboard.read_entries(path, 50)
                self.assertEqual(entries, [])
                self.assertTrue(errors)

    def test_rerun_replaces_floor_even_zero_and_no_target_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dims = scoreboard.dimensions(ROOT / "third_party/RoboDojo")
            public = root / "W.md"
            public.write_text("\n".join(f"| {'Gen' if dim == 'Generalization' else 'Open'} / {task} | 80 |" for dim, tasks in dims.items() for task in tasks))
            new = root / "new"
            self.result(new, "solve_equation", [0]*50)
            args = argparse.Namespace(robodojo=ROOT / "third_party/RoboDojo", public=public, machine=root,
                hist_root=[], new_root=[new], tier2=False, line=55)
            for version in ("1.13.0", "1.13.1", "1.14.0"):
                with self.subTest(version=version):
                    self.plan(new, ["solve_equation"], version, policy="HarnessPolicyWallCodexSkills55")
                    _, rows, _, _ = scoreboard.build(args)
                    self.assertEqual(rows["solve_equation"]["floor"], 80)
                    self.assertEqual(rows["solve_equation"]["selected"], 0)
                    self.assertEqual(rows["solve_equation"]["status"], "MEASURED")
                    self.assertIn("NOT CERTIFIED", scoreboard.render(args))
                    self.assertEqual(rows["align_blocks"]["status"], "FLOOR")
            for invalid in ({"version": "1.12.1"}, {"version": "1.14.1"}, {"version": None},
                            {"seed": 1}, {"eval_num": "50"}, {"identity": ""}, {"policy": "HarnessPolicyWall"}):
                with self.subTest(invalid=invalid):
                    self.plan(new, ["solve_equation"], **invalid)
                    with self.assertRaises(ValueError):
                        scoreboard.build(args)

    def test_old_out_and_new_r2_latest_root_wins_even_lower(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dims = scoreboard.dimensions(ROOT / "third_party/RoboDojo")
            public = root / "W.md"
            public.write_text("\n".join(f"| Open / {task} | 80 |" for tasks in dims.values() for task in tasks))
            old, new = root / "old", root / "r2"
            self.result(old, "general_pickup", [1]*50)
            self.plan(old, ["general_pickup"], "1.13.0", policy="HarnessPolicyWallCodexSkills55")
            self.result(new, "general_pickup", [0]*50, policy="HarnessPolicyWallCodexSkills55R2")
            self.plan(new, ["general_pickup"])
            args = argparse.Namespace(robodojo=ROOT / "third_party/RoboDojo", public=public, machine=root,
                hist_root=[], new_root=[old, new], tier2=False, line=55)
            _, rows, _, _ = scoreboard.build(args)
            self.assertEqual(rows["general_pickup"]["selected"], 0)
            self.assertEqual(rows["general_pickup"]["measured_source"], "r2")
            self.assertEqual(rows["general_pickup"]["identity"], "fixture-1.14.0")

    def test_missing_planned_task_cannot_fallback_to_floor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(scoreboard.whole_task({}, "solve_equation", set(), conservative=True)["score"], 0)
            self.assertFalse(scoreboard.whole_task({}, "solve_equation", set(), conservative=True)["complete"])


if __name__ == "__main__":
    unittest.main()
