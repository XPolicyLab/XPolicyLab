import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.report import DIMENSIONS, aggregate, load_episodes, markdown


def episode(task, reward, success, status="finished", calls=4):
    return {"task": task, "seed": 0, "reward": reward, "success": success, "status": status,
            "model_calls": calls, "cost_usd": 0.0}


class ReportTests(unittest.TestCase):
    def test_scores_pool_random_variants_and_count_errors(self):
        result = aggregate([
            episode("stack_bowls", 1.0, True),
            episode("stack_bowls_random", 0.5, False),
            episode("stack_bowls", None, False, status="error"),
            episode("push_T", 0.2, False),
        ])
        bowls = result["tasks"]["stack_bowls"]
        self.assertEqual(bowls["episodes"], 3)
        self.assertAlmostEqual(bowls["score"], 50.0)
        self.assertAlmostEqual(bowls["sr"], 100 / 3)
        self.assertEqual(bowls["errors"], 1)
        generalization = result["dimensions"]["Generalization"]
        self.assertEqual((generalization["tasks"], generalization["of"]), (2, 12))
        self.assertAlmostEqual(generalization["score"], 35.0)
        self.assertIsNone(result["overall"])
        self.assertIn("| Generalization | 2/12 | 35.00 |", markdown(result))

    def test_overall_needs_every_dimension(self):
        episodes = [episode(task, 0.5, False) for tasks in DIMENSIONS.values() for task in tasks]
        result = aggregate(episodes)
        self.assertEqual(sum(len(tasks) for tasks in DIMENSIONS.values()), 42)
        self.assertAlmostEqual(result["overall"]["score"], 50.0)
        self.assertEqual(result["dimensions"]["Moving targets"]["tasks"], 2)

    def test_episodes_the_simulator_dropped_are_not_scored(self):
        unstable = {**episode("push_T", None, False, status="error"),
                    "error": "Unscorable: RoboDojo marked task=push_T layout=1 unstable after step 9; not scorable"}
        crash = {**episode("push_T", None, False, status="error"), "error": "RuntimeError: worker died"}
        result = aggregate([episode("push_T", 1.0, True), unstable, crash])
        push = result["tasks"]["push_T"]
        self.assertEqual((push["episodes"], push["dropped"], push["errors"]), (2, 1, 1))
        self.assertAlmostEqual(push["sr"], 50.0)
        only_dropped = aggregate([{**unstable, "task": "swap_T"}])
        self.assertIsNone(only_dropped["tasks"]["swap_T"]["score"])
        self.assertEqual(only_dropped["dimensions"]["Memory"]["tasks"], 0)
        self.assertIn("| push_T | 2 | 50.00 | 50.00 |", markdown(result))
        self.assertIn("| 1 | 1 |", markdown(result))  # Errors, Dropped

    def test_superseded_runs_are_not_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("seed-0.unstable-1", "seed-1", "seed-1.error-2", "seed-2"):
                directory = Path(tmp) / "t" / name
                directory.mkdir(parents=True)
                (directory / "summary.json").write_text(json.dumps(episode("push_T", 0.0, False)))
            self.assertEqual(len(load_episodes([tmp])), 2)


if __name__ == "__main__":
    unittest.main()
