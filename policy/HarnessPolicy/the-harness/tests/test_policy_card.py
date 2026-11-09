"""scripts/policy_card.py: a policy's scores alone, sorted into the kinds its description names."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import policy_card  # noqa: E402


def write(root, task, seed, reward, success=False, status="finished"):
    directory = Path(root) / task / f"seed-{seed}"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps({"task": task, "seed": seed, "reward": reward,
                                                        "success": success, "status": status}))


class PolicyCardTests(unittest.TestCase):
    def test_tasks_sorted_into_kinds(self):
        with tempfile.TemporaryDirectory() as root:
            write(root, "pour_liquid_into_cup", 0, 1.0, True)
            write(root, "pour_liquid_into_cup", 1, 0.4)
            write(root, "stack_bowls", 0, 0.3)
            write(root, "stack_bowls_random", 1, 0.0)
            write(root, "solve_equation", 0, 0.0)
            write(root, "solve_equation", 1, 0.0, status="error")
            result = policy_card.card([root])
        kinds = {kind: [entry[0] for entry in tasks] for kind, tasks in result["kinds"].items()}
        self.assertEqual(kinds, {"reliable": ["pour_liquid_into_cup"], "partial": ["stack_bowls"],
                                 "unreliable": ["solve_equation"]})
        self.assertEqual(result["kinds"]["partial"][0], ("stack_bowls", "Generalization", 0.15, 2))

    def test_no_episodes_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit):
                policy_card.card([root])


if __name__ == "__main__":
    unittest.main()
