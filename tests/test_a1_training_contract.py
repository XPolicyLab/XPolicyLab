"""Exercise the batch-size guard in each trainer without importing GPU dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class A1TrainingContractTests(unittest.TestCase):
    def test_per_rank_batch_size_in_both_training_loops(self):
        path = Path(__file__).resolve().parents[1] / "policy/A1/A1/a1/train.py"
        tree = ast.parse(path.read_bytes())
        for name in ("Trainer", "VLATrainer"):
            with self.subTest(trainer=name):
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
                guards = [n for n in ast.walk(cls) if isinstance(n, ast.Assert)
                          and any(isinstance(v, ast.Name) and v.id == "batch_size"
                                  for v in ast.walk(n.test))]
                self.assertEqual(len(guards), 1)
                guard = compile(ast.Module(guards, type_ignores=[]), str(path), "exec")
                env = dict(self=SimpleNamespace(cfg=SimpleNamespace(global_train_batch_size=8)),
                           get_world_size=lambda: 2, get_local_world_size=lambda: 2)
                exec(guard, dict(env, batch_size=4))
                with self.assertRaises(AssertionError):
                    exec(guard, dict(env, batch_size=3))
