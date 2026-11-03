"""Check that the public policy bundle exposes evaluation-only entry points."""
import ast
from pathlib import Path
import unittest

POLICY = Path(__file__).resolve().parents[1]


class InferenceScopeTests(unittest.TestCase):
    def test_no_dedicated_training_files(self):
        forbidden = {
            'train.py', 'trainer.py', 'dino_teacher.py', 'flow_multisample.py',
            'action_time_weights.py', 'qwen_far_aux.py', 'export_checkpoint.py',
        }
        self.assertFalse(forbidden & {p.name for p in POLICY.rglob('*.py')})
        for path in POLICY.rglob('*.sh'):
            self.assertFalse(path.name.startswith('train'), path)

    def test_policy_and_backend_have_no_loss_entry_points(self):
        for rel in (
            'runtime/models/modeling_xvla.py',
            'runtime/hy_vla/modeling_xvla_adapter.py',
            'runtime/models/action_hub.py',
            'runtime/models/vlm_backends/base.py',
            'runtime/models/vlm_backends/qwen3_5.py',
        ):
            tree = ast.parse((POLICY / rel).read_text())
            names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            self.assertFalse(names & {
                'compute_loss', 'compute_flow_matching_loss', 'compute_lm_loss',
                'compute_lm_losses', 'compute_vlm_loss', 'compute_vlm_fast_ce_loss',
                '_sample_flow_matching_time', '_aggregate_xvla_loss',
            }, rel)
        tree = ast.parse((POLICY / 'runtime/models/modeling_xvla.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'XVLA')
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertIn('generate_actions', methods)
        self.assertNotIn('forward', methods)


if __name__ == '__main__':
    unittest.main()
