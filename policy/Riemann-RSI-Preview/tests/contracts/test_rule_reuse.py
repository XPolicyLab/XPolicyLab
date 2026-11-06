"""Variant reuse changes public static shape priors, never workflow or scoring."""
import json
from pathlib import Path
import tempfile
import unittest

from rsi.rule_reuse import implementation_mode
from rsi.policy_catalog import catalog
from rsi.task_contract import verify

ROOT = Path(__file__).resolve().parents[2]


class RuleReuseTests(unittest.TestCase):
    def test_random_uses_the_concrete_standard_workflow_with_its_own_official_route(self):
        source='sort_nesting_dolls_by_size';random=source+'_random'
        self.assertEqual(implementation_mode(ROOT,source),source)
        self.assertEqual(implementation_mode(ROOT,random),source)
        self.assertEqual(verify(ROOT,random),verify(ROOT,source))
        routes=catalog()
        self.assertEqual(routes[random]['official_eval_nums'],25)
        self.assertEqual(routes[source]['official_action_horizon'],1050)
        self.assertEqual(routes[random]['official_action_horizon'],1050)

    def test_reuse_cannot_override_workflow_or_inject_runtime_scene_parameters(self):
        mode='sort_nesting_dolls_by_size_random'
        original=json.loads((ROOT/'tasks'/mode/'composition.json').read_text())
        for changed in [{'workflow_overrides':{'steps':[]}},
                        {'parameter_overrides':{'layout_id':7}},
                        {'implementation_mode':'../other'}]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);task=root/'tasks'/mode;task.mkdir(parents=True)
                (task/'composition.json').write_text(json.dumps(dict(original,**changed)))
                with self.assertRaises(ValueError):implementation_mode(root,mode)


if __name__ == '__main__':
    unittest.main()
