"""Exercise current workflow references without importing any task runtime."""
import json
from pathlib import Path
import tempfile
import unittest

from rsi.workflow_validation import Exports, validate

ROOT = Path(__file__).resolve().parents[2]


class WorkflowDeclarationsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.task = self.root / 'tasks/sample'
        self.task.mkdir(parents=True)
        self.module = self.task / 'skills/action/implementation.py'
        self.module.parent.mkdir(parents=True)
        self.module.write_text('raise RuntimeError("must never import task runtime")\n'
                               'def run(context): pass\nclass Failure(Exception): pass\n')
        self.composition = {'native_workflow': 'workflow.json', 'graph_files': []}
        self.graph = {'schema': 'riemann-declarative-skill-workflow-v1',
                      'mode': 'rule_only', 'official_action_horizon': 10,
                      'wam_reselection': False, 'max_operations': 10,
                      'bindings': {'act': 'skills.action.implementation:run'},
                      'errors': {'failure': 'skills.action.implementation:Failure'},
                      'steps': [{'call': 'act'}]}

    def check(self):
        (self.task / 'workflow.json').write_text(json.dumps(self.graph))
        validate(self.root, 'sample', self.composition)

    def test_all_current_modes_are_canonical_and_reachable_without_models(self):
        from tools.check_source_contracts import check
        result = check(ROOT)
        self.assertEqual(result['verified_modes'], 54)
        self.assertFalse(result['sdk_or_model_loaded'])

    def test_validation_does_not_execute_task_import_side_effects(self):
        self.check()

    def test_unknown_schema_is_rejected(self):
        self.graph['schema'] = 'unknown-experimental-graph'
        with self.assertRaisesRegex(ValueError, 'Unknown executable workflow schema'):
            self.check()

    def test_missing_export_is_rejected(self):
        self.graph['bindings']['act'] = 'skills.action.implementation:missing'
        with self.assertRaisesRegex(ValueError, 'Missing exported workflow callable'):
            self.check()

    def test_missing_module_is_rejected(self):
        self.graph['bindings']['act'] = 'skills.missing.implementation:run'
        with self.assertRaisesRegex(ValueError, 'Missing declared workflow module'):
            self.check()

    def test_undeclared_graph_call_is_rejected(self):
        self.graph['steps'][0]['call'] = 'missing'
        with self.assertRaises(ValueError):
            self.check()

    def test_nested_program_is_validated(self):
        inner = self.graph
        inner['steps'][0]['call'] = 'missing'
        self.graph = {'schema': 'riemann-mature-teacher-programs-v1',
                      'package': 'tasks.sample', 'programs': {'run': inner}}
        with self.assertRaises(ValueError):
            self.check()

    def test_dangling_state_is_rejected(self):
        self.graph = {'schema': 'riemann-language-rear-table-episode-graph-v8',
                      'entry': 'begin', 'states': {'begin': {
                          'call': 'skills.action.implementation:run', 'next': {'ok': 'missing'}}}}
        with self.assertRaisesRegex(ValueError, 'undeclared continuation'):
            self.check()

    def test_nested_public_command_requires_declared_skill(self):
        self.graph = {'schema': 'riemann-public-command-workflow-v1', 'stages': [],
                      'execution_program': {'stages': [{'skill': 'missing'}]}}
        with self.assertRaisesRegex(ValueError, 'undeclared action skill'):
            self.check()

    def test_cross_task_and_escaping_graph_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'another task'):
            Exports(self.root, 'sample').check('tasks.other.skills.action.implementation:run')
        self.composition['native_workflow'] = '../outside.json'
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate(self.root, 'sample', self.composition)


if __name__ == '__main__':
    unittest.main()
