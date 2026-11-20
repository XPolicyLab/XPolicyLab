"""Current paths, dynamic skill slots and external model identities migrate together."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class PolicyLayout(unittest.TestCase):
    def test_all_54_routes_use_the_declared_canonical_policy_id(self):
        from rsi.policy_catalog import catalog
        routes = catalog(ROOT)
        self.assertEqual(len(routes), 54)
        learned = {name: route for name, route in routes.items() if route['backend'] == 'learned'}
        self.assertEqual(len(learned), 10)
        for name, route in learned.items():
            self.assertEqual(route['policy_id'], 'riemann_1', name)
            facts = json.loads((ROOT / 'tasks' / name / 'task.json').read_text())
            self.assertEqual(facts['policy_id'], 'riemann_1')

    def test_actual_selected_library_resolves_the_renamed_adapter_slots(self):
        from rsi.skills.runtime import Library
        policy = json.loads((ROOT / 'policies/riemann_1/policy.json').read_text())
        for mode in policy['tasks']:
            library = Library(ROOT, mode)
            try:
                self.assertEqual(len(library.bindings), 12, mode)
                self.assertTrue(all(slot.startswith('policies/riemann_1/adapter/')
                                    for slot in library.bindings), mode)
                for module in library.plan['modules']:
                    self.assertTrue((ROOT / module['scaffold']).is_file())
                    fake_numpy = ModuleType('numpy')
                    fake_numpy.ndarray = type('ndarray', (), {})
                    template = ModuleType('XPolicyLab.model_template')
                    template.ModelTemplate = object
                    namespace = {'__riemann_skills__': library,
                                 '__file__': str(ROOT / module['scaffold']), '__name__': '_layout_fixture'}
                    with patch.dict(sys.modules, {'numpy': fake_numpy, 'XPolicyLab.model_template': template}):
                        exec(compile((ROOT / module['scaffold']).read_text(), namespace['__file__'], 'exec'), namespace)
                    self.assertTrue(callable(namespace['Model'].get_action))
                    self.assertTrue(callable(namespace['Model'].update_obs))
            finally:
                library.trace.close()

    def test_official_plugin_location_is_registered_by_both_roles(self):
        from rsi.adapters import register_policy
        for name in ('run_official_eval.py', 'serve_official_policy.py'):
            source = (ROOT / 'tools' / name).read_text()
            self.assertIn('from rsi.adapters import register_policy', source)
            self.assertIn('register_policy()', source)
        package = ModuleType('XPolicyLab')
        package.__path__ = []
        package.policy = ModuleType('XPolicyLab.policy')
        package.policy.__path__ = ['/existing/official/policy']
        with patch.dict(sys.modules, {'XPolicyLab': package, 'XPolicyLab.policy': package.policy}):
            register_policy()
            register_policy()
        self.assertEqual(package.policy.__path__,
                         [str(ROOT / 'rsi/adapters'), '/existing/official/policy'])
        self.assertTrue((ROOT / 'rsi/adapters/Riemann-RSI-Preview/model.py').is_file())
        self.assertTrue((ROOT / 'rsi/adapters/Riemann-RSI-Preview/deploy.py').is_file())

    def test_selected_skill_engine_imports_without_global_hooks(self):
        code = ('import sys; hooks = list(sys.meta_path); from rsi.skills import Library; '
                'from rsi.skills.git_versioning import runtime_source_identity; '
                'assert sys.meta_path == hooks; assert "torch" not in sys.modules')
        subprocess.run([sys.executable, '-I', '-S', '-B', '-c',
                        'import sys; sys.path.insert(0, ' + repr(str(ROOT)) + '); ' + code], check=True)

    def test_local_model_outputs_stay_outside_source_control(self):
        ignored = (ROOT / '.gitignore').read_text().splitlines()
        self.assertIn('/policies/riemann_1/checkpoint', ignored)
        self.assertIn('/policies/riemann_1/weights/', ignored)

    def test_named_official_plugin_exports_the_canonical_model(self):
        import importlib
        from rsi.adapters import register_policy
        canonical = ModuleType('rsi.official_adapter')
        canonical.Model = type('CanonicalModel', (), {})
        package = ModuleType('XPolicyLab')
        package.__path__ = []
        package.policy = ModuleType('XPolicyLab.policy')
        package.policy.__path__ = []
        with patch.dict(sys.modules, {'rsi.official_adapter': canonical,
                                    'XPolicyLab': package, 'XPolicyLab.policy': package.policy}):
            for name in list(sys.modules):
                if name.startswith('XPolicyLab.policy.Riemann-RSI-Preview'):
                    del sys.modules[name]
            register_policy()
            plugin = importlib.import_module('XPolicyLab.policy.Riemann-RSI-Preview.model')
            self.assertIs(plugin.Model, canonical.Model)
            self.assertEqual(Path(plugin.__file__), ROOT / 'rsi/adapters/Riemann-RSI-Preview/model.py')


if __name__ == '__main__':
    unittest.main()
