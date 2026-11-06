"""Exercise renamed policy graphs and portable external resource admission."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class PortableSkillBindings(unittest.TestCase):
    def test_selected_graphs_execute_observation_reset_and_action_contract(self):
        from rsi.skills.runtime import Library
        policy = json.loads((ROOT / 'policies/riemann_1/policy.json').read_text())
        library = Library(ROOT, next(iter(policy['tasks'])))
        try:
            namespace = dict(np=np, _STATE_DIM=16, _QUAT_STARTS=(3, 11), _GRIP_IDXS=(7, 15))
            methods = {}
            for slot, identifier in library.bindings.items():
                if identifier.startswith('policy_'):
                    name = library.registry['skills'][identifier]['symbol']
                    methods[name] = library.bind(slot, namespace)
            sample = np.ones(16, dtype=np.float32); sample[7] = 2.; sample[15] = -1.
            actor = SimpleNamespace(policy=SimpleNamespace(reset=Mock(), act=Mock(return_value=sample)),
                                    _stitch_image=lambda obs: obs['image'],
                                    _state_from_obs=lambda obs: obs['state'],
                                    _task_text=lambda obs: obs['instruction'])
            actor.update_obs = lambda obs: methods['update_obs'](actor, obs)
            actor.get_action = lambda: methods['get_action'](actor)
            methods['reset'](actor)
            with self.assertRaisesRegex(RuntimeError, 'before update_obs'): actor.get_action()
            observation = dict(image='rgb', state='own robot', instruction='unchanged instruction')
            methods['update_obs_batch'](actor, [observation])
            actor.policy.reset.assert_called_once_with('rgb', 'own robot', 'unchanged instruction')
            result = methods['get_action_batch'](actor)[0][0]
            self.assertEqual(result['left_ee_joint_state'].tolist(), [1.])
            self.assertEqual(result['right_ee_joint_state'].tolist(), [0.])
            self.assertAlmostEqual(float(np.linalg.norm(result['left_ee_pose'][3:])), 1.)
            with self.assertRaises(NotImplementedError): methods['update_obs_batch'](actor, [observation]*2)
        finally:
            library.trace.close()

    def test_legacy_aliases_are_explicit_and_preserve_longest_prefix(self):
        from rsi.resource_locations import aliases, relocate_manifest
        env = {'RIEMANN_RESOURCE_ROOTS': json.dumps({'models':'/opt/relocated/models', 'encoder':'/opt/relocated/encoder'}),
               'RIEMANN_RESOURCE_ALIASES': json.dumps({'models':['/opt/legacy'], 'encoder':['/opt/legacy/encoder']})}
        with patch('rsi.resource_locations.local', return_value={}):
            result = relocate_manifest({'/opt/legacy/encoder/vae.pth': {'sha256': 'pinned'}}, environment=env)
            self.assertEqual(result, {'/opt/relocated/encoder/vae.pth': {'sha256': 'pinned'}})
            self.assertEqual(result, relocate_manifest(result, environment=env))
            with self.assertRaisesRegex(ValueError, 'Unknown resource alias'):
                aliases({'RIEMANN_RESOURCE_ALIASES':'{"unknown":["/opt/legacy"]}'})
            with self.assertRaisesRegex(ValueError, 'path lists'):
                aliases({'RIEMANN_RESOURCE_ALIASES':'{"models":"/opt/legacy"}'})

    def test_mature_profile_resource_fields_resolve_under_another_installation(self):
        from rsi.resource_locations import load_manifest
        env = {'RIEMANN_OFFICIAL_ROOT':'/opt/test/sdk', 'RIEMANN_EVAL_PYTHON':'/opt/test/eval/bin/python',
               'RIEMANN_RESOURCE_ROOTS':json.dumps({'official_assets':'/opt/test/assets', 'category_models':'/opt/test/language',
                                                    'table_models':'/opt/test/table'})}
        with patch('rsi.resource_locations.local', return_value={}):
            profile = load_manifest(ROOT / 'tasks/classify_objects_by_language/bindings/profile.json', environment=env)
            self.assertEqual(profile['parameters']['public_model_root'], '/opt/test/language')
            self.assertEqual(profile['parameters']['own_robot_root'], '/opt/test/assets/Robots/x5')
            lock = load_manifest(ROOT / 'tasks/organize_table/bindings/runtime-lock.json', environment=env)
            declared = json.loads((ROOT / 'tasks/organize_table/bindings/runtime-lock.json').read_text())
            members = [name.removeprefix('resource://table_models/') for name in declared['external_resources']
                       if name.startswith('resource://table_models/')]
            self.assertTrue(members)
            for member in members: self.assertIn('/opt/test/table/' + member, lock['external_resources'])

    def test_charger_diagnostics_require_an_explicit_output_scope(self):
        from tasks.plug_in_charger.skills.diagnostics.implementation import _authorized_output
        with patch.dict('os.environ', {}, clear=True):
            self.assertFalse(_authorized_output(Path('/tmp/output')))
        with patch.dict('os.environ', {'RIEMANN_DIAGNOSTICS_ROOT':'/tmp/permitted'}, clear=True):
            self.assertTrue(_authorized_output(Path('/tmp/permitted/episode')))
            self.assertFalse(_authorized_output(Path('/tmp/sibling')))

    def test_insert_key_diagnostics_follow_the_run_binding_or_system_temporary_directory(self):
        from tasks.insert_key.skills.state import implementation as state
        with TemporaryDirectory() as temporary, patch.object(state.trace, 'output', None):
            root = Path(temporary)
            with patch.dict('os.environ', {}, clear=True), \
                    patch.object(state, 'gettempdir', return_value=temporary), \
                    patch('os.getpid', return_value=123):
                actor = state.Model()
                expected = root / 'riemann-rsi/insert_key/123'
                self.assertEqual(state.trace.output, expected / 'trial000')
                self.assertTrue(state.trace.output.is_dir())
                actor.reset()
                self.assertEqual(state.trace.output, expected / 'trial001')
            with patch.dict('os.environ', {'IK_RESULT': str(root / 'official-run/policy')}, clear=True):
                state.Model()
                self.assertEqual(state.trace.output, root / 'official-run/policy/trial000')
                self.assertTrue(state.trace.output.is_dir())
