"""Real SDK import + routing contracts; backend commands remain unchanged."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.serve_official_policy import configure_imports
configure_imports()
from rsi.official_adapter import Model, create_backend
from rsi.policy_catalog import catalog, resolve_task, runtime_settings
from rsi.wam_policy import configuration
from rsi.skills.runtime import Library
from rsi.task_contract import verify

ROOT = Path(__file__).resolve().parents[2]


def observation():
    state = {side + suffix: np.zeros(n) for side in ('left', 'right')
             for suffix, n in (('_arm_joint_state', 6), ('_ee_joint_state', 1), ('_ee_pose', 7))}
    state['left_ee_pose'][3] = state['right_ee_pose'][3] = 1
    return {'instruction': 'Stack the bowls.', 'state': state,
            'vision': {name: {'color': np.zeros((8, 8, 3), dtype=np.uint8)}
                       for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
            'reward': 100, 'layout_id': 43, 'scene': object()}


class OfficialDispatchTests(unittest.TestCase):
    def test_switched_modes_use_one_declared_backend_without_fallback(self):
        for mode, kind in [('align_blocks','learned'),('make_kong','rule'),
                           ('sort_nesting_dolls_by_size_random','rule')]:
            with self.subTest(mode=mode), patch('rsi.wam_policy.create') as wam, \
                 patch('rsi.rule_policy.create') as rule:
                selected = wam if kind == 'learned' else rule
                route = resolve_task('unchanged actual public instruction',mode)[1]
                self.assertEqual(route['backend'],kind)
                self.assertIs(create_backend(mode,route,{}),selected.return_value)
                selected.assert_called_once_with(mode,route,{})
                (rule if kind == 'learned' else wam).assert_not_called()

    def test_exact_54_routes_share_one_runtime_and_disjoint_backends(self):
        routes = catalog()
        self.assertEqual(sum(x['backend'] == 'rule' for x in routes.values()), 44)
        self.assertEqual(sum(x['backend'] == 'learned' for x in routes.values()), 10)
        self.assertEqual(runtime_settings()['expected_qualified_episodes'], 2100)
        from rsi.deployment import locations
        self.assertEqual(runtime_settings()['python'], locations()['python'])
        self.assertEqual(runtime_settings()['official_python_version'], '3.11.15')
        from rsi.rule_configuration import settings
        rule = settings()
        self.assertEqual(rule['policy_source_count'], len(rule['tasks']))
        self.assertEqual(rule['expected_qualified_episodes'],
                         sum(x['official_eval_nums'] for x in rule['tasks'].values()))
        for mode, route in routes.items():
            self.assertEqual(resolve_task('actual same-frame public instruction', mode)[0], mode)
            if route['backend'] == 'rule': settings(mode)  # Actual server-side runtime admission, not CLI help only.

    def test_instruction_dispatch_never_guesses_shared_variant(self):
        self.assertEqual(resolve_task('Insert the key into the keyhole.')[0], 'insert_key')
        self.assertEqual(resolve_task('build a tower')[0], 'build_tower')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            resolve_task('Fold the clothes.')
        self.assertEqual(resolve_task('Fold the clothes.', 'fold_clothes_random')[0], 'fold_clothes_random')

    def test_public_model_filters_metadata_and_preserves_ee_commands(self):
        class Actor:
            def update_obs(self, obs): self.obs = obs
            def get_action(self):
                return [{key: self.obs['state'][key] for key in
                         ('left_ee_pose', 'right_ee_pose', 'left_ee_joint_state', 'right_ee_joint_state')}]
            def reset(self): self.obs = None
        actor = Actor()
        with tempfile.TemporaryDirectory() as folder, patch('rsi.official_adapter.create_backend', return_value=actor) as factory:
            model = Model({'task_name':'stack_bowls', 'action_type':'ee', 'output_dir':folder})
            model.update_obs(observation())
            self.assertEqual(set(actor.obs), {'instruction', 'vision', 'state'})
            self.assertTrue(all(set(view) == {'color'} for view in actor.obs['vision'].values()))
            chunk = model.get_action()
            self.assertEqual(chunk[0]['left_ee_pose'].shape, (7,))
            np.testing.assert_array_equal(chunk[0]['left_ee_pose'], observation()['state']['left_ee_pose'])
            model.prepare_case({'layout_id': 1}); model.on_trial_end({'reward':100})
            factory.assert_called_once()
            model.reset(); self.assertIsNone(actor.obs)

    def test_actual_wam_bindings_compile_without_ambient_import_hooks(self):
        library = Library(ROOT, 'stack_bowls')
        try:
            path = ROOT / 'policies/riemann_1/adapter/riemann_1.0_robodojo/model.py'
            spec = importlib.util.spec_from_file_location('_test_actual_wam_adapter', path)
            module = importlib.util.module_from_spec(spec)
            module.__dict__['__riemann_skills__'] = library
            hooks = list(sys.meta_path)
            spec.loader.exec_module(module)
            self.assertEqual(sys.meta_path, hooks)
            actor = object.__new__(module.Model)
            actor._latest = (np.zeros((2, 2, 3)), np.zeros(16))
            class Policy:
                zero = False
                def reset(self, image, state, instruction):
                    self.instruction = instruction
                def act(self, image, state):
                    if self.zero:
                        return np.zeros(16, dtype=np.float32)
                    action = np.arange(16, dtype=np.float32) / 20
                    action[[3, 11]] = 1
                    return action
            actor.policy = Policy()
            action = actor.get_action()[0]
            self.assertEqual(set(action), {'left_ee_pose', 'right_ee_pose', 'left_ee_joint_state', 'right_ee_joint_state'})
            self.assertEqual(action['left_ee_pose'].shape, (7,))
            from types import SimpleNamespace
            actor.data_cfg = SimpleNamespace(combine_views=lambda views: views['observation.images.cam_high'],
                                             video_height=8, video_width=8)
            actor.reset()
            with tempfile.TemporaryDirectory() as folder, patch('rsi.official_adapter.create_backend', return_value=actor):
                public = Model({'task_name':'stack_bowls', 'action_type':'ee', 'output_dir':folder})
                frame = observation()
                frame['vision']['cam_head']['color'][:] = [3, 17, 231]
                original_image = actor._stitch_image(frame)
                public.update_obs(frame)
                np.testing.assert_array_equal(actor._latest[0], original_image)
                self.assertEqual(actor.policy.instruction, frame['instruction'])
                self.assertEqual(public.get_action()[0]['left_ee_pose'].dtype, np.float32)
                actor.policy.zero = True
                public.update_obs(frame)
                # Preserve the original adapter's zero-quaternion edge case;
                # the router must not add a new rule-specific rejection gate.
                np.testing.assert_array_equal(public.get_action()[0]['left_ee_pose'], np.zeros(7))
        finally:
            library.trace.close()

    def test_batch_reuses_loaded_backend_across_actions_and_reset(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('rsi.official_adapter.create_backend') as factory:
            model = Model({'task_name': 'stack_bowls', 'action_type': 'ee', 'output_dir': folder})
            actor = factory.return_value
            actor.get_action.return_value = []
            try:
                for _ in range(2):
                    model.update_obs_batch([observation()]); self.assertEqual(model.get_action_batch([4]), [[]])
                model.reset()
                model.update_obs_batch([observation()]); model.get_action_batch([9])
                factory.assert_called_once()
                self.assertEqual(actor.update_obs.call_count, 3)
                actor.reset.assert_called_once()
            finally:
                model.close()
            actor.close.assert_called_once()

    def test_multi_environment_batch_fails_without_loading_more_models(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('rsi.official_adapter.create_backend') as factory:
            model = Model({'task_name': 'stack_bowls', 'action_type': 'ee', 'output_dir': folder})
            try:
                with self.assertRaisesRegex(ValueError, 'one environment'):
                    model.update_obs_batch([observation(), observation()])
                factory.assert_called_once()
            finally:
                model.close()

    def test_batch_identity_changes_only_after_reset(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('rsi.official_adapter.create_backend') as factory:
            model = Model({'task_name': 'stack_bowls', 'action_type': 'ee', 'output_dir': folder})
            factory.return_value.get_action.return_value = []
            try:
                model.update_obs_batch([observation()]); model.get_action_batch([4])
                model.update_obs_batch([observation()])
                with self.assertRaisesRegex(ValueError, 'Reset'):
                    model.get_action_batch([9])
                self.assertEqual(factory.return_value.update_obs.call_count, 1)
                for indices in ([4, 9], [-1], [True]):
                    with self.assertRaises(ValueError): model.get_action_batch(indices)
            finally:
                model.close()

    def test_git_managed_contracts_do_not_require_source_hash_snapshots(self):
        for mode in catalog():
            bundle = json.loads((ROOT/'tasks'/mode/'composition.json').read_text())
            self.assertEqual(bundle['source_versioning'], 'git')
            self.assertNotIn('files', bundle)
            self.assertNotIn('static_python_imports', bundle)
            self.assertGreater(verify(ROOT, mode), 0, mode)

    def test_wam_inference_and_normalization_defaults_remain_original(self):
        spec = json.loads((ROOT/'policies/riemann_1/policy.json').read_text())
        cfg = configuration('stack_bowls', {'wam_checkpoint':'/explicit/checkpoint'})
        for key, value in spec['inference'].items(): self.assertEqual(cfg[key], value)
        self.assertEqual(cfg['action_type'], 'ee')
        self.assertEqual(cfg['model_path'], '/explicit/checkpoint/model')
        self.assertEqual(cfg['t5_device'], 'cuda:1')

    def test_wam_checkpoint_paths_use_the_official_policy_relative_resolver(self):
        from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
        with patch('rsi.deployment.model_options', return_value={}), \
                patch('XPolicyLab.utils.checkpoint_resolver.resolve_checkpoint_root',
                      wraps=resolve_checkpoint_root) as resolve:
            cfg = configuration('stack_bowls', {'ckpt_name': 'checkpoints/custom-run'})
            self.assertEqual(Path(cfg['exp_dir']), (ROOT/'checkpoints/custom-run').resolve())
            resolve.assert_called_once_with({'checkpoint_path':'checkpoints/custom-run'},
                                           ROOT/'checkpoints', policy_dir=ROOT, must_exist=False)

    def test_submission_and_named_deployment_keep_batch_env_alignment(self):
        import deploy
        canonical = importlib.import_module('rsi.adapters.Riemann-RSI-Preview.deploy')
        from XPolicyLab.policy.demo_policy.deploy import eval_one_episode
        self.assertIs(deploy.eval_one_episode, eval_one_episode)
        self.assertIs(deploy.eval_one_episode_batch, canonical.eval_one_episode_batch)

        class Environment:
            steps = 0
            taken = []
            def is_episode_end(self): return self.steps == 2
            def get_running_env_idx_list(self): return [2, 9] if self.steps == 0 else [9]
            def get_obs_batch(self, indices): return list(indices)
            def take_action_batch(self, actions, indices):
                self.taken.append((list(actions), list(indices)))
                self.steps += 1
        class Client:
            def call(self, func_name, **kwargs):
                if func_name == 'get_action_batch':
                    return [[index, index] for index in kwargs['env_idx_list']]
        env = Environment()
        deploy.eval_one_episode_batch(env, Client())
        self.assertEqual(env.taken, [([2, 9], [2, 9]), ([9], [9])])

    def test_debug_transport_cannot_replace_simulation(self):
        with patch.dict(os.environ, {'EVAL_ENV_TYPE':'sim'}):
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                create_backend('stack_bowls', {'backend':'learned'}, {'debug_hold':True})
        from rsi.official_adapter import DebugTransportPolicy
        policy = DebugTransportPolicy('ee')
        obs = observation()
        obs['state']['left_ee_pose'][:] = 0
        policy.update_obs(obs)
        np.testing.assert_array_equal(policy.get_action()[0]['left_ee_pose'][3:], [1., 0., 0., 0.])



    def test_gpu_lease_descriptor_is_real_and_inheritable_by_owned_child(self):
        import fcntl
        import subprocess
        from rsi.gpu_resources import gpu_leases
        with tempfile.TemporaryDirectory() as folder:
            descriptors = []
            with gpu_leases([2], directory=Path(folder), inventory=lambda gpu: {'index':gpu},
                            lease_descriptors=descriptors):
                self.assertEqual(len(descriptors), 1)
                with (Path(folder)/'gpu2.lock').open('r+') as competing:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competing, fcntl.LOCK_EX|fcntl.LOCK_NB)
                subprocess.run([sys.executable, '-c', 'import os,sys; os.fstat(int(sys.argv[1]))',
                                str(descriptors[0])], pass_fds=tuple(descriptors), check=True)
            with (Path(folder)/'gpu2.lock').open('r+') as released:
                fcntl.flock(released, fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_real_rule_factory_and_ack_bridge_preserve_episode_output_scope(self):
        from types import SimpleNamespace
        import importlib
        from skills.shared.orchestration.public_mature_workflow import implementation as mature
        actual_import = importlib.import_module
        controller = SimpleNamespace(__name__='tasks.general_pickup.controller')
        seen = []
        def load(name, *args, **kwargs):
            return controller if name == 'tasks.general_pickup.controller' else actual_import(name, *args, **kwargs)
        def run(controller, profile, initial, exchange, output, *, horizon):
            seen.append(output.name)
            self.assertEqual(horizon, 200)
            exchange({key: initial['state'][key].copy() for key in
                      ('left_arm_joint_state', 'right_arm_joint_state', 'left_ee_joint_state', 'right_ee_joint_state')})
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(mature, 'run', side_effect=run), \
                patch('rsi.rule_policy.importlib.import_module', side_effect=load):
            model = Model({'task_name':'general_pickup', 'action_type':'joint',
                           'output_dir':str(Path(folder)/'policy')})
            try:
                for _ in range(2):
                    model.reset()
                    model.update_obs(observation())
                    self.assertEqual(model.get_action()[0]['left_arm_joint_state'].shape, (6,))
                    model.update_obs(observation())
            finally:
                model.close()
            self.assertEqual(seen, ['episode-0000', 'episode-0001'])
