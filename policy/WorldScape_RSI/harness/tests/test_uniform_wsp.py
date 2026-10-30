"""Exercise instruction preservation, uniform history and configuration boundaries."""
from pathlib import Path
import copy
import inspect
import json
import types
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from test_harness import ROOT, backends, framework, release


@unittest.skipUnless(framework, 'Requires XPolicyLab CPU dependencies')
class UniformWspTests(unittest.TestCase):
    def setUp(self):
        backends.register_vendor()
        from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter import robodojo_model, runtime_config
        self.native, self.config = robodojo_model, runtime_config
        self.wsp = Path(framework)/'policy/WorldScape_RSI'

    def actor(self):
        native = self.native
        model = object.__new__(native.RoboDojoModel)
        model.wam_history_mode = 'rolling'
        model.wam_max_chunk_size = 2
        model.action_steps = 8
        model.observation_history_segments = 8
        model.use_observation_history = True
        model.input_color_order = 'rgb'
        model._resize_image = lambda image: image
        model._rolling_history_by_env = {}
        model._obs_history_by_env = {}
        model._observation_index_by_env = {}
        model._latest_obs_by_env = {}
        model._latest_env_idx_list = []
        model._history_window_by_env = {}
        model.short_window_for_non_memory_tasks = False
        return model

    def observation(self, value, instruction):
        return {'instruction': instruction,
                'vision': {name: {'color': np.full((4, 4, 3), value, np.uint8)}
                           for name in self.native.CAMERA_ORDER}}

    def test_mixed_instructions_keep_equal_capacity_and_evict_old_frames(self):
        model = self.actor()
        labels = ['press_by_number', 'Move the item to the visible support.', '新目标：根据当前观测完成要求']
        for index in range(17):
            model.update_obs_batch([{**self.observation(index, text), 'env_idx': env}
                                    for env, text in enumerate(labels)])
            if index in (0, 8, 16):
                histories = [model._prepare_observation_cameras(env) for env in range(len(labels))]
                for cameras in histories:
                    self.assertEqual(cameras[0].shape[0], 1 if index == 0 else 9)
                    np.testing.assert_array_equal(cameras[0][:, 0, 0, 0],
                                                  np.arange(max(0, index-8), index+1))
                for env, cameras in enumerate(histories):
                    model._rolling_history_by_env[env] = cameras
                    model._obs_history_by_env[env] = [model._latest_obs_by_env[env]]

    def test_actual_observation_adapter_receives_public_instruction_unchanged(self):
        model = self.actor()
        model.default_instruction = 'Fallback'
        model.mode = 'interactive'
        model.t5_prompt_template = None
        model.bundle = object()
        model.device = 'cpu'
        model.visual_input_range = 'zero_one'
        model.robot_action_dim_info = {}
        calls = []
        def adapter_factory(bundle, **kwargs):
            calls.append(kwargs)
            return lambda values: types.SimpleNamespace(prompts=types.SimpleNamespace(), observation=object())
        model._observation_adapter_class = adapter_factory
        for text in ('press_by_number', 'Press according to the currently visible instruction.',
                     'Remember the changing scene.', '新目标：观察并操作。'):
            model._latest_obs_by_env = {0: self.observation(0, text)}
            model._observation_index_by_env = {0: 1}
            with patch.object(self.native, '_state_fields', return_value={}):
                model._adapt_observation(0, cameras=[np.zeros((1,4,4,3),np.uint8)]*3)
            self.assertEqual(calls[-1]['instruction'], text)

    def test_unknown_execution_options_fail_before_resources_load(self):
        for key, value in {'short_window_for_non_memory_tasks': True,
                           'memory_window_chunks': 4, 'ordinary_window_chunks': 1,
                           'press_by_number_prompt': {'enabled': True},
                           'interactive_qwen': {'enabled': True},
                           'instruction_selector': {'example': 'controller'}}.items():
            # The standalone adapter retains optional features. RSI rejects
            # them at its frozen execution boundary.
            data = json.loads((ROOT/'release.json').read_text())
            data['wam_skills']['worldscape-wam-memory']['config'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.validate(data)

    def test_bundled_deployment_and_both_skill_configs_use_supported_fields(self):
        for name in ('wam_deploy.yml', 'wam_deploy_robodojo.yml'):
            cfg = yaml.safe_load((self.wsp/name).read_text(encoding='utf-8'))
            self.config.validate_config(cfg)
        skills = json.loads((ROOT/'configs/wam_skills.json').read_text(encoding='utf-8'))
        for spec in skills.values():
            self.config.validate_config(spec['config'])
        for name, key in (('mem20000_model_config.example.json', 'worldscape-wam-memory'),
                          ('flash50000_model_config.example.json', 'worldscape-wam-nomemory')):
            cfg = json.loads((ROOT/'configs'/name).read_text(encoding='utf-8'))
            self.config.validate_config(cfg)
            for field, value in skills[key]['config'].items():
                self.assertEqual(cfg[field], value, (name, field))
            self.assertEqual(cfg['replan_steps'] // cfg['observation_interval'],
                             cfg['observation_history_segments'])

    def test_rollout_geometry_has_no_freeform_metadata_input(self):
        checkpoint = types.SimpleNamespace(shape=types.SimpleNamespace(action_horizon=24))
        source = {'runtime_profile':'robodojo', 'action_horizon':8}
        result = self.config.resolve_rollout_config(source, checkpoint, {})
        self.assertEqual(result['action_horizon'], 24)
        self.assertEqual(source['action_horizon'], 8)
        self.assertEqual(set(inspect.signature(self.config.resolve_rollout_config).parameters),
                         {'config','checkpoint_model','provenance'})

    def test_flash_skill_and_direct_service_share_inference_settings(self):
        skills = json.loads((ROOT/'configs/wam_skills.json').read_text())
        flash = skills['worldscape-wam-nomemory']['config']
        for name in ('wam_deploy.yml', 'wam_deploy_robodojo.yml'):
            direct = yaml.safe_load((self.wsp/name).read_text())
            for key in ('wam_history_mode', 'use_observation_history', 'use_causal_memory',
                        'reset_independent_blocks', 'observation_interval',
                        'observation_history_segments', 'action_horizon', 'replan_steps',
                        'input_color_order', 'policy_image_interpolation', 'diffusion_view_layout'):
                with self.subTest(config=name, key=key):
                    self.assertEqual(flash[key], direct[key])

    def test_inconsistent_sampling_geometry_fails_before_resource_loading(self):
        data = json.loads((ROOT/'release.json').read_text())
        data['wam_skills']['worldscape-wam-memory']['config']['observation_interval'] = 2
        with self.assertRaisesRegex(ValueError, 'sampling geometry'):
            release.validate(data)

    def test_retired_implementations_and_obsolete_notes_are_absent(self):
        import hashlib
        hashes = json.loads((ROOT.parent/'validation/wam_source.json').read_text())
        for path, expected in hashes.items():
            self.assertEqual(hashlib.sha256((self.wsp/path).read_bytes()).hexdigest(), expected, path)
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect import policy as base
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF.docs import ARX_X5_EEF_DOCS
        self.assertFalse(hasattr(base,'_optional_absent_int'))
        for phrase in ('0.15 m','20 degrees','table surface','score the arms','last step of the task'):
            self.assertNotIn(phrase, ARX_X5_EEF_DOCS)

    def test_release_rejects_unregistered_execution_options(self):
        original=json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
        for spec in original['wam_skills']:
            changed=copy.deepcopy(original)
            changed['wam_skills'][spec]['config']['unregistered_selector']=True
            with self.assertRaises(ValueError): release.validate(changed)

    def test_checkpoint_task_selector_is_disabled_before_any_observation(self):
        from r2si_under_test.wam import enforce_uniform_history
        from unittest.mock import Mock
        backend = self.actor()
        backend.wam_max_chunk_size = 24
        backend.short_window_for_non_memory_tasks = True
        backend.task_window_chunks = (24, 1)
        backend.model_cfg = {"short_window_for_non_memory_tasks": True}
        backend._prediction_index = 0
        backend.reset = Mock()
        enforce_uniform_history(types.SimpleNamespace(backend=backend))
        backend._latest_obs_by_env = {0: self.observation(0, "press_by_number"),
                                      1: self.observation(0, "unrelated new instruction")}
        with patch.object(self.native, "match_memory_task", side_effect=AssertionError("task matching used")):
            backend._resolve_history_windows([0, 1])
        self.assertEqual(backend._history_window_by_env, {})
        self.assertEqual(backend.task_window_chunks, (24, 24))
        with self.assertRaisesRegex(RuntimeError, "active"):
            enforce_uniform_history(types.SimpleNamespace(backend=backend))


if __name__ == '__main__': unittest.main()
