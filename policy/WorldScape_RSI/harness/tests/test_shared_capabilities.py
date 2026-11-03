"""Task-metadata invariance and real Agent prompt/tool-loop regressions.

Model responses and native actions are CPU fixtures, not evidence of task success.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from test_harness import ROOT, DIMS, FakeRelease, backends, contracts, framework, model, obs

PROBE = ROOT / 'vendor/policy/RoboRSI_RoboProbe'
spec = importlib.util.spec_from_file_location('shared_library_fixture', PROBE / 'library.py')
library_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(library_module)
SkillLibrary = library_module.SkillLibrary
TASKS = ('press_by_number', 'general_pickup', 'imitate_sorting_sequence',
         'pack_objects_into_box', 'push_T_random', 'unknown_goal', '../arbitrary')


class SharedCatalogTests(unittest.TestCase):
    def test_catalog_is_not_selected_by_task_metadata(self):
        with self.assertRaises(TypeError):
            SkillLibrary(PROBE / 'skills', 'press_by_number')
        self.assertFalse((PROBE / 'skills/generic/public-task-instructions').exists())
        self.assertFalse((ROOT / 'vendor/policy/RoboDojo_Agent_L3_Inspect/recipes').exists())

    def test_shared_catalog_and_manifest_selected_context_are_complete(self):
        lib = SkillLibrary(PROBE / 'skills')
        self.assertGreater(len(lib.catalog('skill')), 5)
        self.assertTrue({e['scope'] for e in lib.entries.values()} == {'generic'})
        self.assertIn('robodojo-interface', lib.initial())
        # Check payload inclusion by manifest policy, not by catalog name.
        initial = lib.initial()
        self.assertLessEqual(len(initial), 24000)
        for entry in lib.entries.values():
            if entry.get('load_mode') == 'always':
                self.assertIn(entry['text'], initial)
            else:
                self.assertNotIn(entry['text'], initial)
        for key, entry in lib.entries.items():
            text, offset = '', 0
            while True:
                page = lib.read(key, entry['category'], offset)
                text += page['text']
                if page['next_offset'] is None:
                    break
                offset = page['next_offset']
            self.assertEqual(text, entry['text'])
            for stage in entry['stages']:
                lib.read(key, entry['category'], stage=stage)

    def test_old_task_catalog_and_manifest_selectors_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'skills'
            shutil.copytree(PROBE / 'skills', root)
            (root / 'tasks').mkdir(exist_ok=True)
            with self.assertRaises(ValueError):
                SkillLibrary(root)
            (root / 'tasks').rmdir()
            manifest = root / 'generic/goal-interpretation/manifest.json'
            original = json.loads(manifest.read_text(encoding='utf-8'))
            for field in ('task_context_file', 'task_ids', 'aliases', 'keywords', 'routes'):
                data = {**original, field: ['injected']}
                manifest.write_text(json.dumps(data), encoding='utf-8')
                with self.subTest(field=field), self.assertRaises(ValueError):
                    SkillLibrary(root)

    def test_old_inherited_selection_entry_points_are_removed(self):
        obsolete = {'_goal_content_task', '_defer_llm_plan', '_base_task_name', 'use_task_recipe',
                    '_icl_enabled_for_task', '_icl_demonstration', '_icl_message',
                    '_general_pickup_bimanual_enabled', '_general_pickup_bimanual_lift_enabled',
                    '_deferred_llm_chunk', '_append_prior_attempt'}
        for relative in ('RoboDojo_Agent_L3_Inspect/policy.py', 'RoboDojo_Agent_L3_Inspect_EEF/policy.py',
                         'RoboRSI_RoboProbe/policy.py', 'RoboRSI_RoboProbe/library.py'):
            tree = ast.parse((ROOT / 'vendor/policy' / relative).read_text(encoding='utf-8'))
            names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            self.assertFalse(names & obsolete, relative)
            constants = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertFalse(constants & set(TASKS), relative)

    def test_protocol_metadata_is_not_forwarded_to_backend(self):
        calls = []
        for task in TASKS:
            backend = types.SimpleNamespace(reset=lambda: None, close=lambda: None)
            cfg = {'bench_name': 'RoboDojo', 'env_cfg_type': 'arx_x5', 'action_type': 'joint',
                   'policy_name': 'WorldScape_RSI', 'task_name': task, 'eval_batch': False, 'seed': 7}
            with patch.object(model, 'FrozenRelease', return_value=FakeRelease()), \
                 patch.object(contracts, 'action_dimensions', return_value=DIMS), \
                 patch.object(backends, 'build_backend', return_value=backend) as build:
                adapter = model.Model(cfg)
                adapter.reset()
                calls.append(build.call_args)
                adapter.close()
        self.assertTrue(all(call == calls[0] for call in calls))
        self.assertEqual(calls[0].args[2], {'seed': 7, 'env_cfg_type': 'arx_x5'})


@unittest.skipUnless(framework, 'Requires the XPolicyLab CPU dependencies')
class SharedAgentTests(unittest.TestCase):
    def setUp(self):
        backends.register_vendor()

    def actor(self, tool='primitive-control', arguments=None):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import RoboDojoActionSpec
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import ActionSpace
        from XPolicyLab.policy.RoboRSI_RoboProbe.policy import SkillEefPolicy
        self.captured = []

        def complete(messages, tools):
            self.captured.append(copy.deepcopy((messages, tools)))
            return {'choices': [{'message': {'content': None, 'tool_calls': [{'id': 'fixture-call',
                'function': {'name': 'call_tool', 'arguments': json.dumps({'id': tool,
                    'arguments': arguments if arguments is not None else {'op': 'wait', 'steps': 1}})}}]}}]}

        labels = tuple(ActionSpace().flat_labels())
        spec = RoboDojoActionSpec(labels, np.full(len(labels), -3.), np.full(len(labels), 3.),
                                  25., 'CPU fixture', tuple(.2 for _ in labels))
        client = types.SimpleNamespace(complete=complete, config={'url': 'https://fixture.invalid/v1'})
        return SkillEefPolicy(action_spec=spec, env={}, planner=lambda **kw: {}, client=client)

    def observation(self, task, instruction='Follow the visible goal.'):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Observation
        raw = obs()
        return Observation(images={k:v['color'] for k,v in raw['vision'].items()}, state=raw['state'],
                           instruction=instruction, extra={'task': task, 'task_id': task, 'layout_id': 99})

    def test_real_prepare_preserves_instruction_under_every_metadata_name(self):
        from XPolicyLab.policy.RoboRSI_RoboProbe.contracts import public_observation
        baseline = None
        for task in TASKS:
            actor = self.actor()
            observation = self.observation(task)
            actor.prepare(observation)
            current = (actor._messages, actor._tools, actor.library.catalog('skill'))
            if baseline is None:
                baseline = copy.deepcopy(current)
            self.assertEqual(current, baseline, task)
            self.assertEqual(actor._goal_text, 'Goal: Follow the visible goal.')
            self.assertEqual(public_observation(observation).extra, {})
            # Exercise inherited audit_config too, catching removed-helper references.
            self.assertEqual(actor.audit_config()['skills']['selection'], 'agent_from_current_context')

    def test_changed_instruction_changes_goal_not_catalog(self):
        actor = self.actor(); before = actor.library.catalog('skill')
        for instruction in ('Observe the changing pattern.', 'Align these two visible shapes.', 'press_by_number'):
            actor.reset(); actor.prepare(self.observation('unknown', instruction))
            self.assertEqual(actor._goal_text, 'Goal: ' + instruction)
            self.assertEqual(actor.library.catalog('skill'), before)

    def test_actual_agent_request_and_wait_do_not_change_with_task_metadata(self):
        reference = None
        for task in TASKS:
            actor = self.actor()
            chunk = actor.act(self.observation(task))
            self.assertEqual(len(chunk.actions), 1)
            self.assertEqual(len(self.captured), 1)
            if reference is None:
                reference = self.captured[0]
            self.assertEqual(self.captured[0], reference, task)

    def test_agent_selected_wsp_tool_is_dispatched_without_task_override(self):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import MotionOutcome
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action, ActionChunk
        from test_harness import hold
        for task in TASKS:
            for chosen in ('worldscape-wam-memory', 'worldscape-wam-nomemory'):
                actor = self.actor(chosen, {})
                called = []
                def dispatch(key, observation):
                    called.append(key)
                    return MotionOutcome(ActionChunk([Action(hold())]), 'CPU fixture')
                actor.wam_call = dispatch
                actor.act(self.observation(task))
                self.assertEqual(called, [chosen])



if __name__ == '__main__':
    unittest.main()
