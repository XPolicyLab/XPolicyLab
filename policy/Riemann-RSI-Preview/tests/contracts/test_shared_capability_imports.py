"""Exercise real dynamic capability bindings and preserved sampling contracts."""
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rsi.policy_catalog import catalog

ROOT = Path(__file__).resolve().parents[2]


class SharedCapabilityImports(unittest.TestCase):
    def test_fold_workflows_issue_their_actual_first_scene_view(self):
        api = SimpleNamespace(command_memory=lambda: [], instruction=lambda: 'Fold the clothes.',
                              issued_action_count=lambda: 0)
        for mode in ('fold_clothes', 'fold_clothes_random'):
            with self.subTest(mode=mode):
                controller = importlib.import_module('tasks.'+mode+'.controller')
                workflow = controller.execution()
                request = workflow.next_request(api)
                self.assertEqual(request['skill'], 'garment_scene_view')
                self.assertEqual(workflow.action_bindings[request['skill']],
                                 'tasks/'+mode+'/skills/fold_scene_geometry/implementation.py')
                self.assertEqual(set(controller.describe()['action_tools']), set(workflow.allowed_skills))

    def test_fold_motion_dispatch_uses_its_declared_shared_implementation(self):
        from harness.runtime.manipulation import action_program, ACTION_LIMITS, ACTION_BINDINGS
        from skills.shared.motion.garment_command_execution.implementation import GarmentProgram
        state = {side+suffix:np.zeros(width) for side in ('left','right')
                 for suffix,width in (('_arm_joint_state',6),('_ee_joint_state',1),('_ee_pose',7))}
        state['left_ee_pose'][3] = state['right_ee_pose'][3] = 1.
        observation = dict(instruction='Fold the clothes.', state=state,
            vision={camera:{'color':np.zeros((8,8,3),dtype=np.uint8)}
                    for camera in ('cam_head','cam_left_wrist','cam_right_wrist')})
        original = dict(ACTION_BINDINGS)
        for mode in ('fold_clothes','fold_clothes_random'):
            with self.subTest(mode=mode):
                controller = importlib.import_module('tasks.'+mode+'.controller')
                request = dict(skill='garment_return_home', side='auto',
                               steps=ACTION_LIMITS['garment_return_home'], reason='Declared public home stage')
                program = action_program(observation, request,
                                         action_bindings=controller.execution().action_bindings)
                self.assertIsInstance(program, GarmentProgram)
        self.assertEqual(ACTION_BINDINGS, original)

    def test_insertion_workflow_constructs_and_dispatches_task_owned_tools(self):
        from tasks.insert_tubes import controller
        from harness.runtime.manipulation import ACTION_BINDINGS
        original = dict(ACTION_BINDINGS)
        workflow = controller.execution()
        self.assertEqual(set(workflow.action_bindings), set(controller.TOOLS))
        request = dict(skill='finish_tube_sequence', side='left', stage='home_left',
                       steps=128, reason='Original terminal home boundary')
        api = object()
        with patch('importlib.import_module') as load:
            result = controller.run(api, request)
            load.assert_called_once_with('tasks.insert_tubes.skills.finish_tube_sequence.implementation')
            load.return_value.run.assert_called_once_with(api, request)
            self.assertIs(result, load.return_value.run.return_value)
        self.assertEqual(ACTION_BINDINGS, original)

    def test_insertion_workflow_rejects_an_undeclared_tool(self):
        from tasks.insert_tubes import controller
        from harness.execution import ExecutionError
        missing = dict(controller.action_tools())
        missing.pop('grasp_tube')
        with patch.object(controller, 'action_tools', return_value=missing):
            with self.assertRaises(ExecutionError):
                controller.execution()

    def test_command_declarations_resolve_one_consistent_api_and_workflow(self):
        for mode, route in catalog().items():
            declaration = ROOT/'tasks'/mode/'bindings/source-binding.json'
            if not declaration.is_file():
                continue
            with self.subTest(mode=mode):
                binding = json.loads(declaration.read_text())
                public = importlib.import_module(binding['public_api_module'])
                motion = importlib.import_module(binding['motion_module'])
                contract = importlib.import_module(binding['contract_module'])
                self.assertIs(contract.PhasedProgram, public.PhasedProgram)
                self.assertTrue(callable(motion.NativeJointExecutor))
                self.assertEqual(binding['official_horizon'], route['official_action_horizon'])
                composition = json.loads((declaration.parents[1]/'composition.json').read_text())
                for skill, path in composition.get('native_action_tools', {}).items():
                    self.assertEqual(contract.ACTION_BINDINGS[skill], path)
                controller = importlib.import_module('tasks.'+mode+'.controller')
                execution = getattr(controller, 'execution', None)
                if callable(execution):
                    self.assertIsNotNone(execution())
                else:
                    self.assertTrue(callable(controller.resolve))

    def test_fold_graphs_resolve_every_real_stage_and_share_motion_implementations(self):
        shared = None
        for mode in ('fold_clothes', 'fold_clothes_random'):
            task = ROOT/'tasks'/mode
            graph = json.loads((task/'workflow.json').read_text())
            composition = json.loads((task/'composition.json').read_text())
            bindings = composition['native_action_tools']
            for stage in graph['stages']:
                path = bindings[stage['skill']]
                module = importlib.import_module(path.removesuffix('.py').replace('/', '.'))
                self.assertTrue(callable(module.run))
            motion = {k:v for k,v in bindings.items() if v.startswith('skills/shared/motion/')}
            self.assertTrue(motion)
            if shared is not None:
                self.assertEqual(motion, shared)
            shared = motion

    def test_garment_joint_samples_keep_peer_arm_jaws_and_final_target(self):
        from skills.shared.motion.garment_joint_sampling.implementation import sampled_joint_path, GarmentOwnModel, _declared_rigid_geometry
        joints = {'left':np.zeros(6), 'right':np.full(6,.2)}
        targets = {'left_ee_joint_state':np.array([.3]), 'right_ee_joint_state':np.array([.8])}
        api = SimpleNamespace(own_joint_states=lambda:joints, targets=lambda:targets)
        goal = np.full(6,.1)
        samples = sampled_joint_path(api, {'left':goal})
        self.assertEqual(len(samples),16)
        np.testing.assert_array_equal(samples[-1]['left_arm_joint_state'],goal)
        for sample in samples:
            np.testing.assert_array_equal(sample['right_arm_joint_state'],joints['right'])
            self.assertEqual(sample['left_ee_joint_state'],targets['left_ee_joint_state'].tolist())
            self.assertEqual(sample['right_ee_joint_state'],targets['right_ee_joint_state'].tolist())
        self.assertEqual(GarmentOwnModel().eef(np.zeros(6)).shape,(4,4))
        boxes, hulls, normals = _declared_rigid_geometry()
        self.assertEqual(set(boxes),set(hulls))
        self.assertEqual(set(boxes),set(normals))

    def test_planned_start_and_observed_start_keep_distinct_execution_contracts(self):
        from skills.shared.motion.observed_start_phase_execution.implementation import PublicRobotAPI as Observed
        from skills.shared.motion.planned_start_phase_execution.implementation import PublicRobotAPI as Planned
        self.assertIsNot(Observed, Planned)
        self.assertNotIn('start_state',inspect.signature(Observed.dense_own_trajectory).parameters)
        self.assertIn('start_state',inspect.signature(Planned.dense_own_trajectory).parameters)


if __name__ == '__main__':
    unittest.main()
