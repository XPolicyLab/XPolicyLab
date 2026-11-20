"""Public language-pour Rule routing, ABI and expected observation failures."""
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = ('Pour the liquid from the red bottle into the black bowl, '
               'from the turquoise bottle into the white bowl, '
               'and from the violet bottle into the brown bowl.')


class PourRuleIntegration(unittest.TestCase):
    def test_one_official_rule_route_and_no_pilot_layout_admission(self):
        from rsi.policy_catalog import catalog
        route = catalog(ROOT)['pour_by_language']
        self.assertEqual(route['backend'], 'rule')
        self.assertEqual(route['official_eval_nums'], 50)
        self.assertEqual(route['official_action_horizon'], 800)
        learned = json.loads((ROOT/'policies/riemann_1/policy.json').read_text())
        self.assertNotIn('pour_by_language', learned['tasks'])
        facts = json.loads((ROOT/'tasks/pour_by_language/task.json').read_text())
        self.assertNotIn('evaluation_cohort', facts)
        from tasks.pour_by_language.bindings.official_policy import verify_resources
        self.assertEqual(verify_resources(ROOT/'tasks/pour_by_language')['robot_urdf'], ROOT/'assets/robots/x5/X5A.urdf')

    def test_current_public_frame_and_missing_rgb_evidence_produce_finite_hold(self):
        from tasks.pour_by_language.controller import Model
        from tasks.pour_by_language.skills.public_observation.implementation import Observation, read_finger_joints
        state = {}
        for side, y in [('left',.15), ('right',-.15)]:
            state[side+'_arm_joint_state'] = np.zeros(6)
            state[side+'_ee_joint_state'] = np.array([1.])
            state[side+'_ee_pose'] = np.array([.25,y,.85,1.,0.,0.,0.])
        images = {name:np.zeros((480,640,3),dtype=np.uint8) for name in ('cam_head','cam_left_wrist','cam_right_wrist')}
        actor = Model(robot_urdf=ROOT/'assets/robots/x5/X5A.urdf', instruction=INSTRUCTION)
        try:
            actor.update(Observation.create(state,images,0))
            command = actor.next_action()
            self.assertEqual(set(command),{side+suffix for side in ('left','right') for suffix in ('_arm_joint_state','_ee_joint_state')})
            self.assertTrue(all(np.isfinite(value).all() for value in command.values()))
            actor.update(Observation.create(state,images,1))
            self.assertEqual(actor.instruction,INSTRUCTION)
            actor.next_action()
            self.assertIsNotNone(actor.failure)
            with self.assertRaises(PermissionError): read_finger_joints(None)
        finally:
            actor.close()
