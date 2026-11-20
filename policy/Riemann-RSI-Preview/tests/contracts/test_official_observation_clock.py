"""Only regressions for the interfaces exposed by the first formal wave."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import numpy as np

from skills.shared.motion.declared_x5_command_targets.implementation import clock_environment
from skills.shared.motion.public_observation_command_clock.implementation import PublicCommandClock, PublicHistory
from skills.shared.motion.public_conveyor_workspace.implementation import (
    install_conveyor_planner_capacity, DECK_PROXY_NAME, CAMERA_PROXY_NAME,
)
from tests.contracts.test_official_standard_adapter import environment, observation as small_observation


def observation(value=0.):
    frame = small_observation(value)
    for camera in frame['vision'].values():
        camera['color'] = np.full((480, 640, 3), [11, 23, 47], np.uint8)
    return frame


class PublicObservationClockTests(unittest.TestCase):
    def test_existing_robot_factories_keep_the_exact_environment_and_clock_inputs(self):
        existing = environment()
        self.assertIs(clock_environment(existing), existing)

    def test_real_fold_and_liquid_factories_preserve_sampling_and_current_feedback(self):
        from tasks.fold_clothes.skills.native_public_state.implementation import create_environment as fold
        from tasks.fold_clothes_random.skills.native_public_state.implementation import create_environment as fold_random
        from tasks.pour_liquid_into_cup.skills.public_robot.implementation import create_environment as liquid
        from tasks.pour_liquid_into_cup_random.skills.public_robot.implementation import create_environment as liquid_random
        for factory in (fold, fold_random, liquid, liquid_random):
            for speed in (.9, 1.25):
                outputs = []
                with TemporaryDirectory() as temporary:
                    for index, maker in enumerate((lambda frame, rec, args: environment(), factory)):
                        history = PublicHistory(observation(), Path(temporary) / str(index))
                        actual = maker(history.current_public_observation, history,
                            NS(official_eval=True, public_instruction='Perform the declared task.'))
                        original_manager = actual.robot_manager
                        history.public_environment = actual
                        sent = []
                        clock = PublicCommandClock(clock_environment(actual), history,
                            lambda a: sent.append(deepcopy(a)) or observation(.7), time_scale=speed)
                        clock.execute([{'left_arm_joint_state': np.full(6, i / 1000.)}
                                       for i in range(37)], 'original_move')
                        clock.execute([{'left_ee_joint_state': [i / 10000.] * 2}
                                       for i in range(17)], 'original_jaw_wait')
                        outputs.append(sent)
                        self.assertIs(actual.robot_manager, original_manager)
                        self.assertEqual(actual.take_action_cnt[0], len(sent))
                        self.assertAlmostEqual(float(history.state_left_arm[-1][0]), .7, places=6)
                    self.assertEqual(*outputs)

    def test_original_conveyor_hook_reserves_exact_named_slots_before_planner_capture(self):
        class Planner:
            def _build_robot_and_scene_cfg(self):
                return {'robot': 'original'}, {'cuboid': {'table': {'pose': [0]}}}
        first = install_conveyor_planner_capacity(Planner)
        second = install_conveyor_planner_capacity(Planner)
        self.assertTrue(first['installed_now'])
        self.assertFalse(second['installed_now'])
        robot, scene = Planner()._build_robot_and_scene_cfg()
        self.assertEqual(robot, {'robot': 'original'})
        self.assertEqual(set(scene['cuboid']), {'table', DECK_PROXY_NAME, CAMERA_PROXY_NAME})
        self.assertEqual(scene['cuboid']['table'], {'pose': [0]})

    def test_fold_perception_uses_only_its_declared_single_card_without_a_broker(self):
        from harness.runtime.gpu_profiles import (
            public_perception_cuda_device, public_perception_resource,
            prepare_public_perception_resource,
        )
        from tasks.fold_clothes.skills.native_binding.implementation import prepare_single_gpu_perception
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = prepare_single_gpu_perception(root, 4)
            self.assertEqual(config['policy_physical_gpu'], 4)
            self.assertEqual(config['perception_physical_gpu'], 4)
            self.assertIsNone(config['model_resource_broker'])
            self.assertEqual(public_perception_cuda_device(root, 3), 0)
            with public_perception_resource(root):
                self.assertEqual(public_perception_cuda_device(root, 3), 0)
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                prepare_single_gpu_perception(root, 5)
            with self.assertRaisesRegex(ValueError, 'Only the current single-card'):
                prepare_public_perception_resource(root / 'unselected',
                    dict(native_gpu_resource_profile='debug-five-simulators-int4-planner-v1'))


if __name__ == '__main__': unittest.main()
