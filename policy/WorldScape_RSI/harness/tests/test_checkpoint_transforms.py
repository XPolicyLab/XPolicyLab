"""Exercise native transform validation without loading model weights."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

from test_harness import ROOT
from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.robodojo_model import (
    _validate_joint_transform_bundle,
)

source = ROOT.parents[1] / 'WorldScape_RSI/worldscape-policy/src/worldscape_policy/checkpoint/transforms.py'
spec = importlib.util.spec_from_file_location('native_transforms_under_test', source)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)


class CheckpointTransformTests(unittest.TestCase):
    def bundle(self, modes):
        def fields(prefix):
            result = []
            for index, (name, width) in enumerate([
                ('left_joint', 6), ('left_gripper', 1),
                ('right_joint', 6), ('right_gripper', 1),
            ]):
                mode = modes[index % len(modes)]
                statistics = ({'q01': [-1.] * width, 'q99': [1.] * width}
                              if mode == 'q99' else
                              {'mean': [0.] * width, 'std': [1.] * width,
                               'clip_min': [-5.] * width, 'clip_max': [5.] * width})
                result.append(native.TransformField(
                    prefix + '.' + name, width, mode, statistics, None, True))
            return tuple(result)
        embodiment = native.EmbodimentTransform(0, 14, 14, fields('state'), fields('action'))
        return types.SimpleNamespace(
            transform=types.SimpleNamespace(embodiment=embodiment),
            max_state_dim=14, max_action_dim=14)

    def test_checkpoint_owned_q99_and_mean_std_are_preserved(self):
        for modes in [('q99',), ('mean_std',)]:
            with self.subTest(modes=modes):
                bundle = self.bundle(modes)
                before = bundle.transform.embodiment.to_dict()
                _validate_joint_transform_bundle(bundle, {'arm_dim': [6, 6], 'ee_dim': [1, 1]})
                self.assertEqual(before, bundle.transform.embodiment.to_dict())

    def test_native_mixed_normalization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'consistently'):
            _validate_joint_transform_bundle(self.bundle(('q99', 'mean_std')),
                                             {'arm_dim': [6, 6], 'ee_dim': [1, 1]})

    def test_native_artifact_still_rejects_invalid_normalization(self):
        for mode, stats in [('unknown', {}), ('mean_std', {'mean': [0.]}),
                            ('mean_std', {'mean': [0.], 'std': [float('nan')]}),
                            ('q99', {'q01': [-1.], 'q99': [1., 2.]})]:
            with self.subTest(mode=mode, stats=stats), self.assertRaises(ValueError):
                native.TransformField('state.left_gripper', 1, mode, stats, None, True)

    def test_robot_layout_checks_still_apply_to_mean_std(self):
        bundle = self.bundle(('mean_std',))
        with self.assertRaises(ValueError):
            _validate_joint_transform_bundle(bundle, {'arm_dim': [7, 7], 'ee_dim': [1, 1]})
        bundle.max_action_dim = 13
        with self.assertRaisesRegex(ValueError, 'padded widths'):
            _validate_joint_transform_bundle(bundle, {'arm_dim': [6, 6], 'ee_dim': [1, 1]})
