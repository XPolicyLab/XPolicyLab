"""Load current robot/perception resources without a GPU or model weights."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


class RuntimeResources(unittest.TestCase):
    def test_current_task_resource_tables_point_to_matching_files(self):
        for mode in ('fold_clothes', 'fold_clothes_random'):
            data = json.loads((ROOT / 'tasks' / mode / 'resources.json').read_text())
            for relative, record in data['files'].items():
                with self.subTest(task=mode, resource=relative):
                    self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), record['sha256'])
        data = json.loads((ROOT / 'tasks/insert_tubes/bindings/runtime-lock.json').read_text())
        for relative, expected in data['robot_resources'].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_robot_planner_resolves_the_contained_urdf_and_original_joint_commands(self):
        from tasks.classify_objects_by_language.skills.own_joint_planner.implementation import OriginalOwnJointPlanner
        root = ROOT / 'assets/robots/x5'
        settings = yaml.safe_load((root / 'curobo.yml').read_text())
        self.assertEqual((root / settings['robot_cfg']['kinematics']['urdf_path']).resolve(), root / 'X5A.urdf')
        options = []
        class Planner:
            def __init__(self, **kwargs): options.append(kwargs)
            def plan_joint(self, current, goal):
                positions = np.stack([current, goal])
                return dict(status='Success', position=positions, velocity=np.zeros_like(positions))
        planner = OriginalOwnJointPlanner(root, planner_factory=Planner)
        current, goal = np.zeros(6), np.full(6, .1)
        result = planner('left', current, goal)
        np.testing.assert_array_equal(result['position'], np.stack([current, goal]))
        self.assertEqual(options[0]['yml_path'], str(root / 'curobo.yml'))
        self.assertEqual(options[0]['dt'], .004)

    def test_public_wrist_geometry_loads_the_current_robot_description(self):
        from skills.shared.geometry.public_wrist_view.implementation import own_kinematics, gripper_joint_envelope
        from skills.shared.geometry.garment_camera_projection.implementation import wrist_optical_from_link6
        self.assertEqual(own_kinematics().forward(np.zeros(6)).shape, (4, 4))
        self.assertTrue(np.isfinite(gripper_joint_envelope()).all())
        transform = wrist_optical_from_link6()
        self.assertEqual(transform.shape, (4, 4))
        self.assertTrue(np.isfinite(transform).all())

    def test_garment_templates_load_through_the_actual_perception_skill(self):
        from skills.shared.perception.current_garment_landmarks.implementation import public_shapes
        metadata = json.loads((ROOT / 'assets/perception/garments/metadata.json').read_text())
        shapes = public_shapes()
        self.assertEqual(set(shapes), set(metadata))
        self.assertTrue(shapes)
        for name, shape in shapes.items():
            self.assertEqual(shape['landmarks'], metadata[name]['landmarks_public_local_m'])
            self.assertEqual(shape['outline'].shape[1], 2)
            self.assertTrue(np.isfinite(shape['outline']).all())

    def test_sam_resource_relocation_preserves_every_declared_model_member(self):
        from rsi.resource_locations import load_manifest, reference_for
        source = ROOT / 'configs/models/sam3.json'
        declared = json.loads(source.read_text())
        self.assertEqual(declared['path'], 'resource://models/sam3-robodojo-v21')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                RIEMANN_RESOURCE_ROOTS=json.dumps({'models': directory})):
            actual = load_manifest(source)
            self.assertEqual(actual['path'], str(Path(directory).resolve() / 'sam3-robodojo-v21'))
            self.assertEqual(reference_for(actual['path']), declared['path'])
            self.assertEqual(actual['files'], declared['files'])


if __name__ == '__main__': unittest.main()
