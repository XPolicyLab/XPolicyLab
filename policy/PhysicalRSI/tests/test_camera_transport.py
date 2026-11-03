from pathlib import Path
import importlib.util
import numpy as np


def test_batch_transport_retains_current_calibration_and_excludes_scene_truth():
    path = Path(__file__).resolve().parents[1] / 'asset_revision/overlays/transport_wire.py'
    spec = importlib.util.spec_from_file_location('observed_wire', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Backend:
        def update_obs_batch(self, observations):
            self.observations = observations

    backend = Backend()
    wire = module.Batch10Wire(backend)
    rgb = np.zeros((8, 8, 3), np.uint8)
    k, t = np.eye(3), np.eye(4)
    wire.update_obs_batch([{'env_idx': 2, 'state': {}, 'instruction': 'align blocks',
                           'layout_id': 100, 'asset_path': 'hidden.usd',
                           'vision': {'camera': {'color': rgb, 'intrinsic_matrix': k,
                                                'extrinsic_matrix': t, 'depth': np.ones((8, 8)),
                                                'segmentation_ids': np.zeros((8, 8))}}}])
    observed = backend.observations[0]
    assert set(observed) == {'env_idx', 'state', 'instruction', 'vision'}
    camera = observed['vision']['camera']
    assert set(camera) == {'color', 'intrinsic_matrix', 'extrinsic_matrix'}
    assert camera['color'] is rgb and camera['intrinsic_matrix'] is k and camera['extrinsic_matrix'] is t

    wire.update_obs_batch([dict(observed, env_idx=i) for i in range(20)])
    assert [row['env_idx'] for row in backend.observations] == list(range(20))
    assert all(row['vision']['camera']['intrinsic_matrix'] is k for row in backend.observations)
    assert all(row['vision']['camera']['extrinsic_matrix'] is t for row in backend.observations)
