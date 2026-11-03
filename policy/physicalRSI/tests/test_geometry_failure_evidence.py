from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry import semantic_support


def test_failed_geometry_preserves_exact_camera_inputs_for_replay(monkeypatch,tmp_path):
    monkeypatch.setenv('PHYSICALRSI_GEOMETRY_EVIDENCE_DIR',str(tmp_path))
    vision={}
    for index,name in enumerate(['cam_head','cam_left_wrist','cam_right_wrist']):
        transform=np.eye(4);transform[:3,3]=[.3*index,-.2,1.7]
        vision[name]={'color':np.full((12,16,3),index*40,np.uint8),'intrinsic_matrix':np.array([[20.,0,8],[0,20.,6],[0,0,1]]),'extrinsic_matrix':transform,'excluded_field':'not camera input'}
    obs={'vision':vision,'state':{'unused':'not captured'}}
    failure=GeometryUnavailable('Missing calibrated support')
    def reject(*a):raise failure
    monkeypatch.setattr(semantic_support,'_observe_semantic_plane',reject)
    with pytest.raises(GeometryUnavailable) as caught:semantic_support.observe_semantic_plane(obs,None)
    assert caught.value is failure
    paths=list(tmp_path.glob('failure-*.npz'));assert len(paths)==1
    with np.load(paths[0],allow_pickle=False) as saved:
        assert len(saved.files)==10 and str(saved['error'])=='Missing calibrated support'
        restored={'vision':{}}
        for name in vision:
            restored['vision'][name]={field:saved[f'vision/{name}/{field}'] for field in ('color','intrinsic_matrix','extrinsic_matrix')}
            for field in restored['vision'][name]:np.testing.assert_array_equal(restored['vision'][name][field],vision[name][field])
            np.testing.assert_allclose(Camera.from_observation(restored,name).unproject([[8,6]],[1.]),Camera.from_observation(obs,name).unproject([[8,6]],[1.]))
