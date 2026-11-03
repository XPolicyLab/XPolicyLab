from pathlib import Path
from types import SimpleNamespace
import runpy
import numpy as np
import pytest
from test_multiview_circle import views_for
from physicalrsi_observed_geometry import GeometryUnavailable

fit=runpy.run_path(str(Path(__file__).resolve().parents[1]/'asset_revision/overlays/pour_perception.py'))['observed_liquid_cup_opening']

@pytest.mark.parametrize('center,radius',[([0,0,.8],.023),([.08,.02,.9],.042)])
def test_cup_opening_uses_observed_views_not_known_dimensions(center,radius):
    views=views_for(center,radius);obs={'vision':{}}
    lookup={}
    for name,(camera,mask) in zip(['cam_head','cam_right_wrist'],views):
        obs['vision'][name]={'color':camera.image,'intrinsic_matrix':camera.k,'extrinsic_matrix':camera.transform}
        lookup[id(camera.image)]=mask
    def segment(image,prompts,**kwargs):
        assert prompts==[('cup opening',.2),('inside of cup',.2)]
        return [SimpleNamespace(masks=[lookup[id(image)]])]
    result=fit(obs,views[0][1],[0,0,.6],segment)
    np.testing.assert_allclose(result['center'],center,atol=.002)
    assert abs(result['radius']-radius)<.0015
    obs['vision'].pop('cam_right_wrist')
    with pytest.raises(GeometryUnavailable,match='second view'):fit(obs,views[0][1],[0,0,.6],segment)
    with pytest.raises(GeometryUnavailable,match='selected container'):fit(obs,np.zeros_like(views[0][1]),[0,0,.6],segment)
