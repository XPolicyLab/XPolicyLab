"""The liquid bridge must get bottle extent from RGB rather than learned depth."""
from pathlib import Path
import runpy
from types import SimpleNamespace
import numpy as np
import pytest
from test_observed_radial_profile import camera, PROFILE
from physicalrsi_observed_geometry.radial_profile import render_profile
from physicalrsi_observed_geometry import GeometryUnavailable

fit_bottle=runpy.run_path(str(Path(__file__).resolve().parents[1]/'asset_revision/overlays/pour_perception.py'))['observed_bottle_fit']

@pytest.mark.parametrize('scale',[.7,1.2])
def test_bridge_preserves_observed_height_and_world_support(scale):
    cam=camera()
    mask=render_profile(PROFILE*scale,np.eye(3),[0,-.1,.8],cam,(480,640))
    # No depth or surface method: bottle shape must not depend on depth scale.
    scene=SimpleNamespace(camera=cam,mask=lambda m:np.asarray(m,bool),depth_calibration_report={'observed_plane':[0,0,.8]})
    fit=fit_bottle(mask,scene)
    assert abs(np.ptp(fit['points_local'][:,2])-.22*scale)<.02*scale
    assert fit['pose'][2,3]==pytest.approx(.8)
    assert fit['error_px']<4 and fit['iou']>.8
    assert fit['hidden_geometry_verified'] is False
    with pytest.raises(GeometryUnavailable):fit_bottle(np.zeros_like(mask),scene)
