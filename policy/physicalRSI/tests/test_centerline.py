import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.centerline import centerline_cap

@pytest.mark.parametrize('axis',[[.2,.1,1],[-.3,.2,.8],[1,.2,.3]])
def test_unknown_axis_and_cap(axis):
 axis=np.array(axis,float);axis/=np.linalg.norm(axis);cap=np.array([.04,0,.8]);body=cap-.08*axis;views=[]
 for pos,angle in [([0,-.4,1.5],30),([.1,.3,1.4],-25)]:
  t=np.eye(4);t[:3,3]=pos;t[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix();c=Camera(np.zeros((480,640,3),np.uint8),[[600,0,320],[0,600,240],[0,0,1]],t);uv=c.project([cap,body])[0];views.append((c,uv[1],uv[0]-uv[1]))
 c=views[0][0];uv=c.project([cap,body])[0];fit=centerline_cap(views,c,*uv)
 np.testing.assert_allclose(fit['cap'],cap,atol=1e-8);np.testing.assert_allclose(fit['axis_to_cap'],axis,atol=1e-8)
 with pytest.raises(GeometryUnavailable,match='separation'):centerline_cap([views[0],views[0]],c,*uv)
