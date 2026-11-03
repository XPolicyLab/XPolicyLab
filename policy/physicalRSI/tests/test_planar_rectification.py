from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.planar_rectification import rectify_support


def test_current_calibration_and_sloped_support_determine_sampling():
    yy,xx=np.indices((80,100));image=np.stack([xx,yy,np.zeros_like(xx)],axis=-1).astype('uint8')
    transform=np.eye(4);transform[:3,3]=[.1,-.2,1.7]
    cam=Camera(image,[[60,0,50],[0,65,40],[0,0,1]],transform)
    plane=np.array([.1,-.05,.9])
    result=rectify_support(image,cam,plane,shape=(5,7),origin=(.08,-.18),resolution=.005)
    y,x=np.indices((5,7));world=np.stack([.08+x*.005,-.18-y*.005,np.zeros_like(x)],axis=-1)
    world[:,:,2]=world[:,:,0]*plane[0]+world[:,:,1]*plane[1]+plane[2]
    pixels,_=cam.project(world.reshape(-1,3))
    np.testing.assert_allclose(result[:,:,:2].reshape(-1,2),pixels,atol=.6)
    shifted=transform.copy();shifted[2,3]+=.23
    other=Camera(image,cam.k,shifted)
    np.testing.assert_array_equal(result,rectify_support(image,other,plane+[0,0,.23],shape=(5,7),origin=(.08,-.18),resolution=.005))
    with pytest.raises(GeometryUnavailable):rectify_support(image,cam,None)
