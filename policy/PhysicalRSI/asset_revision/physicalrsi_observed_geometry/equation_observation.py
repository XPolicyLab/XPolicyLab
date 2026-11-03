"""Per-observation geometry context for glyphs with no supplied object heights."""
import cv2
import numpy as np
from . import Camera,GeometryUnavailable
from .support_observation import observe_support_geometry
from .rgb_support import correspondences
from .object_height import measured_height


class EquationObservation:
    def __init__(self,observation,segment_fn):
        if observation is None:raise GeometryUnavailable('Current calibrated observation required for equation geometry')
        self.cameras=[Camera.from_observation(observation,'cam_head')]+[
            Camera.from_observation(observation,n) for n in observation['vision'] if n!='cam_head']
        self.plane=observe_support_geometry(observation,segment_fn,
            occluders=('robot arm','number-shaped object','wooden circular pad','operator disc'))['plane']
        self.matches={id(c):correspondences(self.cameras[0],c) for c in self.cameras[1:]}

    def rectification(self,x,y):
        z=self.plane[0]*x+self.plane[1]*y+self.plane[2]
        uv,_=self.cameras[0].project(np.stack([x,y,z],axis=-1).reshape(-1,3))
        return uv[:,0].reshape(x.shape),uv[:,1].reshape(y.shape)

    def height(self,rectified_mask,u,v):
        camera=self.cameras[0]
        selection=np.asarray(rectified_mask,bool)&np.isfinite(u)&np.isfinite(v)
        x=np.rint(u[selection]).astype(int);y=np.rint(v[selection]).astype(int)
        inside=(x>=0)&(x<camera.width)&(y>=0)&(y<camera.height)
        mask=np.zeros(camera.image.shape[:2],'uint8');mask[y[inside],x[inside]]=1
        mask=cv2.dilate(mask,np.ones((3,3),'uint8')).astype(bool)
        return measured_height(self.cameras,mask,self.plane,matcher=lambda a,b:self.matches[id(b)])
