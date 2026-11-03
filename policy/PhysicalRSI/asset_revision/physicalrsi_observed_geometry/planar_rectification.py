"""Project a declared workspace raster onto an observed support plane."""
import cv2
import numpy as np
from . import GeometryUnavailable


def rectify_support(image, camera, plane, *, shape=(370, 1000), origin=(-.5, .04), resolution=.001):
    plane=np.asarray(plane,float)
    if plane.shape!=(3,) or not np.isfinite(plane).all():
        raise GeometryUnavailable('Observed z=ax+by+c support plane required')
    if np.asarray(image).shape[:2]!=(camera.height,camera.width):
        raise GeometryUnavailable('Rectification image must match current calibration')
    yy,xx=np.indices(shape,dtype=float)
    x=origin[0]+xx*resolution;y=origin[1]-yy*resolution
    world=np.stack([x,y,plane[0]*x+plane[1]*y+plane[2]],axis=-1)
    pixels,_=camera.project(world.reshape(-1,3))
    pixels=pixels.reshape(*shape,2).astype(np.float32)
    return cv2.remap(image,pixels[:,:,0],pixels[:,:,1],cv2.INTER_LINEAR)
