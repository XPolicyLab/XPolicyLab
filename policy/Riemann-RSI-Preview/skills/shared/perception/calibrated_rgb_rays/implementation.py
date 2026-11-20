"""Calibrated image-index rays; continuous optical centres include half pixels."""
import numpy as np
from scipy.spatial.transform import Rotation
class Camera:
 def __init__(self,calibration):
  self.origin=np.asarray(calibration['head_origin_m'],float)
  self.R=Rotation.from_euler('xyz',calibration['head_euler_deg'],degrees=True).as_matrix()
  self.focal=float(calibration['head_focal_pixels'])
 def project(self,points):
  p=np.asarray(points,float);local=(p-self.origin)@self.R
  if np.any(local[...,2]>=-1e-6):raise ValueError('Projection behind the declared public camera')
  return np.stack([319.5+self.focal*local[...,0]/-local[...,2],239.5-self.focal*local[...,1]/-local[...,2]],axis=-1)
 def plane(self,pixels,height):
  p=np.asarray(pixels,float);r=np.stack([(p[...,0]-319.5)/self.focal,-(p[...,1]-239.5)/self.focal,-np.ones(p.shape[:-1])],axis=-1)@self.R.T
  if np.any(np.abs(r[...,2])<1e-7):raise ValueError('Public image ray parallel to plane')
  d=(float(height)-self.origin[2])/r[...,2]
  if np.any(d<=0):raise ValueError('Public plane hypothesis behind camera')
  return self.origin+d[...,None]*r
