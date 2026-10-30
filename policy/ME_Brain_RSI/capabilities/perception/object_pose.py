"""Metric planar object pose from same-instance RGB correspondences only."""
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from .vision_gate import camera,rotation,F
K=np.array([[F,0,320],[0,F,240],[0,0,1]],float)
CV_AXES=np.diag([1.,-1.,-1.])

def estimate(tracker,pose):
 points=np.ascontiguousarray(tracker.points[:,0],dtype=np.float64)
 if len(points)<4 or cv2.contourArea(cv2.convexHull(points.astype('float32')))<100:raise ValueError('pose_feature_geometry_insufficient')
 objects=np.column_stack([tracker.world[:,:2]-tracker.source,np.zeros(len(points))]).astype(np.float64)
 solved=cv2.solvePnPGeneric(objects,points,K,None,flags=cv2.SOLVEPNP_IPPE)
 if not solved[0]:raise ValueError('pose_ippe_unsolved')
 transform=camera(pose);basis=transform[:3,:3]@CV_AXES;candidates=[]
 for rv,tv in zip(solved[1],solved[2]):
  if not np.isfinite(rv).all() or not np.isfinite(tv).all():continue
  rc,_=cv2.Rodrigues(rv);tv=np.asarray(tv).ravel()
  if np.min((objects@rc.T+tv)[:,2])<=.01:continue
  delta=basis@rc
  if delta[2,2]<.2:continue
  projected=cv2.projectPoints(objects,rv,tv,K,None)[0][:,0];error=float(np.sqrt(np.mean(np.sum((projected-points)**2,axis=1))))
  if error>2.5:continue
  root=transform[:3,3]+basis@tv-delta@np.array([0,0,tracker.height/2])
  local=rotation(pose).T@(root-np.asarray(pose[:3]))
  contact_distance=float(np.linalg.norm(local-tracker.offset));bound=tracker.source_radius+tracker.uncertainty
  object_rotation=delta@Rotation.from_euler('z',tracker.initial_yaw,degrees=True).as_matrix()
  prior=rotation(pose)@tracker.object_relative if tracker.object_relative is not None else Rotation.from_euler('z',tracker.initial_yaw,degrees=True).as_matrix()
  angle=float(Rotation.from_matrix(prior.T@object_rotation).magnitude())
  if angle>.8:continue
  candidates.append(dict(score=error+3*angle,root=root,local=local,object_rotation=object_rotation,error_px=error,contact_distance_m=contact_distance,contact_bound_m=bound,attached=contact_distance<=bound,uncertainty_m=max(tracker.uncertainty,tracker.height/2+error*float(tv[2])/F)))
 if not candidates:raise ValueError('pose_geometry_rejected')
 return min(candidates,key=lambda p:p['score'])
