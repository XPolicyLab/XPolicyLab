"""registered texture gate plus an independently confirmed public-CAD ring observation."""
from pathlib import Path
import json,numpy as np,cv2
from .vision_gate import Tracker,wrist_project,rotation
from .source_surface import initialize
from .contour import RingEvidence
from .silhouette import Silhouette
from .object_pose import estimate
class Relation(Tracker):
 def __init__(self,rgb,pose,source,height,operator,offset,uncertainty,appearance,source_mask,label,yaw_deg=0):
  super().__init__(rgb,pose,source,height,operator,source_mask,appearance)
  initialize(self,rgb,pose,source_mask)
  self.offset=np.asarray(offset);self.uncertainty=float(uncertainty)
  self.height=height;self.initial_yaw=yaw_deg;self.measured_offset=None;self.object_relative=None
  ys,xs=np.nonzero(source_mask);self.source_radius=float(np.max(np.hypot(xs/1000-.5-source[0],.12-ys/1000-source[1])))
  self.initial_root_z=.765+height/2
  uv=wrist_project(np.array([*source,self.z]),pose)
  rims=wrist_project(np.array([[source[0]+.035,source[1],self.z],[source[0],source[1]+.035,self.z]]),pose)
  self.pixel_bound=.10*float(np.max(np.linalg.norm(rims-uv,axis=1)))
  self.closed_pose=np.asarray(pose);self.streak=0;self.confirmed=False;self.texture_lost=False;self.ring=None;self.ring_initialization=None
  try:self.ring=RingEvidence(rgb,pose,source,self.z,self.source_surface,label,self.pixel_bound)
  except ValueError as e:self.ring_initialization=str(e)
  self.silhouette=None;self.silhouette_initialization=None
  if self.ring is not None:
   try:self.silhouette=Silhouette(self.ring,rgb,pose)
   except ValueError as e:self.silhouette_initialization=str(e)
 def update(self,rgb,pose):
  ring=None;previously_confirmed=self.ring is not None and self.ring.confirmed
  if self.ring is not None:
   try:ring=self.ring.update(rgb,pose)
   except ValueError as e:self.ring.streak=0;ring=dict(passed=False,rejected=str(e))
  texture=None
  try:
   if not self.texture_lost:texture=super().update(rgb,pose)
  except ValueError as e:
   if str(e)!='wrist_tracking_lost':raise
   self.texture_lost=True
  measured=None;pose_rejection=None
  if texture is not None:
   try:measured=estimate(self,pose)
   except (ValueError,cv2.error) as e:pose_rejection=str(e)
  silhouette=None;outer_confirmed=self.silhouette is not None and self.silhouette.confirmed
  if self.silhouette is not None:
   corroborated=(texture is not None and texture['attached_error_px']<self.pixel_bound) or (ring is not None and ring.get('passed',False))
   try:silhouette=self.silhouette.update(rgb,pose,corroborated)
   except ValueError as e:self.silhouette.streak=0;silhouette=dict(passed=False,rejected=str(e))
  if measured is not None:evidence={**texture,'attached_error_px':measured['error_px']};channel='RGB_planar_object_pose'
  elif texture is not None:evidence=texture;channel='texture'
  elif previously_confirmed and ring is not None and ring.get('passed'):
   evidence={k:ring[k] for k in ['attached_error_px','static_error_px']};channel='independent_CAD_ring'
  elif outer_confirmed and silhouette is not None and silhouette.get('passed'):
   evidence={k:silhouette[k] for k in ['attached_error_px','static_error_px']};channel='independent_outer_silhouette'
  else:raise ValueError('wrist_tracking_lost')
  self.attached=measured['attached'] if measured is not None else evidence['attached_error_px']<self.pixel_bound
  discriminated=evidence['static_error_px']>evidence['attached_error_px']+6
  if measured is not None:
   self.root=measured['root'];self.measured_offset=measured['local'];self.object_relative=rotation(pose).T@measured['object_rotation']
   self.uncertainty=max(self.uncertainty,measured['uncertainty_m'])
  else:self.root=np.asarray(pose[:3])+rotation(pose)@(self.measured_offset if self.measured_offset is not None else self.offset)
  lift=float(self.root[2]-self.initial_root_z)
  self.streak=self.streak+1 if self.attached and discriminated and lift>.012 else 0
  if self.streak>=2:self.confirmed=True
  object_rotation=rotation(pose)@self.object_relative if self.object_relative is not None else None
  return {**evidence,'object_rotation_matrix':object_rotation,'measured_offset_local':self.measured_offset,'pose_rejection':pose_rejection,'pose_contact_distance_m':measured['contact_distance_m'] if measured else None,'pose_contact_bound_m':measured['contact_bound_m'] if measured else None,'silhouette_evidence':silhouette,'silhouette_initialization_rejection':self.silhouette_initialization,'evidence_channel':channel,'ring_evidence':ring,'ring_initialization_rejection':self.ring_initialization,'texture_chain_lost':self.texture_lost,'attached':self.attached,'attached_bound_px':self.pixel_bound,'lift_m':lift,'streak':self.streak,'confirmed':self.confirmed,'estimated_root_xyz':self.root.tolist(),'relation_uncertainty_m':self.uncertainty}
