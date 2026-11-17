"""Wrist geometry comes from public robot calibration; all object evidence is RGB."""
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from ..rgb_reader.reader import project
D=np.load(Path(__file__).with_name('robot_geometry.npz'))
CAM=D['camera']; F=640*13/20.955

def rotation(pose):return Rotation.from_quat(np.asarray(pose[3:])[[1,2,3,0]]).as_matrix()
def camera(pose):
 t=np.eye(4);t[:3,:3]=rotation(pose);t[:3,3]=pose[:3];return t@CAM

def wrist_project(points,pose):
 t=camera(pose);p=(np.asarray(points)-t[:3,3])@t[:3,:3]
 z=-p[...,2];return np.stack([320+F*p[...,0]/z,240-F*p[...,1]/z],axis=-1)

def wrist_unproject(uv,pose,z):
 t=camera(pose);r=np.column_stack([(uv[:,0]-320)/F,-(uv[:,1]-240)/F,-np.ones(len(uv))])@t[:3,:3].T
 return t[:3,3]+r*((z-t[2,3])/r[:,2])[:,None]

def warp_table(rgb,pose):
 yy,xx=np.mgrid[:500,:1000];p=np.stack([xx/1000-.5,.12-yy/1000,np.full_like(xx,.770,dtype=float)],axis=-1)
 uv=wrist_project(p,pose)
 return cv2.remap(rgb,uv[...,0].astype('float32'),uv[...,1].astype('float32'),cv2.INTER_LINEAR)

class Fingers:
 def __init__(self):
  self.templates=[]
  for j in np.linspace(0,.044,45):
   pair=[]
   for name,sign in [('link7',1),('link8',-1)]:
    v=D[name].copy();v[:,1]+=sign*j
    p=(v-CAM[:3,3])@CAM[:3,:3]
    uv=np.column_stack([320+F*p[:,0]/-p[:,2],240-F*p[:,1]/-p[:,2]])
    hull=cv2.convexHull(np.round(np.clip(uv,-2000,2000)).astype('int32'))
    mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(mask,hull,255)
    edge=cv2.Canny(mask,50,100);edge[:8]=0;edge[-8:]=0;edge[:,:8]=0;edge[:,-8:]=0
    # Mask silhouettes include both tips and side walls; image border excluded.
    yy,xx=np.nonzero(edge);pair.append((xx,yy,mask))
   self.templates.append((float(j),pair))
 def fit(self,rgb):
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  edges=cv2.Canny(gray,55,130);dist=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
  best=[]
  for side in range(2):
   costs=[]
   for j,pair in self.templates:
    xx,yy,mask=pair[side]
    costs.append(float(np.mean(np.minimum(dist[yy,xx],15))) if len(xx) else 99)
   idx=int(np.argmin(costs));j,pair=self.templates[idx];xx,yy,mask=pair[side]
   best.append(dict(joint_rgb_m=j,edge_error_px=costs[idx],edge_support=float(np.mean(dist[yy,xx]<3)) if len(xx) else 0))
  return best
 def envelope(self,pose):
  r=rotation(pose);lowest=99
  for j in [0,.044]:
   for name in ['link6','link7','link8']:
    v=D[name].copy()
    if name!='link6':v[:,1]+=(1 if name=='link7' else -1)*j
    lowest=min(lowest,float((v@r.T)[:,2].min()))
  return float(pose[2]+lowest)
 def overlay(self,rgb,fits):
  im=rgb.copy()
  for side,f in enumerate(fits):
   idx=int(round(f['joint_rgb_m']/.001));xx,yy,_=self.templates[idx][1][side];im[yy,xx]=[0,255,0]
  return im

class Tracker:
 """RGB feature tracks, compared against static versus attached hypotheses."""
 def __init__(self,rgb,pose,source,height,operator,source_mask=None,appearance=None):
  self.z=.765+height;self.source=np.array(source);self.pose=np.array(pose)
  uv=wrist_project(np.array([*source,self.z]),pose)
  # Radius from static size; features themselves require visible piece foreground.
  rim=wrist_project(np.array([[source[0]+.035,source[1],self.z],[source[0],source[1]+.035,self.z]]),pose)
  radius=float(np.max(np.linalg.norm(rim-uv,axis=1)))
  yy,xx=np.mgrid[:480,:640];roi=(xx-uv[0])**2+(yy-uv[1])**2<radius**2
  hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
  fg=((hsv[:,:,1]<100)&(hsv[:,:,2]>150)) if operator else ((hsv[:,:,1]>65)&(hsv[:,:,2]>65))
  if not operator and appearance is not None:fg=appearance_mask(rgb,appearance)
  if source_mask is not None:
   world=wrist_unproject(np.column_stack([xx.ravel(),yy.ravel()]),pose,self.z)
   u=((world[:,0]+.5)*1000).reshape(xx.shape).astype('float32')
   v=((.12-world[:,1])*1000).reshape(xx.shape).astype('float32')
   interior=cv2.erode(np.asarray(source_mask,np.uint8),np.ones((5,5),np.uint8))
   observed_support=cv2.remap(interior,u,v,cv2.INTER_NEAREST)>0
   fg=fg&observed_support
  mask=(roi&fg).astype('uint8')*255
  self.mask=mask
  self.gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  self.points=cv2.goodFeaturesToTrack(self.gray,50,.015,4,mask=mask)
  if self.points is None or len(self.points)<4:raise ValueError('wrist_piece_features_insufficient')
  world=wrist_unproject(self.points[:,0,:],pose,self.z)
  localdist=np.linalg.norm(world[:,:2]-self.source,axis=1)
  keep=localdist<.034
  self.points=self.points[keep];self.world=world[keep]
  if len(self.points)<4:raise ValueError('wrist_piece_association_insufficient')
  self.initial_points=self.points.copy();self.n0=len(self.points)
  from .reference_tracks import ReferenceTracks
  self.reference_tracks=ReferenceTracks(rgb,mask)
  self.reference_pose=np.asarray(pose).copy()
  self.appearance=appearance;self.operator=operator
 def update(self,rgb,pose):
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  new,st,err=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,winSize=(21,21),maxLevel=3)
  if new is None:raise ValueError('wrist_tracking_lost')
  back,bst,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,new,None,winSize=(21,21),maxLevel=3)
  keep=(st[:,0]>0)&(bst[:,0]>0)&(np.linalg.norm(back[:,0]-self.points[:,0],axis=1)<1.5)&(err[:,0]<35)
  self.points=new[keep];self.world=self.world[keep];self.initial_points=self.initial_points[keep];self.gray=gray
  renewal=None;renewal_rejected=None
  if len(self.points)<max(8,int(np.ceil(self.n0*.35))):
   hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
   foreground=((hsv[:,:,1]<100)&(hsv[:,:,2]>150)) if self.operator else appearance_mask(rgb,self.appearance) if self.appearance is not None else ((hsv[:,:,1]>65)&(hsv[:,:,2]>65))
   try:
    ref,observed,audit=self.reference_tracks.match(rgb,foreground,(self.initial_points,self.points))
    if len(observed)>len(self.points):
     self.points=observed;self.initial_points=ref
     self.world=wrist_unproject(ref[:,0],self.reference_pose,self.z)
     renewal=audit
   except ValueError as e:renewal_rejected=str(e)
  if len(self.points)<4:raise ValueError('wrist_tracking_lost')
  now=self.points[:,0];static=wrist_project(self.world,pose)
  # Unchanged wrist-to-piece relation predicts constant pixels for translation.
  attached=self.initial_points[:,0]
  return dict(features=len(now),initial_features=self.n0,reference_renewal=renewal,reference_renewal_rejected=renewal_rejected,static_error_px=float(np.median(np.linalg.norm(now-static,axis=1))),attached_error_px=float(np.median(np.linalg.norm(now-attached,axis=1))),pixels=now.tolist())

def surface_evidence(head,wrist,pose,source,height,operator):
 """Two RGB views under static calibration, photometric height hypotheses.
 This is estimated surface height, never simulator depth. Retain uncertainty.
 """
 yy,xx=np.mgrid[:76,:76];x=source[0]+(xx-38)*.001;y=source[1]-(yy-38)*.001
 def feature(a):
  h=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
  return (((h[:,:,1]<95)&(h[:,:,2]>150)) if operator else ((h[:,:,1]>65)&(h[:,:,2]>65))).astype('uint8')
 scores=[]
 for thick in np.linspace(.002,.03,29):
  z=.765+thick;u,v=project(x,y,z)
  a=cv2.remap(head,u.astype('float32'),v.astype('float32'),cv2.INTER_LINEAR)
  uv=wrist_project(np.stack([x,y,np.full_like(x,z)],axis=-1),pose)
  b=cv2.remap(wrist,uv[:,:,0].astype('float32'),uv[:,:,1].astype('float32'),cv2.INTER_LINEAR)
  fa,fb=feature(a),feature(b)
  # Exclude the surrounding light pad/table and finger walls using central piece radius.
  roi=(xx-38)**2+(yy-38)**2<28**2
  fa=fa&roi;fb=fb&roi
  score=float(np.logical_and(fa,fb).sum()/max(1,np.logical_or(fa,fb).sum()))
  scores.append([float(thick),score])
 best=max(scores,key=lambda x:x[1]);supported=[h for h,s in scores if s>=best[1]-.08]
 return dict(estimated_top_thickness_m=best[0],iou=best[1],height_support_m=[min(supported),max(supported)],public_thickness_m=height,consistent=bool(best[1]>.3 and abs(best[0]-height)<.009),hypotheses=scores)

def appearance_model(rgb,mask):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 valid=(mask>0)&(hsv[:,:,1]>65)&(hsv[:,:,2]>65)
 hues=hsv[:,:,0][valid].astype(float)
 if not len(hues):raise ValueError('selected_instance_color_missing')
 center=int(np.bincount(hues.astype(int),minlength=180).argmax())
 distance=np.abs((hues-center+90)%180-90)
 return dict(hue=center,radius=float(np.quantile(distance,.95)+2),pixels=int(len(hues)))
def appearance_mask(rgb,appearance):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 distance=np.abs((hsv[:,:,0].astype(float)-appearance['hue']+90)%180-90)
 return (distance<=appearance['radius'])&(hsv[:,:,1]>65)&(hsv[:,:,2]>65)
