"""Public RGB metric reconstruction with unknown vessel dimensions."""
import cv2
import numpy as np
from scipy.optimize import least_squares

class GeometryObservationError(ValueError):
 pass

class CalibratedRGBView:
 def __init__(self,image,pose,focal,exclusion=None):
  self.image=np.asarray(image);self.pose=np.asarray(pose,float);self.focal=float(focal)
  if self.image.shape!=(480,640,3) or self.image.dtype!=np.uint8 or self.pose.shape!=(4,4) or not np.isfinite(self.pose).all():raise GeometryObservationError('Only calibrated current public RGB')
  self.exclusion=np.zeros((480,640),bool) if exclusion is None else np.asarray(exclusion,bool)
  grey=cv2.GaussianBlur(cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY),(0,0),.8)
  edges=cv2.Canny(grey,30,85)>0
  self.edge_distance=cv2.distanceTransform((~edges).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
 def project(self,points):
  p=(np.asarray(points)-self.pose[:3,3])@self.pose[:3,:3];valid=p[...,2]<-1e-5
  uv=np.stack([319.5+self.focal*p[...,0]/np.maximum(-p[...,2],1e-8),239.5-self.focal*p[...,1]/np.maximum(-p[...,2],1e-8)],-1)
  valid&=(uv[...,0]>=0)&(uv[...,0]<640)&(uv[...,1]>=0)&(uv[...,1]<480)
  u=np.clip(np.rint(uv[...,0]).astype(int),0,639);v=np.clip(np.rint(uv[...,1]).astype(int),0,479)
  return uv,valid

def bilinear(image,uv):
 u=np.clip(uv[:,0],0,638.999);v=np.clip(uv[:,1],0,478.999);x=np.floor(u).astype(int);y=np.floor(v).astype(int);a=u-x;b=v-y
 return (1-a)*(1-b)*image[y,x]+a*(1-b)*image[y,x+1]+(1-a)*b*image[y+1,x]+a*b*image[y+1,x+1]

def ring_points(x,y,z,r,count=72):
 t=np.linspace(0,2*np.pi,count,endpoint=False);return np.c_[x+r*np.cos(t),y+r*np.sin(t),np.full(count,z)]

def fit_horizontal_rim(views,xy,radius,table_z):
 if len(views)<2:raise GeometryObservationError('Single-view rim height/scale is ambiguous')
 xy=np.asarray(xy,float);radius=float(radius);selected=[]
 for view in views:
  uv,visible=view.project(ring_points(*xy,table_z+.06,radius))
  # A mostly clipped ring cannot provide dependable geometric residuals.
  if visible.sum()>=58:
   u=np.clip(np.rint(uv[:,0]).astype(int),0,639);v=np.clip(np.rint(uv[:,1]).astype(int),0,479)
   supported=visible & ~view.exclusion[v,u]
   if supported.sum()>=24:selected.append((view,np.where(supported)[0]))
 if len(selected)<2:raise GeometryObservationError('Need an independent visible rim arc; request new view')
 if max(np.linalg.norm(a[0].pose[:3,3]-b[0].pose[:3,3]) for a in selected for b in selected)<.025:raise GeometryObservationError('Metric baseline unavailable')
 lower=np.r_[xy-.05,table_z+.001,max(.006,radius*.55)];upper=np.r_[xy+.05,table_z+.12,min(.16,radius*1.6)]
 def residual(parameters):
  points=ring_points(*parameters);errors=[]
  for view,indices in selected:
   uv,visible=view.project(points);u=np.clip(np.rint(uv[:,0]).astype(int),0,639);v=np.clip(np.rint(uv[:,1]).astype(int),0,479);visible&=~view.exclusion[v,u];dist=bilinear(view.edge_distance,uv);mask=visible[indices];errors.extend(np.where(mask,dist[indices],0.)/np.sqrt(max(.2,float(mask.mean()))));errors.append(max(0.,.5-float(mask.mean()))*20.)
  return np.asarray(errors)
 candidates=[]
 for height in np.linspace(.005,.105,5):
  x0=np.r_[xy,table_z+height,radius]
  fit=least_squares(residual,np.clip(x0,lower+1e-6,upper-1e-6),bounds=(lower,upper),loss='soft_l1',f_scale=1.2,x_scale=[.03,.03,.06,.03],max_nfev=130)
  err=residual(fit.x);candidates.append((float(np.mean(np.minimum(err,8)**2)),fit,err))
 _,fit,err=min(candidates,key=lambda x:x[0]);rank=np.linalg.matrix_rank(fit.jac,tol=1e-4);cov=np.linalg.pinv(fit.jac.T@fit.jac)*max(.64,float(np.mean(err**2)));sigma=np.sqrt(np.maximum(0,np.diag(cov)));rms=float(np.sqrt(np.mean(err**2)))
 if rank<4 or rms>2.7 or sigma[2]>.012 or max(sigma[:2])>.008:raise GeometryObservationError('Rim evidence is inconsistent or metrically ambiguous')
 return dict(xy=fit.x[:2].tolist(),rim_height=float(fit.x[2]-table_z),radius=float(fit.x[3]),pixel_rms=rms,metric_sigma=sigma.tolist(),views=len(selected),dimensions_from_RGB=True)

def membership(view, mask, points):
 uv,visible=view.project(points);u=np.clip(np.rint(uv[...,0]).astype(int),0,639);v=np.clip(np.rint(uv[...,1]).astype(int),0,479)
 inside=mask[v,u]>0
 # Positive observed colour overrides a conservative predicted own-robot box.
 visible &= ~view.exclusion[v,u] | inside
 return inside,visible

def reconstruct_bottle(views, masks, colour, table_z, plane):

 if len(views)<2:raise GeometryObservationError('Need at least two calibrated views')
 hm=masks[0][colour];ys,xs=np.nonzero(hm)
 if len(xs)<100:raise GeometryObservationError('No head silhouette')
 sample=np.array([[xs.min(),ys.max()],[xs.max(),ys.max()],[xs.min(),ys.min()],[xs.max(),ys.min()]])
 bounds=np.concatenate([plane(sample,table_z+z)[:,:2] for z in [.02,.11]])
 lo=bounds.min(0)-.015;hi=bounds.max(0)+.015;pitch=.0015
 gx=np.arange(lo[0],hi[0]+pitch,pitch);gy=np.arange(lo[1],hi[1]+pitch,pitch);xx,yy=np.meshgrid(gx,gy);centres=[];radii=[]
 for z in [.025,.045,.065,.085,.105]:
  points=np.stack([xx,yy,np.full(xx.shape,table_z+z)],-1);good=np.ones(xx.shape,bool);second=np.zeros(xx.shape,bool)
  for i,(view,ms) in enumerate(zip(views,masks)):
   inside,visible=membership(view,ms[colour],points)
   good&=~visible|inside
   if i==0:good&=visible
   else:second|=visible&inside
  good&=second
  dt=cv2.distanceTransform(good.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);y,x=np.unravel_index(dt.argmax(),dt.shape)
  if dt[y,x]*pitch>.006:centres.append([gx[x],gy[y]]);radii.append(float(dt[y,x]*pitch))
 if len(centres)<3:raise GeometryObservationError('Insufficient two-view body support')
 xy=np.median(centres,axis=0);spread=float(np.max(np.linalg.norm(np.array(centres)-xy,axis=1)))
 if spread>.012:raise GeometryObservationError('Cross-section axes disagree across heights')
 angles=np.linspace(0,2*np.pi,48,endpoint=False);cs=np.c_[np.cos(angles),np.sin(angles)]
 profile=[]
 def fits(z,r):
  points=np.c_[xy[0]+r*cs[:,0],xy[1]+r*cs[:,1],np.full(len(cs),table_z+z)]
  for i,(view,ms) in enumerate(zip(views,masks)):
   inside,visible=membership(view,ms[colour],points)
   if i==0 and visible.sum()<40:return False
   if visible.sum()>=6 and np.mean(inside[visible])<.96:return False
  return True
 for z in np.arange(.006,.40,.002):
  if not fits(z,0):profile.append([float(z),0]);continue
  low,high=0.,.10
  for _ in range(11):
   mid=(low+high)/2
   if fits(z,mid):low=mid
   else:high=mid
  profile.append([float(z),low])
 profile=np.array(profile);positive=profile[profile[:,1]>.004]
 if len(positive)<15:raise GeometryObservationError('Incomplete observed vertical profile')
 apparent=float(positive[-1,0]);upper=positive[(positive[:,0]>.65*apparent)&(positive[:,0]<.90*apparent)];neck=float(np.median(upper[:,1]));height=float(positive[positive[:,1]>=.90*neck][-1,0]);body=float(np.quantile(profile[(profile[:,0]>.02)&(profile[:,0]<.55*height),1],.9));start=float(profile[(profile[:,0]>.5*height)&(profile[:,0]<height)&(profile[:,1]<=1.25*neck)&(profile[:,1]>=.8*neck)][0,0]);grasp=start+.4*(height-start)
 return dict(xy=xy.tolist(),height=height,body_radius=body,neck_radius=neck,neck_start=start,grasp_height=grasp,body_axis_spread=spread,body_slice_centres=centres,profile=np.vstack([[0.,float(positive[0,1])],profile[profile[:,0]<=height]]).tolist(),height_interval=[height-.004,height+.004],dimensions_from_RGB=True)
