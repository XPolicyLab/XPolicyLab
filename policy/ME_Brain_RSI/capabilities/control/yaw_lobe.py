"""Public radius constrained planar circular end-cap observation from RGB."""
import numpy as np,cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
RADIUS=.0044

def observe(rgb,ee,hole,cam,K):
 T=np.eye(4);T[:3,3]=ee[:3];T[:3,:3]=Rotation.from_quat(np.roll(ee[3:],-1)).as_matrix();T=T@cam
 p=(np.linalg.inv(T)@np.r_[hole[:3,3],1])[:3]*[1,-1,-1]
 if p[2]<=.02:return None,dict(reason='behind_camera')
 uv=K@p;uv=uv[:2]/uv[2];u,v=np.rint(uv).astype(int);h,w=rgb.shape[:2]
 if not(0<=u<w and 0<=v<h):return None,dict(reason='outside_image')
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);y0,y1=max(0,v-50),min(h,v+51);x0,x1=max(0,u-50),min(w,u+51)
 threshold,local=cv2.threshold(gray[y0:y1,x0:x1],0,1,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
 m=np.zeros((h,w),np.uint8);m[y0:y1,x0:x1]=local
 n,l,stats,cents=cv2.connectedComponentsWithStats(m);ids=[i for i in range(1,n) if 25<stats[i,4]<2400 and 5<max(stats[i,2:4])<85 and np.linalg.norm(cents[i]-uv)<40]
 if not ids:return None,dict(reason='no_component')
 i=min(ids,key=lambda i:np.linalg.norm(cents[i]-uv));cts,_=cv2.findContours((l==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);px=cv2.convexHull(max(cts,key=len)).reshape(-1,2)
 rays=(np.c_[px,np.ones(len(px))]@np.linalg.inv(K).T)*[1,-1,-1];rays=rays@T[:3,:3].T
 if np.any(abs(rays[:,2])<.1):return None,dict(reason='grazing')
 xyz=T[:3,3]+rays*((hole[2,3]-T[2,3])/rays[:,2])[:,None];axis=hole[:2,0];axis=axis/np.linalg.norm(axis);cross=np.array([-axis[1],axis[0]]);basis=np.column_stack([axis,cross]);origin=hole[:2,3];xy=(xyz[:,:2]-origin)@basis;limit=float(xy[:,0].min()+RADIUS);arc=xy[xy[:,0]<=limit]
 if len(arc)<5:return None,dict(reason='few_rim_pixels',rim_pixels=len(arc))
 start=np.array([xy[:,0].min()+RADIUS,(arc[:,1].min()+arc[:,1].max())/2]);fit=least_squares(lambda c:(np.linalg.norm(arc-c,axis=1)-RADIUS)*1000,start,bounds=(start-.006,start+.006),loss='soft_l1',f_scale=.4,max_nfev=25)
 residual=np.linalg.norm(arc-fit.x,axis=1)-RADIUS;span=float(np.ptp(arc[:,1]));rms=float(np.sqrt(np.mean(residual**2)));angles=np.unwrap(np.arctan2((arc-fit.x)[:,1],(arc-fit.x)[:,0]));angular_span=float(np.ptp(angles));world=origin+basis@fit.x;shift=float(np.linalg.norm(world-origin));accepted=bool(span>.005 and rms<.0007 and angular_span>np.deg2rad(100) and shift<.020)
 point=np.r_[world,hole[2,3]];proj=(np.linalg.inv(T)@np.r_[point,1])[:3]*[1,-1,-1];projected=K@proj;projected=projected[:2]/projected[2]
 info=dict(local_threshold=float(threshold),accepted=accepted,reason='circular_rim' if accepted else 'rim_support_reject',rim_pixels=len(arc),transverse_span_m=span,radial_rms_m=rms,angular_span_deg=float(np.rad2deg(angular_span)),shift_m=shift,centre_uv=projected.tolist(),world=point.tolist(),prior_uv=uv.tolist())
 return (point if accepted else None),info
