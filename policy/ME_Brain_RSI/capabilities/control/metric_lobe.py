"""Complete RGB silhouette negative extent and symmetry recover circular-lobe centre."""
import numpy as np,cv2
from scipy.spatial.transform import Rotation

def observe(rgb,ee,hole,cam,K):
 T=np.eye(4);T[:3,3]=ee[:3];T[:3,:3]=Rotation.from_quat(np.roll(ee[3:],-1)).as_matrix();T=T@cam
 spacing=.00025;coordinates=(np.arange(281)-140)*spacing;xx,yy=np.meshgrid(coordinates,coordinates);world=np.stack([xx+hole[0,3],yy+hole[1,3],np.full_like(xx,hole[2,3]),np.ones_like(xx)],-1)
 camera=(world@np.linalg.inv(T).T)[...,:3]*[1,-1,-1];proj=camera@K.T;uv=proj[...,:2]/proj[...,2:];valid=(camera[...,2]>.02)&(uv[...,0]>0)&(uv[...,0]<rgb.shape[1]-1)&(uv[...,1]>0)&(uv[...,1]<rgb.shape[0]-1)
 if not valid.all():return None,dict(accepted=False,reason='plane_roi_outside_camera')
 plane=cv2.remap(rgb,uv[...,0].astype('float32'),uv[...,1].astype('float32'),cv2.INTER_LINEAR);gray=cv2.cvtColor(plane,cv2.COLOR_RGB2GRAY);threshold,m=cv2.threshold(gray,0,1,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
 n,l,st,ce=cv2.connectedComponentsWithStats(m);ids=[i for i in range(1,n) if 400<st[i,4]<3500 and np.linalg.norm(ce[i]-[140,140])<80 and st[i,0]>0 and st[i,1]>0 and st[i,0]+st[i,2]<281 and st[i,1]+st[i,3]<281]
 if not ids:return None,dict(accepted=False,reason='no_complete_component')
 i=min(ids,key=lambda i:np.linalg.norm(ce[i]-[140,140]));component=(l==i).astype('uint8');cts,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);fill=np.zeros_like(m);cv2.drawContours(fill,cts,-1,1,-1);y,x=np.nonzero(fill);points=np.c_[coordinates[x],coordinates[y]];centroid=points.mean(0);points-=centroid;_,sv,vt=np.linalg.svd(points,full_matrices=False);direction=vt[0];prior=hole[:2,0]
 if direction@prior<0:direction=-direction
 normal=np.array([-direction[1],direction[0]]);length=float(np.ptp(points@direction));width=float(np.ptp(points@normal));area=len(points)*spacing**2;ratio=float(sv[0]/sv[1]);angle=float(np.rad2deg(np.arctan2(np.linalg.det(np.stack([prior,direction])),prior@direction)))
 accepted=bool(.015<length<.022 and .0065<width<.011 and .00008<area<.00016 and 1.4<ratio<3 and abs(angle)<30)
 longitudinal=points@direction;transverse=points@normal;offset=centroid+direction*(float(longitudinal.min())+.0044)+normal*(float(transverse.min()+transverse.max())/2);point=np.r_[hole[:2,3]+offset,hole[2,3]];shift=float(np.linalg.norm(offset));accepted=accepted and shift<.020
 return (point if accepted else None),dict(accepted=bool(accepted),world=point.tolist(),shift_m=shift,reason='complete_metric_lobe_extent' if accepted else 'geometry_reject',yaw_delta_deg=angle,length_m=length,width_m=width,area_m2=area,axis_ratio=ratio,threshold=threshold,direction=direction.tolist())
