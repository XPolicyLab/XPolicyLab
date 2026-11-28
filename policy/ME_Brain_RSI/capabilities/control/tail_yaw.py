"""Current settled wrist RGB gives aperture extension direction, not object identity."""
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from .yaw_lobe import observe as lobe
from .hole_servo import REL,K

def observe(rgb,ee,hole):
 point,info=lobe(rgb,ee,hole,REL,K)
 if point is None:return None,dict(accepted=False,reason='lobe_unavailable',lobe=info)
 T=np.eye(4);T[:3,3]=ee[:3];T[:3,:3]=Rotation.from_quat(np.roll(ee[3:],-1)).as_matrix();T=T@REL
 uv=np.array(info['prior_uv']);u,v=np.rint(uv).astype(int);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape
 mask=np.zeros((h,w),np.uint8);y0,y1=max(0,v-50),min(h,v+51);x0,x1=max(0,u-50),min(w,u+51)
 _,local=cv2.threshold(gray[y0:y1,x0:x1],0,1,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU);mask[y0:y1,x0:x1]=local
 n,l,st,ce=cv2.connectedComponentsWithStats(mask);ids=[j for j in range(1,n) if 25<st[j,4]<2400 and 5<max(st[j,2:4])<85 and np.linalg.norm(ce[j]-uv)<40]
 if not ids:return None,dict(accepted=False,reason='component_unavailable')
 j=min(ids,key=lambda j:np.linalg.norm(ce[j]-uv));yy,xx=np.nonzero(l==j)
 rays=(np.c_[xx,yy,np.ones(len(xx))]@np.linalg.inv(K).T)*[1,-1,-1];rays=rays@T[:3,:3].T
 if np.any(abs(rays[:,2])<.1):return None,dict(accepted=False,reason='grazing')
 xyz=T[:3,3]+rays*((hole[2,3]-T[2,3])/rays[:,2])[:,None];xy=xyz[:,:2]-point[:2];rad=np.linalg.norm(xy,axis=1);prior=hole[:2,0];tail=xy[(rad>.006)&(rad<.016)&(xy@prior>0)]
 if len(tail)<12:return None,dict(accepted=False,reason='tail_pixels',pixels=len(tail))
 _,sv,vt=np.linalg.svd(tail-tail.mean(0),full_matrices=False);direction=vt[0]
 if direction@prior<0:direction=-direction
 cross=np.array([-direction[1],direction[0]]);angle=float(np.arctan2(np.linalg.det(np.stack([prior,direction])),prior@direction));ratio=float(sv[0]/max(sv[1],1e-9));length=float(np.ptp(tail@direction));width=float(np.ptp(tail@cross))
 accepted=bool(ratio>1.2 and .006<length<.012 and .003<width<.008 and abs(angle)<np.deg2rad(30))
 record=dict(accepted=accepted,reason='wrist_tail_direction' if accepted else 'tail_geometry_reject',pixels=len(tail),axis_ratio=ratio,yaw_delta_deg=float(np.rad2deg(angle)),tail_length_m=length,tail_width_m=width,lobe_world=point.tolist(),direction=direction.tolist())
 rotation=np.column_stack([np.r_[direction,0.],np.r_[cross,0.],[0.,0.,1.]])
 return (rotation if accepted else None),record
