"""Triangulate human-selected corresponding RGB pixels using camera calibration."""
import json, sys
import numpy as np

def ray(cal, uv):
    k=np.asarray(cal['intrinsic_matrix'],float)
    t=np.asarray(cal['extrinsic_matrix'],float).copy()
    t[:3,3]-=np.asarray(cal.get('env_origin',[0,0,0]))
    d=np.linalg.solve(k, [*uv,1])*[1,-1,-1]
    d=t[:3,:3]@d
    return t[:3,3], d/np.linalg.norm(d)

def triangulate(cal1,p1,cal2,p2):
    a,u=ray(cal1,p1); b,v=ray(cal2,p2)
    if abs(np.dot(u,v))>.995: raise ValueError('Insufficient stereo baseline')
    s,t=np.linalg.lstsq(np.column_stack([u,-v]),b-a,rcond=None)[0]
    if min(s,t)<=0: raise ValueError('Point behind camera')
    p=a+s*u; q=b+t*v
    gap=float(np.linalg.norm(p-q))
    if gap>.025: raise ValueError(f'Inconsistent correspondence ray gap {gap*100:.2f} cm')
    return {'point_cm':((p+q)*50).round(2).tolist(),'ray_gap_cm':round(gap*100,3),
            'source':'RGB correspondence + camera calibration; no object state'}

if __name__=='__main__':
    raw=json.load(open(sys.argv[1])); c=raw['calibration']
    p=list(map(float,sys.argv[2:6]))
    print(json.dumps(triangulate(c['cam_head'],p[:2],c['cam_right_wrist'],p[2:])))
