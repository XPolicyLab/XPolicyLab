# Copyright © 2026 Martingale. All Rights Reserved.
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from martingale.hosts import robodojo_kinematics as kin
_DATA=json.loads(Path(__file__).with_name('robot_envelope_bounds.json').read_text())
_BOUNDS={name:np.asarray(value,float)for name,value in _DATA['bounds'].items()}
_BOUNDS['link7'][1,1]+=.044
_BOUNDS['link8'][0,1]-=.044


def fixed(pos,rpy=(0,0,0)):
    value=np.eye(4);value[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();value[:3,3]=pos;return value


def boxes(arm,q):
    frames={'base_link':kin._BASE_T[arm]}
    frames.update({'link'+str(i+1):f for i,f in enumerate(kin.frames(arm,q))})
    ee=frames['link6']
    frames.update(camera_base=ee@fixed([.057,0,0],[0,0,-3.1416]),
        link7=ee@fixed([.08657,.024896,-.0002436]),link8=ee@fixed([.08657,-.0249,-.00024366]))
    frames['camera']=frames['camera_base']@fixed([-.0275,0,.05],[0,.3491,np.pi])
    return [(name,T[:3,:3]@b.mean(axis=0)+T[:3,3],T[:3,:3],(b[1]-b[0])/2)
            for name,b in _BOUNDS.items()for T in [frames[name]]]


def separation(a,b):
    _,ca,ra,ha=a;_,cb,rb,hb=b
    crosses=np.cross(ra.T[:,None,:],rb.T[None,:,:]).reshape(-1,3)
    axes=np.concatenate((ra.T,rb.T,crosses));length=np.linalg.norm(axes,axis=1)
    axes=axes[length>1e-8]/length[length>1e-8,None]
    return float(np.max(np.abs(axes@(cb-ca))-np.abs(axes@ra)@ha-np.abs(axes@rb)@hb))


def screen(starts,trajectory,margin=.005):
    if not trajectory or not np.isfinite(margin)or margin<0:raise ValueError('Nonempty trajectory and nonnegative margin required')
    best=float('inf');closest=None
    for index,q in enumerate([starts,*trajectory]):
        for arm in ('left','right'):
            if np.asarray(q[arm]).shape!=(6,)or not np.isfinite(q[arm]).all():raise ValueError('Finite six-joint robot state required')
        for a in boxes('left',q['left']):
            for b in boxes('right',q['right']):
                gap=separation(a,b)
                if gap<best:best=gap;closest={'waypoint':index,'bodies':[a[0],b[0]]}
    return {'passed':best>margin,'minimum_separating_axis_gap_m':best,'required_margin_m':margin,
            'closest':closest,'sample_count':len(trajectory)+1,'full_finger_travel':True,
            'scope':'Sampled commanded own-robot inter-arm OBBs only; no scene/held-object or continuous-path certification'}
