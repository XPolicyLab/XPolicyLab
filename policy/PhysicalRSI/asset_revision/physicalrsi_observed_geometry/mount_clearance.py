"""Exclude the fixed robot mount footprint from its supporting collision slab.

Only robot base collision geometry defines the footprint. Moving robot links,
self-collision checks, and unrelated scene obstacles remain unchanged.
"""
from copy import deepcopy
import numpy as np


def clear_fixed_mount(scene, robot, *, support_name='table'):
    result=deepcopy(scene)
    box=result['cuboid'][support_name]
    dims=np.asarray(box['dims'],float);pose=np.asarray(box['pose'],float)
    if (dims.shape!=(3,) or pose.shape!=(7,) or not np.isfinite(dims).all()
            or not np.isfinite(pose).all() or np.any(dims<=0)
            or not np.allclose(np.abs(pose[3:]),[1,0,0,0],atol=1e-8)):
        raise ValueError('Axis-aligned support slab in robot frame required')
    kin=robot['robot_cfg']['kinematics']
    spheres=kin['collision_spheres'].get(kin.get('base_link','base_link'),[])
    low=pose[:3]-dims/2;high=pose[:3]+dims/2
    bounds=[]
    for sphere in spheres:
        center=np.asarray(sphere['center'],float);radius=float(sphere['radius'])
        if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:
            raise ValueError('Valid robot base collision spheres required')
        if center[2]-radius<high[2] and center[2]+radius>low[2]:
            bounds.append((center-radius,center+radius))
    if not bounds:return result
    minimum=np.min([b[0] for b in bounds],axis=0)
    maximum=np.max([b[1] for b in bounds],axis=0)
    x0,x1=max(low[0],minimum[0]),min(high[0],maximum[0])
    y0,y1=max(low[1],minimum[1]),min(high[1],maximum[1])
    if x1<=x0 or y1<=y0:return result
    regions=[(low[0],x0,low[1],high[1]),(x1,high[0],low[1],high[1]),
             (x0,x1,low[1],y0),(x0,x1,y1,high[1])]
    del result['cuboid'][support_name]
    for i,(a,b,c,d) in enumerate(regions):
        if b>a and d>c:
            name=f'{support_name}_mount_region_{i}'
            if name in result['cuboid']:raise ValueError('Mount region name collision')
            region=deepcopy(box)
            region.update(dims=[b-a,d-c,float(dims[2])],
                          pose=[(a+b)/2,(c+d)/2,float(pose[2]),*pose[3:].tolist()])
            result['cuboid'][name]=region
    return result
