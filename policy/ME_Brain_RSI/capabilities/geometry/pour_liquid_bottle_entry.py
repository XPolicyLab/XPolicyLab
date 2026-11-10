import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .geometry import quat_matrix


def envelope(source,upright):
    profile=np.asarray(source['profile'])
    profile=np.vstack([[0.,source['base_radius']],profile,[source['height'],source['rim_radius']]])
    z=np.linspace(0,source['height'],max(3,int(np.ceil(source['height']/.003))+1))
    radii=np.interp(z,profile[:,0],profile[:,1]);theta=np.linspace(0,2*np.pi,96,endpoint=False)
    shell=np.concatenate([np.c_[r*np.cos(theta),r*np.sin(theta),np.full(len(theta),h-source['height'])] for h,r in zip(z,radii)])
    return shell@upright


def required_entry_raise(start,end,offset,points,cup_xy,cup_radius,cup_top,table_top,entry_at_start):
    r0,r1=quat_matrix(start[3:]),quat_matrix(end[3:]);a0=start[:3]+r0@offset;a1=end[:3]+r1@offset
    angle=Rotation.from_matrix(r0.T@r1).magnitude();reach=np.linalg.norm(points,axis=1).max()
    n=max(2,int(np.ceil((np.linalg.norm(a1-a0)+reach*angle)/.003)))
    ts=np.linspace(0,1,n+1);rs=Slerp([0,1],Rotation.from_matrix([r0,r1]))(ts).as_matrix()
    dz=0.;unresolved=0.;minimum=None;at=None
    for u,r in zip(ts,rs):
        p=points@r.T+(1-u)*a0+u*a1
        inside=np.linalg.norm(p[:,:2]-cup_xy,axis=1)<cup_radius
        gap=float(p[inside,2].min()-cup_top) if inside.any() else float('inf')
        deficit=max(0.,-gap,float(table_top-p[:,2].min()))
        if minimum is None or gap<minimum:minimum=gap;at=float(u)
        weight=1-u if entry_at_start else u
        if weight>1e-8:dz=max(dz,deficit/weight)
        else:unresolved=max(unresolved,deficit)
    return dz,dict(samples=n+1,minimum_gap_m=minimum if np.isfinite(minimum) else None,minimum_fraction=at,unresolved_endpoint_m=unresolved)


def raise_entry(start,entry,next_pose,offset,source,target,table_z,clearance):
    entry=np.array(entry,float).copy();next_pose=np.array(next_pose,float).copy();original_z=float(entry[2]);original_next_z=float(next_pose[2]);points=envelope(source,quat_matrix(entry[3:]))
    m=target['measured'];radius=max(m['rim_radius'],m['base_radius'])+float(np.linalg.norm(m.get('xy_sample_half_range',0.)))
    xy=np.asarray(target['xy']);top=table_z+m['height']+clearance
    for pose in (entry,next_pose):
        r=quat_matrix(pose[3:]);mouth=pose[:3]+r@offset
        pose[2]+=max(0.,top-float(mouth[2]+(points@r.T)[:,2].min()))
    _,endpoint=required_entry_raise(entry,next_pose,offset,points,xy,radius,top,table_z+clearance,True)
    next_pose[2]+=endpoint['unresolved_endpoint_m']
    checks=[]
    for first in (True,False):
        a,b=(entry,next_pose) if first else (start,entry)
        dz,check=required_entry_raise(a,b,offset,points,xy,radius,top,table_z+clearance,first)
        entry[2]+=dz;checks.append(check)
    after=[]
    for a,b,first in ((start,entry,False),(entry,next_pose,True)):
        dz,check=required_entry_raise(a,b,offset,points,xy,radius,top,table_z+clearance,first);after.append(dict(additional_raise=dz,**check))
    return entry,next_pose,dict(method='measured_axial_bottle_anchor_slerp_entry_only',entry_raise_m=float(entry[2]-original_z),first_tilt_raise_m=float(next_pose[2]-original_next_z),cup_radius_m=float(radius),clearance_m=float(clearance),before=checks,after=after,approximate=True)
