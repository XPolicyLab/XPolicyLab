import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .geometry import quat_matrix,bounded_anchor_pose
from .tool_surface import surfaces

SURFACES=surfaces()
SURFACE_ERROR=.003+np.sqrt(3)*.0015
SWEEP_ERROR=.002

class RimUpdateGuard:
    def __init__(self,source,grasp,upright,tcp,target,table_z):
        self.source=source;self.upright=upright;self.target=target;self.table_z=table_z
        finger=SURFACES['link7'];selected=np.abs(finger[:,0]-tcp[0])<=source['rim_radius']
        heights=grasp+((finger[selected]-tcp)@upright.T)[:,2]
        profile=np.asarray(source['profile']);valid=(profile[:,0]>=heights.min())&(profile[:,0]<=heights.max())
        width=2*float(profile[valid,1].max() if valid.any() else np.interp(grasp,profile[:,0],profile[:,1]))
        parts=[]
        for name,points in SURFACES.items():
            p=points.copy()
            if name=='link7':p[:,1]+=width/2
            if name=='link8':p[:,1]-=width/2
            parts.append(p)
        self.tool=np.concatenate(parts)
        theta=np.linspace(0,2*np.pi,48,endpoint=False)
        self.shell=np.concatenate([np.c_[r*np.cos(theta),r*np.sin(theta),np.full(len(theta),z-source['height'])] for z,r in profile])
        self.shell=np.vstack([self.shell,[0.,0.,-source['height']]])
        self.width=width

    def gap(self,pose,anchor,upright=None):
        r=quat_matrix(pose[3:]);tool=self.tool@r.T+pose[:3]
        body=(self.shell@(self.upright if upright is None else upright)+anchor)@r.T+pose[:3]
        t=self.target['measured'];xy=np.array(self.target['xy'])
        radius=t['rim_radius'];top=self.table_z+t['height']
        radial_uncertainty=float(np.linalg.norm(t.get('xy_sample_half_range',0.)))+t['fit_rms_mm']/1000
        height_uncertainty=t.get('height_sample_half_range',0.)
        def cup(points,margin):
            d=np.c_[np.abs(np.linalg.norm(points[:,:2]-xy,axis=1)-radius)-radial_uncertainty,np.abs(points[:,2]-top)-height_uncertainty]
            distance=np.linalg.norm(np.maximum(d,0.),axis=1)+np.minimum(np.max(d,axis=1),0.)
            return float(distance.min()-margin)
        return np.array([cup(tool,SURFACE_ERROR+SWEEP_ERROR),float(tool[:,2].min()-self.table_z-SURFACE_ERROR-SWEEP_ERROR),cup(body,SWEEP_ERROR),float(body[:,2].min()-self.table_z-SWEEP_ERROR)])

    def allow(self,current,old_anchor,new_anchor,poses,translation,rotation):
        old=poses[0];new=old.copy();new[:3]+=quat_matrix(old[3:])@(old_anchor-new_anchor)
        old_command=bounded_anchor_pose(current,old,old_anchor,translation,rotation)
        new_command=bounded_anchor_pose(current,new,new_anchor,translation,rotation)
        checks=[];r0=quat_matrix(current[3:]);r1=quat_matrix(old_command[3:])
        angle=Rotation.from_matrix(r0.T@r1).magnitude()
        reach=max(np.linalg.norm(self.tool,axis=1).max(),self.source['height'])
        count=max(2,int(np.ceil((max(np.linalg.norm(old_command[:3]-current[:3]),np.linalg.norm(new_command[:3]-current[:3]))+reach*angle)/SWEEP_ERROR))+1)
        fractions=np.linspace(0,1,count);rs=Slerp([0,1],Rotation.from_matrix([r0,r1]))(fractions).as_quat()
        for fraction,q in zip(fractions,rs):
            a=np.r_[current[:3]*(1-fraction)+old_command[:3]*fraction,q[3],q[:3]]
            b=np.r_[current[:3]*(1-fraction)+new_command[:3]*fraction,q[3],q[:3]]
            checks.append((self.gap(a,new_anchor),self.gap(b,new_anchor)))
        for pose in poses:
            revised=pose.copy();revised[:3]+=quat_matrix(pose[3:])@(old_anchor-new_anchor)
            checks.append((self.gap(pose,old_anchor),self.gap(revised,new_anchor)))
        before=np.asarray([x[0] for x in checks]);after=np.asarray([x[1] for x in checks])
        allowed=np.where(before>=0.,after>=0.,after>=before-1e-6)
        uncertain_rim=before[:,[0,2]]<0.
        allowed[:,[0,2]]=np.where(uncertain_rim,True,after[:,[0,2]]>=0.)
        return bool(allowed.all()),dict(allowed=bool(allowed.all()),before_min_m=before.min(0).tolist(),after_min_m=after.min(0).tolist(),samples=len(checks),uncertain_rim_samples=int(uncertain_rim.sum()),uncertain_rim_decrease_samples=int(((after[:,[0,2]]<before[:,[0,2]])&uncertain_rim).sum()),opening_width_m=self.width,obstacle_model='RGB measured rim ring uncertainty band and calibrated table; cup interior depth and occluded sidewall are unknown',scope='Approximate local command sweep and future anchor-waypoint endpoints relative to observed rim; positive rim separation retained; within measured rim uncertainty is unknown rather than proven contact; table risk may not worsen; no full collision-free guarantee')


    def allow_reorientation(self,current,anchor,old_poses,new_poses,new_upright,translation,rotation):
        old_command=bounded_anchor_pose(current,old_poses[0],anchor,translation,rotation)
        new_command=bounded_anchor_pose(current,new_poses[0],anchor,translation,rotation)
        r0=quat_matrix(current[3:]);ro=quat_matrix(old_command[3:]);rn=quat_matrix(new_command[3:])
        reach=max(np.linalg.norm(self.tool,axis=1).max(),self.source['height'])
        angle=max(Rotation.from_matrix(r0.T@ro).magnitude(),Rotation.from_matrix(r0.T@rn).magnitude())
        distance=max(np.linalg.norm(old_command[:3]-current[:3]),np.linalg.norm(new_command[:3]-current[:3]))
        count=max(2,int(np.ceil((distance+reach*angle)/SWEEP_ERROR))+1)
        fractions=np.linspace(0,1,count)
        old_rs=Slerp([0,1],Rotation.from_matrix([r0,ro]))(fractions)
        new_rs=Slerp([0,1],Rotation.from_matrix([r0,rn]))(fractions)
        m0=current[:3]+r0@anchor;mo=old_command[:3]+ro@anchor;mn=new_command[:3]+rn@anchor
        checks=[]
        for t,ra,rb in zip(fractions,old_rs,new_rs):
            qa=ra.as_quat();qb=rb.as_quat()
            a=np.r_[(1-t)*m0+t*mo-ra.as_matrix()@anchor,qa[3],qa[:3]]
            b=np.r_[(1-t)*m0+t*mn-rb.as_matrix()@anchor,qb[3],qb[:3]]
            checks.append((self.gap(a,anchor,new_upright),self.gap(b,anchor,new_upright)))
        for old,new in zip(old_poses,new_poses):checks.append((self.gap(old,anchor,self.upright),self.gap(new,anchor,new_upright)))
        before=np.array([x[0] for x in checks]);after=np.array([x[1] for x in checks])
        allowed=np.where(before>=0.,after>=0.,after>=before-1e-6)
        uncertain=before[:,[0,2]]<0.
        allowed[:,[0,2]]=np.where(uncertain,True,after[:,[0,2]]>=0.)
        return bool(allowed.all()),dict(allowed=bool(allowed.all()),before_min_m=before.min(0).tolist(),after_min_m=after.min(0).tolist(),samples=len(checks),uncertain_rim_samples=int(uncertain.sum()),uncertain_rim_decrease_samples=int(((after[:,[0,2]]<before[:,[0,2]])&uncertain).sum()),scope='Approximate immediate anchor-interpolated rotation sweep and future endpoints using observed held-axis proposal; measured rim/table only, occluded sidewall and interior unknown')
