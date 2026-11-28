import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import quat_matrix

def align(a,b):
    a=np.asarray(a)/np.linalg.norm(a);b=np.asarray(b)/np.linalg.norm(b)
    cross=np.cross(a,b);size=np.linalg.norm(cross);dot=float(np.clip(a@b,-1.,1.))
    if size<1e-9:
        if dot<0:raise ValueError('Axis sign ambiguous')
        return np.eye(3)
    return Rotation.from_rotvec(cross/size*np.arctan2(size,dot)).as_matrix()

def circle_normals(ellipse,k,prior_normal):
    center=np.array(ellipse[0]);axes=np.array(ellipse[1])/2
    if min(axes)<6:raise ValueError('Rim too small for normal recovery')
    ang=np.deg2rad(ellipse[2]);rot=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    theta=np.arccos(np.clip(abs(prior_normal[2]),0,1));phi=np.arctan2(prior_normal[1],prior_normal[0])
    ray=np.linalg.solve(k,[*center,1.]);angles=np.linspace(0,2*np.pi,64,endpoint=False);out=[]
    def shape(p):
        z=np.exp(p[2]);c=np.array([p[0]*z,p[1]*z,z])
        n=np.array([np.sin(p[3])*np.cos(p[4]),np.sin(p[3])*np.sin(p[4]),-np.cos(p[3])])
        a=np.cross(n,[0,1,0]);a/=np.linalg.norm(a);b=np.cross(n,a)
        return c,n,c+np.cos(angles)[:,None]*a+np.sin(angles)[:,None]*b
    def residual(p):
        c,n,points=shape(p);uv=points@k.T;uv=uv[:,:2]/uv[:,2:]
        return (np.linalg.norm((uv-center)@rot/axes,axis=1)-1)*np.mean(axes)
    for shift in [0,np.pi]:
        p0=[ray[0],ray[1],np.log(k[0,0]/max(axes)),max(theta,.01),phi+shift]
        fit=least_squares(residual,p0,bounds=([-2,-2,np.log(1.5),0,-20],[2,2,np.log(1000),np.pi/2-.001,20]),max_nfev=100)
        c,n,_=shape(fit.x);rms=float(np.sqrt(np.mean(fit.fun**2)))
        if not np.isfinite(np.r_[c,n,rms]).all() or rms>1.:continue
        out.append(dict(center_unit_radius=c,normal=n,rms=rms))
    if len(out)<2:raise ValueError('Conic branches unresolved')
    return out

class AxisWarmup(ValueError):
    pass

class HeldAxisTracker:
    def __init__(self,rim_tracker,upright):
        self.rim=rim_tracker
        self.initial_axis=upright.T@np.array([0.,0.,1.])
        self.cv_to_ee=self.rim.camera_to_ee[:3,:3]@self.rim.cv_to_gl
        prior=self.cv_to_ee.T@self.initial_axis
        if prior[2]>0:raise ValueError('Opening outward direction not observable')
        choices=circle_normals(self.rim.reference,self.rim.k,prior)
        costs=[float(np.arccos(np.clip(x['normal']@prior,-1,1))) for x in choices]
        order=np.argsort(costs)
        if costs[order[1]]-costs[order[0]]<np.deg2rad(20):raise ValueError('Initial normal branch ambiguous')
        self.reference_normal=choices[order[0]]['normal'];self.last_normal=self.reference_normal.copy()
        self.previous=None;self.accepted=0;self.last_observation_step=-1

    def observe(self,vision,state,side,anchor,source,current_axis,perception,observation_step):
        if observation_step<=self.last_observation_step:raise ValueError('Duplicate or out-of-order axis observation')
        self.last_observation_step=int(observation_step)
        try:return self._observe(vision,state,side,anchor,source,current_axis,perception)
        except AxisWarmup:raise
        except ValueError:
            self.previous=None
            raise

    def _observe(self,vision,state,side,anchor,source,current_axis,perception):
        choices=circle_normals(self.rim.ellipse,self.rim.k,self.last_normal)
        current=np.asarray(state[side+'_ee_pose']);r=quat_matrix(current[3:]);mouth=current[:3]+r@anchor
        candidates=[]
        for choice in choices:
            normal=choice['normal']
            delta=align(self.reference_normal,normal)
            axis=self.cv_to_ee@delta@self.cv_to_ee.T@self.initial_axis
            axis/=np.linalg.norm(axis)
            segment=perception.camera.project([mouth-r@axis*source['height'],mouth])
            candidates.append(dict(axis=axis,normal=normal,pixels=segment,rms=choice['rms']))
        if len(candidates)<2:raise ValueError('Normal branch continuity unavailable')
        rgb=vision['cam_head']['color'];masks=perception.masks(rgb,'bottle')
        nominal=perception.camera.project([mouth-r@current_axis*source['height'],mouth])
        segment_length=float(np.linalg.norm(nominal[1]-nominal[0]))
        if segment_length<25:raise ValueError('Head bottle axis foreshortened')
        observations=[]
        for mask,score,key in masks:
            yy,xx=np.where(mask>0)
            if len(xx)<80 or score<.5:continue
            points=np.c_[xx,yy].astype(float)
            if xx.min()<2 or yy.min()<2 or xx.max()>rgb.shape[1]-3 or yy.max()>rgb.shape[0]-3:continue
            center=points.mean(0);values,vectors=np.linalg.eigh(np.cov(points.T));direction=vectors[:,-1]
            axial=(points-center)@direction;length=np.quantile(axial,.98)-np.quantile(axial,.02)
            width=2*np.sqrt(max(values[0],0.))
            if length<.65*segment_length or length>1.5*segment_length or values[1]<2*values[0]:continue
            distance=float(np.linalg.norm(center-nominal.mean(0)))
            if distance>.4*segment_length:continue
            observations.append((distance,direction,length,width,float(score),key))
        if not observations:raise ValueError('Head silhouette incomplete or obscured')
        _,direction,length,width,score,key=min(observations,key=lambda x:x[0])
        tolerance=float(np.arctan2(max(2.,width*.25),length))
        for c in candidates:
            v=c['pixels'][1]-c['pixels'][0];v/=np.linalg.norm(v)
            c['head_error']=float(np.arccos(np.clip(abs(v@direction),0,1)))
        candidates.sort(key=lambda x:x['head_error']);best=candidates[0]
        if best['head_error']>tolerance or candidates[1]['head_error']-best['head_error']<2*tolerance:raise ValueError('Head view does not disambiguate the rim normal')
        if best['normal']@self.last_normal<np.cos(np.deg2rad(20)):raise ValueError('Normal change lacks temporal support')
        proposed=best['axis']
        if self.previous is None:
            self.previous=proposed.copy();raise AxisWarmup('Awaiting independent normal agreement')
        disagreement=float(np.arccos(np.clip(proposed@self.previous,-1,1)))
        self.previous=proposed.copy()
        if disagreement>2*tolerance:raise ValueError('Axis estimate unstable')
        rotation=align(current_axis,proposed);vector=Rotation.from_matrix(rotation).as_rotvec();angle=float(np.linalg.norm(vector))
        if angle>.015:vector*=.015/angle
        new_axis=Rotation.from_rotvec(vector).apply(current_axis)
        self.last_normal=best['normal'].copy();self.accepted+=1
        return new_axis,dict(head_error_rad=best['head_error'],head_tolerance_rad=tolerance,other_branch_error_rad=candidates[1]['head_error'],observed_axis=proposed.tolist(),previous_model_axis=np.asarray(current_axis).tolist(),increment_rad=min(angle,.015),head_sam_score=score,head_sam_request=key,normal_cv=best['normal'].tolist(),measurement='Relative unit-circle normal change, selected by current head silhouette; no absolute circle depth estimate',accepted_observations=self.accepted)
