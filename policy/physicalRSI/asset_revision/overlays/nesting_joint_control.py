"""Policy-owned differential IK: public URDF, six legal proprio fields, no env imports."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from physicalrsi_observed_geometry.robot import base_from_observation

class JointControl:
 def __init__(self):
  self.urdf=Path(__file__).parent/'X5A.urdf'
  tree=ET.parse(self.urdf).getroot();self.chain=[];self.checked=set();self.bases={}
  for name in ['joint1','joint2','joint3','joint4','joint5','joint6']:
   j=tree.find(f"joint[@name='{name}']");o=j.find('origin');limit=j.find('limit')
   self.chain.append((np.fromstring(o.get('xyz'),sep=' '),Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix(),np.fromstring(j.find('axis').get('xyz'),sep=' '),float(limit.get('lower')),float(limit.get('upper'))))
 def reset(self):
  self.bases.clear()
 def bind(self,state,joints):
  observation={'state':dict(state,**{arm+'_arm_joint_state':joints[arm] for arm in ['left','right']})}
  for arm in ['left','right']:
   if arm not in self.bases:
    pose=np.asarray(base_from_observation(observation,arm,self.urdf))
    self.bases[arm]=(pose[:3],Rotation.from_quat(pose[[4,5,6,3]]).as_matrix())
 def base(self,arm):
  if arm not in self.bases:raise ValueError('Current episode robot base has not been observed')
  p,r=self.bases[arm]
  return p.copy(),r.copy()
 def fk(self,q,arm):
  p,rot=self.base(arm)
  for v,(t,r,axis,_,_) in zip(q,self.chain):p=p+rot@t;rot=rot@r@Rotation.from_rotvec(axis*v).as_matrix()
  return p,Rotation.from_matrix(rot)
 def differential_fk(self,q,arm):
  p,rot=self.base(arm);origins=[];axes=[]
  for v,(t,r,axis,_,_) in zip(q,self.chain):
   p=p+rot@t;rot=rot@r;origins.append(p.copy());axes.append(rot@axis);rot=rot@Rotation.from_rotvec(axis*v).as_matrix()
  axes=np.asarray(axes).T
  return p,rot,np.cross(axes.T,p-np.asarray(origins)).T,axes
 def solve(self,target,q,arm):
  target=np.asarray(target,float);q=np.asarray(q,float);tr=Rotation.from_quat(np.roll(target[3:],-1))
  cache={}
  def evaluate(x):
   if 'x' in cache and np.array_equal(x,cache['x']):return cache['residual'],cache['jacobian']
   p,r,jp,jr=self.differential_fk(x,arm);phi=Rotation.from_matrix(tr.as_matrix().T@r).as_rotvec();theta=np.linalg.norm(phi)
   a,b,c=phi;skew=np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])
   coefficient=1/12+theta*theta/720 if theta<1e-4 else (1-.5*theta/np.tan(.5*theta))/(theta*theta)
   inverse_left=np.eye(3)-.5*skew+coefficient*(skew@skew)
   residual=np.r_[p-target[:3],phi*.12,(x-q)*.001]
   jacobian=np.vstack([jp,.12*inverse_left@tr.as_matrix().T@jr,np.eye(6)*.001])
   cache.update(x=x.copy(),residual=residual,jacobian=jacobian)
   return residual,jacobian
  def residual(x):
   return evaluate(x)[0]
  lo=np.maximum(q-.16,[x[3] for x in self.chain]);hi=np.minimum(q+.16,[x[4] for x in self.chain]);initial=np.clip(q,lo+1e-8,hi-1e-8)
  fit=least_squares(residual,initial,jac=lambda x:evaluate(x)[1],bounds=(lo,hi),max_nfev=28,ftol=1e-6,xtol=1e-6,gtol=1e-6)
  p,r=self.fk(fit.x,arm)
  return fit.x,{'position_error_m':float(np.linalg.norm(p-target[:3])),'rotation_error_rad':float((tr.inv()*r).magnitude()),'max_joint_delta':float(np.max(np.abs(fit.x-q))),'nfev':fit.nfev}
 def solve_empty_position(self,target,q,arm,tip_floor):
  target=np.asarray(target,float);q=np.asarray(q,float)
  def residual(x):
   p,_,_,_=self.differential_fk(x,arm)
   return np.r_[p-target[:3],(x-q)*.001]
  def jacobian(x):return np.vstack([self.differential_fk(x,arm)[2],np.eye(6)*.001])
  lo=np.maximum(q-.16,[x[3] for x in self.chain]);hi=np.minimum(q+.16,[x[4] for x in self.chain])
  fit=least_squares(residual,np.clip(q,lo+1e-8,hi-1e-8),jac=jacobian,bounds=(lo,hi),max_nfev=28,ftol=1e-6,xtol=1e-6,gtol=1e-6)
  tips=[]
  for alpha in np.linspace(0,1,5):
   p,r=self.fk(q+alpha*(fit.x-q),arm);tips.append(p+r.apply([.13257000232458113,0,0]))
  safe=bool(min(t[2] for t in tips)>=tip_floor)
  p,r=self.fk(fit.x,arm)
  return (fit.x if safe else None),{'position_error_m':float(np.linalg.norm(p-target[:3])),'max_joint_delta':float(np.max(np.abs(fit.x-q))),'minimum_interpolated_tip_z':float(min(t[2] for t in tips)),'tip_floor':float(tip_floor),'clearance_valid':safe,'control':'released_empty_position','nfev':fit.nfev}
 def convert(self,poses,state,joints,empty_clearance=None):
  self.bind(state,joints)
  output={};diagnostics={}
  for arm in ['left','right']:
   q=joints[arm];actual=np.asarray(state[arm+'_ee_pose']);fk,r=self.fk(q,arm)
   fkerror=float(np.linalg.norm(fk-actual[:3]));rerror=float((r.inv()*Rotation.from_quat(np.roll(actual[3:],-1))).magnitude())
   if fkerror>.008 or rerror>.035:
    # Relevant ABI mismatch: preserve current pose; never use an uncalibrated joint controller.
    target=q;diag={'fk_mismatch':True,'fk_error_m':fkerror,'fk_rotation_error_rad':rerror}
   else:
    if empty_clearance is not None and arm in empty_clearance:
     target,diag=self.solve_empty_position(poses[arm+'_ee_pose'],q,arm,empty_clearance[arm])
     if target is None:
      clearance_diag=diag;target,diag=self.solve(poses[arm+'_ee_pose'],q,arm);diag['empty_clearance_rejected']=clearance_diag
    else:target,diag=self.solve(poses[arm+'_ee_pose'],q,arm)
    diag.update(fk_error_m=fkerror,fk_rotation_error_rad=rerror)
   output[arm+'_arm_joint_state']=target.tolist();output[arm+'_ee_joint_state']=list(poses[arm+'_ee_joint_state']);diagnostics[arm]=diag
  return output,diagnostics
