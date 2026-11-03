"""Numerical reachability in the robot base inferred from current proprioception."""
from pathlib import Path
import json
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
model=pin.buildModelFromUrdf(str(Path(__file__).parent/'assets/X5A.urdf'))
data=model.createData();fid=model.getFrameId('link6')
from physicalrsi_observed_geometry.robot import base_from_observation
def fk(q):
 full=pin.neutral(model);full[:6]=q
 pin.framesForwardKinematics(model,data,full)
 return data.oMf[fid].copy()
def ik(arm,pose,previous,observation):
 base=np.asarray(base_from_observation(observation,arm,Path(__file__).parent/'assets/X5A.urdf'))
 target=pin.SE3(Rotation.from_quat(base[[4,5,6,3]]).as_matrix(),base[:3]).inverse()*pin.SE3(Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix(),np.asarray(pose[:3]))
 def residual(q):
  actual=fk(q)
  return np.r_[actual.translation-target.translation,pin.log3(actual.rotation.T@target.rotation)]
 seeds=[previous,np.zeros(6),np.array([0.,1.,-1.,0.,1.,0.])]
 candidates=[]
 for q in seeds:
  q=np.clip(q,model.lowerPositionLimit[:6]+1e-6,model.upperPositionLimit[:6]-1e-6)
  opt=least_squares(residual,q,bounds=(model.lowerPositionLimit[:6],model.upperPositionLimit[:6]),max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  r=residual(opt.x);pe=float(np.linalg.norm(r[:3]));ae=float(np.linalg.norm(r[3:]))
  candidates.append(dict(q=opt.x.tolist(),position_error_m=pe,rotation_error_rad=ae,nfev=opt.nfev,found=bool(pe<.003 and ae<np.deg2rad(2))))
  if candidates[-1]['found']:break
 return min(candidates,key=lambda r:r['position_error_m']+r['rotation_error_rad'])
