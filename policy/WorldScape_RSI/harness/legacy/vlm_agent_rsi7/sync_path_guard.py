"""Private CPU FK guard for synchronized motion. Finite sampled checks, not a
continuous collision certificate or a tabletop-object model. Fail closed.
Uses robot joints, URDF and configured spheres only, never object poses.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
import yaml

TCP_MIN=.30
PAIR_MARGIN=.015
JOINT_GRID=.01
HAND_RADIUS=.065
HELD_RADIUS=.065
POLICY='sampled_fk_order_distance_spheres_v1'

def rot(axis,angle):
 a=np.asarray(axis,float);a=a/np.linalg.norm(a)
 x,y,z=a;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
 return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)

def transform(xyz=(0,0,0),rpy=(0,0,0)):
 t=np.eye(4);r,p,y=rpy;t[:3,:3]=rot([0,0,1],y)@rot([0,1,0],p)@rot([1,0,0],r);t[:3,3]=xyz
 return t

def pose_matrix(pose):
 a=np.asarray(pose,float);assert a.shape==(7,) and np.isfinite(a).all()
 w,x,y,z=a[3:]/np.linalg.norm(a[3:]);t=np.eye(4);t[:3,3]=a[:3]
 t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
 return t

def pose_error(a,b):
 d=np.linalg.norm(a[:3,3]-b[:3,3])
 ang=math.acos(float(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
 return float(d),float(ang)

class Model:
 def __init__(self,yml):
  kin=yaml.safe_load(Path(yml).read_text())['robot_cfg']['kinematics']
  root=ET.parse(kin['urdf_path']).getroot();self.base=kin['base_link'];self.ee=kin['tool_frames'][0]
  self.joints=[]
  for j in root.findall('joint'):
   o=j.find('origin');axis=j.find('axis')
   xyz=[float(v) for v in (o.get('xyz','0 0 0') if o is not None else '0 0 0').split()]
   rpy=[float(v) for v in (o.get('rpy','0 0 0') if o is not None else '0 0 0').split()]
   ax=[float(v) for v in (axis.get('xyz') if axis is not None else '1 0 0').split()]
   self.joints.append((j.get('name'),j.get('type'),j.find('parent').get('link'),j.find('child').get('link'),transform(xyz,rpy),ax))
  self.spheres=kin['collision_spheres']
  assert isinstance(self.spheres,dict) and self.spheres,'Missing sphere model'
 def fk(self,q,names,base):
  vals=dict(zip(names,q));out={self.base:base};todo=list(self.joints)
  for _ in range(len(todo)+1):
   if not todo:break
   progressed=False
   for j in todo[:]:
    name,kind,parent,child,origin,axis=j
    if parent not in out:continue
    # Finger joints are replaced by a conservative hand envelope, not modeled at guessed opening.
    if kind!='fixed' and name not in vals:
     if child in ('link7','link8'):todo.remove(j);continue
     raise ValueError('Unmapped moving joint '+name)
    motion=np.eye(4)
    if kind in ('revolute','continuous'):motion[:3,:3]=rot(axis,float(vals[name]))
    elif kind=='prismatic':motion[:3,3]=np.asarray(axis)*float(vals[name])
    elif kind!='fixed':raise ValueError('Unsupported joint '+kind)
    out[child]=out[parent]@origin@motion;todo.remove(j);progressed=True
   if not progressed and todo:raise ValueError('Unresolved URDF tree')
  assert self.ee in out
  return out
 def geometry(self,q,names,base,offset,closed):
  fk=self.fk(q,names,base);ee=fk[self.ee];tcp=ee[:3,3]+ee[:3,0]*offset
  spheres=[]
  for link,ls in self.spheres.items():
   t=fk[link]
   for s in ls:spheres.append([*(t[:3,:3]@np.asarray(s['center'])+t[:3,3]),float(s['radius'])])
  # Cover wrist-to-tip hand corridor including fingers with overlapping spheres.
  for d in np.linspace(0,offset,9):
   spheres.append([*(ee[:3,3]+ee[:3,0]*d),HAND_RADIUS])
  if closed:spheres.append([*tcp,HELD_RADIUS])
  return tcp,np.asarray(spheres),ee

def points_guard(left,right):
 l=np.asarray(left,float);r=np.asarray(right,float)
 if not np.isfinite([l,r]).all():return 'nonfinite_tcp'
 if l[0]>=r[0]:return 'left_right_order'
 if np.linalg.norm(l-r)<TCP_MIN:return 'tcp_distance'
 return None

def sphere_margin(l,r):
 delta=l[:,None,:3]-r[None,:,:3]
 return float(np.min(np.linalg.norm(delta,axis=-1)-l[:,None,3]-r[None,:,3]))

def check_samples(samples):
 """Pure checks for analytic fixtures; samples contain TCP and modeled spheres."""
 minimum=float('inf');dist=float('inf');count=0
 for s in samples:
  count+=1;reason=points_guard(s['left'][0],s['right'][0])
  if reason:return dict(ok=False,reason=reason,sample=count)
  dist=min(dist,float(np.linalg.norm(s['left'][0]-s['right'][0])))
  gap=sphere_margin(s['left'][1],s['right'][1]);minimum=min(minimum,gap)
  if gap<PAIR_MARGIN:return dict(ok=False,reason='modeled_cross_arm_sphere_overlap',sample=count,minimum_sphere_gap_m=gap)
 return dict(ok=bool(count),reason=None if count else 'empty_path',samples=count,minimum_tcp_distance_m=dist if count else None,minimum_sphere_gap_m=minimum if count else None)

def validate(r,env,poses,starts,actions,keys,targets,offset,grippers):
 audit=dict(policy=POLICY,continuous_collision_certified=False,tabletop_objects_checked=False,
            mesh_collision_checked=False,held_object_model='fixed conservative TCP envelope, no object pose',
            fallback='zero motion; inspect fresh state then explicit serial command',
            joint_grid_rad=JOINT_GRID,tcp_min_m=TCP_MIN,pair_margin_m=PAIR_MARGIN)
 try:
  assert actions and len(actions)<=10000
  robots=dict(r.target_arms(env));models={};bases={};names={}
  for a in ('left','right'):
   planner=env.robot_manager.planner[robots[a].robot_name]
   models[a]=Model(planner.yml_path);names[a]=list(planner.active_joints_name)
   bases[a]=pose_matrix(robots[a].entity_origin_pose)
   _,_,ee=models[a].geometry(starts[a],names[a],bases[a],offset,grippers.get(a,1)<.8)
   observed=pose_matrix([*poses[a][0],*poses[a][1]])
   dp,da=pose_error(ee,observed)
   if dp>.008 or da>.04:raise ValueError(f'{a} start FK mismatch {dp:.6f}m {da:.6f}rad')
   _,_,last=models[a].geometry(actions[-1][keys[a]],names[a],bases[a],offset,grippers.get(a,1)<.8)
   dp,da=pose_error(last,pose_matrix([*targets[a]['pos'],*targets[a]['quat']]))
   if dp>.008 or da>.04:raise ValueError(f'{a} goal FK mismatch {dp:.6f}m {da:.6f}rad')
  previous={a:np.asarray(starts[a],float) for a in names};samples=[]
  def sample(qs):
   return {a:models[a].geometry(qs[a],names[a],bases[a],offset,grippers.get(a,1)<.8)[:2] for a in names}
  samples.append(sample(previous))
  for act in actions:
   current={a:np.asarray(act[keys[a]],float) for a in names}
   assert all(current[a].shape==previous[a].shape and np.isfinite(current[a]).all() for a in names)
   n=max(1,int(math.ceil(max(float(np.max(np.abs(current[a]-previous[a]))) for a in names)/JOINT_GRID)))
   if len(samples)+n>20000:raise ValueError('dense path sample limit')
   for k in range(1,n+1):samples.append(sample({a:previous[a]+(current[a]-previous[a])*k/n for a in names}))
   previous=current
  audit.update(check_samples(samples))
 except Exception as exc:audit.update(ok=False,reason='guard_error',error=f'{type(exc).__name__}: {exc}')
 return audit

