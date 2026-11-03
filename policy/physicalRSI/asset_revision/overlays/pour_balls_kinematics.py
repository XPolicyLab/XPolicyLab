"""Public URDF FK and policy-side reachability scoring; no simulator imports."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

class Kinematics:
 def __init__(self):
  root=ET.parse(Path(__file__).with_name('X5A.urdf')).getroot();self.origins=[];self.axes=[];self.lower=[];self.upper=[]
  for i in range(1,7):
   j=root.find(f"joint[@name='joint{i}']");o=j.find('origin');t=np.eye(4);t[:3,3]=np.fromstring(o.get('xyz'),sep=' ');t[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix();self.origins.append(t);self.axes.append(np.fromstring(j.find('axis').get('xyz'),sep=' '));self.lower.append(float(j.find('limit').get('lower')));self.upper.append(float(j.find('limit').get('upper')))
  self.lower=np.array(self.lower);self.upper=np.array(self.upper)
  self.finger_hulls=dict(np.load(Path(__file__).with_name('finger_hulls.npz')));self.self_ignore=json.loads(Path(__file__).with_name('self_collision_ignore.json').read_text());self.spheres=json.loads(Path(__file__).with_name('collision_spheres.json').read_text());self.fixed=[]
  for joint in root.findall('joint'):
   if joint.get('type')!='fixed':continue
   o=joint.find('origin');t=np.eye(4)
   if o is not None:t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');t[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
   self.fixed.append((joint.find('parent').get('link'),joint.find('child').get('link'),t))
 def fk(self,q,jacobian=False):
  t=np.eye(4);positions=[];axes=[]
  for o,axis,angle in zip(self.origins,self.axes,q):
   t=t@o;positions.append(t[:3,3].copy());axes.append(t[:3,:3]@axis);rot=np.eye(4);rot[:3,:3]=R.from_rotvec(axis*angle).as_matrix();t=t@rot
  if not jacobian:return t
  axes=np.array(axes);j=np.r_[np.cross(axes,t[:3,3]-np.array(positions)).T,axes.T];return t,j
 def solve(self,target,q,position_tolerance=.003,rotation_tolerance=.025):
  q=np.array(q,float).copy();best=(1e9,q.copy(),None)
  for _ in range(70):
   t,j=self.fk(q,True);e=np.r_[target[:3,3]-t[:3,3],R.from_matrix(target[:3,:3]@t[:3,:3].T).as_rotvec()];d=np.linalg.norm(e[:3]);r=np.linalg.norm(e[3:]);score=d+.1*r
   if score<best[0]:best=(score,q.copy(),(d,r))
   if d<position_tolerance and r<rotation_tolerance:break
   e[3:]*=.2;j[3:]*=.2;delta=j.T@np.linalg.solve(j@j.T+np.eye(6)*.00008,e);delta*=min(1.,.22/max(np.max(np.abs(delta)),1e-9));q=np.clip(q+delta,self.lower+1e-5,self.upper-1e-5)
  return best[1],best[2]
 def world_spheres(self,q,base,exclude=()):
  t=base.copy();links={'base_link':t}
  for k,(o,axis,angle) in enumerate(zip(self.origins,self.axes,q)):
   rot=np.eye(4);rot[:3,:3]=R.from_rotvec(axis*angle).as_matrix();t=t@o@rot;links[f'link{k+1}']=t
  for parent,child,tf in self.fixed:
   if parent in links:links[child]=links[parent]@tf
  points=[];radii=[]
  for name,spheres in self.spheres.items():
   if name not in links or name in exclude:continue
   tf=links[name]
   for sphere in spheres:points.append(tf[:3,:3]@sphere['center']+tf[:3,3]);radii.append(sphere['radius'])
  return np.array(points),np.array(radii)
 def collision(self,q,base,tree):
  points,radii=self.world_spheres(q,base)
  return max(0.,float(np.max(radii+.006-tree.query(points)[0])))
 def tool_separation(self,ee,tree):
  links={'link6':ee}
  for parent,child,tf in self.fixed:
   if parent in links:links[child]=links[parent]@tf
  points=[];radii=[]
  for name,spheres in self.spheres.items():
   if name not in links:continue
   tf=links[name]
   for sphere in spheres:points.append(tf[:3,:3]@sphere['center']+tf[:3,3]);radii.append(sphere['radius'])
  return float(np.min(tree.query(points)[0]-np.array(radii)))
 def self_separation(self,q):
  t=np.eye(4);links={'base_link':t}
  for j,(o,axis,angle) in enumerate(zip(self.origins,self.axes,q)):
   rot=np.eye(4);rot[:3,:3]=R.from_rotvec(axis*angle).as_matrix();t=t@o@rot;links[f'link{j+1}']=t
  for parent,child,tf in self.fixed:
   if parent in links:links[child]=links[parent]@tf
  clouds={name:(np.array([links[name][:3,:3]@s['center']+links[name][:3,3] for s in spheres]),np.array([s['radius'] for s in spheres])) for name,spheres in self.spheres.items() if name in links}
  names=list(clouds);gap=1.
  for i,a in enumerate(names):
   for b in names[i+1:]:
    if b in self.self_ignore.get(a,[]) or a in self.self_ignore.get(b,[]):continue
    x,rx=clouds[a];y,ry=clouds[b];gap=min(gap,float((np.linalg.norm(x[:,None]-y[None,:],axis=2)-rx[:,None]-ry[None,:]).min()))
  return gap
 def finger_separation(self,ee,tree,aperture=None,distance_upper_bound=np.inf):
  widths=[0.,.022,.044] if aperture is None else [float(np.clip(aperture,0.,.044))]
  points=[]
  for width in widths:
   for name,sign in [('link7',1),('link8',-1)]:
    rel=self.finger_hulls[name]+[.08657,sign*(.024898+width),-.0002436]
    points.append(rel@ee[:3,:3].T+ee[:3,3])
  return float(tree.query(np.concatenate(points),distance_upper_bound=distance_upper_bound)[0].min())
 def table_separation(self,q,base,support_plane):
  points,radii=self.world_spheres(q,base,exclude=('base_link','link1'))
  normal=np.asarray(support_plane['normal']);offset=float(support_plane['offset']);gap=float(np.min(points@normal+offset-radii));ee=base@self.fk(q)
  for width in [0.,.044]:
   for name,sign in [('link7',1),('link8',-1)]:
    rel=self.finger_hulls[name]+[.08657,sign*(.024898+width),-.0002436]
    gap=min(gap,float(((rel@ee[:3,:3].T+ee[:3,3])@normal+offset).min()))
  return gap
 def rank(self,stages,base,q,tree,support_plane):
  inv=np.linalg.inv(base);res=[];travel=0.;penetrations=[];q=np.array(q,float)
  # Whole skill chain, including contact, transport, pour and return.
  for s in stages:
   if 'pose' not in s or s['name'] in ['close','release','relay_release','align_lip']:continue
   if s['name'].startswith('upright_') or s['name'].startswith('recover_'):continue
   p=np.array(s['pose']);target=np.eye(4);target[:3,3]=p[:3];target[:3,:3]=R.from_quat(p[[4,5,6,3]]).as_matrix();nq,(d,r)=self.solve(inv@target,q);travel+=np.linalg.norm(nq-q);q=nq;penetration=self.collision(q,base,tree) if tree is not None else 0.;penetrations.append(penetration);res.append({'stage':s['name'],'position_error':d,'rotation_error':r,'vase_penetration_m':penetration,'self_separation_m':self.self_separation(q),'table_separation_m':self.table_separation(q,base,support_plane),'finger_separation_m':self.finger_separation(base@self.fk(q),tree) if tree is not None else 1.})
  return max(x['position_error']+.1*x['rotation_error'] for x in res)+.0001*travel+3*max(penetrations)+4*max(0.,-min(x['self_separation_m'] for x in res))+3*max(0.,.015-min(x['finger_separation_m'] for x in res))+3*max(0.,.005-min(x['table_separation_m'] for x in res)),res
