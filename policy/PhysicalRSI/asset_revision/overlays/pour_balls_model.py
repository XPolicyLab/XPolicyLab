import os,json,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing
import numpy as np,cv2
cv2.setNumThreads(1)
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.spatial import cKDTree
from XPolicyLab.model_template import ModelTemplate
from .perception import registration,pose,matrix,PixelOcclusion,observed_cup_fit,observed_opening_fit,opening_in_frame
from physicalrsi_observed_geometry import Scene, Camera, GeometryUnavailable
from .sam.sam3_http import segment_many_http
from .kinematics import Kinematics
from .return_path import plan_return,bounded_joint_step,clearance

KEYS=['left_ee_pose','right_ee_pose','left_ee_joint_state','right_ee_joint_state']
def copy(s):return {k:np.asarray(s[k],float).reshape(-1).tolist() for k in KEYS}
def err(a,b):return np.linalg.norm(np.array(a[:3])-b[:3]),(R.from_quat(np.array(a)[[4,5,6,3]]).inv()*R.from_quat(np.array(b)[[4,5,6,3]])).magnitude()
def bounded(a,b):
 d,r=err(a,b);f=min(1.,.012/max(d,1e-9),.085/max(r,1e-9));q=Slerp([0,1],R.from_quat(np.array([a,b])[:,[4,5,6,3]]))([f]).as_quat()[0]
 return np.r_[np.array(a[:3])+f*(np.array(b[:3])-a[:3]),q[[3,0,1,2]]].tolist()

def mouth_geometry(points,frame=None):
 if frame is not None:points=(points-frame[:3,3])@frame[:3,:3]
 top=float(points[:,2].max());rim=points[points[:,2]>top-.003];xy=(rim[:,:2].min(0)+rim[:,:2].max(0))/2
 radius=float(np.quantile(np.linalg.norm(rim[:,:2]-xy,axis=1),.10))
 center=np.r_[xy,top]
 if frame is not None:center=frame[:3,:3]@center+frame[:3,3]
 return center,max(.003,min(.008,.25*radius))

def aperture_geometry(points,frame=None,opening=None):
 if opening is not None:
  center=np.asarray(opening['center']).copy()
  if frame is not None:center=frame[:3,:3]@center+frame[:3,3]
  return center,float(opening['radius'])
 local=points if frame is None else (points-frame[:3,3])@frame[:3,:3]
 top=float(local[:,2].max());sl=local[local[:,2]>top-.008]
 xy=(sl[:,:2].min(0)+sl[:,:2].max(0))/2
 inner=float(np.quantile(np.linalg.norm(sl[:,:2]-xy,axis=1),.10))
 center=np.r_[xy,top]
 if frame is not None:center=frame[:3,:3]@center+frame[:3,3]
 return center,max(.012,inner)

def wall_cloud(points,frame,inner,opening=None):
 if opening is not None:
  center=np.asarray(opening['center']);normal=np.asarray(opening['normal']);normal=normal/np.linalg.norm(normal)
  local=(points-frame[:3,3])@frame[:3,:3];delta=local-center;axial=delta@normal;radial=delta-axial[:,None]*normal
  keep=~((abs(axial)<.012)&(np.linalg.norm(radial,axis=1)<inner))
  ref=np.eye(3)[np.argmin(abs(normal))];x=np.cross(normal,ref);x/=np.linalg.norm(x);y=np.cross(normal,x)
  angle=np.linspace(0,2*np.pi,180,endpoint=False);rim=center+inner*(np.cos(angle)[:,None]*x+np.sin(angle)[:,None]*y)
  return np.concatenate([points[keep],rim@frame[:3,:3].T+frame[:3,3]])
 local=(points-frame[:3,3])@frame[:3,:3]
 top=float(local[:,2].max());center,_=aperture_geometry(points,frame,opening)
 xy=(np.linalg.inv(frame)@np.r_[center,1])[:2]
 r=np.linalg.norm(local[:,:2]-xy,axis=1)
 opening=(local[:,2]>top-.012)&(r<inner+.006)
 return points[~opening]

_candidate_planner=None
def evaluate_candidate(fits,state,folder,allow_transfer,direction,symmetry):
 global _candidate_planner
 if _candidate_planner is None:
  _candidate_planner=Model.__new__(Model);_candidate_planner.kin=Kinematics()
 start=time.perf_counter();plan=_candidate_planner.trajectory(fits,{'state':state},folder,allow_transfer,direction,symmetry)
 arm=plan['arm'];q=np.asarray(state[arm+'_arm_joint_state'],float).reshape(-1)[:6]
 base=matrix(np.asarray(state[arm+'_ee_pose']).reshape(-1))@np.linalg.inv(_candidate_planner.kin.fk(q))
 vf=fits['vase'];points=vf['points_local']@vf['pose'][:3,:3].T+vf['pose'][:3,3]
 _,inner=aperture_geometry(points,vf['pose'],vf.get('opening_local'));walls=wall_cloud(points,vf['pose'],inner,vf.get('opening_local'))
 score,residuals=_candidate_planner.kin.rank(plan['stages'],base,q,cKDTree(walls if len(walls) else points),fits['_support_plane'])
 return score,plan,direction,symmetry,residuals,time.perf_counter()-start

class Model(ModelTemplate):
 def __init__(self,model_cfg):
  self.log_lock=threading.Lock();self.log_pending=[]
  self.kin=Kinematics();self.pool=ThreadPoolExecutor(max_workers=2)
  self.candidate_pool=ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn'))
  self.out=Path(os.environ['BALLS_TRACE']);self.out.mkdir(parents=True,exist_ok=True);self.round=-1;self.reset()
 def set_trace_directory(self,case_meta=None):
  self.out=Path(case_meta['trace_directory']);self.out.mkdir(parents=True,exist_ok=True)
 def reset(self):self.slots={};self.obs={};self.round+=1
 def log(self,**d):
  line=json.dumps(d,default=lambda a:a.tolist())+'\n'
  with self.log_lock:self.log_pending.append(line)
 def flush_log(self):
  with self.log_lock:
   lines=self.log_pending;self.log_pending=[]
  if lines:
   with (self.out/'events.jsonl').open('a') as f:f.writelines(lines)
 def initial(self,o,allow_transfer=True,retry=None,cup_observation=None):
  start=time.perf_counter()
  scene=Scene.from_observation(o)
  i=o['env_idx'];folder=self.out/(f'batch{self.round}_env{i}'+('' if allow_transfer else '_handover')+('_retry' if retry else ''));folder.mkdir(exist_ok=True);rgb=np.asarray(o['vision']['cam_head']['color'],np.uint8)[:,:,:3];cv2.imwrite(str(folder/'initial.png'),rgb[:,:,::-1])
  sam_start=time.perf_counter();raw=segment_many_http(rgb,[('yellow cup',.2),('vase',.2),('black vase',.2),('turquoise vase',.2),('white vase',.2),('white bottle',.2),('table surface',.2)],timeout_s=600);sam_seconds=time.perf_counter()-sam_start;fits={}
  cup_masks=list(raw[0].masks);vase_masks=[m for group in raw[1:-1] for m in group.masks if not any((m&cm).sum()/max(1,m.sum())>.5 for cm in cup_masks)]
  support=np.zeros(rgb.shape[:2],bool)
  for mask in raw[-1].masks:support |= mask
  for mask in cup_masks+vase_masks:support &= ~cv2.dilate(mask.astype('uint8'),np.ones((7,7),'uint8')).astype(bool)
  fits['_support_plane']=scene.support_plane(support)
  groups=[cup_masks,vase_masks]
  for label,g in zip(['cup','vase'],groups):
   if label=='cup' and cup_observation is not None:
    fits['cup']=cup_observation['fit'];selected_cup_mask=cup_observation['mask']
    cv2.imwrite(str(folder/'cup_confirmed.png'),selected_cup_mask.astype('uint8')*255)
    continue
   candidates=[]
   for j,m in enumerate(g):
    cv2.imwrite(str(folder/f'{label}_{j}.png'),m.astype('uint8')*255)
    if m.sum()<50:continue
    if label=='cup':
     try:fit=observed_cup_fit(m,scene,fits['_support_plane'])
     except GeometryUnavailable:
      # Non-frustum observations retain the generic RGB/depth estimator.
      fit=registration(m,label,scene=scene)
    else:fit=registration(m,label,scene=scene)
    candidates.append(dict(fit,mask_index=j))
   if not candidates:raise ValueError(label+' no mask')
   fit=min(candidates,key=lambda x:x['error_px']);fits[label]=fit
   if fit['error_px']>6:raise ValueError(label+' silhouette error '+str(fit['error_px']))
   if label=='cup':selected_cup_mask=cup_masks[fit['mask_index']]
  if retry is not None:
   reference=np.asarray(retry['cup_mask'],bool);current=np.asarray(selected_cup_mask,bool)
   overlap=float((reference&current).sum()/max(1,(reference|current).sum()))
   centroid_motion=float(np.linalg.norm(np.argwhere(reference).mean(0)-np.argwhere(current).mean(0)))
   cup=fits['cup']['points_local']@fits['cup']['pose'][:3,:3].T+fits['cup']['pose'][:3,3]
   center=(cup.min(0)+cup.max(0))/2;center_shift=float(np.linalg.norm(center-retry['cup_center']))
   self.log(event='approach_retry_rgb_support_consistency',env=i,mask_iou=overlap,mask_centroid_motion_px=centroid_motion,center_shift_m=center_shift,evidence='same-episode initial supported silhouette; contact not independently measured')
   if overlap<.60 or centroid_motion>4. or center_shift>.012:
    raise ValueError('retry initial supported-cup RGB consistency not confirmed')
  try:
   opening=observed_opening_fit(o,vase_masks[fits['vase']['mask_index']],fits['_support_plane'],segment_many_http)
   fits['vase']['opening_local']=opening_in_frame(opening,fits['vase']['pose'])
   self.log(event='observed_opening',env=i,accepted=True,fit=opening)
  except GeometryUnavailable as exc:
   self.log(event='observed_opening',env=i,accepted=False,reason=str(exc))
  (folder/'geometry.json').write_text(json.dumps(fits,default=lambda x:x.tolist(),indent=2))
  vf=fits['vase'];vase_points=vf['points_local']@vf['pose'][:3,:3].T+vf['pose'][:3,3];tree=cKDTree(vase_points)
  if allow_transfer:
   cup_points=fits['cup']['points_local']@fits['cup']['pose'][:3,:3].T+fits['cup']['pose'][:3,3];center=(cup_points.min(0)+cup_points.max(0))/2;mouth,_=aperture_geometry(vase_points,vf['pose'],vf.get('opening_local'))
   if center[0]*mouth[0]<0:
    fits['_relay_center'],choices=self.relay_support(o,center,cup_points,vase_points,fits['_support_plane'])
    self.log(event='relay_support_selection',env=i,center=fits['_relay_center'],candidates=choices)
  geometry_start=time.perf_counter();jobs=[]
  for direction in range(16):
   yaw=(direction//4)*np.pi/2;approach=R.from_euler('z',yaw).apply([0.,1.,0.]);toward=np.array([[0.,-1.,0.],[0.,1.,0.],[-1.,0.,0.],[1.,0.,0.]][direction%4])
   if abs(np.dot(np.cross([0.,0.,1.],toward),approach))<.99:continue
   for symmetry in range(2):
    jobs.append(self.candidate_pool.submit(evaluate_candidate,fits,o['state'],folder,allow_transfer,direction,symmetry))
  computed=[f.result() for f in jobs];candidates=[x[:5] for x in computed]
  self.log(event='phase_timing',env=i,phase='initial' if allow_transfer else 'reacquire',wall_seconds=time.perf_counter()-start,sam_http_seconds=sam_seconds,registration_io_seconds=geometry_start-start-sam_seconds,candidate_wall_seconds=time.perf_counter()-geometry_start,candidate_cpu_worker_seconds=sum(x[5] for x in computed))
  def selection_key(candidate):
   score,plan,direction,symmetry,res=candidate
   def stage_ok(x):
    pour=str(x.get('stage','')).startswith('tilt_') or x.get('stage') in ['above_mouth','align_lip','unseat_lip','staging_tilt','withdraw_tilted']
    return x['position_error']<.01 and x['rotation_error']<.08 and x['vase_penetration_m']<.002 and x['self_separation_m']>=0. and x['finger_separation_m']>=(.006 if pour else .015) and x['table_separation_m']>=.005
   feasible=all(stage_ok(x) for x in res)
   late=[x for x in plan['stages'] if x['name'] in ['tilt_1.62','tilt_1.76','tilt_1.9']]
   heights=[max(0.,float((matrix(x['pose'])@np.r_[plan['lip_local'],1])[2]-aperture_geometry(vase_points,vf['pose'],vf.get('opening_local'))[0][2])) for x in late]
   height=float(np.mean(heights)) if heights else 0.
   plan['selection_quality']={'feasible':feasible,'mean_lip_height_m':height,'pour_drop_m':height,'aperture_m':plan.get('aperture_radius')}
   return (0,height,score) if feasible else (1,score,height)
  if retry is not None:
   candidates=[x for x in candidates if (x[2],x[3])!=tuple(retry['excluded'])]
   candidates=[x for x in candidates if selection_key(x)[0]==0]
   if not candidates:raise ValueError('no feasible alternate RGB grasp')
  best=min(candidates,key=selection_key);self.log(event='kinematic_selection',env=i,selected_direction=best[2],selected_symmetry=best[3],score=best[0],candidates=[{'direction':d,'symmetry':sym,'score':score,'residuals':res,'quality':plan['selection_quality']} for score,plan,d,sym,res in candidates],official_state=o['state'])
  selected=best[1];arm=selected['arm'];q=np.asarray(o['state'][arm+'_arm_joint_state'],float).reshape(-1)[:6]
  base=matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1))@np.linalg.inv(self.kin.fk(q))
  goal,errors=self.kin.solve(np.linalg.inv(base)@matrix(selected['stages'][0]['pose']),q,position_tolerance=.0005,rotation_tolerance=.004)
  selected['stages'][0]['joints']=goal.tolist()
  self.log(event='selected_pregrasp_skill_choice',env=i,direction=best[2],symmetry=best[3],start_joints=q,goal_joints=goal,ik_errors=errors,ranking='unchanged_registered')
  best[1]['selected_grasp']=(best[2],best[3]);best[1]['allow_transfer']=allow_transfer
  cp=fits['cup']['points_local']@fits['cup']['pose'][:3,:3].T+fits['cup']['pose'][:3,3]
  best[1]['initial_cup_center']=(cp.min(0)+cp.max(0))/2;best[1]['initial_cup_mask']=np.asarray(selected_cup_mask,bool).copy()
  best[1]['cup_points_local']=fits['cup']['points_local'];best[1]['cup_inferred_surfaces']=fits['cup'].get('hidden_surfaces_reconstructed',False);best[1]['support_plane']=fits['_support_plane']
  best[1]['vase_points']=vase_points;best[1]['vase_fit']=vf
  if best[1].get('aperture_radius') is not None:
   best[1]['mouth'],_=aperture_geometry(vase_points,vf['pose'],vf.get('opening_local'));best[1]['mouth_tolerance']=max(.003,min(.008,.25*best[1]['aperture_radius']))
  else:
   best[1]['mouth'],best[1]['mouth_tolerance']=mouth_geometry(vase_points,vf['pose'])
  best[1]['origin_joints']={arm:np.asarray(o['state'][arm+'_arm_joint_state'],float).reshape(-1)[:6].tolist() for arm in ['left','right']};return best[1]
 def relay_support(self,o,center,cup_points,vase_points,support_plane):
  origins=[matrix(np.asarray(o['state'][a+'_ee_pose']).reshape(-1)) for a in ['left','right']];mid=(origins[0][:3,3]+origins[1][:3,3])/2
  receiver='right' if center[0]<0 else 'left';q0=np.asarray(o['state'][receiver+'_arm_joint_state'],float).reshape(-1)[:6];base=matrix(np.asarray(o['state'][receiver+'_ee_pose']).reshape(-1))@np.linalg.inv(self.kin.fk(q0));inv=np.linalg.inv(base);tree=cKDTree(vase_points);choices=[]
  for forward in [.19,.25,.31,.37]:
   point=np.array([mid[0],mid[1]+forward,center[2]])
   cup_gap=float(tree.query(cup_points+point-center)[0].min())
   for yaw in np.arange(4)*np.pi/2:
    zrot=R.from_euler('z',yaw).as_matrix();approach=zrot@np.array([0.,np.sin(1.3),-np.cos(1.3)]);radial=zrot@np.array([1.,0.,0.])
    for symmetry in range(2):
     rot=np.column_stack([approach,radial,np.cross(approach,radial)])@np.diag([1.,(-1.)**symmetry,(-1.)**symmetry]);grasp=np.eye(4);grasp[:3,:3]=rot;grasp[:3,3]=point+[0,0,.010]-.15757000232458113*approach
     q=q0.copy();gaps=[];errors=[];travel=0.
     for offset in [-.05*approach,np.zeros(3),np.array([0.,0.,.1])]:
      target=grasp.copy();target[:3,3]+=offset;nq,er=self.kin.solve(inv@target,q);travel+=float(np.linalg.norm(nq-q));q=nq;errors.append(er)
      gaps.append(min(self.kin.self_separation(q),self.kin.table_separation(q,base,support_plane)-.005,.006-self.kin.collision(q,base,tree),self.kin.finger_separation(base@self.kin.fk(q),tree)-.009))
     feasible=cup_gap>.025 and min(gaps)>=0 and all(d<.01 and a<.08 for d,a in errors)
     choices.append({'point':point.tolist(),'yaw':float(yaw),'symmetry':symmetry,'feasible':feasible,'min_gap_m':min(gaps),'cup_vase_gap_m':cup_gap,'error':max(d+.1*a for d,a in errors),'travel':travel})
  best=min(choices,key=lambda x:(not x['feasible'],-x['min_gap_m'] if x['feasible'] else x['error']+max(0,-x['min_gap_m']),x['travel']))
  return np.asarray(best['point']),choices
 def trajectory(self,fits,o,folder,allow_transfer,direction,symmetry):
  yaw=(direction//4)*np.pi/2;direction=direction%4
  cup=fits['cup']['pose'];vase=fits['vase']['pose'];cp=fits['cup']['points_local']@cup[:3,:3].T+cup[:3,3];vp=fits['vase']['points_local']@vase[:3,:3].T+vase[:3,3]
  center=(cp.min(0)+cp.max(0))/2;top=cp[:,2].max();vtop=vp[:,2].max();mouth,_=mouth_geometry(vp,vase)
  aperture_center,aperture_radius=aperture_geometry(vp,vase,fits['vase'].get('opening_local'));mouth=aperture_center.copy();vtop=float(aperture_center[2]);mouth[2]+=.008
  cup_top=fits['cup']['points_local'];cup_rim=cup_top[cup_top[:,2]>cup_top[:,2].max()-.003];cup_open=float(np.quantile(np.linalg.norm(cup_rim[:,:2]-cup_rim[:,:2].mean(0),axis=1),.10))
  pitcher=aperture_radius<cup_open+.006
  walls=wall_cloud(vp,vase,aperture_radius,fits['vase'].get('opening_local'));wall_tree=cKDTree(walls if len(walls) else vp)
  destination='right' if mouth[0]>0 else 'left';arm=destination if center[0]*mouth[0]>=0 or not allow_transfer else ('right' if center[0]>0 else 'left');side=1 if arm=='right' else -1
  handover=allow_transfer and arm!=destination
  radial=np.array([side,0.,0.]);approach=np.array([0.,np.sin(1.3),-np.cos(1.3)]);yaw_rot=R.from_euler('z',yaw).as_matrix();radial=yaw_rot@radial;approach=yaw_rot@approach;rot=np.column_stack([approach,radial,np.cross(approach,radial)])
  alternative=rot@np.diag([1.,-1.,-1.]);current=matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1))[:3,:3]
  rot=[rot,alternative][symmetry]
  g=np.eye(4);g[:3,:3]=rot;g[:3,3]=center+np.array([0.,0.,.010])-.15757000232458113*approach
  # Tilt towards the opposite cup lip, keeping that lip over the vase opening.
  toward=np.array([[0.,-1.,0.],[0.,1.,0.],[-1.,0.,0.],[1.,0.,0.]][direction]);axis=np.cross([0.,0.,1.],toward);lip=center+cup_open*toward;lip[2]=top;offset=g[:3,3]-lip
  stages=[]
  def add(name,t,grip,settle=2,outlet_point=None):
   stage={'name':name,'pose':pose(t).tolist(),'grip':grip,'settle':settle}
   if outlet_point is not None:stage['outlet_local']=(np.linalg.inv(g)@np.r_[outlet_point,1])[:3]
   stages.append(stage)
  pre=g.copy();pre[:3,3]-=.05*approach;add('pregrasp',pre,1.);add('contact',g,1.,5);add('close',g,0.,18)
  lift=g.copy();lift[2,3]=g[2,3]+.10;add('lift',lift,0.,8)
  if handover:
   transfer=g.copy();transfer[:3,3]=g[:3,3]+fits['_relay_center']-center+[0.,0.,.10];add('relay_transfer',transfer,0.,3)
   placed=transfer.copy();placed[2,3]-=.10;add('relay_place',placed,0.,8);add('relay_release',placed,1.,12)
   back=placed.copy();back[2,3]+=.08;add('relay_retreat',back,1.);add('relay_origin',matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1)),1.,5)
   stages.append({'name':'reacquire'})
   return dict(history=[],lip_local=(np.linalg.inv(g)@np.r_[lip,1])[:3],grasp_local=np.linalg.inv(cup)@g,arm=arm,stages=stages,index=0,ticks=0,settled=0,best=99.,stagnant=0,folder=folder,origin=copy(o['state']),aperture_radius=aperture_radius,vase_wall_points=walls,pitcher=pitcher)
  rr=R.from_rotvec(axis*.6).as_matrix();lift_pivot=lip+(lift[:3,3]-g[:3,3]);tilted=g.copy();tilted[:3,:3]=rr@rot;tilted[:3,3]=lift_pivot+rr@offset;add('staging_tilt',tilted,0.,5)
  # Episode-observed cup surface and known robot fingertip envelope.
  finger_center=g[:3,3]+.15757000232458113*approach
  fingers=np.array([finger_center+x*radial+y*approach+z*np.cross(approach,radial)
                    for x in [-.052,.052] for y in [-.030,.030] for z in [-.012,.012]])
  envelope=np.concatenate([cp,fingers]);vxy=(vp[:,:2].min(0)+vp[:,:2].max(0))/2
  radius=np.max(np.linalg.norm(vp[:,:2]-vxy,axis=1))+.012
  vase_tree=cKDTree(vp);cup_sample=cp[np.linspace(0,len(cp)-1,min(3000,len(cp))).astype(int)]
  def outlet(angle):
   radius=cup_open
   point=center+radius*toward;point[2]=top;return point
  def pivot(angle):
   anchor=outlet(angle);rr=R.from_rotvec(axis*angle).as_matrix();rel=(cup_sample-anchor)@rr.T
   chosen=None
   # Search only late narrow pours; early transfer and wide-mouth geometry stay centered.
   offsets=[0.]
   if pitcher and angle>=1.18:
    offsets += [x for x in [.004,-.004,.008,-.008] if abs(x)<=aperture_radius-.010]
   for z in np.arange(vtop-.010,vtop+.202,.002):
    for along in offsets:
     q=np.r_[mouth[:2]+along*toward[:2],z];translated=rel+q
     if float(wall_tree.query(translated,distance_upper_bound=.005)[0].min())<.005:continue
     tool=g.copy();tool[:3,:3]=rr@rot;tool[:3,3]=q+rr@(g[:3,3]-anchor)
     if self.kin.tool_separation(tool,wall_tree)<.005:continue
     if self.kin.finger_separation(tool,wall_tree,distance_upper_bound=.006)<.006:continue
     chosen=q;break
    if chosen is not None:break
   if chosen is not None:return chosen
   all_rel=(envelope-anchor)@rr.T;within=np.linalg.norm(all_rel[:,:2]+mouth[:2]-vxy,axis=1)<radius
   h=vtop+.015-all_rel[within,2].min() if within.any() else mouth[2]
   q=mouth.copy();q[2]=max(mouth[2],h);return q
  high=pivot(.6);above=tilted.copy();above[:3,3]=high+rr@offset;add('above_mouth',above,0.,outlet_point=outlet(.6))
  base=above.copy();add('align_lip',base,0.,outlet_point=outlet(.6))
  angles=[.75,.89,1.04,1.18,1.33,1.47,1.62,1.76] if pitcher else [.75,.89,1.04,1.18,1.33,1.47,1.62,1.76,1.9]
  for angle in angles:
   rr=R.from_rotvec(axis*angle).as_matrix();t=g.copy();t[:3,:3]=rr@rot;t[:3,3]=pivot(angle)+rr@(g[:3,3]-outlet(angle));add('tilt_'+str(round(angle,2)),t,0.,1 if angle<angles[-1] else 30,outlet_point=outlet(angle))
  # Withdraw while still tilted, then right the cup away from vase volume.
  final=angles[-1]
  source_pivot=lift_pivot.copy();rr=R.from_rotvec(axis*final).as_matrix()
  exit_pivot=pivot(final);exit_pivot[2]=max(exit_pivot[2],vtop+.015-((cp-outlet(final))@rr.T)[:,2].min())
  unseat=g.copy();unseat[:3,:3]=rr@rot;unseat[:3,3]=exit_pivot+rr@(g[:3,3]-outlet(final));add('unseat_lip',unseat,0.,3)
  withdrawal_pivot=source_pivot.copy();withdrawal_pivot[2]=max(source_pivot[2],exit_pivot[2]);withdraw=g.copy();withdraw[:3,:3]=rr@rot;withdraw[:3,3]=withdrawal_pivot+rr@offset;add('withdraw_tilted',withdraw,0.,3)
  lower=withdraw.copy();lower[:3,3]=source_pivot+rr@offset;add('lower_source_tilt',lower,0.,3)
  for angle in np.linspace(1.7,0.,10):
   rr=R.from_rotvec(axis*angle).as_matrix();t=g.copy();t[:3,:3]=rr@rot;t[:3,3]=source_pivot+rr@offset;add('upright_'+str(round(angle,2)),t,0.)
  add('return_high',lift,0.);add('return_cup',g,0.,3);add('release',g,1.,12);add('retreat',pre,1.);add('origin',matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1)),1.,8)
  return dict(history=[],lip_local=(np.linalg.inv(g)@np.r_[lip,1])[:3],grasp_local=np.linalg.inv(cup)@g,arm=arm,stages=stages,index=0,ticks=0,settled=0,best=99.,stagnant=0,folder=folder,origin=copy(o['state']),aperture_radius=aperture_radius,vase_wall_points=walls,pitcher=pitcher)
 def rebind(self,i,o,s,observe_only=False):
  scene=Scene.from_observation(o)
  start=time.perf_counter();rgb=np.asarray(o['vision']['cam_head']['color'],np.uint8)[:,:,:3];groups=segment_many_http(rgb,[('yellow cup',.2),('robot arm',.3),('vase',.2),('black vase',.2),('turquoise vase',.2),('white vase',.2)],timeout_s=600);sam_seconds=time.perf_counter()-start
  if not len(groups[0].masks):self.log(event='visual_binding_absent',env=i,stage=s['stages'][s['index']]['name']);return
  mask=max(groups[0].masks,key=lambda m:m.sum());occ=np.zeros(mask.shape,bool)
  for m in groups[1].masks:occ|=m
  occ &= ~mask
  vf=s['vase_fit'];vase_masks=[m for group in groups[2:] for m in group.masks if m.sum()>50 and (m&mask).sum()/max(1,m.sum())<.5]
  if vase_masks and not observe_only:
   vfits=[registration(m,vf['key'],prior=np.asarray(vf['pose']),visibility=PixelOcclusion(occ|mask,scene.camera),scene=scene,template=vf['points_local']) for m in vase_masks]
   vfit=min(vfits,key=lambda f:f['error_px']);accepted=vfit['error_px']<3 and vfit['iou']>.45
   if accepted and vf.get('opening_local') is not None:
    try:
     opening=observed_opening_fit(o,vase_masks[next(j for j,f in enumerate(vfits) if f is vfit)],s['support_plane'],segment_many_http)
     vfit['opening_local']=opening_in_frame(opening,vfit['pose'])
    except GeometryUnavailable as exc:
     accepted=False;self.log(event='opening_rebind_rejected',env=i,reason=str(exc))
   self.log(event='vase_binding',env=i,stage=s['stages'][s['index']]['name'],accepted=accepted,fit=vfit)
   if accepted:
    vp=vfit['points_local']@vfit['pose'][:3,:3].T+vfit['pose'][:3,3];mouth,inner=aperture_geometry(vp,vfit['pose'],vfit.get('opening_local'));tolerance=max(.003,min(.008,.25*inner));shift=mouth-s['mouth']
    for stage in s['stages'][s['index']:]:
     if 'pose' in stage and (stage['name'].startswith('tilt_') or stage['name'] in ['above_mouth','align_lip','unseat_lip','relay_transfer','relay_place','relay_release','relay_retreat']):
      target=matrix(stage['pose']);target[:3,3]+=shift;stage['pose']=pose(target).tolist()
    s.update(vase_fit=vfit,vase_points=vp,mouth=mouth,mouth_tolerance=tolerance,aperture_radius=inner,vase_wall_points=wall_cloud(vp,vfit['pose'],inner,vfit.get('opening_local')))
    self.log(event='mouth_update',env=i,shift_m=shift,tracking_margin_m=tolerance)
  current=matrix(np.asarray(o['state'][s['arm']+'_ee_pose']).reshape(-1));prior=current@np.linalg.inv(s['grasp_local']);views=[];wrist=np.asarray(o['vision']['cam_'+s['arm']+'_wrist']['color'],np.uint8)[:,:,:3]
  wrist_start=time.perf_counter();wg=segment_many_http(wrist,[('yellow cup',.2),('robot gripper',.3)],timeout_s=600);sam_seconds+=time.perf_counter()-wrist_start
  if len(wg[0].masks):
   wm=max(wg[0].masks,key=lambda m:m.sum());wo=np.zeros(wm.shape,bool)
   for m in wg[1].masks:wo|=m
   wo &= ~wm
   if wm.sum()>100:views=[{'mask':wm,'occlusion':wo,'camera':Camera.from_observation(o,'cam_'+s['arm']+'_wrist')}]
   cv2.imwrite(str(s['folder']/('wrist_bind_'+s['stages'][s['index']]['name']+'.png')),wrist[:,:,::-1]);cv2.imwrite(str(s['folder']/('wrist_mask_'+s['stages'][s['index']]['name']+'.png')),wm.astype('uint8')*255)
  fit=registration(mask,'cup',prior=prior,visibility=PixelOcclusion(occ,scene.camera),extra_views=views,scene=scene,template=s['cup_points_local'],inferred_template=s.get('cup_inferred_surfaces',False));name=s['stages'][s['index']]['name']
  cv2.imwrite(str(s['folder']/f'bind_{name}.png'),rgb[:,:,::-1]);cv2.imwrite(str(s['folder']/f'bind_{name}_mask.png'),mask.astype('uint8')*255)
  accepted=fit['error_px']<3 and fit['iou']>.45 and all(e<4 for e in fit.get('view_errors_px',[]))
  self.log(event='visual_binding',env=i,stage=name,accepted=accepted,fit=fit,prior=prior)
  if observe_only:
   local=np.linalg.inv(np.asarray(fit['pose']))@current
   self.log(event='rgb_hold_review',env=i,t=s.get('total'),stage=name,accepted=accepted,translation_error_m=float(np.linalg.norm(local[:3,3]-s['grasp_local'][:3,3])),rotation_error_rad=float(R.from_matrix(local[:3,:3]@s['grasp_local'][:3,:3].T).magnitude()),wrist_views=len(views),fit=fit,prior=prior)
   return
  if accepted and name in ['relay_transfer','relay_place','relay_release']:
   prev_relay=s.get('relay_binding'); drift=float(np.linalg.norm(np.asarray(fit['pose'])[:3,3]-np.asarray(prev_relay['pose'])[:3,3])) if prev_relay else 1.0
   s['relay_streak']=s.get('relay_streak',0)+1 if prev_relay and drift<=0.010 else 0
   self.log(event='relay_rebind',env=i,stage=name,drift_m=drift,streak=s['relay_streak'])
   s['relay_binding']=fit
   if s['relay_streak']<1: return
  if not accepted:
   self.log(event='phase_timing',env=i,phase='relative_rgb_rebind_rejected',wall_seconds=time.perf_counter()-start,sam_http_seconds=sam_seconds,cpu_geometry_io_seconds=time.perf_counter()-start-sam_seconds)
   return

  if name in ['tilt_1.62','tilt_1.76','tilt_1.9']:
   prior_binding=s.get('last_accepted_binding',{})
   low_motion = float(prior_binding.get('motion_m',1)) <= 0.008
   if low_motion and s.get('last_low_motion',False):
    s['final_tilt_rgb_confirmations']=s.get('final_tilt_rgb_confirmations',0)+1
   s['last_low_motion']=low_motion
  prev=s.get('last_accepted_binding',{})
  # registered: measure actual RGB binding displacement between accepted fits.
  motion_m=1.0
  if prev and isinstance(prev.get('fit'),dict):
   try: motion_m=float(np.linalg.norm(np.asarray(fit['pose'])[:3,3]-np.asarray(prev['fit']['pose'])[:3,3]))
   except Exception: motion_m=1.0
  s['last_motion_m']=motion_m
  if name in ['tilt_1.62','tilt_1.76','tilt_1.9']:
   same_binding=bool(prev) and motion_m<=0.008 and abs(float(fit.get('error_px',99))-float(prev.get('fit',{}).get('error_px',99)))<2.0
   # registered monotonic signed RGB flow proxy: cup binding displacement toward mouth may not reverse.
   signed_flow=False
   if prev and isinstance(prev.get('fit'),dict):
    try:
     dz=float(np.asarray(fit['pose'])[2,3]-np.asarray(prev['fit']['pose'])[2,3])
     signed_flow=dz>=-0.002
    except Exception: signed_flow=False
   same_binding = same_binding and signed_flow
   s['signed_flow_streak']=s.get('signed_flow_streak',0)+1 if same_binding else 0
   if s['signed_flow_streak']<3:
    self.log(event='adaptive_release_hold',env=i,stage=name,motion_m=motion_m,streak=s['signed_flow_streak'])
    s['last_accepted_binding']={'stage':name,'total':s.get('total',0),'fit':fit,'motion_m':motion_m,'confidence':float(s.get('last_binding_confidence',0.0))}
    return
  # registered: signed occupancy-flow proxy requires four monotonic consensus observations.
  if name in ['tilt_1.62','tilt_1.76','tilt_1.9']:
   same = bool(prev) and abs(float(fit.get('error_px',99))-float(prev.get('fit',{}).get('error_px',99))) < 2.0
   s['plateau_rebind_agree_streak'] = s.get('plateau_rebind_agree_streak',0)+1 if same else 0
   if s['plateau_rebind_agree_streak'] < 5:
    self.log(event='occupancy_flow_slope_guard_pending',env=i,stage=name,agree=same,streak=s['plateau_rebind_agree_streak'])
    s['last_accepted_binding']={'stage':name,'total':s.get('total',0),'fit':fit,'motion_m':float(s.get('last_motion_m',1.0)),'confidence':float(s.get('last_binding_confidence',0.0))}
    return
  s['last_accepted_binding']={'stage':name,'total':s.get('total',0),'fit':fit,'motion_m':float(s.get('last_motion_m',1.0)),'confidence':float(s.get('last_binding_confidence',0.0))}
  new_local=np.linalg.inv(fit['pose'])@current;correction=np.linalg.inv(s['grasp_local'])@new_local
  for stage in s['stages'][s['index']:]:
   if 'pose' in stage and stage['name'] not in ['origin','relay_origin']:stage['pose']=pose(matrix(stage['pose'])@correction).tolist()
   if 'outlet_local' in stage:stage['outlet_local']=(np.linalg.inv(correction)@np.r_[stage['outlet_local'],1])[:3]
  s['lip_local']=(np.linalg.inv(new_local)@s['grasp_local']@np.r_[s['lip_local'],1])[:3];s['grasp_local']=new_local
  walls=s.get('vase_wall_points',s['vase_points']);vase_tree=cKDTree(walls);vtop=float(s['vase_points'][:,2].max());aperture=float(np.asarray(o['state'][s['arm']+'_ee_joint_state']).reshape(-1)[0])*.054-.01
  qcursor=np.asarray(o['state'][s['arm']+'_arm_joint_state'],float).reshape(-1)[:6];base=current@np.linalg.inv(self.kin.fk(qcursor));invbase=np.linalg.inv(base)
  cp=s['cup_points_local'];cup_sample=cp[::max(1,len(cp)//3000)];rim=cp[cp[:,2]>cp[:,2].max()-.002]
  for stage in s['stages'][s['index']:]:
   if 'pose' not in stage or not (stage['name'].startswith('tilt_') or stage['name'] in ['above_mouth','align_lip','unseat_lip','relay_transfer','relay_place','relay_release','relay_retreat']):continue
   target=matrix(stage['pose']);cup_target=target@np.linalg.inv(new_local);cup_world=cup_sample@cup_target[:3,:3].T+cup_target[:3,3];rim_world=rim@cup_target[:3,:3].T+cup_target[:3,3]
   low=max(-.10,vtop-.008-float(rim_world[:,2].min()))
   if stage['name']=='unseat_lip':low=max(0.,vtop+.015-float(cup_world[:,2].min()))
   chosen=None;gap=None;dz=None
   for dz in np.arange(low,max(.122,low+.002),.002):
    trial=target.copy();trial[2,3]+=dz;translated=cup_world+[0.,0.,dz]
    if float(vase_tree.query(translated)[0].min())<.005:continue
    gap=self.kin.finger_separation(trial,vase_tree,aperture)
    if gap<.006 or self.kin.tool_separation(trial,vase_tree)<.005:continue
    qnew,errors=self.kin.solve(invbase@trial,qcursor)
    if errors[0]>=.01 or errors[1]>=.08 or self.kin.self_separation(qnew)<0 or self.kin.table_separation(qnew,base,s['support_plane'])<.005:continue
    chosen=trial;qcursor=qnew;break
   self.log(event='finger_clearance_replan',env=i,stage=stage['name'],aperture_m=aperture,accepted=chosen is not None,height_delta_m=float(dz) if chosen is not None else None,finger_gap_m=gap)
   if chosen is not None:stage['pose']=pose(chosen).tolist()
  self.log(event='phase_timing',env=i,phase='relative_rgb_rebind',wall_seconds=time.perf_counter()-start,sam_http_seconds=sam_seconds,cpu_geometry_io_seconds=time.perf_counter()-start-sam_seconds)
 def prepare_clearance_return(self,i,o,s):
  started=time.perf_counter();binding=s.get('last_accepted_binding',{})
  if binding.get('stage') not in ['tilt_1.62','tilt_1.76','tilt_1.9'] or ((s.get('final_tilt_rgb_confirmations',0)<5) if (binding.get('confidence',0)>=0.82 and binding.get('motion_m',1)<=0.008) else (s.get('final_tilt_rgb_confirmations',0)<8)) or s.get('total',0)-binding.get('total',-1000)>90:
   self.log(event='clearance_return_retained_original',env=i,reason='no recent accepted final-tilt RGB binding');return
  targets={x['name']:matrix(x['pose']) for x in s['stages'] if x['name'] in ['return_high','withdraw_tilted']}
  if len(targets)!=2:return
  arm=s['arm'];q=np.asarray(o['state'][arm+'_arm_joint_state'],float).reshape(-1)[:6];current=matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1))
  aperture=float(np.asarray(o['state'][arm+'_ee_joint_state']).reshape(-1)[0])*.054-.01
  goals,details=plan_return(self.kin,q,current,s['grasp_local'],s['cup_points_local'],s['vase_points'],targets,aperture,s['support_plane'])
  self.log(event='clearance_return_plan',env=i,t=s.get('total'),accepted=goals is not None,skill_choices=details)
  self.log(event='phase_timing',phase='clearance_return_plan',env=i,wall_seconds=time.perf_counter()-started,cpu_geometry_io_seconds=time.perf_counter()-started)
  if goals is None:return
  tail=[dict(x) for x in s['stages'] if x['name'] in ['return_cup','release','retreat','origin']]
  original=[dict(x) for x in s['stages'][s['index']:]]
  joint_stages=[{'name':'clearance_return_'+str(j),'pose':pose(g['matrix']).tolist(),'joints':g['joints'],'grip':0.,'settle':1} for j,g in enumerate(goals)]
  s['stages']=s['stages'][:s['index']]+joint_stages+tail
  s['return_original_tail']=original;s['return_aperture']=aperture
  s['ticks']=s['settled']=s['stagnant']=0;s['best']=99.
 def update_obs_batch(self,obs):
  self.ids=[o['env_idx'] for o in obs]
  futures=[(o,self.pool.submit(self.initial,o)) for o in obs if o['env_idx'] not in self.slots]
  for o,f in futures:
   try:self.slots[o['env_idx']]=f.result()
   except Exception as e:self.slots[o['env_idx']]={'failure':str(e),'hold':copy(o['state']),'folder':self.out};self.log(event='initial_failure',env=o['env_idx'],error=str(e))
  for o in obs:self.obs[o['env_idx']]=o
 def early_handover(self,i,o,s):
  scene=Scene.from_observation(o)
  started=time.perf_counter()
  rgb=np.asarray(o['vision']['cam_head']['color'],np.uint8)[:,:,:3]
  groups=segment_many_http(rgb,[('yellow cup',.2),('robot arm',.3)],timeout_s=600)
  sam_seconds=time.perf_counter()-started
  if not len(groups[0].masks):
   self.log(event='phase_timing',phase='early_handover_visibility',env=i,wall_seconds=time.perf_counter()-started,sam_http_seconds=sam_seconds);return False
  candidates=[]
  robot=np.zeros(rgb.shape[:2],bool)
  for m in groups[1].masks:robot |= m
  for mask in groups[0].masks:
   if mask.sum()<50:continue
   fit=registration(mask,'cup',scene=scene,visibility=PixelOcclusion(robot & ~mask,scene.camera))
   candidates.append((fit,mask))
  if not candidates:return False
  fit,mask=min(candidates,key=lambda x:x[0]['error_px'])
  placed=next(x for x in s['stages'] if x['name']=='relay_place')
  expected_pose=matrix(placed['pose'])@np.linalg.inv(s['grasp_local'])
  expected_points=s['cup_points_local']@expected_pose[:3,:3].T+expected_pose[:3,3]
  observed_points=fit['points_local']@fit['pose'][:3,:3].T+fit['pose'][:3,3]
  shift=float(np.linalg.norm((observed_points.min(0)+observed_points.max(0)-expected_points.min(0)-expected_points.max(0))/2))
  self.log(event='phase_timing',phase='early_handover_visibility',env=i,wall_seconds=time.perf_counter()-started,sam_http_seconds=sam_seconds)
  visible_fraction=fit['visible_model_points']/max(1,fit['visible_model_points']+fit['occluded_model_points'])
  qualified=fit['error_px']<=3. and fit['iou']>=.65 and shift<=.030 and visible_fraction>=.5
  self.log(event='early_handover_visibility',env=i,t=s.get('total'),error_px=fit['error_px'],iou=fit['iou'],relay_center_shift_m=shift,qualified=qualified,source_homing_ticks=s['ticks'])
  if not qualified:
   s.pop('handover_rgb_candidate',None);return False
  center=(observed_points.min(0)+observed_points.max(0))/2
  previous=s.get('handover_rgb_candidate')
  s['handover_rgb_candidate']={'center':center.copy(),'tick':s['ticks']}
  stable=previous is not None and s['ticks']>previous['tick'] and np.linalg.norm(center-previous['center'])<=.004
  self.log(event='handover_rgb_temporal_confirmation',env=i,t=s.get('total'),visible_fraction=visible_fraction,stable=bool(stable),center=center,previous_center=previous['center'] if previous else None,fit=fit)
  if not stable:return False
  try:ns=self.initial(o,False,cup_observation={'fit':fit,'mask':mask.copy()})
  except Exception as exc:
   self.log(event='early_handover_deferred',env=i,error=str(exc));return False
  if ns['arm']==s['arm']:
   self.log(event='early_handover_deferred',env=i,error='RGB destination association did not switch arms');return False
  ns.update(origin=s['origin'],origin_joints=s['origin_joints'],total=s['total']-1,approach_retries=s.get('approach_retries',0),background_origin_joints={s['arm']:s['origin_joints'][s['arm']]})
  self.slots[i]=ns
  self.log(event='early_handover_started',env=i,t=s['total']-1,source=s['arm'],destination=ns['arm'],source_homing_ticks=s['ticks'])
  return True
 def cross_arm_gap(self,observed,proposed):
  clouds=[]
  for arm in ['left','right']:
   q=np.asarray(observed[arm+'_arm_joint_state'],float).reshape(-1)[:6]
   base=matrix(np.asarray(observed[arm+'_ee_pose']).reshape(-1))@np.linalg.inv(self.kin.fk(q))
   clouds.append(self.kin.world_spheres(proposed[arm],base))
  (a,ra),(b,rb)=clouds
  return float(np.min(np.linalg.norm(a[:,None]-b[None,:],axis=2)-ra[:,None]-rb[None,:]))
 def joint_action(self,i):
  ee=self.action(i);o=self.obs[i];s=self.slots[i];result={};diagnostics={}
  stage=s['stages'][s['index']]['name'] if 'stages' in s and s['index']<len(s['stages']) else 'finished'
  for arm in ['left','right']:
   q=np.asarray(o['state'][arm+'_arm_joint_state'],float).reshape(-1)[:6]
   if arm!=s.get('arm'):
    qgoal=np.asarray(s.get('background_origin_joints',{}).get(arm,q));errors=(0.,0.)
   elif (stage=='pregrasp' or stage.startswith('approach_backtrack_') or stage.startswith('clearance_return_')) and 'hold' not in s:
    qgoal=np.asarray(s['stages'][s['index']]['joints']);errors=(0.,0.)
   elif stage in ['origin','relay_origin','finished'] and 'origin_joints' in s and 'hold' not in s:
    qgoal=np.array(s['origin_joints'][arm]);errors=(0.,0.)
   else:
    current=matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1));base=current@np.linalg.inv(self.kin.fk(q));target=np.linalg.inv(base)@matrix(ee[arm+'_ee_pose']);qgoal,errors=self.kin.solve(target,q,position_tolerance=.0005,rotation_tolerance=.004)
   delta=qgoal-q;delta*=min(1.,.04/max(np.max(np.abs(delta)),1e-9));command=q+delta
   if arm==s.get('arm') and stage=='pregrasp' and 'hold' not in s:
    current=matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1));base=current@np.linalg.inv(self.kin.fk(q));command=bounded_joint_step(self.kin,q,qgoal,base);delta=command-q
   if arm==s.get('arm') and stage.startswith('clearance_return_') and 'hold' not in s:
    current=matrix(np.asarray(o['state'][arm+'_ee_pose']).reshape(-1));base=current@np.linalg.inv(self.kin.fk(q));command=bounded_joint_step(self.kin,q,qgoal,base);delta=command-q
    gap=clearance(self.kin,command,base,s['grasp_local'],s['cup_points_local'][::max(1,len(s['cup_points_local'])//1200)],cKDTree(s['vase_points']),s['return_aperture'],s['support_plane'])
    if gap<0:
     command=q.copy();delta=command-q;s['stages']=s['stages'][:s['index']]+s.pop('return_original_tail');s['ticks']=s['settled']=s['stagnant']=0;s['best']=99.
     self.log(event='clearance_return_tracking_revert',env=i,t=s.get('total'),gap_m=gap)
   result[arm+'_arm_joint_state']=command.tolist();result[arm+'_ee_joint_state']=[1.] if arm in s.get('background_origin_joints',{}) else ee[arm+'_ee_joint_state']
   previous=s.get('last_joint_observation',{}).get(arm,q)
   diagnostics[arm]={'observed':q.tolist(),'commanded':command.tolist(),'command_delta_max':float(np.max(np.abs(delta))),'observed_delta_max':float(np.max(np.abs(q-previous))),'ik_position_error':errors[0],'ik_rotation_error':errors[1]}
  if s.get('background_origin_joints'):
   proposed={a:np.asarray(result[a+'_arm_joint_state']) for a in ['left','right']}
   current={a:np.asarray(o['state'][a+'_arm_joint_state'],float).reshape(-1)[:6] for a in ['left','right']}
   gap=self.cross_arm_gap(o['state'],proposed)
   if gap<.006:
    source=next(iter(s['background_origin_joints']));dest=s['arm'];serial=dict(proposed);serial[dest]=current[dest]
    source_only_gap=self.cross_arm_gap(o['state'],serial)
    if source_only_gap>=.006:result[dest+'_arm_joint_state']=current[dest].tolist()
    else:
     for a in ['left','right']:result[a+'_arm_joint_state']=current[a].tolist()
    self.log(event='handover_cross_arm_guard',env=i,t=s.get('total'),combined_gap_m=gap,source_only_gap_m=source_only_gap)
   for a,target in list(s['background_origin_joints'].items()):
    if float(np.max(np.abs(current[a]-target)))<.015:
     del s['background_origin_joints'][a];self.log(event='background_origin_reached',env=i,t=s.get('total'),arm=a)
   for a in ['left','right']:
    diagnostics[a]['commanded']=result[a+'_arm_joint_state'];diagnostics[a]['command_delta_max']=float(np.max(np.abs(np.asarray(result[a+'_arm_joint_state'])-current[a])))
  s['last_joint_observation']={arm:np.asarray(o['state'][arm+'_arm_joint_state'],float).reshape(-1)[:6].copy() for arm in ['left','right']}
  self.log(event='continuous_joint_command',env=i,t=s.get('total'),stage=stage,joints=diagnostics)
  return result
 def get_action_batch(self,ids=None):
  start=time.perf_counter()
  futures=[self.pool.submit(self.joint_action,i) for i in (self.ids if ids is None else ids)]
  result=[[f.result()] for f in futures];self.log(event='phase_timing',phase='action_batch',wall_seconds=time.perf_counter()-start,env_count=len(futures));self.flush_log();return result
 def action(self,i):
  o=self.obs[i];s=self.slots[i];a=copy(o['state']);t=s.get('total',0);s['total']=t+1
  if t%20==0:
   for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
    rgb=np.asarray(o['vision'][cam]['color'],np.uint8)[:,:,:3];cv2.imwrite(str(s['folder']/f'{cam}_{t:03d}.jpg'),rgb[:,:,::-1])
  if 'hold' in s:return s['hold']
  if s['index']>=len(s['stages']):return s['origin']
  stage=s['stages'][s['index']]
  if stage['name']=='withdraw_tilted' and not s.get('clearance_return_attempted'):
   s['clearance_return_attempted']=True;self.prepare_clearance_return(i,o,s);stage=s['stages'][s['index']]
  if stage['name']=='relay_origin' and s['ticks']>=8 and s['ticks']%8==0:
   try:early=self.early_handover(i,o,s)
   except Exception as exc:
    self.log(event='early_handover_deferred',env=i,error=str(exc));early=False
   if early:return self.action(i)
  if stage['name']=='approach_retry':
   try:
    ns=self.initial(o,s['allow_transfer'],retry={'cup_center':s['initial_cup_center'],'cup_mask':s['initial_cup_mask'],'excluded':s['selected_grasp']})
    ns.update(origin_joints=s['origin_joints'],origin=s['origin'],total=t,approach_retries=1,background_origin_joints=s.get('background_origin_joints',{}))
    self.slots[i]=ns;self.log(event='approach_retry_accepted',env=i,t=t,selected_grasp=ns['selected_grasp'])
   except Exception as exc:
    s['hold']=copy(o['state']);s['hold'][s['arm']+'_ee_joint_state']=[1.]
    self.log(event='approach_retry_rejected',env=i,t=t,error=str(exc));return s['hold']
   return self.action(i)
  if stage['name']=='pregrasp' and not s.get('approach_retries'):
   q=np.asarray(o['state'][s['arm']+'_arm_joint_state'],float).reshape(-1)[:6]
   history=s.setdefault('approach_observed_path',[])
   if not history or np.max(np.abs(q-np.asarray(history[-1]['joints'])))>.04:
    history.append({'pose':a[s['arm']+'_ee_pose'][:],'joints':q.tolist()})
  if stage['name']=='reacquire':
   try:
    self.slots[i]=self.initial(o,False);self.slots[i]['origin_joints']=s['origin_joints'];self.slots[i]['origin']=s['origin'];self.slots[i]['total']=t;self.slots[i]['approach_retries']=s.get('approach_retries',0)
   except Exception as exc:s['hold']=copy(o['state']);self.log(event='relay_perception_failure',env=i,error=str(exc));return s['hold']
   return self.action(i)
  if stage['name'] in ['align_lip','tilt_1.18','tilt_1.62','tilt_1.76','tilt_1.9'] and not stage.get('visual_done'):
   self.rebind(i,o,s);stage['visual_done']=True
  key=s['arm']+'_ee_pose';d,r=err(a[key],stage['pose']);s['ticks']+=1
  if not s.get('recovering') and not stage['name'].startswith('approach_') and stage['name'] not in ['pregrasp','contact','close']:
   hist=s['history']
   if not hist or err(hist[-1],a[key])[0]>.018 or err(hist[-1],a[key])[1]>.15:hist.append(a[key][:])
  pouring=stage['name'].startswith('tilt_') or stage['name'] in ['above_mouth','align_lip']
  outlet_local=stage.get('outlet_local',s['lip_local']);actual_outlet=(matrix(a[key])@np.r_[outlet_local,1])[:3];target_outlet=(matrix(stage['pose'])@np.r_[outlet_local,1])[:3]
  lip_error=np.linalg.norm(actual_outlet-target_outlet);planar_error=float(np.linalg.norm(actual_outlet[:2]-s['mouth'][:2]))
  outlet_tracking_error=float(np.linalg.norm(actual_outlet[:2]-target_outlet[:2]))
  capture_radius=max(.003,s['aperture_radius']-.010)
  lip_reached=pouring and outlet_tracking_error<s['mouth_tolerance'] and planar_error<capture_radius and abs(actual_outlet[2]-target_outlet[2])<.010 and r<.08

  lifted=stage['name']=='lift' and a[key][2]-s['stages'][2]['pose'][2]>=.08 and r<.08
  pos_gate=.015 if stage['name']=='pregrasp' else .008;rot_gate=.12 if stage['name']=='pregrasp' else .08
  reached=lip_reached if pouring else ((d<pos_gate and r<rot_gate) or lifted)
  if stage['name'].startswith('approach_backtrack_') or stage['name'].startswith('clearance_return_'):
   q=np.asarray(o['state'][s['arm']+'_arm_joint_state'],float).reshape(-1)[:6]
   joint_error=float(np.max(np.abs(q-np.asarray(stage['joints']))))
   reached=reached and joint_error<.015
   d=max(d,joint_error*.1)
  if reached:s['settled']+=1
  else:s['settled']=0
  e=d+.08*r
  if e<s['best']-.0005:s['best']=e;s['stagnant']=0
  else:s['stagnant']+=1
  self.log(env=i,t=t,stage=stage['name'],gripper_observed=a[s['arm']+'_ee_joint_state'],gripper_command=stage['grip'],position_error=d,rotation_error=r,lip_error=lip_error,outlet_planar_error=planar_error,outlet_tracking_error=outlet_tracking_error,capture_radius=capture_radius,mouth_tolerance=s['mouth_tolerance'],observed=a[key],target=stage['pose'])
  a[key]=bounded(a[key],stage['pose']);a[s['arm']+'_ee_joint_state']=[stage['grip']]
  if s['settled']>=stage['settle']:
   if stage['name'] in ['lift','relay_transfer','relay_place']:
    self.rebind(i,o,s,observe_only=True)
   s['index']+=1;s['ticks']=s['settled']=s['stagnant']=0;s['best']=99.
  elif s['stagnant']>60 and stage['name']=='pregrasp' and not s.get('approach_retries') and s.get('approach_observed_path'):
   reverse=[dict(p,name='approach_backtrack_'+str(j),grip=1.,settle=1) for j,p in enumerate(reversed(s['approach_observed_path']))]
   s['stages']=reverse+[{'name':'approach_retry','pose':reverse[-1]['pose'],'grip':1.,'settle':1}]
   s['index']=0;s['ticks']=s['settled']=s['stagnant']=0;s['best']=99.;s['approach_retries']=1
   self.log(event='approach_backtrack_started',env=i,t=t,waypoints=len(reverse))
  elif s['stagnant']>60 and not stage['name'].startswith('approach_') and not s.get('recovering') and s['history']:
   # Reverse observed reachable poses instead of locking a tilted cup indefinitely.
   reverse=[{'name':'recover_'+str(j),'pose':q,'grip':0.,'settle':1} for j,q in enumerate(reversed(s['history']))]
   tail=[dict(x) for x in s['stages'] if x['name'] in ['return_cup','release','retreat','origin']]
   if tail:
    s['stages']=reverse+tail;s['index']=0;s['ticks']=s['settled']=s['stagnant']=0;s['best']=99.;s['recovering']=True
    self.log(event='reverse_recovery',env=i,t=t,stage=stage['name'],waypoints=len(reverse))
   else:
    s['hold']=copy(o['state']);self.log(event='failed_hold',env=i,t=t,stage=stage['name'])
  elif s['stagnant']>60:
   s['hold']=copy(o['state']);s['hold'][s['arm']+'_ee_joint_state']=[stage['grip']];self.log(event='failed_hold',env=i,t=t,stage=stage['name'])
  return a
 def update_obs(self,o):self.update_obs_batch([o])
 def get_action(self):return self.get_action_batch([0])[0]


# Local extraction prototype; the original source remains unchanged.
from physicalrsi_skill_library.manipulation_programs import execute_memory
class StructuredModel(Model):
    _operation_revision = 'ae767f1377823de0f14e7d8b3eb2a80fb5527fb477d8a6df8d8f5691cebe381e'
    _operation_bindings = [('vision.schedule', '_memory_vision_schedule'), ('vision.bind', '_memory_vision_bind'), ('observation.commit', '_memory_observation_commit')]
    _memory_path = str(__import__('pathlib').Path(__file__).with_name('structured_memory.json'))
    def _memory_vision_schedule(self, packet, context):
        obs=packet
        self.ids = [o['env_idx'] for o in obs]
        futures = [(o, self.pool.submit(self.initial, o)) for o in obs if o['env_idx'] not in self.slots]
        return (obs,futures)
    def _memory_vision_bind(self, packet, context):
        obs,futures=packet
        for o, f in futures:
            try:
                self.slots[o['env_idx']] = f.result()
            except Exception as e:
                self.slots[o['env_idx']] = {'failure': str(e), 'hold': copy(o['state']), 'folder': self.out}
                self.log(event='initial_failure', env=o['env_idx'], error=str(e))
        return obs
    def _memory_observation_commit(self, packet, context):
        obs=packet
        for o in obs:
            self.obs[o['env_idx']] = o
    def update_obs_batch(self,obs):
        return execute_memory(self,'action',obs)

Model = StructuredModel
