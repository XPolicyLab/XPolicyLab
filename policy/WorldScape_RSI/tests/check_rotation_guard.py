import importlib.util, os, types, sys
from pathlib import Path
import numpy as np

from ..harness import main_route as m
m.live_poses=lambda e:{a:(p.copy(),q.copy()) for a,(p,q) in e.poses.items()}
m.steps_used=lambda e:e.step
class Settle:
 def __init__(self,*a): pass
class Env:
 def __init__(self):
  self.step=0; self.poses={a:(np.array([-.2 if a=='left' else .2,-.2,1.]),np.array([1.,0.,0.,0.])) for a in ('left','right')}
 def is_episode_end(self): return False
def run(fault_at=None, orientation=False, enhanced='1'):
 os.environ['ROBODAWN_RSI2_HARNESS_V1']=enhanced
 e=Env(); start=e.poses['left']; cmd=m.commands.Command.from_dict({'kind':'rotate','arm':'left','axis':'pitch','value':15})
 target=m.resolve(cmd,*start,None,m.TCP_OFFSET_M)
 def execute(env,action,report,*args,**kw):
  env.step+=1; report['executed']+=1
  pos,quat=target['pos'].copy(),target['quat'].copy()
  if fault_at is not None and env.step>=fault_at:
   if orientation:
    q=np.array([np.cos(np.pi/4),0,np.sin(np.pi/4),0]);pos=m.wrist_of(m.tcp_of(*start,m.TCP_OFFSET_M),q,m.TCP_OFFSET_M);quat=q
   else:pos[0]+=.031
  env.poses['left']=(pos,quat)
 hooks=types.SimpleNamespace(settle_module=types.SimpleNamespace(GripperSettle=Settle),joint_module=None,
  stop_reason=lambda *a:None, execute_control=execute,gripper_real=lambda e:{'left':.75,'right':1.},
  gripper_closed_floor=lambda e:{'left':.185,'right':.185},
  plan_keypose=lambda *a:([{},{}],{'plan_status':'Success','plan_points':2}))
 report,stop=m.run_sequence(e,{'commands':[cmd.to_dict(),{'kind':'wait','value':3}],'options':{'motion_settle_steps':5}}, {},hooks,{},float('inf'),.04,{'left':0.,'right':1.})
 return e,report,stop
for enhanced in ('0','1'):
 for fault_at in (1,2,3):
  e,r,s=run(fault_at,False,enhanced)
  assert e.step==fault_at,(enhanced,fault_at,e.step)
  assert len(r['command_results'])==1 and r['batch_aborted']=='command_failed'
  result=r['command_results'][0]
  assert result['blocked_reason']=='rotation_path_guard' and not result['ok'] and result['guard_step']==fault_at
  assert result['settle_executed']==max(0,fault_at-2) and s is None
e,r,s=run();assert all(x['ok'] for x in r['command_results']) and len(r['command_results'])==2
e,r,s=run(1,True);assert e.step==1 and r['command_results'][0]['blocked_reason']=='rotation_path_guard'
assert r['command_results'][0]['max_tcp_excursion_cm']<1e-6
print('PASS: 8 cases, normal rotation, motion and settling excursion, batch cancellation, enhanced on/off, independent angular guard.')
