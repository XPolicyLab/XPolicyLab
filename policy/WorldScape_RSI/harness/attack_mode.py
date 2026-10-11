"""Shared attack-only diagnostic adapter. No native reward updates or privileged geometry reads."""
import json,os,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def profile(task):return json.loads((ROOT/'configs/task_profiles'/f'{task}.json').read_text())
def enabled():
 on=os.environ.get('RDA_ATTACK_MODE','0')=='1'
 if on and os.environ.get('RDA_STAGE')!='experiment':raise RuntimeError('attack feedback allowed only in experiment stage')
 return on

def metadata(task):
 p=profile(task);core=json.loads((ROOT/'configs/shared_core.json').read_text())
 return {'core_version':core['version'],'core_hash':core['sha256'],'task_profile_version':p['version'],'task_profile_hash':hashlib.sha256((ROOT/'configs/task_profiles'/f'{task}.json').read_bytes()).hexdigest(),'owner':os.environ.get('RDA_WORKER_ID'),'attack_mode':enabled(),'reward_feedback':enabled(),'evaluation_track':'attack_feedback' if enabled() else 'no_feedback','icl_release':p.get('icl_release')}

class AttackClient:
 def __init__(self,client,task):self.client=client;self.task=task;self.control={};self.feedback=None
 def __getattr__(self,k):return getattr(self.client,k)
 def begin(self):self.control={}
 def complete(self,messages,**kwargs):
  if not enabled():return self.client.complete(messages,**kwargs)
  p=profile(self.task)
  instruction=('DEVELOPMENT ATTACK MODE. Native accumulated process-score feedback is enabled, final success unchanged. '
   'Never infer grasp from TCP arrival or gripper opening alone. Use current images: locate -> approach -> close -> short test lift -> confirm object moved with hand -> transport -> verify placement/release. '
   'Keep explicit phase, object, confirmed events, failure reason and remaining step budget in memory. Preserve clearance/return budget. A recovery must change the failed approach. Demonstration coordinates are not current coordinates. '
   'After completing all listed settlement prerequisites for a registered milestone, submit a CHECK-ONLY reply with commands:[] and attack_checkpoint:{id:<listed id>,settled:true,evidence:<current visual/action evidence>}. '
   'The check inspects scores ALREADY accumulated by normal physics steps, adds no hidden steps, and does not trigger reward recomputation. A missed eligible milestone ends this attempt immediately, without auto-home. '
   'Do not check after a short test lift when the milestone requires a larger lift. Do not check incomplete releases, delayed triggers, or final100 before native success. Never use absence of score on terminal-only tasks to stop. '
   'On irrecoverable visually confirmed failure use commands:[] and early_abort:<reason> to stop without automatic home. For incomplete but recoverable phases use ordinary actions and reobserve. '
   'Task execution guidance: '+str(p.get('execution_guidance','Use phase_contract; do not invent completed stages.'))+' Phase contract: '+json.dumps(p.get('phase_contract',[]))+'. Registered milestones: '+json.dumps(p['milestones'],ensure_ascii=False)+'. Latest diagnostic: '+json.dumps(self.feedback,ensure_ascii=False))
  msgs=list(messages)+[{'role':'user','content':instruction}]
  result=self.client.complete(msgs,**kwargs)
  try:
   s=result['text'].strip();s=re.sub(r'^```(?:json)?\s*|\s*```$','',s);d=json.loads(s)
   self.control={k:d[k] for k in ('attack_checkpoint','early_abort') if k in d}
   if self.control and d.get('commands') != []:
    self.control={'attack_checkpoint':{'id':'invalid_nonempty_commands'}}
  except (ValueError,KeyError,TypeError):self.control={}
  return result

class AttackSession:
 def __init__(self,env,task):
  self.env=env;self.task=task;self.on=enabled();self.p=profile(task);self.checked=set()
  if self.on and hashlib.sha256(Path(self.p['source']).read_bytes()).hexdigest()!=self.p['source_sha256']:raise RuntimeError('Scoring source changed; review milestone profile')
 def snapshot(self):
  if not self.on:return None
  rm=self.env.reward_manager
  # Pure read of already accumulated PROCESS scores. Exclude final-only scoring and all ground-truth poses.
  try:score=float(rm._score_for_state(0,0.0,rm.score_meta,rm.score_achieved,rm.score_completed_count))
  except (AttributeError,IndexError,KeyError,TypeError) as exc:return {'available':False,'error':type(exc).__name__}
  return {'available':True,'process_score':score,'units':'0-100','supports_milestones':bool(self.p['milestones'])}
 def checkpoint(self,control):
  if not self.on:return {'status':'disabled','abort':False}
  snap=self.snapshot();base={'diagnostic':snap,'abort':False}
  if control.get('early_abort'):
   return dict(base,status='explicit_early_abort',abort=True,reason=str(control['early_abort']))
  c=control.get('attack_checkpoint')
  if not isinstance(c,dict):return dict(base,status='invalid_checkpoint')
  item=next((m for m in self.p['milestones'] if m['id']==c.get('id')),None)
  if item is None or c.get('settled') is not True or not str(c.get('evidence','')).strip():return dict(base,status='invalid_checkpoint',reason='registered id, settled=true and evidence required')
  if c['id'] in self.checked:return dict(base,status='already_checked')
  if not snap.get('available'):return dict(base,status='score_unavailable',reason='no automatic abort')
  # A scalar cannot establish task-specific completion by itself; declaration supplies the phase boundary.
  passed=snap['process_score']+1e-6>=item['expected_score']
  self.checked.add(c['id'])
  return dict(base,status='milestone_pass' if passed else 'milestone_missed',abort=not passed,milestone=c['id'],expected_score=item['expected_score'],evidence=c['evidence'])
 def log(self,event):
  if not self.on:return
  p=Path(os.environ['RDA_RUN_DIR'])/'attack_events.jsonl'
  with p.open('a') as f:f.write(json.dumps(dict(event,steps_used=int(self.env.take_action_cnt[0])),ensure_ascii=False)+'\n')
