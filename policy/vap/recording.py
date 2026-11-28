"""Record observed frames without changing simulator stepping or model input."""
import json
import os
from pathlib import Path
import time
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit
class EpisodeRecorder:
 def __init__(self,env,episode_id):
  self.enabled=bool(os.environ.get('VAP_RUN_DIR')) and os.environ.get('EVAL_ENV_TYPE')!='debug'
  if not self.enabled:return
  from .deploy import _active_layout_id
  self.layout=_active_layout_id(env)
  self.root=Path(os.environ['VAP_RUN_DIR'])/'episodes'/('layout_%03d'%self.layout)/episode_id
  self.root.mkdir(parents=True)
  self.meta={'episode_id':episode_id,'layout_id':self.layout,'status':'running','start_time':time.time()}
  (self.root/'episode.json').write_text(json.dumps(self.meta,indent=2))
 def capture(self,obs,step):
  if not self.enabled:return
  frame=self.root/('frame_%06d.jpg'%step)
  frame.write_bytes(encode_image_bit(decode_image_bit(obs['vision']['cam_head']['color'])))
  row={'step':step,'real_frame':str(frame),'image_basis':'recorded_observation','time':time.time()}
  with (self.root/'actual_execution.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
 def finish(self,steps):
  if not self.enabled:return
  self.meta.update(status='execution_returned',action_steps=steps,end_time=time.time())
  (self.root/'episode.json').write_text(json.dumps(self.meta,indent=2))
