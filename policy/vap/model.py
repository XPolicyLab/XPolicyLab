"""VAP adapter using official XPolicyLab transport and fixed-thread model ownership."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import runpy
import sys
import time
import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .config_env import values, MAPPING
HERE=Path(__file__).resolve().parent

class Model(ModelTemplate):
 def __init__(self,model_cfg):
  self.model_cfg=dict(model_cfg)
  if model_cfg.get('env_cfg_type')!='arx_x5' or model_cfg.get('action_type')!='ee':
   raise ValueError('This checkpoint supports env_cfg_type=arx_x5, action_type=ee only')
  if model_cfg.get('eval_batch'):raise ValueError('VAP supports eval_batch=false only')
  self.dims=get_robot_action_dim_info(model_cfg['env_cfg_type'])
  if len(self.dims['arm_dim'])!=2 or self.dims['ee_dim']!=[1,1]:raise ValueError(self.dims)
  self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='vap-model-owner')
  try:self.backend=self.executor.submit(self._initialize).result()
  except BaseException:self.executor.shutdown(wait=False);raise
 def _initialize(self):
  os.environ.update(values(self.model_cfg))
  sys.path.insert(0,str(HERE/'upstream'))
  checkpoint=resolve_checkpoint_root(self.model_cfg,HERE/'checkpoints',policy_dir=HERE)
  if not checkpoint.is_dir():raise ValueError('Pass a checkpoint directory containing config.yml, normalizer_action.pth and model.safetensors')
  os.environ['VAP_SOURCE_CHECKPOINT']=str(checkpoint)
  os.environ.setdefault('VAP_REPO',str(HERE/'upstream'))
  os.environ.setdefault('VAP_NORMALIZER_STATS',str(HERE/'configs/action_stats.json'))
  os.environ.setdefault('VAP_RUN_DIR',str(HERE/'outputs'/('server_'+str(time.time_ns()))))
  run=Path(os.environ['VAP_RUN_DIR']);run.mkdir(parents=True,exist_ok=True)
  runpy.run_path(str(HERE/'prepare_checkpoint.py'),run_name='__main__')
  view=run/'runtime/checkpoint_online'
  os.environ['VAP_CHECKPOINT']=str(view);os.environ['VAP_CONFIG']=str(view/'config.yml')
  cfg=dict(self.model_cfg,vap_repo=str(HERE/'upstream'),vap_checkpoint=str(view),vap_config=str(view/'config.yml'))
  from .backend import Model as Backend
  backend=Backend(cfg)
  text_device=os.environ.get('VAP_TEXT_ENCODER_DEVICE') or self.model_cfg.get('text_encoder_device')
  if text_device:
   from .text_device import DeviceRoutedQwenPrompter
   base=backend.policy.runtime.model._model
   base.prompter=DeviceRoutedQwenPrompter(base.prompter,text_device)
   print('VAP_TEXT_DEVICE',text_device,flush=True)
  (run/'model_ready.json').write_text(json.dumps({'checkpoint':str(checkpoint),'config':cfg,'owner_thread':'vap-model-owner','resolved_runtime_options':{key:os.environ.get(key) for key in list(MAPPING.values())+['VAP_TEXT_ENCODER_DEVICE','VAP_TEXT_ENCODER_GPU','CUDA_VISIBLE_DEVICES']},'model_code':str(HERE/'upstream'),'ready_unix':time.time()},indent=2))
  return backend
 def update_obs(self,obs):return self.executor.submit(self.backend.update_obs,obs).result()
 def get_action(self):
  start=time.monotonic();actions=self.executor.submit(self.backend.get_action).result()
  keys={f'{side}_ee_pose':7 for side in ('left','right')}
  keys.update({f'{side}_ee_joint_state':self.dims['ee_dim'][i] for i,side in enumerate(('left','right'))})
  for action in actions:
   if set(action)!=set(keys):raise ValueError('Unexpected action keys: '+str(action.keys()))
   for key,dim in keys.items():
    value=np.asarray(action[key])
    if value.shape!=(dim,) or not np.isfinite(value).all():raise ValueError('Invalid action '+key)
  print('VAP_ACTION_VERIFIED',json.dumps({'count':len(actions),'finite':True,'seconds':time.monotonic()-start}),flush=True)
  return actions
 def reset(self):return self.executor.submit(self.backend.reset).result()
 def update_obs_batch(self,obs_list):raise NotImplementedError('eval_batch=false; one server per environment')
 def get_action_batch(self,env_idx_list=None):raise NotImplementedError('eval_batch=false; one server per environment')
