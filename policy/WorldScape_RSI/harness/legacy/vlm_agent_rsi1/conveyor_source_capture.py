"""Preserve raw observed RGB and exact policy/execution records for ICL review."""
import hashlib,json,os
from pathlib import Path
def value(x):
 if hasattr(x,'detach'):return x.detach().cpu().tolist()
 if hasattr(x,'tolist'):return x.tolist()
 raise TypeError(type(x).__name__)
def save(label,data):
 root=Path(os.environ['ROBODAWN_ICL_SOURCE']);root.mkdir(parents=True,exist_ok=True)
 out=root/(label+'.json');tmp=out.with_suffix('.tmp')
 tmp.write_text(json.dumps(data,ensure_ascii=False,default=value));tmp.replace(out)
def observation(label,obs,step):
 import numpy as np
 from PIL import Image
 root=Path(os.environ['ROBODAWN_ICL_SOURCE']);root.mkdir(parents=True,exist_ok=True)
 images=[]
 for camera,entry in obs['vision'].items():
  color=entry['rgb'] if 'rgb' in entry else entry.get('color')
  if color is None:raise ValueError('camera has no RGB: '+camera)
  if hasattr(color,'detach'):color=color.detach().cpu().numpy()
  a=np.asarray(color)
  if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3:raise ValueError('unexpected original RGB format')
  f=root/(label+'_'+camera+'.png');Image.fromarray(a).save(f)
  images.append({'camera':camera,'path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'resolution':[a.shape[1],a.shape[0]],'control_step':step})
 save(label,{'kind':'terminal_observation' if label=='terminal' else 'pre_decision_observation','control_step':step,'simulated_time_s':step*.04,'images':images,'original_observation_fields':{k:v for k,v in obs.items() if k!='vision'}})
