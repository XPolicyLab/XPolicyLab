import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,'/ML-vePFS/model_partner/ken/RoboDojo')
from env.seed_manager.seed_manager import SeedManager
m=SeedManager({'num_envs':1,'task_name':'general_pickup','config_name':'arx_x5','seed':0})
m.init_eval()
rows=[]
for i in range(8,28):
    p=Path(m.seed_info[i]['scene_layout']).resolve()
    rows.append({'eval_seed':0,'layout_id':i,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len({r['sha256'] for r in rows})==20
print(json.dumps(rows,indent=2))
