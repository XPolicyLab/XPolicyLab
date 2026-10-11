"""Private dataset selector; original evaluator and task code remain unchanged."""
import hashlib
import json
import os
from pathlib import Path
import runpy
from env.seed_manager.seed_manager import SeedManager

original = SeedManager.init_eval
def selected(self, *args, **kwargs):
    original(self, *args, **kwargs)
    index = int(os.environ['RSI5_LAYOUT_ID'])
    assert self.task_name == 'general_pickup' and index in self.seed_list
    path = Path(self.seed_info[index]['scene_layout'])
    manifest = {'eval_seed':self.eval_seed,'layout_id':index,'layout_path':str(path),
                'layout_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'selection':'private harness restricts existing seed queue; native layout unchanged'}
    Path(os.environ['RSI5_RUN_DIR'],'layout_manifest.json').write_text(json.dumps(manifest,indent=2))
    self.seed_list = [index]
    self.idx = self.st_idx = 0
    self.ed_idx = 1
SeedManager.init_eval = selected
runpy.run_path('/ML-vePFS/model_partner/ken/RoboDojo/src/eval_client/main.py',run_name='__main__')
