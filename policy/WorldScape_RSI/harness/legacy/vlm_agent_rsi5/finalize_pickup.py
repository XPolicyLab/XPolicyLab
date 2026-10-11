import ast
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import h5py

run=Path(sys.argv[1]); quality=sys.argv[2]
log=(run/'console.log').read_text(errors='replace')
end=re.search(r'episode ended: (.*?), decisions=(\d+), steps=(\d+), success=(\w+)',log)
assert end
raw=next((run/'trajectories').glob('*.hdf5'))
with h5py.File(raw) as f:
    attrs={k:v.item() if hasattr(v,'item') else v for k,v in f.attrs.items()}
    keys=[]; f.visit(keys.append)
    assert not any('reward_feedback' in k or 'privileged' in k for k in keys)
match=re.search(r'Video is saved to `([^`]+)`',log)
ev=(Path('/ML-vePFS/model_partner/ken/RoboDojo')/match[1]).parent
native=json.loads((ev/'_result.json').read_text())
feedback=[json.loads(x) for x in (run/'reward_feedback.jsonl').read_text().splitlines()]
assert feedback[-1]['native_step']==int(end[3])
assert attrs['native_step_limit']==200 and int(end[3])<=200
assert bool(attrs['success'])==(end[4]=='True')==feedback[-1]['native_success']
telemetry=[json.loads(p.read_text()) for p in (run/'bridge').glob('*/telemetry.json')]
skill=re.search(r'\[vlm_agent\] task skill: (\{.*\})',(run/'server.log').read_text())
times=re.findall(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z',log)
start=datetime.strptime(run.name[-15:],'%Y%m%d-%H%M%S')
summary=dict(run=str(run),steps=int(end[3]),step_lim=200,decisions=int(end[2]),native_result=native,
             capture=attrs,quality=quality,layout=json.loads((run/'layout_manifest.json').read_text()),
             input_tokens=None,output_tokens=None,wall_time_approx_s=(datetime.fromisoformat(times[-1])-start).total_seconds(),
             simulation_seconds=int(end[3])/25,video_seconds=(int(end[3])+1)/25,
             bridge_elapsed_s=sum(x.get('bridge_elapsed_s',0) for x in telemetry),
             task_skill=ast.literal_eval(skill[1]),uses_privileged_reward_feedback=True,
             diagnostics_in_sft_observation=False)
(run/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'raw':str(raw),'summary':summary},indent=2))
