"""Replay actual attempt5 boundaries; never contacts the live bridge."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from pickup_supervisor_adapter import PickupSupervisorAdapter,snapshot
from rsi_supervisor_v1 import Evidence

here=Path(__file__).parent
run=Path('/ML-vePFS/model_partner/ken/run_logs/rsi5-general_pickup-20261001-144655')
rows=[json.loads(x) for x in (run/'reward_feedback.jsonl').read_text().splitlines()]
assert [r['native_step'] for r in rows]==[0,38,55,63,77,89,100]
# Exercise the real private Agent report handler without importing its GPU stack.
tree=ast.parse((here/'agent.py').read_text())
method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
            if isinstance(n,ast.FunctionDef) and n.name=='note_execution')
module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
scope={};exec(compile(module,'actual_agent_report_handler','exec'),scope)
agent=SimpleNamespace(pickup_feedback=None)
a=PickupSupervisorAdapter();initial=snapshot(rows[0],terminal=False)
baseline=a.supervisor.feedback(initial);a.begin('approach',initial)
trace=[{'replay':True,'baseline':baseline,'explicit_initial_baseline':0}]
for r in rows[1:]:
    step=r['native_step'];current={}
    if step==38: current={'approach_clear':Evidence(True,step,'RSI5 live RGB assessment at attempt5 turn2')}
    if step==89: current={'object_retained':Evidence(True,step,'RSI5 live RGB assessment at attempt5 turn6')}
    s=snapshot(r,terminal=step==100,current=current)
    packet=a.after(s,kind='recorded_primitive',action_ok=step!=100,checkpoint_step=step,
                   phase_complete=step in (38,89,100))
    scope['note_execution'](agent,{'privileged_reward_feedback':packet})
    assert agent.pickup_feedback is packet
    trace.append({'replay':True,'native_step':step,**packet,
                  'agent_prompt_fragment':'Privileged collection diagnostics (not RGB SFT observation): '+json.dumps(agent.pickup_feedback)})
    if step==38:a.begin('grasp',s)
    if step==89:a.begin('lift',s)
assert trace[-1]['acceptance']['decision']=='stop_success'
assert all(t.get('feedback',{}).get('score_event') is None for t in trace)
out=run/'supervisor_v1_offline_replay.json'
out.write_text(json.dumps(trace,indent=2))
print(out)
