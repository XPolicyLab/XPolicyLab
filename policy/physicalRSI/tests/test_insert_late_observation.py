"""Late re-observation only schedules remaining insertion motion when visible."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from source_edits import LATE_INSERT_OBSERVATION


@pytest.mark.parametrize('guide_only,expected',[(False,('realign','guide','retreat')),(True,('guide','retreat'))])
def test_late_pose_replan_uses_remaining_stages_and_propagates_feasibility(guide_only,expected):
    calls=[];context={'arm':'right','tube_length':.173};estimate={'visible':True,'axis_in_gripper':[0,0,1]}
    def observe(env,arm,length):
        assert (env,arm,length)==(2,'right',.173)
        return estimate
    runtime=SimpleNamespace(contexts={2:{'tube1':context}},perception=SimpleNamespace(held_pose=observe))
    def replan(*args,**kwargs):calls.append((args,kwargs));return False
    ns={'_refresh_delivery':replan};exec(LATE_INSERT_OBSERVATION,ns)
    assert not ns['_refresh_observed_insertion'](runtime,2,{},'tube1',{},guide_only=guide_only)
    assert calls[0][0][-1] is estimate
    assert calls[0][1]['delivery_phases']==expected
    assert context=={'arm':'right','tube_length':.173}


def test_occluded_pose_keeps_existing_visual_fallback_available():
    runtime=SimpleNamespace(contexts={0:{'tube0':{'arm':'left','tube_length':.12}}},perception=SimpleNamespace(held_pose=lambda *args:{'visible':False}))
    def forbidden(*args,**kwargs):raise AssertionError('Invisible pose must not replan')
    ns={'_refresh_delivery':forbidden};exec(LATE_INSERT_OBSERVATION,ns)
    assert not ns['_refresh_observed_insertion'](runtime,0,{},'tube0',{},guide_only=True)
