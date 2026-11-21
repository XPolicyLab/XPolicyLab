"""Inject failures around declared main-operation boundaries in real task code."""
import importlib
import numpy as np
import pytest
from support import observation
from XPolicyLab.policy.go3_agent_preview.gm.runtime.session import Session

def empty(*args,**kw):
    if False:yield

def session(task,program):
    value=Session(task,program,None)
    value.observe(observation())
    return value

def finish(value,limit=30):
    for _ in range(limit):
        value.next_action()
        if value.finished or value.switch_paused:break
    return value

@pytest.mark.parametrize("after_final_release",[False,True])
def test_cover_final_retreat_error_does_not_switch_after_release(monkeypatch,after_final_release):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs.cover_blocks")
    monkeypatch.setattr(p,"locate_cubes",lambda *_:(["red","green","blue"],[np.array([x,0.]) for x in (-.2,0,.2)]))
    monkeypatch.setattr(p,"locate_cups",lambda *_:[np.array([x,.1]) for x in (-.2,0,.2)])
    calls=[]
    def move(ctx,*args):
        calls.append(args)
        if len(calls)==6:
            ctx.memory["last_cup_released"]=after_final_release
            raise p.m.PlanError("final retreat injection")
        if False:yield
        return True
    monkeypatch.setattr(p,"move_cup",move)
    monkeypatch.setattr(p.m,"ready",empty);monkeypatch.setattr(p.m,"home",empty)
    value=finish(session("cover_blocks",p.run))
    assert len(calls)==6 and value.switch_paused is (not after_final_release)

@pytest.mark.parametrize("completed",[False,True])
def test_xylophone_outer_checkpoint_suppressed_after_final_hit(monkeypatch,completed):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs.play_Xylophone")
    objects={"mallet":{"residual":0.,"xy":np.zeros(2),"axis":np.ones(2)},"xylophone":{"residual":0.}}
    monkeypatch.setattr(p.manip,"parked",empty);monkeypatch.setattr(p.manip,"finish",empty)
    monkeypatch.setattr(p.V,"xylophone_scene",lambda *_:objects)
    monkeypatch.setattr(p,"_plans",lambda *a:{"arm":"left"})
    def play(ctx,*a):
        ctx.memory["all_keys_struck"]=completed
        raise p.m.PlanError("boundary injection")
        yield
    monkeypatch.setattr(p,"_play",play)
    value=finish(session("play_Xylophone",p.run))
    assert value.switch_paused is (not completed)


def test_xylophone_final_hit_flag_precedes_retreat(monkeypatch):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs.play_Xylophone")
    monkeypatch.setattr(p,"settle_tcp",empty);monkeypatch.setattr(p,"_head_check",lambda *a:True)
    monkeypatch.setattr(p.m,"grip",empty);monkeypatch.setattr(p.m,"hold",empty)
    monkeypatch.setattr(p.manip,"high_to",empty)
    def move(ctx,*a,**kw):
        if ctx.memory.get("all_keys_struck"):raise p.m.PlanError("final high retreat")
        if False:yield
    monkeypatch.setattr(p.m,"move",move)
    pose=np.eye(4)
    plan={"arm":"left","pick":pose,"lift":pose,"rotate":pose,"targets":[(pose,pose)],"delta":np.zeros(3),"source":pose,"points":[pose],"opening":1.,"above":pose}
    value=session("play_Xylophone",lambda ctx:p._play(ctx,plan))
    # _play expects a held mallet throughout; avoid synthetic empty gripper rejection.
    value.ctx.obs["state"]["left_ee_joint_state"][:]=.4
    finish(value)
    assert value.ctx.memory.get("all_keys_struck") is True

@pytest.mark.parametrize("failure_at",["required","cleanup"])
def test_press_required_sequence_checkpoint_excludes_final_cleanup(monkeypatch,failure_at):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs.press_by_number")
    monkeypatch.setattr(p,"read_cards",lambda *a:(1,1))
    monkeypatch.setattr(p,"locate_buttons",lambda *a:{k:np.zeros(2) for k in p.BTN_XY})
    calls=[]
    def stroke(ctx,*a):
        calls.append(a)
        if failure_at=="required" and len(calls)==3:raise p.m.PlanError("stroke failed")
        if False:yield
    def cleanup(ctx,*a,**kw):
        if failure_at=="cleanup":raise p.m.PlanError("home failed")
        if False:yield
    monkeypatch.setattr(p,"stroke",stroke);monkeypatch.setattr(p.m,"grip",empty)
    monkeypatch.setattr(p.m,"home",cleanup);monkeypatch.setattr(p.m,"hold",empty)
    value=finish(session("press_by_number",p.run))
    if failure_at=="required":
        assert len(calls)==3 and value.switch_paused
        assert value.ctx.switch_request.source=="press_by_number.required_stroke_plan_error"
    else:
        assert len(calls)==4 and value.finished and not value.switch_paused
        assert value.ctx.switch_request is None


@pytest.mark.parametrize("task",["classify_objects","classify_objects_by_language"])
def test_classify_initial_required_scan_plan_error_switches_before_object_work(monkeypatch,task):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs."+task)
    base=getattr(p,"base",p)
    history=[]
    def scan(ctx,arm,*a,**kw):
        history.append(("scan",arm))
        raise p.m.PlanError("required initial scan unreachable")
        yield
    def measure(*a,**kw):
        history.append(("measure",))
        raise AssertionError("object work must not start after scan abandonment")
    monkeypatch.setattr(base,"go_scan",scan);monkeypatch.setattr(p,"measure_view",measure)
    value=session(task,p.run);value.next_action()
    assert value.switch_paused and history==[("scan","left")]
    assert value.ctx.switch_request.source==task+".initial_scan_plan_error"


@pytest.mark.parametrize("task",["classify_objects","classify_objects_by_language"])
def test_classify_later_scan_error_does_not_use_initial_checkpoint(monkeypatch,task):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs."+task)
    base=getattr(p,"base",p)
    calls=[]
    def scan(ctx,arm,*a,**kw):
        calls.append(arm)
        if len(calls)>2:raise p.m.PlanError("post-flow scan")
        if False:yield
    monkeypatch.setattr(base,"go_scan",scan);monkeypatch.setattr(p,"measure_view",lambda *a:[])
    monkeypatch.setattr(p.m,"home",empty)
    value=finish(session(task,p.run))
    assert len(calls)>2 and value.finished and not value.switch_paused
    assert value.ctx.switch_request is None


def test_dolls_empty_initial_scene_keeps_original_second_perception_before_switch(monkeypatch):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs.sort_nesting_dolls_by_size")
    calls=[];cleanup=[]
    def perceive(ctx,label,*a):
        calls.append(label)
        if False:yield
        return []
    def finish_task(*a):cleanup.append("finish");yield
    monkeypatch.setattr(p,"perceive",perceive);monkeypatch.setattr(p,"tuck",empty)
    monkeypatch.setattr(p.S,"one_per_rank",lambda *a,**kw:([],None))
    monkeypatch.setattr(p.m,"move_joints",empty);monkeypatch.setattr(p,"finish",finish_task)
    value=finish(session("sort_nesting_dolls_by_size",p.run))
    assert calls==["initial","second"] and not cleanup and value.switch_paused
    assert value.ctx.switch_request.source=="sort_nesting_dolls_by_size.initial_perception_exhausted"


@pytest.mark.parametrize("latest,home_error",[("unfinished",False),("repaired",False),("unobserved",False),("repaired",True)])
def test_organize_keyboard_checkpoint_follows_all_original_jobs_and_corrections(monkeypatch,latest,home_error):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs.organize_table")
    bad={"xy":np.array([-.1,0.]),"yaw":0.,"size":next(iter(p.KEYBOARDS.values()))}
    good={**bad,"xy":p.kb_target(bad)}
    assert not p.kb_ok(bad) and p.kb_ok(good)
    history=[];y_calls=[]
    def scene(ctx,tag,*a,**kw):
        history.append("look:"+tag)
        kb={"unfinished":bad,"repaired":good,"unobserved":None}[latest] if tag=="final2" else bad
        return {"keyboard":kb,"figurine":None,"alarm":None,"mouse":None,"drawer":None,"mat":None}
    def options(*a,**kw):
        if False:yield
        return "main",{}
    def job(name):
        def work(*a,**kw):
            history.append(name)
            yield
            return True
        return work
    def y_push(*a,**kw):
        history.append("y");y_calls.append(1)
        yield
        # The first original correction fails. Other-object work and the final
        # correction must still run before deciding whether Code is exhausted.
        if len(y_calls)==1 or latest=="unfinished":raise p.m.PlanError("keyboard correction injection")
    def home(*a,**kw):
        history.append("home")
        if home_error:raise p.m.PlanError("completed cleanup injection")
        if False:yield
    monkeypatch.setattr(p,"look",scene);monkeypatch.setattr(p,"alarm_options",options)
    monkeypatch.setattr(p,"alarm_job",job("alarm"));monkeypatch.setattr(p,"mouse_job",job("mouse"))
    monkeypatch.setattr(p,"figurine_job",job("figurine"));monkeypatch.setattr(p,"clear_arm",empty)
    monkeypatch.setattr(p,"keyboard_x_push",job("x"));monkeypatch.setattr(p,"keyboard_y_push",y_push)
    monkeypatch.setattr(p.m,"hold",empty);monkeypatch.setattr(p.m,"home",home)
    value=finish(session("organize_table",p.run))
    assert [h for h in history if h in {"alarm","mouse","x","y","figurine"}]==["alarm","mouse","x","y","figurine","x","y"]
    assert history.index("look:final2")>history.index("figurine")
    if latest=="unfinished":
        assert value.switch_paused and "home" not in history
        assert value.ctx.switch_request.source=="organize_table.keyboard_recovery_exhausted"
    else:
        assert value.finished and not value.switch_paused and value.ctx.switch_request is None
        assert history[-1]=="home"
        if home_error:assert "completed cleanup injection" in value.ctx.failure


@pytest.mark.parametrize("remaining",[False,True])
def test_bottles_latest_confirmed_candidates_checked_after_both_original_workers_end(monkeypatch,remaining):
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs.put_bottles_into_dustbin")
    xy=np.array([.2,-.1])
    candidate={"xy":xy,"kind":"standing","off":0.,"n":100,
               "standing":{"c":xy,"h":.2},"lying":{"c":xy,"u":np.array([1.,0.]),"len":.2}}
    history=[];home_calls=[];holder={}
    def candidates(*a,**kw):
        ctx=holder["session"].ctx
        history.append((ctx.step,ctx.memory["left_done"],ctx.memory["right_done"]))
        return [candidate] if remaining else []
    def play(*a,**kw):
        yield
    def home(*a,**kw):
        home_calls.append(True)
        if False:yield
    monkeypatch.setattr(p.S,"head_candidates",candidates)
    monkeypatch.setattr(p.S,"arm_mask",lambda shape,*a,**kw:np.zeros(shape,bool))
    monkeypatch.setattr(p,"reachable",lambda *a,**kw:False)
    monkeypatch.setattr(p,"plan_safe",lambda arm,q,*a,**kw:([q],[]));monkeypatch.setattr(p,"play",play)
    monkeypatch.setattr(p.m,"plan_joint",lambda *a,**kw:[]);monkeypatch.setattr(p.m,"follow",home)
    value=session("put_bottles_into_dustbin",p.run);holder["session"]=value
    finish(value,limit=80)
    assert value.ctx.memory["left_done"] and value.ctx.memory["right_done"]
    assert len(history)>5 and history[-1][1:]==(True,True)
    # An observed but unreachable bottle does not interrupt either worker's
    # original idle/park recovery. An empty current view follows original home.
    if remaining:
        assert value.switch_paused and not home_calls
        assert value.ctx.switch_request.source=="put_bottles_into_dustbin.remaining_bottles_after_recovery"
    else:
        assert value.finished and not value.switch_paused and home_calls==[True]
        assert value.ctx.switch_request is None
