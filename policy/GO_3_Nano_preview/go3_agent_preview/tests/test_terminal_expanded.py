"""Typed caller checkpoints preserve recovery and zero-tick pure-code comparison."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from support import observation
from XPolicyLab.policy.go3_agent_preview.gm.lib.motion import plan as m
from XPolicyLab.policy.go3_agent_preview.gm.runtime.session import (
    Ctx, Session, SwitchReason, SwitchRequest, SWITCH_SOURCES)
from conftest import setup_model


def session(program, *, active=True, task="cover_blocks", source="cover_blocks.aborted"):
    value=Session(task,program,None,respect_switch=active)
    value.observe(observation())
    return value


def test_nested_parallel_marker_pauses_before_other_arm_and_finally():
    history=[]
    def program(ctx):
        def left():
            try:
                ctx.cmd_q["left"][:]=.8
                yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED,
                    detail="synthetic final caller abandonment", source="cover_blocks.aborted")
                history.append("left after")
                yield
            finally:
                history.append("cleanup")
        def right():
            history.append("right advanced")
            ctx.cmd_q["right"][:]=.9
            yield
        yield from m.parallel(m.parallel(left()),right())
    value=session(program)
    action=value.next_action()
    assert value.switch_paused and history==[]
    assert np.allclose(action["left_arm_joint_state"],0)
    assert np.allclose(action["right_arm_joint_state"],0)
    # Preserve the paused generator until controller/worker discard; do not close.
    value.ctx.memory["keep_generator"]=value.gen


def test_pure_code_parallel_marker_same_actions_and_tick_count():
    def make(marker):
        def program(ctx):
            def left():
                if marker:
                    yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED,
                        detail="synthetic marker only",source="cover_blocks.aborted")
                ctx.cmd_q["left"][:]=.2
                yield
                ctx.cmd_q["left"][:]=.3
                yield
            def right():
                ctx.cmd_q["right"][:]=.4
                yield
                ctx.cmd_q["right"][:]=.5
                yield
            yield from m.parallel(left(),right())
        return program
    marked=session(make(True),active=False)
    baseline=session(make(False),active=False)
    for _ in range(3):
        a=marked.next_action();b=baseline.next_action()
        assert a.keys()==b.keys()
        assert all(np.array_equal(a[k],b[k]) for k in a)
    assert marked.ctx.step==baseline.ctx.step==3
    assert marked.finished==baseline.finished


def test_forged_marker_and_infrastructure_exception_never_pause():
    def forged(ctx):
        yield SwitchRequest(SwitchReason.CODE_ABORTED,"forged","cover_blocks.aborted",1)
    def infrastructure(ctx):
        raise OSError("synthetic infrastructure failure")
        yield
    def arbitrary_plan_error(ctx):
        raise m.PlanError("unbound runtime error")
        yield
    for program in (forged,infrastructure,arbitrary_plan_error):
        value=session(program);value.next_action()
        assert value.finished and not value.switch_paused
        assert value.ctx.switch_request is None


def test_source_registry_alias_and_wrong_task_rejected():
    alias=Ctx("make_toast_random",None)
    request=alias.switch_to_wam(SwitchReason.CODE_ABORTED,
        detail="no code plan for this task",source="make_toast.no_code_plan")
    assert isinstance(request,SwitchRequest)
    with pytest.raises(ValueError):
        Ctx("cover_blocks",None).switch_to_wam(SwitchReason.CODE_ABORTED,
            detail="cross task",source="make_toast.no_code_plan")


def test_dual_arm_partial_code_action_discarded_on_same_wam_rpc(setup_model):
    def program(ctx):
        def first():
            ctx.cmd_q["right"][:]=.8
            yield
        def second():
            ctx.cmd_q["left"][:]=.9
            yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED,
                detail="second arm confirms whole task abandonment",source="cover_blocks.aborted")
            yield
        yield from m.parallel(first(),second())
    model,backend=setup_model(program)
    model.update_obs(observation());action=model.get_action()[0]
    assert np.allclose(action["left_arm_joint_state"],.1)
    assert np.allclose(action["right_arm_joint_state"],0)
    assert backend.calls==1
    assert model.skill_switch.episodes[0].branch=="wam"
    events=model.skill_switch.episodes[0].counts
    model.close()


def test_actual_cover_retry_success_does_not_pause(monkeypatch):
    from XPolicyLab.policy.go3_agent_preview.gm.programs import cover_blocks as program
    monkeypatch.setattr(program,"locate_cubes",lambda *_:(["red","green","blue"],[np.array([x,0.]) for x in (-.2,0,.2)]))
    monkeypatch.setattr(program,"locate_cups",lambda *_:[np.array([x,.1]) for x in (-.2,0,.2)])
    attempts=[]
    def move(*args):
        attempts.append(1)
        yield
        return len(attempts)>1
    def noop(*args,**kw):yield
    monkeypatch.setattr(program,"move_cup",move)
    monkeypatch.setattr(program.m,"ready",noop);monkeypatch.setattr(program.m,"home",noop)
    value=session(program.run)
    for _ in range(12):
        value.observe(observation());value.next_action()
        if value.finished:break
    assert len(attempts)==7 and value.finished
    assert not value.switch_paused and value.ctx.switch_request is None


@pytest.mark.parametrize("task,source,prepare",[
    ("align_blocks","align_blocks.perception_abandoned",lambda p,mp: (mp.setattr(p,"measure_cubes",lambda *_:[]),mp.setattr(p,"measure_bar",lambda *_:None))),
    ("fasten_screws","fasten_screws.pairs_unavailable",lambda p,mp:mp.setattr(p,"head_scan",_empty_generator)),
    ("fold_clothes","fold_clothes.garment_unavailable",lambda p,mp:mp.setattr(p.see,"measure",lambda *_:None)),
])
def test_actual_initial_terminal_hook(task,source,prepare,monkeypatch):
    import importlib
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs."+task)
    prepare(p,monkeypatch)
    value=session(p.run,task=task,source=source);value.next_action()
    assert value.switch_paused and value.ctx.switch_request.source==source


def _empty_generator(*args,**kw):
    if False:yield
    return []


NO_CODE_PLAN=("fill_egg_holder","hang_mugs","make_toast","pack_objects_into_box",
              "pick_from_conveyor_by_image","store_laptop_and_headphones","sweep_blocks")


@pytest.mark.parametrize("task",NO_CODE_PLAN)
def test_task_without_code_plan_requests_the_switch_on_its_first_action(task):
    import importlib
    p=importlib.import_module("XPolicyLab.policy.go3_agent_preview.gm.programs."+task)
    value=session(p.run,task=task)
    action=value.next_action()
    assert value.switch_paused and not value.finished and value.ctx.step==1
    assert value.ctx.switch_request.source==task+".early_handoff"
    assert value.ctx.switch_request.sequence==1
    # The paused tick holds the observed state; the controller discards it.
    assert all(np.array_equal(action[k],value.ctx.obs["state"][k]) for k in action)


def test_all_program_marker_sources_registered():
    root=Path(__import__("XPolicyLab.policy.go3_agent_preview.gm.programs",fromlist=["x"]).__file__).parent
    found=[]
    for path in root.rglob("*.py"):
        if path.name.startswith("._") or "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="switch_to_wam":
                source_node=next(k.value for k in node.keywords if k.arg=="source")
                if not isinstance(source_node,ast.Constant):
                    continue  # Dynamic helper is validated against group D manifest.
                source=source_node.value
                task=source.split(".",1)[0]
                assert source in SWITCH_SOURCES.get(task,set()),(path,source)
                found.append(source)
    assert len(found)>=90


def test_same_rpc_code_tick_precedes_switch_and_never_repeats(setup_model,tmp_path):
    def program(ctx):
        ctx.cmd_q["left"][:]=.8
        yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED,
            detail="event order abandonment",source="cover_blocks.aborted")
        yield
    model,_=setup_model(program)
    for _ in range(3):
        model.update_obs(observation());model.get_action()
    model.close()
    events=[json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()]
    kinds=[e["event_type"] for e in events]
    switch_index=kinds.index("skill_switch")
    assert "code_tick" in kinds[:switch_index]
    assert "code_tick" not in kinds[switch_index:]
