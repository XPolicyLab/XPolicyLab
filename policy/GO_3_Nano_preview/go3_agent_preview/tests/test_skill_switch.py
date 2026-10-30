"""Default explicit skill switch, action/cache/reset and deadline behavior."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pytest
from support import cfg, observation, DIMS, Backend
from conftest import setup_model
from XPolicyLab.policy.go3_agent_preview.gm.runtime.session import SwitchReason
from XPolicyLab.policy.go3_agent_preview.gm import model as gm_module
from XPolicyLab.policy.go3_agent_preview.gm.isolated import IsolatedGm
from XPolicyLab.policy.go3_agent_preview.model import Model
from XPolicyLab.policy.go3_agent_preview.wam.backend_client import GeActBackendClient
def failing_program(ctx):
    ctx.cmd_q["left"][:] = 0.75
    ctx.failure = "synthetic terminal code abandonment"
    yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED, detail=ctx.failure, source="cover_blocks.aborted")
    ctx.cmd_q["left"][:] = 0.75
    yield


def test_explicit_abandonment_switches_once_and_discards_code_action(setup_model, tmp_path):
    model, backend = setup_model(failing_program)
    for _ in range(4):
        model.update_obs(observation(7))
        chunk = model.get_action()
        assert len(chunk) == 1
        assert np.max(chunk[0]["left_arm_joint_state"]) <= 0.3 + 1e-6
    assert backend.calls == 2
    assert not model.code._sessions
    model.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(ev["event_type"] == "skill_switch" for ev in events) == 1
    assert any(ev["event_type"] == "code_event" and ev["features"]["kind"] == "skill_switch_requested" for ev in events)


def test_normal_program_end_does_not_trigger(setup_model):
    def program(ctx):
        ctx.cmd_q["left"][:] = 0.02
        yield
    model, backend = setup_model(program)
    for _ in range(4):
        model.update_obs(observation(0))
        assert np.allclose(model.get_action()[0]["left_arm_joint_state"], 0.02)
    assert model.skill_switch.episodes[0].status["done"]
    assert model.skill_switch.episodes[0].switch_step is None
    assert backend.calls == 0
    model.close()


def test_mixed_batch_reordering_shrinking_and_chunk_consumption(setup_model):
    def program(ctx):
        if ctx.obs["env_idx"] == 42:
            ctx.failure = "synthetic failure"
            yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED, detail=ctx.failure, source="cover_blocks.aborted")
            return
        while True:
            ctx.cmd_q["left"] += 0.01
            yield
    model, backend = setup_model(program)
    for n in (1, 2, 3):
        model.update_obs_batch([observation(42), observation(7)])
        chunks = model.get_action_batch([7, 42])
        assert [len(chunk) for chunk in chunks] == [1, 1]
        assert np.allclose(chunks[0][0]["left_arm_joint_state"], 0.01 * n)
        assert np.allclose(chunks[1][0]["left_arm_joint_state"], 0.1 * n)
    assert backend.calls == 1
    model.update_obs_batch([observation(7)])
    assert np.allclose(model.get_action_batch([7])[0][0]["left_arm_joint_state"], 0.04)
    assert 42 not in model.skill_switch.episodes
    assert 42 not in model.wam._pending_by_env
    model.close()


def test_reset_discards_wam_chunk_and_next_episode_starts_from_code(setup_model):
    model, backend = setup_model(failing_program)
    model.update_obs(observation(0))
    model.get_action()
    assert len(model.skill_switch.episodes[0].cache) == 2
    model.reset()
    assert not model.skill_switch.episodes
    assert not model.wam._pending_by_env
    model.update_obs(observation(0))
    assert model.skill_switch.episodes[0].branch == "gm"
    assert np.allclose(model.get_action()[0]["left_arm_joint_state"], 0.1)
    assert backend.calls == 2
    model.close()



NO_CODE_PLAN = ("fill_egg_holder", "hang_mugs", "hang_mugs_random", "make_toast", "make_toast_random",
                "pack_objects_into_box", "pack_objects_into_box_random", "pick_from_conveyor_by_image",
                "store_laptop_and_headphones", "store_laptop_and_headphones_random",
                "sweep_blocks", "sweep_blocks_random")


@pytest.mark.parametrize("task", NO_CODE_PLAN)
def test_task_without_code_plan_runs_wam_from_the_first_action(setup_model, tmp_path, task):
    loader = gm_module.GmModel(cfg(task))
    run = loader._load(task).run
    loader.close()
    model, backend = setup_model(run, extra={"task_name": task})
    obs = observation(5, instruction="synthetic current task instruction")
    obs["state"]["left_arm_joint_state"][:] = 0.23
    for n in (1, 2):
        model.update_obs(obs)
        action = model.get_action()[0]
        # Both actions come from the chunk WAM inferred on the current observation.
        assert np.allclose(action["left_arm_joint_state"], 0.1 * n)
    assert backend.calls == 1
    sent = backend.observed[0]
    assert np.allclose(np.asarray(sent["state"]).reshape(-1)[:6], 0.23)
    assert sent["prompt"] == obs["instruction"]
    episode = model.skill_switch.episodes[5]
    assert episode.branch == "wam" and episode.switch_step == 0
    assert episode.reason == task.removesuffix("_random") + ".early_handoff"
    assert not model.code._sessions
    model.close()
    record = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    assert (record["switch_count"], record["switch_step"], record["wam_calls"], record["control_steps"]) == (1, 0, 1, 2)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["control_step"] for e in events if e["event_type"] == "first_wam_action"] == [0]


def test_wam_error_enters_finite_controlled_hold_without_retries(setup_model):
    backend = Backend(fail=True)
    model, _ = setup_model(failing_program, backend=backend)
    for _ in range(5):
        model.update_obs(observation(0))
        action = model.get_action()[0]
        assert np.allclose(action["left_arm_joint_state"], 0)
        assert np.allclose(action["left_ee_joint_state"], 1)
    assert backend.calls == 1
    assert model.skill_switch.episodes[0].stopped
    model.close()


def test_out_of_range_wam_action_is_not_executed(setup_model):
    class InvalidBackend(Backend):
        def infer(self, **kwargs):
            result = super().infer(**kwargs)
            result["result"]["left_arm"][0][0] = 100
            return result
    model, backend = setup_model(failing_program, backend=InvalidBackend())
    model.update_obs(observation(0))
    assert np.allclose(model.get_action()[0]["left_arm_joint_state"], 0)
    assert model.skill_switch.episodes[0].stopped
    model.close()


def test_backend_uses_shared_deadline_across_metadata_and_inference():
    client = GeActBackendClient("ws://unused", request_timeout_s=30)
    client.deadline_monotonic = time.monotonic() + 0.05
    assert 0 < client._remaining(30) <= 0.05
    client.deadline_monotonic = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        client._remaining(30)


def test_stuck_code_is_killed_other_env_continues_and_reset_has_no_stale_write(monkeypatch, tmp_path):
    marker = tmp_path / "stale-worker-write"
    monkeypatch.setattr(gm_module, "get_robot_action_dim_info", lambda _: DIMS)

    def program(ctx):
        if ctx.obs["env_idx"] == 42:
            time.sleep(0.5)
            marker.write_text("bad")
        ctx.cmd_q["left"][:] = 0.05
        yield

    monkeypatch.setattr(gm_module.GmModel, "_load", lambda self, name: SimpleNamespace(run=program))
    model = IsolatedGm({**cfg(), "geact_request_timeout_s": 0.1})
    # Fork only in this synthetic test to propagate the monkeypatched program;
    # production explicitly uses spawn and imports the real unmodified program.
    model.context = mp.get_context("fork")
    model.update_obs_batch([observation(42), observation(7)])
    stuck_pid = model.workers[42][0].pid
    started = time.monotonic()
    chunks = model.get_action_batch([7, 42])
    assert time.monotonic() - started < 0.8
    assert np.allclose(chunks[0][0]["left_arm_joint_state"], 0.05)
    assert model.status(42)["failure"] == "code_compute_timeout"
    with pytest.raises(ProcessLookupError):
        os.kill(stuck_pid, 0)
    model.reset()
    model.update_obs_batch([observation(7)])
    assert np.allclose(model.get_action_batch([7])[0][0]["left_arm_joint_state"], 0.05)
    time.sleep(0.5)
    assert not marker.exists()
    model.close()


def test_current_state_switch_records_first_action_prompt_and_distinct_latency_sources(setup_model, tmp_path):
    model, backend = setup_model(failing_program)
    obs = observation(7)
    obs["instruction"] = "synthetic current task instruction"
    model.update_obs(obs)
    model.get_action()
    model.update_obs(obs)
    model.get_action()
    model.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    first = [ev for ev in events if ev["event_type"] == "first_wam_action"]
    assert len(first) == 1
    assert first[0]["branch_after"] == "wam"
    assert first[0]["control_step"] == 0
    saved = np.load(first[0]["observation_ref"], allow_pickle=False)
    assert saved["instruction"].item() == obs["instruction"]
    assert int(saved["env_idx"]) == 7
    assert np.array_equal(saved["state__left_arm_joint_state"], obs["state"]["left_arm_joint_state"])
    assert np.array_equal(saved["rgb__cam_head"], obs["vision"]["cam_head"]["color"])
    episode = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    assert len(episode["wam_inference_latencies_ms"]) == 1
    assert len(episode["policy_get_action_latencies_ms"]) == 2
    assert episode["rpc_latencies_ms"] == episode["wam_inference_latencies_ms"]
    assert all(ms >= 0 for ms in episode["policy_get_action_latencies_ms"])
    assert len([ev for ev in events if ev["event_type"] == "policy_get_action"]) == 2


def test_authored_native_limits_allow_legal_action_outside_gm_planner_range(setup_model):
    class NativeActionBackend(Backend):
        def infer(self, **kwargs):
            result = super().infer(**kwargs)
            result["result"]["left_arm"][0][1] = -0.2
            return result
    limits = [[-10, 10]] * 5 + [[-3.14, 3.14]]
    model, backend = setup_model(failing_program, backend=NativeActionBackend(),
        extra={})
    model.update_obs(observation(0))
    assert np.isclose(model.get_action()[0]["left_arm_joint_state"][1], -0.2)
    assert not model.skill_switch.episodes[0].stopped
    model.close()



def test_wam_is_lazy_and_no_implicit_failure_signal_switches(setup_model):
    def program(ctx):
        ctx.failure="logged diagnostic only"
        ctx.log("retry", attempt=1000)
        ctx.cmd_g["left"]=0.0
        yield
        raise RuntimeError("infrastructure exception")
    model,backend=setup_model(program)
    assert model.wam is None and not model.test_wam_builds
    for _ in range(3):
        model.update_obs(observation());model.get_action()
    assert model.wam is None and not model.test_wam_builds and backend.calls==0
    assert model.skill_switch.episodes[0].branch=="gm"
    model.close()


def test_invalid_wam_chunk_stops_for_episode_and_reset_recovers(setup_model):
    class Invalid(Backend):
        def infer(self,**kw):
            value=super().infer(**kw)
            value["result"]["left_arm"][0][0]=float("nan")
            return value
    model,backend=setup_model(failing_program,backend=Invalid())
    for _ in range(3):
        obs=observation();obs["state"]["left_arm_joint_state"][:]=.23
        model.update_obs(obs)
        assert np.allclose(model.get_action()[0]["left_arm_joint_state"],.23)
    assert backend.calls==1
    model.reset();model.update_obs(observation());model.get_action()
    assert backend.calls==2
    model.close()


def test_default_isolation_context_and_existing_timeout_contract(monkeypatch):
    monkeypatch.setenv("XPL_REQUEST_TIMEOUT_S","40")
    value=IsolatedGm({**cfg(),"geact_request_timeout_s":600})
    assert value.context.get_start_method()=="spawn"
    assert value.timeout_s==value.startup_timeout_s==30
    value.close()


def test_unknown_observation_indices_and_duplicate_batches_rejected(setup_model):
    def program(ctx):
        while True:yield
    model,_=setup_model(program)
    with pytest.raises(RuntimeError):model.get_action_batch([7])
    with pytest.raises(ValueError):model.update_obs_batch([observation(7),observation(7)])
    model.update_obs(observation(7))
    with pytest.raises(ValueError):model.get_action_batch([7,7])
    model.close()

def test_implicit_batch_indices_survive_mixed_subsets_and_wam_single_obs(setup_model):
    def program(ctx):
        if ctx.obs["env_idx"]==0:
            yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED,detail="env zero abandons",source="cover_blocks.aborted")
        while True:
            ctx.cmd_q["left"]+=.01
            yield
    model,backend=setup_model(program)
    for tick in (1,2,3):
        observations=[observation(),observation()]
        for obs in observations:obs.pop("env_idx")
        model.update_obs_batch(observations)
        chunks=model.get_action_batch([1,0])
        assert np.allclose(chunks[0][0]["left_arm_joint_state"],.01*tick)
        assert np.allclose(chunks[1][0]["left_arm_joint_state"],.1*tick)
        assert all("env_idx" not in obs for obs in observations)
        assert model.skill_switch.episodes[1].observed["env_idx"]==1
    assert backend.calls==1 and not model.skill_switch.episodes[0].stopped
    model.reset()
    model.update_obs_batch(observations)
    reset_chunks=model.get_action_batch([1,0])
    assert np.allclose(reset_chunks[0][0]["left_arm_joint_state"],.01)
    assert np.allclose(reset_chunks[1][0]["left_arm_joint_state"],.1)
    assert backend.calls==2
    model.close()


def test_implicit_nonzero_batch_index_can_switch_to_wam(setup_model):
    def program(ctx):
        if ctx.obs["env_idx"]==1:
            yield ctx.switch_to_wam(SwitchReason.CODE_ABORTED,detail="env one abandons",source="cover_blocks.aborted")
        while True:yield
    model,backend=setup_model(program)
    observations=[observation(),observation()]
    for obs in observations:obs.pop("env_idx")
    model.update_obs_batch(observations)
    chunks=model.get_action_batch([0,1])
    assert np.allclose(chunks[1][0]["left_arm_joint_state"],.1)
    assert backend.calls==1 and not model.skill_switch.episodes[1].stopped
    model.close()
