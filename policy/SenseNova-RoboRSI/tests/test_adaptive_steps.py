"""Offline runtime limits and reviewed-action ACKs; no simulator or model API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module, util
from pathlib import Path
import queue

import pytest


POLICY = Path(__file__).resolve().parents[1]
SOURCE = POLICY / "vendor/robodojo_adaptive/source"
MIN_ENV = "ROBORSI_ADAPTIVE_MIN_STEPS"
fixtures = import_module(
    "XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend"
)


def source_module(name, relative):
    spec = util.spec_from_file_location(name, SOURCE / relative)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def adaptive_steps():
    return source_module("_adaptive_steps_offline_test", "adaptive_steps.py")


@pytest.mark.parametrize(
    ("environment", "minimum", "default"),
    [({}, 5, 5), ({MIN_ENV: "2"}, 2, 5),
     ({MIN_ENV: "10"}, 10, 10), ({MIN_ENV: "50"}, 50, 50)],
)
def test_resolved_limits_and_runtime_reexport(adaptive_steps, environment, minimum, default):
    expected = {
        "approved_steps_min": minimum,
        "approved_steps_max": 50,
        "approved_steps_default": default,
    }
    before = dict(environment)
    assert adaptive_steps.resolve_adaptive_steps(environment) == expected
    runtime = import_module("XPolicyLab.policy.SenseNova-RoboRSI.runtime_config")
    assert runtime.resolve_adaptive_steps(environment) == expected
    assert environment == before


def test_resolver_reads_environment_only_when_not_supplied(adaptive_steps, monkeypatch):
    monkeypatch.setenv(MIN_ENV, "10")
    assert adaptive_steps.resolve_adaptive_steps()["approved_steps_min"] == 10
    assert adaptive_steps.resolve_adaptive_steps({})["approved_steps_min"] == 5
    monkeypatch.delenv(MIN_ENV)
    assert adaptive_steps.resolve_adaptive_steps()["approved_steps_min"] == 5


@pytest.mark.parametrize("value", ["", " ", "1", "51", "5.0", "true", "false"])
def test_invalid_minimum_is_rejected(adaptive_steps, value):
    with pytest.raises(ValueError, match=MIN_ENV):
        adaptive_steps.resolve_adaptive_steps({MIN_ENV: value})


@pytest.mark.parametrize(
    ("minimum", "span"),
    [(None, 4), (None, True), (None, 5.0), (None, 51), ("10", 5)],
)
def test_invalid_span_is_rejected_without_movement(
    adaptive_steps, experiment, backend, reference, vla, minimum, span
):
    if minimum is not None:
        experiment.cfg["options"].update(
            adaptive_steps.resolve_adaptive_steps({MIN_ENV: minimum})
        )
    proposal = fixtures.propose(experiment)
    with pytest.raises(reference.InvalidAction):
        experiment.request(fixtures.review(proposal, approved_steps=span))
    assert backend.control_step == 0
    assert experiment.vla_pending["proposal_id"] == proposal["proposal_id"]
    with pytest.raises(queue.Empty):
        backend.next_action(timeout=0.01)


@pytest.mark.parametrize(
    ("minimum", "explicit_span", "expected_steps"),
    [("2", 2, 2), ("10", None, 10)],
)
def test_configured_span_requires_one_real_ack_per_step(
    adaptive_steps, experiment, backend, vla, minimum, explicit_span, expected_steps
):
    experiment.cfg["options"].update(
        adaptive_steps.resolve_adaptive_steps({MIN_ENV: minimum})
    )
    proposal = fixtures.propose(experiment)
    request = fixtures.review(proposal)
    if explicit_span is not None:
        request["approved_steps"] = explicit_span
    request_ids = set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(experiment.request, request)
        try:
            for step in range(expected_steps):
                pending = backend.next_action(timeout=2)
                assert pending.request_id not in request_ids
                request_ids.add(pending.request_id)
                assert backend.control_step == step
                assert not future.done()
                backend.acknowledge(pending.request_id, fixtures.observation())
                assert backend.control_step == step + 1
            result = future.result(timeout=2)
        finally:
            if not future.done():
                backend.cancel()
    assert result["executed_steps"] == expected_steps
    assert len(request_ids) == expected_steps
    assert backend.control_step == expected_steps
    assert result["observation"]["control_step"] == expected_steps
    assert experiment.vla_pending is None
    with pytest.raises(queue.Empty):
        backend.next_action(timeout=0.01)


@pytest.fixture
def robot_client(monkeypatch):
    client = source_module("_adaptive_robot_client_test", "code/policylab/robot.py")
    requests = []

    def call(request):
        requests.append(request)
        if request["op"] == "vla_propose":
            return {"proposal_id": "next-unmoved-proposal"}
        return {"done": False}

    monkeypatch.setattr(client, "call", call)
    return client, requests


def invoke_helper(client, helper, **kwargs):
    parameters = {
        "proposal_id": "reviewed-proposal",
        "decision": "accept",
        "reason": "Synthetic measured-state review.",
        "assessment": {"current_subgoal": "Synthetic alignment."},
        **kwargs,
    }
    if helper == "vla_review_and_propose":
        parameters["next_subgoal"] = "Inspect the updated synthetic state."
    return getattr(client, helper)(**parameters)


@pytest.mark.parametrize("helper", ["vla_review", "vla_review_and_propose"])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_helpers_omit_unspecified_span_for_host_default(robot_client, helper, explicit_none):
    client, requests = robot_client
    kwargs = {"approved_steps": None} if explicit_none else {}
    invoke_helper(client, helper, **kwargs)
    assert requests[0]["op"] == "vla_review"
    assert "approved_steps" not in requests[0]
    assert [request["op"] for request in requests] == (
        ["vla_review", "vla_propose"] if helper.endswith("and_propose") else ["vla_review"]
    )


@pytest.mark.parametrize("helper", ["vla_review", "vla_review_and_propose"])
@pytest.mark.parametrize("span", [4, True, 5.0, 51])
def test_helpers_do_not_clamp_or_coerce_explicit_invalid_span(robot_client, helper, span):
    client, requests = robot_client
    invoke_helper(client, helper, approved_steps=span)
    assert requests[0]["approved_steps"] == span
    assert type(requests[0]["approved_steps"]) is type(span)
