"""A cold start must neither consume nor extend an original episode budget."""

from importlib import import_module

import json

import pytest

CaseWatchdog = import_module("XPolicyLab.policy.SenseNova-RoboRSI.case_budget").CaseWatchdog
write_execution_budget = import_module("XPolicyLab.policy.SenseNova-RoboRSI.case_budget").write_execution_budget


def test_startup_and_execution_have_separate_deadlines(tmp_path):
    marker = tmp_path / "execution-budget.json"
    budget = CaseWatchdog(1800, started_at=1000)
    assert budget.remaining(marker, now=2000) == 800
    write_execution_budget(marker, 86400, started_at=2100)
    assert budget.remaining(marker, now=2200) == 86300
    assert budget.phase == "agent"


def test_resume_replay_and_agent_share_one_remaining_budget(tmp_path):
    marker = tmp_path / "execution-budget.json"
    budget = CaseWatchdog(1800, started_at=1000)
    write_execution_budget(marker, 600, phase="replay", started_at=1100)
    assert budget.remaining(marker, now=1100) == 600
    write_execution_budget(marker, 520, phase="agent", started_at=1180)
    assert budget.remaining(marker, now=1200) == 500
    assert budget.execution_deadline == 1700


def test_receipt_rewrite_cannot_extend_execution_deadline(tmp_path):
    marker = tmp_path / "execution-budget.json"
    budget = CaseWatchdog(1800, started_at=1000)
    write_execution_budget(marker, 600, started_at=1100)
    assert budget.remaining(marker, now=1200) == 500
    write_execution_budget(marker, 600, started_at=1300)
    assert budget.remaining(marker, now=1400) == 300
    marker.unlink()
    assert budget.remaining(marker, now=1701) == -1


def test_marker_after_startup_deadline_does_not_revive_case(tmp_path):
    marker = tmp_path / "execution-budget.json"
    budget = CaseWatchdog(20, started_at=1000)
    write_execution_budget(marker, 86400, started_at=1021)
    assert budget.remaining(marker, now=1022) == -2
    assert budget.phase == "startup"


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        write_execution_budget(tmp_path / "execution-budget.json", value)


def test_future_receipt_is_rejected(tmp_path):
    marker = tmp_path / "execution-budget.json"
    write_execution_budget(marker, 600, started_at=1100)
    with pytest.raises(ValueError, match="clock"):
        CaseWatchdog(started_at=1000).remaining(marker, now=1050)


def test_invalid_receipt_fails_closed(tmp_path):
    marker = tmp_path / "execution-budget.json"
    marker.write_text(json.dumps({"format": "unsupported"}))
    with pytest.raises(ValueError, match="Unsupported"):
        CaseWatchdog(started_at=1000).remaining(marker, now=1050)
