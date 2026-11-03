"""The outer case runner preserves replay identity and original execution time."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    configured = os.environ.get("ROBORSI_CASE_RUNNER", "").strip()
    if not configured:
        pytest.skip("Set ROBORSI_CASE_RUNNER to test the separately supplied case supervisor")
    source = Path(configured)
    if not source.is_absolute() or not source.is_file():
        pytest.fail("ROBORSI_CASE_RUNNER must be an absolute path to a case supervisor file")
    spec = importlib.util.spec_from_file_location("_case_runner_budget_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.P = Path(__file__).resolve().parents[1]
    module.CONDA = Path("/synthetic/conda")
    return module


@pytest.fixture
def resume_request(tmp_path):
    path = tmp_path / "resume-request.json"
    record = {
        "format": "xpolicylab_own_episode_resume_v1",
        "identity": {
            "task": "align_blocks", "eval_seed": 0, "layout": 7,
            "run_id": "fixture-align_blocks-s0-layout7",
        },
        "native_identity": {"scene_sha256": "a" * 64},
        "config": {
            "episode_seconds": 700,
            "test_episodes": [
                {"task": "align_blocks", "eval_seed": 0, "layout_id": 7}
            ],
        },
    }
    path.write_text(json.dumps(record))
    return path, record


def test_resume_preserves_run_layout_and_remaining_budget(runner, resume_request):
    path, record = resume_request
    identity, remaining = runner.resume_request_identity(path, "align_blocks", 7, 86400)
    assert identity == record["identity"]
    assert remaining == 700


def test_wrong_layout_is_not_relabelled(runner, resume_request):
    path, _ = resume_request
    with pytest.raises(ValueError, match="layout"):
        runner.resume_request_identity(path, "align_blocks", 8, 86400)


def test_old_receipt_only_manifest_is_rejected(runner, resume_request):
    path, record = resume_request
    record["format"] = "receipt_only_v1"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="receipt-only"):
        runner.resume_request_identity(path, "align_blocks", None, 86400)


def test_native_scene_hash_required(runner, resume_request):
    path, record = resume_request
    record.pop("native_identity")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scene"):
        runner.resume_request_identity(path, "align_blocks", None, 86400)


def test_wrapper_exports_request_hash_layout_and_budget_file(runner, resume_request, tmp_path):
    path, _ = resume_request
    marker = tmp_path / "attempt/execution-budget.json"
    env = runner.case_env(
        "synthetic-unused-test-key", tmp_path, "fixture", tmp_path / "trajectory",
        path, layout_id=7, execution_budget_file=marker,
    )
    assert env["ROBORSI_RESUME_REQUEST"] == str(path)
    assert env["ROBORSI_RESUME_REQUEST_SHA256"] == runner.sha(path)
    assert env["ROBORSI_NATIVE_LAYOUT_ID"] == "7"
    assert env["ROBORSI_EXECUTION_BUDGET_FILE"] == str(marker)
    assert "ROBORSI_RESUME_MANIFEST" not in env


def test_fresh_wrapper_does_not_replace_native_queue(runner, tmp_path):
    env = runner.case_env("synthetic-unused-test-key", tmp_path, "fixture", tmp_path)
    assert "ROBORSI_NATIVE_LAYOUT_ID" not in env
    assert "ROBORSI_RESUME_REQUEST" not in env


def test_watchdog_is_loaded_without_model_dependencies(runner, tmp_path):
    budget = runner.load_execution_watchdog()(20, started_at=1000)
    assert budget.remaining(tmp_path / "missing.json", now=1010) == 10
