"""Verify the PR contains a standalone RoboDojo source and evaluation payload."""

from __future__ import annotations

from importlib import import_module

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

load_submission = import_module("XPolicyLab.policy.SenseNova-RoboRSI.preflight").load_submission
validate_source_bundle = import_module("XPolicyLab.policy.SenseNova-RoboRSI.preflight").validate_source_bundle

POLICY_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def submission_copy(tmp_path):
    """Use only PR-visible source files, without runtime or ignored archives."""
    shutil.copy2(POLICY_DIR / "submission.json", tmp_path / "submission.json")
    shutil.copytree(
        POLICY_DIR / "vendor",
        tmp_path / "vendor",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return tmp_path


@pytest.mark.parametrize("change", ["missing", "modified", "other_benchmark"])
def test_reject_source_bundle_changes(submission_copy, change):
    source = submission_copy / "vendor/roborsi_source"
    if change == "missing":
        (source / "robots/robodojo/worker.py").unlink()
    elif change == "modified":
        with (source / "robots/robodojo/native.py").open("a") as stream:
            stream.write("\n# Unexpected evaluator change\n")
    else:
        unexpected = source / "robots/robotwin/env_server.py"
        unexpected.parent.mkdir(parents=True)
        unexpected.write_text("raise RuntimeError('not part of this submission')\n")
    with pytest.raises(ValueError, match="source bundle is missing or changed"):
        load_submission(submission_copy)


def test_clean_source_imports_and_contains_complete_robodojo_runtime(
    submission_copy, tmp_path
):
    metadata = load_submission(submission_copy)
    files = validate_source_bundle(metadata, submission_copy)
    assert {Path(name).parts[1] for name in files if name.startswith("robots/")} == {
        "__init__.py",
        "robodojo",
    }
    assert {
        Path(name).parts[1] for name in files if name.startswith("benchmarks/")
    } == {
        "build_inputs.py",
        "package.py",
        "robodojo",
    }
    source = submission_copy / metadata["source"]["bundle_path"]
    program = r"""
import importlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(root))
for name in (
    "roborsi.agents.config", "roborsi.embodied_agent", "roborsi.planner.codex_cli",
    "robots.robodojo.benchmark_service", "robots.robodojo.worker",
    "robots.robodojo.native", "robots.robodojo.eef_native",
    "robots.robodojo.agents.icl_pi05.bwrap_transport",
    "robots.robodojo.agents.icl_pi05.service",
    "robots.robodojo.agents.icl_pi05.runtime_records",
    "benchmarks.robodojo.prepare",
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), name
from roborsi.benchmarks.runtime_source import benchmark_source_digest, benchmark_source_files
runtime_files = benchmark_source_files("robodojo")
assert len(runtime_files) == 48
assert all((root / name).is_file() for name in runtime_files)
# Original cfa5915 runtime digest, also computed from the previous complete
# 417-file source export. Removing private image build inputs changes none of
# the actual native evaluation sources covered by this independent identity.
assert benchmark_source_digest(root, "robodojo") == "faf3ee08f1c0f8d760df76fdd5cfabc54cb24de5ce23afd926dcd0ae6227a0b2"
assert sorted(path.name for path in (root / "benchmarks/robodojo").iterdir()) == ["prepare.py"]
assert sorted(path.name for path in (root / "robots").iterdir()) == ["__init__.py", "robodojo"]
print(json.dumps({"source": str(root), "benchmark": "robodojo", "imports": "passed"}))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("ROBORSI_SOURCE", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(source)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["imports"] == "passed"


def test_vendor_subset_has_no_author_mount_paths_and_keeps_transport():
    vendor = POLICY_DIR / "vendor"
    for path in vendor.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        assert b"/mnt/afs/" not in raw, path.relative_to(vendor)
        assert b"/mnt/afs_e/" not in raw, path.relative_to(vendor)
    legacy = vendor / "roborsi_source/robots/robodojo/agents/icl_pi05"
    assert not (legacy / "resources").exists()
    assert not (legacy / "source").exists()
    assert (legacy / "bwrap_transport.py").is_file()
