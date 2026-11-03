"""Exercise install.sh with fake package metadata and pip; never install packages.

The installer's real metadata-reading Python runs against controlled versions.
Every pip command is recorded, not executed. These tests cover shell control
flow and constraint propagation, not pip resolution or a clean installation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


POLICY_DIR = Path(__file__).resolve().parents[1]
MOCK_PYTHON = r'''
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

args = sys.argv[1:]
if args[:2] == ["-m", "pip"]:
    log = Path(os.environ["MOCK_PIP_LOG"])
    previous = log.read_text().splitlines() if log.exists() else []
    constraints = [args[i + 1] for i, arg in enumerate(args) if arg == "-c"]
    record = {"args": args, "constraints": [
        {"path": name, "content": Path(name).read_text()} for name in constraints
    ]}
    with log.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if len(previous) + 1 == int(os.environ.get("MOCK_FAIL_PIP_CALL", "0")):
        print("Simulated pip dependency resolution failure", file=sys.stderr)
        raise SystemExit(37)
elif args and args[0] == "-":
    installed = json.loads(os.environ["MOCK_INSTALLED"])
    def installed_version(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]
    code = sys.stdin.read()
    sys.argv = ["-", *args[1:]]
    with patch("importlib.metadata.version", installed_version):
        exec(compile(code, "<installer-metadata>", "exec"), {"__name__": "__main__"})
else:
    raise SystemExit("Unexpected Python invocation in installer fixture")
'''


def run_case(root, name, installed, *, extra=None, missing_extra=False, fail_call=0):
    case = root / name
    case.mkdir()
    temporary = case / "temporary files"
    temporary.mkdir()
    pip_log = case / "pip.jsonl"
    env = dict(os.environ)
    env.pop("RSIWAM_CONSTRAINTS", None)
    env.update(
        PATH=str(root / "bin") + os.pathsep + env.get("PATH", ""),
        TMPDIR=str(temporary),
        MOCK_INSTALLED=json.dumps(installed),
        MOCK_PIP_LOG=str(pip_log),
        MOCK_FAIL_PIP_CALL=str(fail_call),
    )
    extra_path = case / "site constraints.txt"
    if extra is not None:
        extra_path.write_text(extra)
        env["RSIWAM_CONSTRAINTS"] = str(extra_path)
    if missing_extra:
        env["RSIWAM_CONSTRAINTS"] = str(extra_path)
    result = subprocess.run(
        ["bash", str(root / "XPolicyLab/policy/RSIWAM/install.sh")],
        env=env, text=True, capture_output=True, check=False,
    )
    calls = [json.loads(line) for line in pip_log.read_text().splitlines()] if pip_log.exists() else []
    assert not list(temporary.iterdir()), f"Temporary constraint leak in {name}"
    for call in calls:
        pins = call["constraints"][0]
        assert pins["content"] == "".join(f"{key}=={installed[key]}\n" for key in ("torch", "torchvision"))
        assert not Path(pins["path"]).exists(), f"Temporary pins survive exit in {name}"
        assert "constraints-reference.txt" not in " ".join(call["args"])
        requirements = root / "XPolicyLab/policy/RSIWAM/bundle/requirements.txt"
        assert call["constraints"][1] == {"path": str(requirements), "content": requirements.read_text()}
        if extra is not None:
            assert call["constraints"][2] == {"path": str(extra_path), "content": extra}
        else:
            assert len(call["constraints"]) == 2
    if calls:
        assert len({call["constraints"][0]["path"] for call in calls}) == 1
    return result, calls


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix="rsiwam-installer-check-") as location:
        root = Path(location)
        policy = root / "XPolicyLab/policy/RSIWAM"
        policy.mkdir(parents=True)
        shutil.copyfile(POLICY_DIR / "install.sh", policy / "install.sh")
        (policy / "bundle").mkdir()
        shutil.copyfile(POLICY_DIR / "bundle/requirements.txt", policy / "bundle/requirements.txt")
        (root / "XPolicyLab/setup_policy_server.py").touch()
        (root / "bin").mkdir()
        python = root / "bin/python"
        python.write_text(f"#!{sys.executable}\n" + MOCK_PYTHON)
        python.chmod(0o755)
        a800 = {"torch": "2.5.1+cu118", "torchvision": "0.20.1+cu118"}
        for missing in ("torch", "torchvision"):
            result, calls = run_case(root, "missing-" + missing, {k: v for k, v in a800.items() if k != missing})
            assert result.returncode != 0 and not calls
            assert f"requires preinstalled {missing}" in result.stderr
            checks.append("missing_" + missing + "_fails_before_pip")
        for name, versions in (
            ("a800", a800),
            ("h800", {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130"}),
        ):
            result, calls = run_case(root, name, versions)
            assert result.returncode == 0 and len(calls) == 3
            assert "--no-deps" in calls[1]["args"]
            assert "-r" in calls[0]["args"] and "-e" in calls[2]["args"]
            checks.append(name + "_versions_pinned_for_all_three_pip_commands")
        checks.append("policy_dependency_bounds_propagated_to_xpl_install")
        result, calls = run_case(root, "extra", a800, extra="numpy==1.26.4\n")
        assert result.returncode == 0 and len(calls) == 3
        checks.append("optional_constraints_propagated_without_replacing_pins")
        result, calls = run_case(root, "missing-extra", a800, missing_extra=True)
        assert result.returncode != 0 and not calls and "readable constraints file" in result.stderr
        checks.append("missing_optional_constraints_fails_before_pip")
        result, calls = run_case(root, "conflict", a800, extra="torch==2.11.0+cu130\n", fail_call=1)
        assert result.returncode == 37 and len(calls) == 1
        checks.append("simulated_constraint_conflict_stops_and_cleans_up")
        result, calls = run_case(root, "late-failure", a800, fail_call=3)
        assert result.returncode == 37 and len(calls) == 3
        checks.append("later_pip_failure_stops_and_cleans_up")
    print(json.dumps({
        "status": "INSTALLER_CPU_CONTRACT_CHECKS_PASSED",
        "checks": checks,
        "all_cases_removed_temporary_constraints": True,
        "real_metadata_script_executed": True,
        "package_versions": "MOCKED",
        "pip": "RECORDED_NOT_EXECUTED",
        "clean_installation_tested": False,
        "gpu_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
