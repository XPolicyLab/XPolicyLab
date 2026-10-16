"""Exercise the policy hook through unmodified official simulator launchers.

Conda and the native main are CPU-only fixtures. The shared launcher is from
XPolicyLab 34d6135d and eval_policy.sh from RoboDojo 726e9aab, byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

POLICY = Path(__file__).resolve().parents[1]
FIXTURES = POLICY / "tests/fixtures"
FIXED_FILES = {
    "official_run_sim_env_client.sh": "ada1dcdce349ce589b1ad4a02c0af6f983fc0dfeaafd184a98dfb447dafb441c",
    "official_eval_policy.sh": "b27ca311a2f0eb9b0ac182d7ed0d7e8e13f08cc5d1467eff6f7df3ffdb905f3e",
}


def _fixture_tree(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "native workspace with spaces"
    policy = root / "XPolicyLab/policy/SenseNova-RoboRSI"
    utils = root / "XPolicyLab/utils"
    policy.mkdir(parents=True)
    utils.mkdir()
    for name in ("setup_eval_env_client.sh", "setup_eval_policy_server.sh", "deploy.yml"):
        shutil.copy(POLICY / name, policy)
    shutil.copytree(
        POLICY / "sim_bootstrap",
        policy / "sim_bootstrap",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("setup_env_client.sh", "resolve_eval_env_type.sh"):
        shutil.copy(POLICY.parents[1] / "utils" / name, utils)
    for source, relative in (
        ("official_run_sim_env_client.sh", "XPolicyLab/utils/run_sim_env_client.sh"),
        ("official_eval_policy.sh", "scripts/eval_policy.sh"),
    ):
        payload = (FIXTURES / source).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == FIXED_FILES[source]
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    for relative, text in {
        "utils/load_file.py": "from sdk_dependency import MARKER\n",
        "env/seed_manager/seed_manager.py": "# native fixture\n",
        "env_cfg/arx_x5.yml": '{"config":{"sim":"fixture"}}\n',
        "env_cfg/sim/fixture.yml": '{"render_interval":1,"scene":{"num_envs":1}}\n',
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    client = root / "src/eval_client/main.py"
    client.parent.mkdir(parents=True)
    client.write_text(
        "import builtins, json, os, sys\n"
        "payload = {'fixture_native_client': True, 'argv': sys.argv, "
        "'bootstrap_bound_before_main': 'utils.load_file' in sys.modules, "
        "'sdk_sitecustomize_chained': getattr(builtins, 'FIXTURE_SDK_CHAINED', False), "
        "'pythonpath': os.environ.get('PYTHONPATH'), "
        "'unbuffered_stdout': sys.stdout.write_through}\n"
        "print('NATIVE_FIXTURE_JSON:' + json.dumps(payload), flush=True)\n"
        "assert payload['bootstrap_bound_before_main']\n"
        "from utils.load_file import MARKER\n"
        "assert MARKER == 'sdk-preserved'\n"
        "state = os.environ.get('ROBORSI_FIXTURE_RETRY_STATE')\n"
        "if state:\n"
        "    try:\n"
        "        with open(state, 'x'):\n"
        "            pass\n"
        "    except FileExistsError:\n"
        "        pass\n"
        "    else:\n"
        "        raise SystemExit(99)\n"
        "raise SystemExit(int(os.environ['ROBORSI_FIXTURE_EXIT_CODE']))\n"
    )
    sdk = tmp_path / "SDK path with spaces"
    sdk.mkdir()
    (sdk / "sdk_dependency.py").write_text("MARKER = 'sdk-preserved'\n")
    (sdk / "sitecustomize.py").write_text(
        "import builtins\nbuiltins.FIXTURE_SDK_CHAINED = True\n"
    )
    conda = tmp_path / "fake conda"
    conda_bin = conda / "bin"
    conda_bin.mkdir(parents=True)
    conda_executable = conda_bin / "conda"
    conda_executable.write_text(
        f"#!/bin/bash\nprintf '%s\\n' {shlex.quote(str(conda))}\n"
    )
    conda_executable.chmod(0o755)
    profile = conda / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "conda() {\n"
        '  printf "%s\\n" "$1" >> "$ROBORSI_FIXTURE_ACTIVATIONS"\n'
        '  if [[ "$1" == activate || "$1" == deactivate ]]; then\n'
        '    export PYTHONPATH="$ROBORSI_FIXTURE_SDK"\n'
        '    export PATH="$ROBORSI_FIXTURE_BIN:$PATH"\n'
        "  fi\n"
        "}\n"
    )
    logger = tmp_path / "record_python.py"
    logger.write_text(
        "import json, os, sys\n"
        "with open(os.environ['ROBORSI_FIXTURE_PYTHON_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'argv': sys.argv[1:], "
        "'pythonpath': os.environ.get('PYTHONPATH')}) + '\\n')\n"
    )
    python = conda_bin / "python"
    python.write_text(
        "#!/bin/bash\n"
        f'{shlex.quote(sys.executable)} -S {shlex.quote(str(logger))} "$@"\n'
        # A real Conda interpreter has the same prefix with and without -S.
        # The test harness may itself be a venv on Python <= 3.13.
        'if [[ "${1:-}" == -S && "${2:-}" == -c ]]; then\n'
        f"  printf '%s\\n' {shlex.quote(str(Path(sys.prefix).resolve()))}\n"
        "  exit 0\n"
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    shutil.copy(python, conda_bin / "python3")
    return {
        "root": root,
        "policy": policy,
        "sdk": sdk,
        "conda_bin": conda_bin,
        "calls": tmp_path / "python-calls.jsonl",
        "activations": tmp_path / "activations.txt",
    }


def _environment(tree: dict[str, Path], exit_code: int) -> dict[str, str]:
    return {
        "PATH": f"{tree['conda_bin']}{os.pathsep}{os.defpath}",
        "HOME": str(tree["root"].parent),
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(tree["root"].parent / "discarded-before-final-launch"),
        "EVAL_ENV_TYPE": "sim",
        "ROBORSI_FIXTURE_BIN": str(tree["conda_bin"]),
        "ROBORSI_FIXTURE_SDK": str(tree["sdk"]),
        "ROBORSI_FIXTURE_ACTIVATIONS": str(tree["activations"]),
        "ROBORSI_FIXTURE_PYTHON_CALLS": str(tree["calls"]),
        "ROBORSI_FIXTURE_EXIT_CODE": str(exit_code),
        "ROBODOJO_MAX_BASH_RETRIES": "1",
        # The unchanged official launcher ignores this obsolete extension.
        "XPL_SIM_PYTHONPATH_PREFIX": "invalid-and-not-used-by-this-policy",
    }


def _native_payloads(result: subprocess.CompletedProcess) -> list[dict]:
    records = [
        json.loads(line.removeprefix("NATIVE_FIXTURE_JSON:"))
        for line in result.stdout.splitlines()
        if line.startswith("NATIVE_FIXTURE_JSON:")
    ]
    assert records, result.stdout + result.stderr
    return records


@pytest.mark.parametrize("selection", ["uv", "explicit"])
def test_controller_server_uses_selected_python_without_caller_overrides(
    tmp_path, selection
):
    tree = _fixture_tree(tmp_path)
    policy = tree["policy"]
    python = policy / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    calls = tmp_path / "server-calls.jsonl"
    (tree["root"] / "controller_dependency.py").write_text("MARKER = 'selected-root'\n")
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from controller_dependency import MARKER\n"
        "with open(os.environ['ROBORSI_TEST_SERVER_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'argv': sys.argv[1:], 'marker': MARKER, "
        "'pythonpath': os.environ.get('PYTHONPATH'), "
        "'python_overrides': [name for name in ('PYTHONHOME', 'VIRTUAL_ENV') if name in os.environ], "
        "'no_user_site': os.environ.get('PYTHONNOUSERSITE'), "
        "'key_present': bool(os.environ.get('CODEX_API_KEY')), "
        "'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\\n')\n"
    )
    python.chmod(0o755)
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "CODEX_API_KEY": "fixture-runtime-key",
        "PYTHONHOME": str(tmp_path / "foreign-python"),
        "PYTHONPATH": str(tree["sdk"]),
        "VIRTUAL_ENV": str(tmp_path / "foreign-venv"),
        "PYTHONNOUSERSITE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ROBORSI_TEST_SERVER_CALLS": str(calls),
    }
    result = subprocess.run(
        [
            "bash",
            str(policy / "setup_eval_policy_server.sh"),
            "RoboDojo",
            "align_blocks",
            "roborsi-codex-v1",
            "arx_x5",
            "joint",
            "0",
            "2",
            "uv" if selection == "uv" else str(python.parent.parent),
            "12345",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [Path(row["argv"][0]).name for row in records] == ["preflight.py", "serve.py"]
    for row in records:
        assert row["pythonpath"] == os.pathsep.join(
            (str(tree["root"]), str(tree["root"] / "XPolicyLab"))
        )
        assert row["python_overrides"] == []
        assert row["no_user_site"] == "1"
        assert row["marker"] == "selected-root"
        assert row["key_present"] is True
        assert row["gpu"] == "2"
    assert "fixture-runtime-key" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "exit_code,retry",
    [
        (0, False),
        (23, False),
        (78, False),
        (99, False),
        (134, False),
        (139, False),
        (0, True),
    ],
)
def test_policy_bootstrap_survives_official_conda_reactivation(
    tmp_path, exit_code, retry
):
    tree = _fixture_tree(tmp_path)
    env = _environment(tree, exit_code)
    if retry:
        env["ROBODOJO_MAX_BASH_RETRIES"] = "2"
        env["ROBORSI_FIXTURE_RETRY_STATE"] = str(tmp_path / "retry-state")
    result = subprocess.run(
        [
            "bash",
            str(tree["policy"] / "setup_eval_env_client.sh"),
            "RoboDojo",
            "align_blocks",
            "roborsi-codex-v1",
            "arx_x5",
            "joint",
            "0",
            "2",
            "FakeSim",
            "fixture-info",
            "12345",
            "127.0.0.1",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    payloads = _native_payloads(result)
    assert len(payloads) == (2 if retry else 1)
    assert all(payload["bootstrap_bound_before_main"] is True for payload in payloads)
    assert all(payload["sdk_sitecustomize_chained"] is True for payload in payloads)
    payload = payloads[-1]
    assert payload["bootstrap_bound_before_main"] is True
    assert payload["sdk_sitecustomize_chained"] is True
    assert payload["unbuffered_stdout"] is True
    assert payload["argv"][0] == "src/eval_client/main.py"
    for key, expected in {
        "--task_name": "align_blocks",
        "--env_cfg_type": "arx_x5",
        "--policy_name": "SenseNova-RoboRSI",
        "--port": "12345",
        "--protocol": "ws",
        "--policy_server_url": "ws://127.0.0.1:12345",
        "--seed": "0",
        "--device_id": "2",
    }.items():
        assert payload["argv"][payload["argv"].index(key) + 1] == expected
    calls = [json.loads(line) for line in tree["calls"].read_text().splitlines()]
    final = [
        row for row in calls if row["argv"][:2] == ["-u", "src/eval_client/main.py"]
    ]
    assert len(final) == (2 if retry else 1)
    bootstrap = str(tree["policy"] / "sim_bootstrap")
    assert all(row["pythonpath"].split(os.pathsep)[0] == bootstrap for row in final)
    assert str(tree["sdk"]) in final[0]["pythonpath"].split(os.pathsep)
    metadata = [row for row in calls if row["argv"] == ["-"]]
    assert len(metadata) == 1
    assert metadata[0]["pythonpath"] == str(tree["sdk"])
    assert "deactivate" in tree["activations"].read_text().splitlines()
    assert tree["activations"].read_text().splitlines().count("activate") >= 3


def test_official_launcher_without_policy_wrapper_loses_startup_hook(tmp_path):
    tree = _fixture_tree(tmp_path)
    env = _environment(tree, 0)
    env.update(
        ROBORSI_SIM_BOOTSTRAP="1",
        ROBORSI_SIM_BENCH_ROOT=str(tree["root"]),
        ROBORSI_SIM_PYTHON_PREFIX=str(Path(sys.prefix).resolve()),
        PYTHONPATH=str(tree["policy"] / "sim_bootstrap"),
    )
    result = subprocess.run(
        [
            "bash",
            str(tree["root"] / "XPolicyLab/utils/run_sim_env_client.sh"),
            "false",
            "FakeSim",
            "12345",
            "RoboDojo",
            "align_blocks",
            "arx_x5",
            "SenseNova-RoboRSI",
            "fixture-info",
            str(tree["root"]),
            "0",
            "2",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert _native_payloads(result)[0]["bootstrap_bound_before_main"] is False
