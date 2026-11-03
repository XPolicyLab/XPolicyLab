"""Exercise installer prerequisites without downloads or production environments."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

POLICY = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def _python(
    path: Path,
    *,
    version: tuple[int, int, int] = (3, 10, 12),
    missing_memfd: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, runpy, sys\n"
        "from pathlib import Path\n"
        "record = {'kind': 'python', 'executable': __file__, 'args': sys.argv[1:], "
        "'key_present': 'CODEX_API_KEY' in os.environ, "
        "'proxy': os.environ.get('HTTPS_PROXY'), "
        "'ca': os.environ.get('SSL_CERT_FILE')}\n"
        "with Path(os.environ['INSTALL_EVENTS']).open('a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        f"sys.version_info = {(*version, 'final', 0)!r}\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    exec(sys.argv[2])\n"
        "    raise SystemExit(0)\n"
        "if Path(sys.argv[1]).name == 'bootstrap.py':\n"
        "    raise SystemExit(76)  # Observe automatic prerequisite delegation offline.\n"
        "if Path(sys.argv[1]).name == 'runtime_capabilities.py':\n"
        f"    if {missing_memfd!r} and '--python-only' in sys.argv:\n"
        "        # Exercise the production early check with the missing API;\n"
        "        # this never reaches a memfd syscall or launches a sandbox.\n"
        "        os.__dict__.pop('memfd_create', None)\n"
        "        sys.argv = sys.argv[1:]\n"
        "        runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "        raise AssertionError('missing memfd was accepted')\n"
        "    if '--python-only' not in sys.argv and os.environ.get('INSTALL_FAIL_SANDBOX'):\n"
        "        print('fixture sandbox capability refusal', file=sys.stderr)\n"
        "        raise SystemExit(74)\n"
        "    print(json.dumps({'test_fixture': True, 'sandbox_executed': False,\n"
        "                      'mode': 'python-only' if '--python-only' in sys.argv else 'sandbox'}))\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[-1:] == ['--metadata-only']:\n"
        "    if not os.environ.get('INSTALL_ADVANCE'):\n"
        "        raise SystemExit(72)  # Stop before installation or network access.\n"
        "    raise SystemExit(0)\n"
        "if os.environ.get('INSTALL_ADVANCE'):\n"
        "    if Path(sys.argv[1]).name == 'tool_install.py':\n"
        "        assert sys.argv[2:] in ([], ['--check-input'])\n"
        "        if os.environ.get('INSTALL_FAIL_TOOLS'):\n"
        "            print('fixture original tools unavailable', file=sys.stderr)\n"
        "            raise SystemExit(75)\n"
        "        print(json.dumps({'test_fixture': True, 'tool_payload_verified': False}))\n"
        "        raise SystemExit(0)\n"
        "    if sys.argv[1] == '-':\n"
        "        # Do not execute inline source preparation or downloads.\n"
        "        assert len(sys.argv) == 3, 'final receipt must follow the final preflight'\n"
        "        assert 'urlopen' in sys.stdin.read()\n"
        "        Path(sys.argv[2], 'runtime').mkdir(exist_ok=True)\n"
        "        raise SystemExit(0)\n"
        "    if Path(sys.argv[1]).name == 'probe.py':\n"
        "        assert Path(sys.argv[1]).parent.name == 'sim_bootstrap'\n"
        "        print(json.dumps({'test_fixture': True, 'sim_namespace_verified': False}))\n"
        "        raise SystemExit(0)\n"
        "    if Path(sys.argv[1]).name in {'icl_assets.py', 'pi05_runtime.py'}:\n"
        "        assert sys.argv[2:] == ['install']\n"
        "        raise SystemExit(0)\n"
        "    if Path(sys.argv[1]).name == 'preflight.py':\n"
        "        assert '--installation-only' in sys.argv\n"
        "        assert sys.argv[sys.argv.index('--verify-icl-task') + 1] == 'align_blocks'\n"
        "        print(json.dumps({'test_fixture': True, 'icl': {'materialization': {'materialization_verified': False}}}))\n"
        "        raise SystemExit(73)  # Stop after observing the exact final entry.\n"
        "raise SystemExit(99)\n"
    )
    path.chmod(0o755)


def _harness(tmp_path: Path):
    policy = tmp_path / "policy"
    policy.mkdir()
    shutil.copy2(POLICY / "install.sh", policy / "install.sh")
    shutil.copy2(POLICY / "runtime_capabilities.py", policy / "runtime_capabilities.py")
    (policy / "requirements.lock").write_text("# fixture only; no dependency install\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "dirname").symlink_to(shutil.which("dirname"))
    for name in ("uv", "setsid"):
        executable = tools / name
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with Path(os.environ['INSTALL_EVENTS']).open('a') as stream:\n"
            "    stream.write(json.dumps({'kind': Path(__file__).name, 'args': sys.argv[1:],\n"
            "                             'key_present': 'CODEX_API_KEY' in os.environ}) + '\\n')\n"
            "raise SystemExit(0 if os.environ.get('INSTALL_ADVANCE') else 99)\n"
        )
        executable.chmod(0o755)
    events = tmp_path / "events.jsonl"
    env = {
        "PATH": str(tools),
        "HOME": str(tmp_path),
        "PYTHONNOUSERSITE": "1",
        "INSTALL_EVENTS": str(events),
    }
    return policy, tools, events, env


def _records(events: Path) -> list[dict]:
    return [json.loads(line) for line in events.read_text().splitlines()]


@pytest.mark.parametrize("selection", ["explicit", "python3.10", "python3"])
def test_python_selection_and_install_secret_boundary(tmp_path, selection):
    policy, tools, events, env = _harness(tmp_path)
    _python(
        tools / "python3",
        version=(3, 12, 3) if selection != "python3" else (3, 10, 12),
    )
    if selection != "python3":
        _python(tools / "python3.10")
    if selection == "explicit":
        chosen = tools / "chosen-python"
        _python(chosen)
        env["ROBORSI_PYTHON"] = str(chosen)
    else:
        chosen = tools / selection
    env.update(
        CODEX_API_KEY="fixture-runtime-key",
        HTTPS_PROXY="http://fixture.invalid:8080",
        SSL_CERT_FILE=str(tmp_path / "fixture-ca.pem"),
    )
    result = subprocess.run(
        [
            BASH,
            "-c",
            '"$1" "$2"; status=$?; '
            '[[ -n "${CODEX_API_KEY:-}" ]] && echo parent-key-retained; '
            'exit "$status"',
            "installer-test",
            BASH,
            str(policy / "install.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 72, result.stderr
    assert "parent-key-retained" in result.stdout
    records = _records(events)
    assert len(records) == 3
    assert all(Path(record["executable"]) == chosen for record in records)
    assert records[-1]["args"][-1] == "--metadata-only"
    assert records[1]["args"] == [
        str(policy / "runtime_capabilities.py"),
        "--python-only",
    ]
    assert all(not record["key_present"] for record in records)
    assert all(record["proxy"] == env["HTTPS_PROXY"] for record in records)
    assert all(record["ca"] == env["SSL_CERT_FILE"] for record in records)
    assert "fixture-runtime-key" not in result.stdout + result.stderr


@pytest.mark.parametrize("version", [(3, 10, 11), (3, 10, 13), (3, 11, 17), (3, 12, 3)])
def test_unsuitable_default_python_delegates_to_bootstrap_before_metadata(
    tmp_path, version
):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3", version=version)
    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 76
    records = _records(events)
    assert len(records) == 2
    assert records[-1]["args"] == [str(policy / "bootstrap.py")]


@pytest.mark.parametrize("version", [(3, 10, 13), (3, 11, 17), (3, 12, 3)])
def test_existing_policy_python_is_checked_before_metadata(tmp_path, version):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10")
    _python(policy / ".venv/bin/python", version=version)
    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "Existing policy environment must use Python 3.10.12" in result.stderr
    records = _records(events)
    assert len(records) == 3
    assert records[0]["args"][0] == records[-1]["args"][0] == "-c"
    assert records[1]["args"][-1] == "--python-only"
    assert not any("--metadata-only" in record["args"] for record in records)


def test_explicit_wrong_python_does_not_fall_back_to_available_target(tmp_path):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10")
    selected = tools / "explicit-python"
    _python(selected, version=(3, 12, 3))
    env["ROBORSI_PYTHON"] = str(selected)
    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "requires Python 3.10.12" in result.stderr
    records = _records(events)
    assert len(records) == 1
    assert records[0]["executable"] == str(selected)


def test_missing_setsid_fails_before_metadata(tmp_path):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10")
    (tools / "setsid").unlink()
    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "Required installation prerequisite is missing: setsid" in result.stderr
    assert len(_records(events)) == 2
    assert _records(events)[-1]["args"][-1] == "--python-only"


@pytest.mark.parametrize("interpreter", ["selected", "existing-policy-env"])
def test_missing_memfd_fails_before_downloads_or_dependency_install(
    tmp_path, interpreter
):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10", missing_memfd=interpreter == "selected")
    if interpreter == "existing-policy-env":
        _python(policy / ".venv/bin/python", missing_memfd=True)
    frozen = policy / "vendor/unchanged-source.py"
    frozen.parent.mkdir()
    frozen.write_text("# Existing upstream source must not be patched.\n")
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    env["CODEX_API_KEY"] = "fixture-runtime-key"

    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == (76 if interpreter == "selected" else 1)
    assert "lacks os.memfd_create or MFD_CLOEXEC" in result.stderr
    assert "no fallback" in result.stderr
    records = _records(events)
    assert len(records) == (3 if interpreter == "selected" else 4)
    assert all(record["kind"] == "python" for record in records)
    if interpreter == "selected":
        assert records[-1]["args"] == [str(policy / "bootstrap.py")]
        records = records[:-1]
    assert records[-1]["args"] == [
        str(policy / "runtime_capabilities.py"),
        "--python-only",
    ]
    assert all(
        record["args"][0] == "-c" or record["args"][-1] == "--python-only"
        for record in records
    )
    assert all(not record["key_present"] for record in records)
    assert not (policy / "runtime").exists()
    assert list(frozen.parent.iterdir()) == [frozen]
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before
    assert "fixture-runtime-key" not in result.stdout + result.stderr


def test_installer_reaches_offline_sandbox_sim_probe_and_selected_icl_check(tmp_path):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10")
    _python(policy / ".venv/bin/python")
    env.update(INSTALL_ADVANCE="1", CODEX_API_KEY="fixture-runtime-key")

    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    # The real shell reaches the final check, whose fixture deliberately stops
    # before publishing any production installation/acceptance receipt.
    assert result.returncode == 73, result.stderr
    records = _records(events)
    capability_calls = [
        record
        for record in records
        if Path(record["args"][0]).name == "runtime_capabilities.py"
    ]
    assert [record["args"][1:] for record in capability_calls] == [
        ["--python-only"],
        ["--python-only"],
        ["--policy-dir", str(policy)],
    ]
    assert [record["args"][0:2] for record in records if record["kind"] == "uv"] == [
        ["pip", "sync"],
        ["pip", "install"],
    ]
    sandbox_index = records.index(capability_calls[-1])
    tool_calls = [
        (index, record)
        for index, record in enumerate(records)
        if Path(record["args"][0]).name == "tool_install.py"
    ]
    assert [record["args"][1:] for _, record in tool_calls] == [["--check-input"], []]
    first_uv = next(
        index for index, record in enumerate(records) if record["kind"] == "uv"
    )
    assert tool_calls[0][0] < first_uv < tool_calls[1][0] < sandbox_index
    simulator_index = next(
        index
        for index, record in enumerate(records)
        if Path(record["args"][0]).name == "probe.py"
    )
    icl_index = next(
        index
        for index, record in enumerate(records)
        if Path(record["args"][0]).name == "icl_assets.py"
    )
    vla_index = next(
        index
        for index, record in enumerate(records)
        if Path(record["args"][0]).name == "pi05_runtime.py"
    )
    assert sandbox_index < simulator_index < icl_index < vla_index < len(records) - 1
    assert records[simulator_index]["args"] == [
        str(policy / "sim_bootstrap/probe.py"),
        "--workspace",
        str(policy.parents[2]),
        "--allow-missing-workspace",
    ]
    assert records[-1]["args"] == [
        str(policy / "preflight.py"),
        "--config",
        str(policy / "deploy.yml"),
        "--installation-only",
        "--verify-icl-task",
        "align_blocks",
    ]
    assert all(not record["key_present"] for record in records)
    sandbox = json.loads((policy / "runtime/sandbox-install.json").read_text())
    assert sandbox == {
        "test_fixture": True,
        "sandbox_executed": False,
        "mode": "sandbox",
    }
    simulator = json.loads((policy / "runtime/sim-bootstrap-install.json").read_text())
    assert simulator["test_fixture"] is True
    assert simulator["sim_namespace_verified"] is False
    final_check = json.loads((policy / "runtime/install-preflight.json").read_text())
    assert final_check["icl"]["materialization"]["materialization_verified"] is False
    assert not (policy / "runtime/install-receipt.json").exists()


def test_sandbox_gate_failure_stops_before_large_icl_and_vla_downloads(tmp_path):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10")
    _python(policy / ".venv/bin/python")
    env.update(INSTALL_ADVANCE="1", INSTALL_FAIL_SANDBOX="1")

    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 74, result.stderr
    assert "fixture sandbox capability refusal" in result.stderr
    records = _records(events)
    assert records[-1]["args"] == [
        str(policy / "runtime_capabilities.py"),
        "--policy-dir",
        str(policy),
    ]
    assert not any(
        Path(record["args"][0]).name in {"probe.py", "icl_assets.py", "pi05_runtime.py"}
        for record in records
    )
    assert not (policy / "runtime/install-receipt.json").exists()


def test_missing_original_tools_stops_before_source_or_dependency_downloads(tmp_path):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10")
    env.update(INSTALL_ADVANCE="1", INSTALL_FAIL_TOOLS="1")
    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 75
    assert "original tools unavailable" in result.stderr
    records = _records(events)
    assert records[-1]["args"] == [str(policy / "tool_install.py"), "--check-input"]
    assert not any(
        record["kind"] == "uv" or record["args"][0] == "-" for record in records
    )


def test_wrong_default_target_uses_available_seed_without_installing(tmp_path):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10", version=(3, 10, 11))
    _python(tools / "python3", version=(3, 11, 17))
    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 76
    records = _records(events)
    assert [Path(row["executable"]).name for row in records] == [
        "python3.10",
        "python3",
    ]
    assert records[-1]["args"] == [str(policy / "bootstrap.py")]


def test_missing_uv_bootstraps_using_explicit_compatible_host(tmp_path):
    policy, tools, events, env = _harness(tmp_path)
    _python(tools / "python3.10")
    env["ROBORSI_PYTHON"] = str(tools / "python3.10")
    (tools / "uv").unlink()
    result = subprocess.run(
        [BASH, str(policy / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 76
    assert _records(events)[-1]["args"] == [str(policy / "bootstrap.py")]
    assert not (policy / "runtime").exists()
