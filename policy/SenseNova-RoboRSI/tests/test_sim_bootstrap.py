"""Fresh-process checks for the opt-in simulator import bootstrap."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

POLICY = Path(__file__).resolve().parents[1]
BOOTSTRAP = POLICY / "sim_bootstrap"
HOOK_ENV = (
    "ROBORSI_SIM_BOOTSTRAP",
    "ROBORSI_SIM_BENCH_ROOT",
    "ROBORSI_SIM_PYTHON_PREFIX",
    "ROBORSI_SIM_BOOTSTRAP_DIR",
)


def native_workspace(root):
    for relative, content in {
        "utils/load_file.py": "MARKER = 'native-load-file'\n",
        "env/seed_manager/seed_manager.py": "# native seed manager\n",
        "scripts/eval_policy.sh": "#!/bin/bash\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


@pytest.fixture
def workspace(tmp_path):
    return native_workspace(tmp_path / "native workspace")


def process_env(
    workspace, *, enabled=True, prefix=None, extra_path=(), bootstrap=BOOTSTRAP
):
    env = os.environ.copy()
    for key in HOOK_ENV:
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(map(str, (bootstrap, *extra_path)))
    if enabled:
        env.update(
            ROBORSI_SIM_BOOTSTRAP="1",
            ROBORSI_SIM_BENCH_ROOT=str(workspace),
            ROBORSI_SIM_PYTHON_PREFIX=str(prefix or Path(sys.prefix).resolve()),
        )
    return env


def run_python(source, env):
    return subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_native_load_file_survives_later_kit_style_finder(workspace):
    result = run_python(
        """
import importlib, importlib.abc, pathlib, sys
class KitFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'utils' or fullname.startswith('utils.'):
            raise AssertionError('foreign finder won: ' + fullname)
sys.meta_path.insert(0, KitFinder())
native = importlib.import_module('utils.load_file')
assert native.MARKER == 'native-load-file'
assert list(sys.modules['utils'].__path__) == [str(pathlib.Path(native.__file__).parent)]
print('app-stub-reached')
""",
        process_env(workspace),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "app-stub-reached"


def test_chain_preserves_external_sitecustomize_identity_and_runs_once(
    workspace, tmp_path
):
    external = tmp_path / "external"
    external.mkdir()
    (external / "sitecustomize.py").write_text(
        "import sitecustomize, sys\n"
        "assert __name__ == 'sitecustomize'\n"
        "assert sitecustomize is sys.modules[__name__]\n"
        "assert sitecustomize.__dict__ is globals()\n"
        "assert 'utils.load_file' in sys.modules\n"
        "sys.external_hook_runs = getattr(sys, 'external_hook_runs', 0) + 1\n"
        "MARKER = 'external-hook'\n"
    )
    result = run_python(
        "import sitecustomize, sys; assert sys.external_hook_runs == 1; "
        "assert sitecustomize.MARKER == 'external-hook'",
        process_env(workspace, extra_path=(external,)),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["disabled", "main-prefix", "vla-prefix"])
def test_unrelated_interpreters_only_chain_external_hook(workspace, tmp_path, mode):
    external = tmp_path / "external"
    external.mkdir()
    (external / "sitecustomize.py").write_text(
        "import sys\nsys.external_hook_runs = getattr(sys, 'external_hook_runs', 0) + 1\n"
    )
    result = run_python(
        "import sys; assert 'utils' not in sys.modules; assert sys.external_hook_runs == 1",
        process_env(
            workspace,
            enabled=mode != "disabled",
            prefix=tmp_path / mode if mode != "disabled" else None,
            extra_path=(external,),
        ),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "external_source",
    [
        "raise RuntimeError('external-hook-failed')\n",
        "import sys, types\nsys.modules['utils'] = types.ModuleType('utils')\n",
    ],
)
def test_external_hook_error_or_rebinding_fails_before_app(
    workspace, tmp_path, external_source
):
    external = tmp_path / "external"
    external.mkdir()
    (external / "sitecustomize.py").write_text(external_source)
    result = run_python(
        "print('app-stub-reached')", process_env(workspace, extra_path=(external,))
    )
    assert result.returncode == 78
    assert "startup refused" in result.stderr
    assert "app-stub-reached" not in result.stdout


def test_external_hook_error_never_echoes_runtime_credentials(workspace, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "sitecustomize.py").write_text(
        'import os\nraise RuntimeError(os.environ["CODEX_API_KEY"])\n'
    )
    sentinel = "synthetic-runtime-credential-must-not-be-logged"
    environment = process_env(workspace, extra_path=(external,))
    environment["CODEX_API_KEY"] = sentinel
    result = run_python("print('app-stub-reached')", environment)
    assert result.returncode == 78
    assert "startup refused: RuntimeError" in result.stderr
    assert sentinel not in result.stdout
    assert sentinel not in result.stderr
    assert "Traceback" not in result.stderr
    assert "app-stub-reached" not in result.stdout


def test_preexisting_foreign_utils_is_not_evicted(workspace):
    result = run_python(
        """
import sys, types
from _roborsi_sim_namespace import BootstrapError, pin_native_utils
foreign = types.ModuleType('utils')
foreign.__file__ = '/foreign/cv2/utils/__init__.py'
sys.modules['utils'] = foreign
try:
    pin_native_utils(sys.argv[1])
except BootstrapError:
    assert sys.modules['utils'] is foreign
else:
    raise AssertionError('foreign module was accepted')
""".replace("sys.argv[1]", repr(str(workspace))),
        process_env(workspace, enabled=False),
    )
    assert result.returncode == 0, result.stderr


def test_already_registered_foreign_finder_cannot_choose_native_source(workspace):
    result = run_python(
        """
import importlib.abc, sys
from _roborsi_sim_namespace import pin_native_utils
class ForeignFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'utils' or fullname.startswith('utils.'):
            raise AssertionError('foreign source must never execute')
sys.meta_path.insert(0, ForeignFinder())
binding = pin_native_utils(WORKSPACE)
assert binding.load_file.MARKER == 'native-load-file'
binding.verify()
""".replace("WORKSPACE", repr(str(workspace))),
        process_env(workspace, enabled=False),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("problem", ["missing", "regular-package", "escaping-source"])
def test_invalid_native_layout_fails_before_app(workspace, tmp_path, problem):
    if problem == "missing":
        (workspace / "utils/load_file.py").unlink()
    elif problem == "regular-package":
        (workspace / "utils/__init__.py").write_text(
            "raise AssertionError('must not run')\n"
        )
    else:
        outside = tmp_path / "foreign.py"
        outside.write_text("MARKER = 'foreign'\n")
        (workspace / "utils/load_file.py").unlink()
        (workspace / "utils/load_file.py").symlink_to(outside)
    result = run_python("print('app-stub-reached')", process_env(workspace))
    assert result.returncode == 78
    assert "app-stub-reached" not in result.stdout


def test_probe_relocates_with_checkout_and_works_with_readonly_native_source(tmp_path):
    root = native_workspace(tmp_path / "relocated checkout with spaces")
    bootstrap = root / "XPolicyLab/policy/SenseNova-RoboRSI/sim_bootstrap"
    shutil.copytree(BOOTSTRAP, bootstrap, ignore=shutil.ignore_patterns("__pycache__"))
    for source in (
        root / "utils/load_file.py",
        root / "env/seed_manager/seed_manager.py",
    ):
        source.chmod(0o444)
    result = subprocess.run(
        [sys.executable, str(bootstrap / "probe.py")],
        env=process_env(root, enabled=False, bootstrap=bootstrap),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["workspace"] == str(root)
    assert payload["sim_namespace_verified"] is True


@pytest.mark.parametrize("allow_missing", [False, True])
def test_probe_missing_workspace_is_explicit_metadata_only(tmp_path, allow_missing):
    root = tmp_path / "no native workspace"
    command = [sys.executable, str(BOOTSTRAP / "probe.py"), "--workspace", str(root)]
    if allow_missing:
        command.append("--allow-missing-workspace")
    result = subprocess.run(
        command,
        env=process_env(root, enabled=False),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == (0 if allow_missing else 78)
    if allow_missing:
        assert "WARNING" in result.stderr
        assert json.loads(result.stdout)["sim_namespace_verified"] is False
        assert json.loads(result.stdout)["status"] == "skipped"


def test_debug_client_does_not_inject_simulator_hook(tmp_path):
    # Copy the real entrypoint into a minimal relocated checkout. The client
    # executable records its environment instead of importing a simulator.
    root = tmp_path / "debug checkout"
    policy = root / "XPolicyLab/policy/SenseNova-RoboRSI"
    policy.mkdir(parents=True)
    shutil.copy(POLICY / "setup_eval_env_client.sh", policy)
    utils = root / "XPolicyLab/utils"
    utils.mkdir(parents=True)
    shutil.copy(POLICY.parents[1] / "utils/resolve_eval_env_type.sh", utils)
    client = tmp_path / "debug-python"
    client.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        f"print(json.dumps({{key: os.environ.get(key) for key in {HOOK_ENV!r}}}))\n"
    )
    client.chmod(0o755)
    env = process_env(root, enabled=False, bootstrap=tmp_path / "no-bootstrap")
    env["EVAL_ENV_TYPE"] = "debug"
    result = subprocess.run(
        [
            "bash",
            str(policy / "setup_eval_env_client.sh"),
            "bench",
            "align_blocks",
            "ckpt",
            "env",
            "action",
            "0",
            "0",
            str(client),
            "info",
            "12345",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert all(
        value is None for value in json.loads(result.stdout.splitlines()[-1]).values()
    )


@pytest.mark.parametrize(
    "mode",
    [
        "sim",
        "sim-sdk-hooks",
        "sim-missing",
        "sim-source-failed",
        "sim-activate-failed",
        "real_world",
    ],
)
def test_standard_client_sim_guard_and_real_world_isolation(tmp_path, mode):
    root = native_workspace(tmp_path / "shell checkout")
    if mode == "sim-missing":
        (root / "utils/load_file.py").unlink()
    policy = root / "XPolicyLab/policy/SenseNova-RoboRSI"
    policy.mkdir(parents=True)
    shutil.copy(POLICY / "setup_eval_env_client.sh", policy)
    shutil.copytree(
        BOOTSTRAP,
        policy / "sim_bootstrap",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    utils = root / "XPolicyLab/utils"
    utils.mkdir(parents=True)
    shutil.copy(POLICY.parents[1] / "utils/resolve_eval_env_type.sh", utils)
    (utils / "setup_env_client.sh").write_text(
        'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
        'conda activate "$3"\n'
        "python -c 'import json, os, sys; "
        'print(json.dumps({"framework_reached": True, "hook": os.environ.get("ROBORSI_SIM_BOOTSTRAP"), '
        '"native_bound": "utils.load_file" in sys.modules}))' + "'\n"
    )
    if mode == "sim-sdk-hooks":
        # Exercise the real framework's second deactivate/activate cycle. The
        # image hook restores a fixed SDK path, discarding the policy prefix.
        shutil.copy(POLICY / "deploy.yml", policy)
        shutil.copy(POLICY.parents[1] / "utils/setup_env_client.sh", utils)
        shutil.copy(
            POLICY / "tests/fixtures/official_run_sim_env_client.sh",
            utils / "run_sim_env_client.sh",
        )
        (root / "scripts/eval_policy.sh").write_text(
            '#!/bin/bash\ncd "$(dirname "$0")/.."\npython -u src/eval_client/main.py\n'
        )
        client = root / "src/eval_client/main.py"
        client.parent.mkdir(parents=True)
        client.write_text(
            "import json, os, sys\n"
            'print(json.dumps({"framework_reached": True, "hook": os.environ.get("ROBORSI_SIM_BOOTSTRAP"), '
            '"native_bound": "utils.load_file" in sys.modules}))\n'
        )
    conda_root = tmp_path / "fake conda"
    conda_bin = conda_root / "bin"
    conda_bin.mkdir(parents=True)
    sdk = tmp_path / "SDK python packages"
    sdk.mkdir()
    (sdk / "sdk_dependency.py").write_text("MARKER = 'sdk-activated'\n")
    if mode == "sim-sdk-hooks":
        (root / "utils/load_file.py").write_text(
            "from sdk_dependency import MARKER\nassert MARKER == 'sdk-activated'\n"
        )
    conda = conda_bin / "conda"
    conda.write_text(f"#!{sys.executable}\nprint({str(conda_root)!r})\n")
    conda.chmod(0o755)
    profile = conda_root / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        # Match the real Isaac activation hook's read of an optional Zsh value.
        ('test -z "$ZSH_VERSION"\n' if mode == "sim-sdk-hooks" else "")
        + ("return 8\n" if mode == "sim-source-failed" else "")
        + "command() {\n"
        '  if [[ "$1" == -v && "$2" == python && "$-" != *u* ]]; then return 17; fi\n'
        '  builtin command "$@"\n'
        "}\n"
        "conda() {\n"
        '  if [[ "${ROBORSI_TEST_ACTIVATE_FAIL:-}" == 1 ]]; then return 9; fi\n'
        '  if [[ "$1" == deactivate && "${ROBORSI_TEST_SDK_HOOK:-}" == 1 ]]; then\n'
        '    export PYTHONPATH="${ROBORSI_TEST_SDK_PATH}"\n'
        "  fi\n"
        '  if [[ "$1" == activate ]]; then\n'
        '    if [[ "${ROBORSI_TEST_SDK_HOOK:-}" == 1 ]]; then test -z "$ZSH_VERSION"; fi\n'
        '    export PATH="${ROBORSI_TEST_CONDA_BIN}:$PATH"\n'
        '    if [[ "${ROBORSI_TEST_SDK_HOOK:-}" == 1 ]]; then\n'
        '      export PYTHONPATH="${ROBORSI_TEST_SDK_PATH}:${PYTHONPATH:-}"\n'
        '      SCRIPT_DIR="${ROBORSI_TEST_SDK_PATH}"\n'
        "    fi\n"
        "  fi\n"
        "}\n"
    )
    selected_python = conda_bin / "python"
    selected_python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1:2] == ['-S']:\n"
        "    print(os.path.realpath(sys.prefix))\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    selected_python.chmod(0o755)
    env = process_env(root, enabled=False, bootstrap=tmp_path / "no-hook")
    env.pop("ZSH_VERSION", None)
    env.update(
        EVAL_ENV_TYPE="sim" if mode.startswith("sim") else mode,
        PATH=str(conda_bin) + os.pathsep + os.environ["PATH"],
        ROBORSI_TEST_CONDA_BIN=str(conda_bin),
        ROBORSI_TEST_ACTIVATE_FAIL="1" if mode == "sim-activate-failed" else "0",
        ROBORSI_TEST_SDK_HOOK="1" if mode == "sim-sdk-hooks" else "0",
        ROBORSI_TEST_SDK_PATH=str(sdk),
    )
    result = subprocess.run(
        [
            "bash",
            str(policy / "setup_eval_env_client.sh"),
            "bench",
            "align_blocks",
            "ckpt",
            "env",
            "action",
            "0",
            "0",
            "FakeSim",
            "info",
            "12345",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if mode in ("sim-missing", "sim-source-failed", "sim-activate-failed"):
        assert result.returncode == (78 if mode == "sim-missing" else 1)
        assert '"framework_reached"' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.splitlines()[-1])
        assert payload["framework_reached"] is True
        assert payload["hook"] == ("1" if mode.startswith("sim") else None)
        assert payload["native_bound"] is mode.startswith("sim")
