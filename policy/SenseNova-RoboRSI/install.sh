#!/usr/bin/env bash
# Install the committed policy dependencies, RoboRSI source, and exact CLI release.
set -euo pipefail

# Installation makes no model request. The caller retains its runtime key.
unset CODEX_API_KEY

POLICY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
POLICY_ENV="${ROBORSI_POLICY_ENV:-${POLICY_DIR}/.venv}"
if [[ -n "${ROBORSI_PYTHON:-}" ]]; then
    PYTHON_BIN="${ROBORSI_PYTHON}"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN=python3.10
else
    PYTHON_BIN=python3
fi

BOOTSTRAP_PYTHON="${ROBORSI_BOOTSTRAP_PYTHON:-python3}"
if [[ -z "${ROBORSI_BOOTSTRAP_PYTHON:-}" ]] && ! command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="${PYTHON_BIN}"
fi

bootstrap_prerequisites() {
    # An explicit interpreter is a contract; never silently replace it.
    if [[ -n "${ROBORSI_PYTHON:-}" ]]; then
        printf '%s\n' 'SenseNova-RoboRSI requires Python 3.10.12 with working memfd support; correct ROBORSI_PYTHON.' >&2
        exit 1
    fi
    exec "${BOOTSTRAP_PYTHON}" "${POLICY_DIR}/bootstrap.py"
}
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    bootstrap_prerequisites
fi
if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info[:3] == (3, 10, 12) else 1)'; then
    bootstrap_prerequisites
fi
if ! "${PYTHON_BIN}" "${POLICY_DIR}/runtime_capabilities.py" --python-only > /dev/null; then
    bootstrap_prerequisites
fi

if [[ -e "${POLICY_ENV}/bin/python" || -L "${POLICY_ENV}/bin/python" ]]; then
    if [[ ! -x "${POLICY_ENV}/bin/python" ]]; then
        printf 'Existing policy interpreter is not executable: %s\n' "${POLICY_ENV}/bin/python" >&2
        exit 1
    fi
    "${POLICY_ENV}/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:3] == (3, 10, 12) else "Existing policy environment must use Python 3.10.12; select another ROBORSI_POLICY_ENV")'
    "${POLICY_ENV}/bin/python" "${POLICY_DIR}/runtime_capabilities.py" --python-only > /dev/null
fi
if ! command -v uv >/dev/null 2>&1; then
    exec "${BOOTSTRAP_PYTHON}" "${POLICY_DIR}/bootstrap.py"
fi
for command in uv setsid; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        printf 'Required installation prerequisite is missing: %s\n' "${command}" >&2
        exit 1
    fi
done

# Fail before creating an environment or downloading anything when release evidence is absent.
"${PYTHON_BIN}" "${POLICY_DIR}/preflight.py" --metadata-only > /dev/null
"${PYTHON_BIN}" "${POLICY_DIR}/tool_install.py" --check-input > /dev/null
if [[ ! -f "${POLICY_DIR}/requirements.lock" ]]; then
    printf '%s\n' 'Missing committed requirements.lock; installation cannot choose dependency versions.' >&2
    exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-${POLICY_DIR}/runtime/uv-cache}"

"${PYTHON_BIN}" - "${POLICY_DIR}" <<'PY'
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

policy = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(policy))
from cli_artifact import unpack_cli_archive
from preflight import (
    load_submission, sha256_file, validate_archive, validate_companions,
    validate_source_bundle, validate_source_manifest, write_source_manifest,
)

metadata = load_submission(policy)
cli = metadata["cli"]
if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
    raise SystemExit("The confirmed Codex standalone release supports Linux x86_64 only")
archive = validate_archive(metadata, policy)
runtime = policy / "runtime"
runtime.mkdir(exist_ok=True)

if not archive.is_file():
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=".codex-download-", delete=False) as stream:
        temporary_archive = Path(stream.name)
        try:
            # URLs contain no credentials; runtime credentials are never used for installation.
            with urlopen(cli["download"]["url"], timeout=60) as response:
                shutil.copyfileobj(response, stream)
        except BaseException:
            temporary_archive.unlink(missing_ok=True)
            raise
    if sha256_file(temporary_archive) != cli["download"]["sha256"]:
        temporary_archive.unlink(missing_ok=True)
        raise SystemExit("Downloaded Codex archive SHA-256 mismatch")
    temporary_archive.replace(archive)

source = runtime / "RoboRSI"
commit = metadata["source"]["commit"]
source_marker = source / ".submission-source-commit"
if source.exists():
    if not source_marker.is_file() or source_marker.read_text().strip() != commit:
        raise SystemExit("runtime/RoboRSI exists without the expected source receipt; move it aside before installing")
    validate_source_manifest(metadata, policy)
else:
    with tempfile.TemporaryDirectory(prefix=".source-", dir=runtime) as temporary:
        staging = Path(temporary)
        expected_files = validate_source_bundle(metadata, policy)
        source_bundle = policy / metadata["source"]["bundle_path"]
        extracted = staging / "extracted"
        shutil.copytree(
            source_bundle, extracted,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        # The PR contains the payload; no local checkout or remote source fetch.
        from preflight import source_file_hashes
        if source_file_hashes(extracted) != expected_files:
            raise SystemExit("RoboRSI source changed while copying the committed bundle")
        (extracted / ".submission-source-commit").write_text(commit + "\n")
        extracted.replace(source)
        write_source_manifest(
            metadata, policy, source,
            artifact_sha256=metadata["source"]["manifest_sha256"],
            origin="committed_bundle",
        )

codex_root = runtime / "codex"
binary = codex_root / "bin" / "codex"
if codex_root.exists():
    if not binary.is_file() or sha256_file(binary) != cli["sha256"]:
        raise SystemExit("runtime/codex exists with an unexpected executable; move it aside before installing")
    validate_companions(metadata, codex_root)
else:
    with tempfile.TemporaryDirectory(prefix=".codex-", dir=runtime) as temporary:
        staging = Path(temporary)
        # A wheel is an authenticated native-CLI distribution, not SDK version evidence.
        # Extract its standalone payload only; the Python wrapper is not installed.
        extracted = unpack_cli_archive(metadata, archive, staging / "extracted")
        candidate = extracted / "bin" / "codex"
        if not candidate.is_file():
            raise SystemExit("Codex artifact must contain bin/codex and its companion release files")
        if sha256_file(candidate) != cli["sha256"]:
            raise SystemExit("Codex executable SHA-256 does not match the confirmed standalone release")
        if not os.access(candidate, os.X_OK):
            raise SystemExit("Codex artifact bin/codex is not executable")
        validate_companions(metadata, extracted)
        extracted.replace(codex_root)
print("Committed RoboRSI source and authenticated Codex artifact are ready.")
PY

if [[ ! -x "${POLICY_ENV}/bin/python" ]]; then
    uv venv --python "${PYTHON_BIN}" "${POLICY_ENV}"
fi
uv pip sync --quiet --python "${POLICY_ENV}/bin/python" --require-hashes "${POLICY_DIR}/requirements.lock"
uv pip install --quiet --python "${POLICY_ENV}/bin/python" --no-deps --no-build-isolation \
    "${POLICY_DIR}/runtime/RoboRSI" -e "${POLICY_DIR}/../.."

# Build the B-aligned tools independently of the Python 3.10.12 policy host.
printf '%s\n' 'Building the pinned model-tool environment...'
"${POLICY_ENV}/bin/python" "${POLICY_DIR}/tool_install.py" > /dev/null

# Exercise the original transport and actual tools before large model downloads.
"${POLICY_ENV}/bin/python" "${POLICY_DIR}/runtime_capabilities.py" \
    --policy-dir "${POLICY_DIR}" > "${POLICY_DIR}/runtime/sandbox-install.json"

# Keep the simulator import fix in the policy, rather than an operator's home.
# A standalone XPolicyLab checkout may be installed before its native workspace.
# The simulation entry point repeats this check strictly once that workspace exists.
BENCH_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"
"${POLICY_ENV}/bin/python" "${POLICY_DIR}/sim_bootstrap/probe.py" \
    --workspace "${BENCH_ROOT}" --allow-missing-workspace \
    > "${POLICY_DIR}/runtime/sim-bootstrap-install.json"

# Complete public-training ICL data is required, including its referenced media.
printf '%s\n' 'Preparing the fixed ICL packages...'
"${POLICY_ENV}/bin/python" "${POLICY_DIR}/icl_assets.py" install > /dev/null

# Install only the selected VLA. The Simate path keeps the vendored OpenPI
# wire codec but never downloads the unused pi0.5 weights/runtime.
VLA_BACKEND="$("${POLICY_ENV}/bin/python" - "${POLICY_DIR}/deploy.yml" <<'PYBACKEND'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1]))
backend=cfg.get("vla_backend", "pi05")
if backend not in ("pi05", "simate_beta"):
    raise SystemExit("Unsupported vla_backend")
print(backend)
PYBACKEND
)"
printf 'Installing VLA backend: %s\n' "${VLA_BACKEND}"
if [[ "${VLA_BACKEND}" == simate_beta ]]; then
    bash "${POLICY_DIR}/install_simate.sh"
else
    "${POLICY_ENV}/bin/python" "${POLICY_DIR}/pi05_runtime.py" install > /dev/null
fi

# Installation validates identity and configuration offline; eval.sh always requires the key.
"${POLICY_ENV}/bin/python" "${POLICY_DIR}/preflight.py" \
    --config "${POLICY_DIR}/deploy.yml" --installation-only \
    --verify-icl-task align_blocks > "${POLICY_DIR}/runtime/install-preflight.json"
"${POLICY_ENV}/bin/python" - "${POLICY_DIR}" "${POLICY_ENV}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

policy = Path(sys.argv[1])
sys.path.insert(0, str(policy))
from preflight import load_submission, sha256_file

metadata = load_submission(policy)
import yaml
backend = yaml.safe_load((policy / "deploy.yml").read_text()).get("vla_backend", "pi05")
receipt = {
    "schema_version": 1,
    "installed_at": datetime.now(timezone.utc).isoformat(),
    "source_commit": metadata["source"]["commit"],
    "agent_label": metadata["agent_label"],
    "cli_version": metadata["cli"]["expected_version"],
    "cli_sha256": sha256_file(policy / "runtime/codex/bin/codex"),
    "cli_companion_sha256": metadata["cli"]["companion_sha256"],
    "requirements_sha256": sha256_file(policy / "requirements.lock"),
    "source_manifest_sha256": sha256_file(policy / "runtime/source-files.json"),
    "icl_catalog_sha256": metadata["icl"]["catalog_sha256"],
    "icl_artifacts_sha256": metadata["icl"]["artifacts_sha256"],
    "hybrid_manifest_sha256": metadata["hybrid"]["manifest_sha256"],
    "vla_backend": backend,
    "vla_install_receipt": "simate-install.json" if backend == "simate_beta" else "runtime/pi05-installed.json",
    **({"simate_requirements_sha256": sha256_file(policy / "simate/requirements.lock")}
       if backend == "simate_beta" else {
           "pi05_requirements_sha256": sha256_file(policy / "pi05/requirements.lock"),
           "pi05_install_receipt": "runtime/pi05-installed.json",
       }),
    "python": str(Path(sys.argv[2]) / "bin/python"),
    "tool_runtime": json.loads((policy / "runtime/tool-runtime.json").read_text()),
    "simulator_bootstrap": json.loads(
        (policy / "runtime/sim-bootstrap-install.json").read_text()
    ),
    "runtime_capabilities": json.loads(
        (policy / "runtime/sandbox-install.json").read_text()
    ),
    "icl_materialization": json.loads(
        (policy / "runtime/install-preflight.json").read_text()
    )["icl"]["materialization"],
}
(policy / "runtime/install-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
printf 'Installation complete. Policy environment: %s\n' "${POLICY_ENV}"
