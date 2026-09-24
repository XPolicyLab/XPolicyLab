#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="${XBRAIN_RELEASE_REPO:-wuchong617/robodojo_xbrain_real}"
DESTINATION="${XBRAIN_RELEASE_DIR:-${SCRIPT_DIR}/checkpoints/xbrain_v1_release_0.1.0}"
PYTHON_BIN="${XBRAIN_PYTHON:-${PYTHON_BIN:-}}"
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "[XBrain_v1][ERROR] Python interpreter not found. Set XBRAIN_PYTHON=/path/to/python." >&2
    exit 1
fi

mkdir -p "${DESTINATION}"
DESTINATION="$(cd "${DESTINATION}" && pwd)"
if [[ "${XBRAIN_SKIP_DOWNLOAD:-0}" != "1" ]]; then
    echo "[XBrain_v1] downloading ${REPO_ID} -> ${DESTINATION}"
    "${PYTHON_BIN}" - "${REPO_ID}" "${DESTINATION}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], repo_type="model", local_dir=sys.argv[2])
PY
fi

required=(
    "piper_x/model_ema/config.json"
    "piper_x/model_ema/inference_config.json"
    "piper_x/model_ema/diffusion_pytorch_model.bin"
    "piper/model_ema/config.json"
    "piper/model_ema/inference_config.json"
    "piper/model_ema/diffusion_pytorch_model.bin"
    "arx_x5/model_ema/config.json"
    "arx_x5/model_ema/inference_config.json"
    "arx_x5/model_ema/diffusion_pytorch_model.bin"
    "norm/robodojo_piperx_0902.json"
    "norm/robodojo_piper_0903.json"
    "norm/robodojo_arx5_0903.json"
    "tokenizer/config.json"
    "tokenizer/tokenizer.json"
    "fast_tokenizer/tokenizer.json"
    "release_manifest.json"
)
for relative_path in "${required[@]}"; do
    if [[ ! -f "${DESTINATION}/${relative_path}" ]]; then
        echo "[XBrain_v1][ERROR] missing release file: ${DESTINATION}/${relative_path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" - "${DESTINATION}" "${SCRIPT_DIR}/runtime/release_manifest.json" "${XBRAIN_VERIFY_WEIGHTS:-1}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
bundled_manifest_path = pathlib.Path(sys.argv[2])
verify_weights = sys.argv[3] == "1"
manifest = json.loads(bundled_manifest_path.read_text(encoding="utf-8"))
downloaded_manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
if downloaded_manifest != manifest:
    raise SystemExit(
        f"release manifest differs from bundled runtime: {root / 'release_manifest.json'}"
    )
for robot, spec in manifest["robots"].items():
    model_dir = root / robot / "model_ema"
    checks = {
        model_dir / "config.json": spec["checkpoint_config_sha256"],
        model_dir / "inference_config.json": spec["inference_config_sha256"],
        root / "norm" / spec["norm_filename"]: spec["norm_sha256"],
    }
    if verify_weights:
        checks[model_dir / spec["weights_filename"]] = spec["weights_sha256"]
    for path, expected in checks.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise SystemExit(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    print(f"[XBrain_v1] verified resources for {robot}")
PY

ENV_FILE="${DESTINATION}/xbrain_env.sh"
{
    printf 'export XBRAIN_PIPERX_CHECKPOINT_PATH=%q\n' "${DESTINATION}/piper_x/model_ema"
    printf 'export XBRAIN_PIPERX_NORM_STATS_PATH=%q\n' "${DESTINATION}/norm/robodojo_piperx_0902.json"
    printf 'export XBRAIN_PIPER_CHECKPOINT_PATH=%q\n' "${DESTINATION}/piper/model_ema"
    printf 'export XBRAIN_PIPER_NORM_STATS_PATH=%q\n' "${DESTINATION}/norm/robodojo_piper_0903.json"
    printf 'export XBRAIN_ARX_X5_CHECKPOINT_PATH=%q\n' "${DESTINATION}/arx_x5/model_ema"
    printf 'export XBRAIN_ARX_X5_NORM_STATS_PATH=%q\n' "${DESTINATION}/norm/robodojo_arx5_0903.json"
    printf 'export XBRAIN_TOKENIZER_PATH=%q\n' "${DESTINATION}/tokenizer"
    printf 'export XBRAIN_FAST_TOKENIZER_PATH=%q\n' "${DESTINATION}/fast_tokenizer"
} > "${ENV_FILE}"

echo "[XBrain_v1] release ready: ${DESTINATION}"
echo "[XBrain_v1] activate resources with: source ${ENV_FILE}"
