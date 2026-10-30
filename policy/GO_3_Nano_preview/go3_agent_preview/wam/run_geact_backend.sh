#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"           # .../policy/go3_agent_preview
BENCH_ROOT="$(cd "$POLICY_DIR/../../.." && pwd)"     # RoboDojo checkout root
SOURCE_ROOT="$SCRIPT_DIR/runtime"

PYTHON_BIN="${PYTHON_BIN:-${WAM_PYTHON_BIN:-python3}}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$POLICY_DIR/deploy.yml}"

# Path overrides may be given relative to the adapter directory, but the script
# changes directory before launching, so resolve them all against one base here.
abs_path() {
  local p="$1"
  [[ -z "$p" ]] && { echo ""; return; }
  case "$p" in
    /*) echo "$p" ;;
    *) echo "$POLICY_DIR/$p" ;;
  esac
}

CKPT="$(abs_path "${CKPT:-}")"
STAT_PATH="$(abs_path "${STAT_PATH:-$SCRIPT_DIR/stats.json}")"
CFG="$(abs_path "${CFG:-$SOURCE_ROOT/config/cfg_e2e60k_infer.py}")"
DEBUG_VIDEO_DIR="$(abs_path "${DEBUG_VIDEO_DIR:-$BENCH_ROOT/outputs/wam_openloop}")"
DEPLOY_CONFIG="$(abs_path "$DEPLOY_CONFIG")"
ROBOT_NAME="${ROBOT_NAME:-${GEACT_ROBOT_NAME:-sl_arxvr}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
DENSE_ACTION_STEPS="${DENSE_ACTION_STEPS:-30}"

# Published assets. When a path is unset the launcher downloads it here instead of
# failing, so a machine without our checkpoints only needs network access once.
# Each repo has its own revision: the pinned one identifies preview_model, and
# Qwen tracks its own default branch.
WAM_MODEL_REPO="${WAM_MODEL_REPO:-ccccsc/preview_model}"
WAM_MODEL_REVISION="${WAM_MODEL_REVISION:-24e3a2553f274b8700f52a3065d7088e80f8da10}"
WAM_QWEN_REPO="${WAM_QWEN_REPO:-Qwen/Qwen3.5-2B}"
WAM_QWEN_REVISION="${WAM_QWEN_REVISION:-main}"
WAM_ASSETS_DIR="$(abs_path "${WAM_ASSETS_DIR:-$POLICY_DIR/.wam-assets}")"

# Download a repo into the asset root unless the marker already exists.
#   $1 repo   $2 revision   $3 local subdir   $4 repo include pattern   $5 marker   $6 label
# The include pattern is what the repo calls the file; the local subdir is where we
# keep it. Those differ for Qwen, whose repo root is not called "qwen3.5-2b".
fetch_asset() {
  local repo="$1" revision="$2" subdir="$3" include="$4" marker="$5" what="$6"
  local dest="$WAM_ASSETS_DIR/$subdir"
  if [[ -e "$marker" ]]; then
    return 0
  fi
  echo "[info] $what not found at $marker; downloading $repo@$revision" >&2
  local downloader=()
  if command -v hf >/dev/null; then
    downloader=(hf download)
  elif "$PYTHON_BIN" -c 'import huggingface_hub' 2>/dev/null; then
    downloader=("$PYTHON_BIN" -m huggingface_hub.commands.huggingface_cli download)
  else
    echo "[error] downloading $repo needs the Hugging Face CLI:" >&2
    echo '        pip install "huggingface_hub[cli]"   (or set the asset path explicitly)' >&2
    return 1
  fi
  mkdir -p "$dest"
  HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
    "${downloader[@]}" "$repo" --revision "$revision" \
      --include "$include" --local-dir "$dest" >&2
  [[ -e "$marker" ]] || { echo "[error] download did not produce $marker" >&2; return 1; }
}

# Resolve the published assets. An explicit path always wins; anything left unset
# is downloaded, so a fresh machine needs network access rather than our checkpoints.
if [[ -z "$CKPT" ]]; then
  CKPT="$WAM_ASSETS_DIR/preview_model/checkpoints"
  fetch_asset "$WAM_MODEL_REPO" "$WAM_MODEL_REVISION" preview_model 'checkpoints/*' \
    "$CKPT/action_model/diffusion_pytorch_model.bin" "checkpoint" || exit 1
fi

if [[ -z "${WAM_QWEN_DIR:-}" ]]; then
  # The Qwen directory itself is the asset: AutoProcessor loads it in place, so
  # fetch the whole repo rather than naming files.
  WAM_QWEN_DIR="$WAM_ASSETS_DIR/qwen3.5-2b"
  fetch_asset "$WAM_QWEN_REPO" "$WAM_QWEN_REVISION" qwen3.5-2b '*' \
    "$WAM_QWEN_DIR/config.json" "Qwen3.5 VLM" || exit 1
fi
if [[ -z "${WAM_VAE_DIR:-}" ]]; then
  WAM_VAE_DIR="$WAM_ASSETS_DIR/preview_model/vae"
  fetch_asset "$WAM_MODEL_REPO" "$WAM_MODEL_REVISION" preview_model 'vae/*' \
    "$WAM_VAE_DIR/config.json" "VAE" || exit 1
fi
if [[ -z "${WAM_LATENT_STATS_PATH:-}" ]]; then
  WAM_LATENT_STATS_PATH="$WAM_VAE_DIR/latent_stats.pth"
  [[ -e "$WAM_LATENT_STATS_PATH" ]] || {
    echo "[error] $WAM_LATENT_STATS_PATH is missing from the VAE download" >&2
    exit 1
  }
fi
export WAM_QWEN_DIR WAM_VAE_DIR WAM_LATENT_STATS_PATH

# The checkpoint's action expert is required; the config and video weights arrive
# with the same download, so a missing one means an interrupted fetch.
for path in "$CKPT/diffusion_pytorch_model.bin" "$CKPT/action_model/diffusion_pytorch_model.bin" "$STAT_PATH" "$CFG"; do
  [[ -e "$path" ]] || { echo "[error] missing $path" >&2; exit 1; }
done

# The backend holds one GPU for its whole lifetime, and the inference code selects
# it as cuda:settings.local_rank. That field is read from LOCAL_RANK, an inherited
# torchrun variable, so a stale LOCAL_RANK=1 with one visible card would ask for a
# device that does not exist. Pin it: CUDA_VISIBLE_DEVICES alone chooses the card.
export LOCAL_RANK=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[info] backend CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (logical device 0)" >&2
fi

export PYTHONPATH="$BENCH_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GEACT_INFER_LEGACY_STATE_LAYOUT="${GEACT_INFER_LEGACY_STATE_LAYOUT:-identity}"

CFG=$("$PYTHON_BIN" - "$CFG" "$DEPLOY_CONFIG" <<'PY'
import sys
import yaml
from XPolicyLab.policy.go3_agent_preview.wam.memory_config import MemoryConfig
with open(sys.argv[2], encoding="utf-8") as stream:
    settings = yaml.safe_load(stream) or {}
print(MemoryConfig.from_mapping(settings).resolve_config(sys.argv[1], sys.argv[2]))
PY
)

for path in "$CKPT/diffusion_pytorch_model.bin" "$CKPT/action_model/diffusion_pytorch_model.bin" "$STAT_PATH" "$CFG"; do
  [[ -e "$path" ]] || { echo "[error] missing $path" >&2; exit 1; }
done

"$PYTHON_BIN" "$SCRIPT_DIR/check_checkpoint.py" --config "$CFG" --ckpt "$CKPT"
mkdir -p "$DEBUG_VIDEO_DIR"
export GEACT_OPENLOOP_DEBUG_VIDEO_DIR="$DEBUG_VIDEO_DIR"

cd "$SOURCE_ROOT"
exec "$PYTHON_BIN" serve_geact.py \
  --deploy-config "$DEPLOY_CONFIG" \
  --cfg-path "$CFG" --ckpt-path "$CKPT" \
  --stat-path "$STAT_PATH" --robot-name "$ROBOT_NAME" \
  --dense-action-steps "$DENSE_ACTION_STEPS" \
  --host "$HOST" --port "$PORT" --seed "${SEED:-42}"
