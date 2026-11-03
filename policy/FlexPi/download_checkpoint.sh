#!/usr/bin/env bash
# Fetch everything a FlexPi evaluation loads, into the paths setup_eval_policy_server.sh reads:
#
#   checkpoints/<ckpt_name>/   config.yaml, dataset_stats.json, step_*.pt         (Hugging Face)
#   FlexPi/checkpoints/        Wan2.2 VAE, umt5-xxl encoder + tokenizer, ~13 GB    (ModelScope, once)
#
#   bash download_checkpoint.sh <hf_repo_id> [ckpt_name] [revision]
#   bash eval.sh <bench_name> <task> <ckpt_name> <env_cfg_type> joint 0 0 0 flexpi <eval_env>
#
# It reads a package at the top of the repo. A repo with one folder per robot
# (flex-pi/flexpi-robodojo-real) needs `huggingface-cli download --include '<robot>/*'`
# (README, RoboDojo_real checkpoints); the script stops with that hint.
#
# Run it in the policy env (bash install.sh). DIFFSYNTH_MODEL_BASE_PATH moves the base
# weights; the server reads the same variable. A checkpoint whose config.yaml keeps
# model.skip_dit_load_from_pretrain: false also loads the Wan2.2-TI2V-5B DiT and the
# ActionDiT init file at boot (README, Installation).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="${1:?usage: bash download_checkpoint.sh <hf_repo_id> [ckpt_name] [revision]}"
CKPT_NAME="${2:-${REPO_ID##*/}}"
REVISION="${3:-main}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${SCRIPT_DIR}/FlexPi/checkpoints}"

PYTHONPATH="${SCRIPT_DIR}/FlexPi/src:${PYTHONPATH:-}" \
python - "${REPO_ID}" "${REVISION}" "${SCRIPT_DIR}/checkpoints/${CKPT_NAME}" <<'PY'
import glob
import os
import sys

from huggingface_hub import snapshot_download

from flexpi.models.helpers.io import ModelConfig

repo_id, revision, dest = sys.argv[1:4]
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=dest,
    allow_patterns=["config.yaml", "dataset_stats.json", "step_*.pt", "README.md"],
)
if not glob.glob(os.path.join(path, "step_*.pt")):
    sys.exit(
        f"[FlexPi] no step_*.pt at the top of {repo_id}. For a repo with one folder per robot run\n"
        f"  huggingface-cli download {repo_id} --include '<robot>/*' --local-dir {os.path.dirname(dest)}/<name>"
    )
print(f"[FlexPi] checkpoint -> {path}")

# Evaluation reads these three even when the checkpoint supplies the DiT weights.
# They are published on ModelScope only, so DIFFSYNTH_DOWNLOAD_SOURCE stays unset.
for repo, pattern in [
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "google/umt5-xxl/"),
]:
    ModelConfig(model_id=repo, origin_file_pattern=pattern).download_if_necessary()
print("[FlexPi] base weights ready")
PY
