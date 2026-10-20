#!/bin/bash
set -euo pipefail

PYTHON_BIN="${SPARK_0_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[check_da3_teacher] ERROR: Python interpreter not found: ${PYTHON_BIN}" >&2
    echo "[check_da3_teacher] Set SPARK_0_PYTHON to the training interpreter." >&2
    exit 1
fi

# Fail fast when the resolved train config ENABLES DA3 spatial alignment but
# the teacher weights are missing on THIS node (each node lazy-loads DA3
# locally on the first training forward, so a missing copy otherwise errors
# minutes into the run).
#
# The enable flag AND the teacher path are read from the YAML itself, never
# from the recipe FILENAME — recipe names have historically both enabled DA3
# without saying "da3" and carried "da3" while setting
# spatial_alignment.enable=false.
#
# Usage: bash scripts/check_da3_teacher.sh <config_yaml>
# Honors SKIP_DA3_CHECK=1 (e.g. an HF-hub teacher id with no local copy).
# Exits 0 when DA3 is disabled or the teacher looks loadable.

if [[ "${SKIP_DA3_CHECK:-0}" == "1" ]]; then
    exit 0
fi

config_yaml="${1:?usage: bash scripts/check_da3_teacher.sh <config_yaml>}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_teacher_dir="$(dirname "${script_dir}")/source_starvla/playground/Pretrained_models/DA3-LARGE"

# A missing config is train.sh's error to report, not this script's.
if [[ ! -f "${config_yaml}" ]]; then
    exit 0
fi

# Resolve enable + teacher_model_path from the YAML in one pass. The path may
# be a local dir or an HF hub id. Relative paths are resolved the way TRAINING
# resolves them: train.sh cd's into source_starvla/ before launching, so
# `./playground/...` in a recipe means source_starvla/playground/... .
read -r da3_enabled teacher_path < <("${PYTHON_BIN}" - "$config_yaml" "$default_teacher_dir" <<'PY'
import os
import sys

import yaml

config_yaml, default_dir = sys.argv[1], sys.argv[2]
with open(config_yaml, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp) or {}
spatial = (cfg.get("framework") or {}).get("spatial_alignment") or {}
enabled = bool(spatial.get("enable", False))
path = spatial.get("teacher_model_path") or default_dir
path = os.path.expanduser(str(path))
if not os.path.isabs(path):
    config_dir = os.path.dirname(os.path.abspath(config_yaml))
    spark_root = os.path.dirname(config_dir)
    candidates = [
        # Training cwd (train.sh cd's here) — how recipes intend ./playground.
        os.path.join(spark_root, "source_starvla", path),
        os.path.join(config_dir, path),
        os.path.join(spark_root, path),
    ]
    for cand in candidates:
        cand = os.path.normpath(cand)
        if os.path.exists(cand):
            path = cand
            break
    else:
        # None exists; report against the training-cwd resolution so the
        # error message points at the path training would actually use.
        path = os.path.normpath(candidates[0])
print("true" if enabled else "false", path)
PY
)

if [[ "${da3_enabled}" != "true" ]]; then
    exit 0
fi

fail() {
    echo "[check_da3_teacher] ERROR: $(basename "${config_yaml}") sets framework.spatial_alignment.enable=true" >&2
    echo "[check_da3_teacher] but $1" >&2
    echo "[check_da3_teacher] Every node needs the teacher: bash scripts/download_da3_large.sh" >&2
    echo "[check_da3_teacher] (or set spatial_alignment.teacher_model_path; SKIP_DA3_CHECK=1 to bypass)." >&2
    exit 1
}

# 1. The depth_anything_3 package must be importable in the training env.
if ! "${PYTHON_BIN}" -c "import depth_anything_3" >/dev/null 2>&1; then
    fail "the 'depth_anything_3' package is not installed in this environment (pip install 'git+https://github.com/ByteDance-Seed/Depth-Anything-3')."
fi

# 2. The teacher path must exist (a local dir) and carry config + weights.
if [[ ! -d "${teacher_path}" ]]; then
    fail "DA3 teacher weights were not found at ${teacher_path} (from spatial_alignment.teacher_model_path)."
fi
if [[ ! -f "${teacher_path}/config.json" ]]; then
    fail "no config.json under ${teacher_path}."
fi
if ! ls "${teacher_path}"/*.safetensors >/dev/null 2>&1 && ! ls "${teacher_path}"/*.bin >/dev/null 2>&1; then
    fail "no weight files (*.safetensors / *.bin) under ${teacher_path} — the download looks incomplete."
fi

exit 0
