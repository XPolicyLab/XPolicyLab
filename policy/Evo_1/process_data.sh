#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 5 ]]; then
    echo "Usage: bash process_data.sh RoboTwin <ckpt_name> <env_cfg_type> joint task=/path/to/lerobot_v21 [...]" >&2
    exit 2
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
shift 4
if [[ "${bench_name}" != RoboTwin || "${action_type}" != joint || ! "${env_cfg_type}" =~ ^(arx_x5|aloha_agilex)$ ]]; then
    echo "[ERROR] Supported: RoboTwin, arx_x5|aloha_agilex, joint." >&2
    exit 2
fi
if [[ ! "${ckpt_name}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "[ERROR] ckpt_name must be a simple run label." >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO1_SOURCE_DIR="${EVO1_SOURCE_DIR:-${SCRIPT_DIR}/upstream}"
DATA_DIR="${SCRIPT_DIR}/processed_data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
python "${SCRIPT_DIR}/prepare_data.py" --output "${DATA_DIR}" "$@"
cd "${EVO1_SOURCE_DIR}/Evo_1"
python -m dataset.compute_normstats_streaming "${DATA_DIR}/config.yaml" --action_horizon 50
# Upstream's CLI can log an error without a nonzero exit; verify the actual outputs.
python - "${DATA_DIR}/config.yaml" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
for name, dataset in config["data_groups"]["aloha_joint"].items():
    stats = json.loads((Path(dataset["path"]) / "meta/stats.json").read_text())
    for feature in ("observation.state", "action"):
        for metric in ("min", "max", "mean", "std", "q01", "q99"):
            values = np.asarray(stats[feature][metric])
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid stats: {name}/{feature}/{metric}")
print("[Evo_1] Per-task statistics verified")
PY
