#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num=100]" >&2
    exit 2
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
if [[ "$bench_name" != "RoboDojo" || "$env_cfg_type" != "arx_x5" || "$action_type" != "joint" || "${5:-100}" != 100 ]]; then
    echo "The reference manifest supports RoboDojo / arx_x5 / joint with 100 episodes per task." >&2
    exit 2
fi
if [[ ! "$ckpt_name" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
    echo "ckpt_name must be a run label, not a path." >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${AWOMO_DATA_ROOT:?Set the directory containing raw RoboDojo HDF5 task folders}"
export PYTHONPATH="$SCRIPT_DIR:$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATA_DIR=${AWOMO05_DATA_DIR:-$SCRIPT_DIR/data/$bench_name-$ckpt_name-$env_cfg_type-$action_type}
python - "$DATA_DIR" "$SCRIPT_DIR/training/dataset_stats.json" <<'PY'
import hashlib
import os
from pathlib import Path
import sys
import tempfile
source = sys.argv[2]
payload = Path(source).read_bytes()
expected = "ef1dddb4ba560b00184ffce1784ee6cc1192a149005648f6305a09530ae64875"
if hashlib.sha256(payload).hexdigest() != expected:
    raise ValueError("SHA256 mismatch: dataset_stats.json")
directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
target = directory / "dataset_stats.json"
with tempfile.NamedTemporaryFile(dir=directory, prefix=".dataset_stats.", delete=False) as stream:
    temporary = Path(stream.name)
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(temporary, 0o644)
try:
    try:
        os.link(temporary, target)
    except FileExistsError:
        if target.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite different normalization parameters: {target}")
finally:
    temporary.unlink()
print(f"Normalization parameters downloaded and verified: {target}")
PY
exec python "$SCRIPT_DIR/training/prepare_data.py" \
    --data-root "$AWOMO_DATA_ROOT" --output "$DATA_DIR/index.pt"
