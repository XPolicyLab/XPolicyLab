#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/etc/profile.d/conda.sh"
BASE_CONFIG="${BASE_CONFIG:-$SCRIPT_DIR/config_xspark.yml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$SCRIPT_DIR/.config_xspark_gpu4.runtime.yml}"
FULL_CONVERT="${FULL_CONVERT:-0}"

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Missing base config: $BASE_CONFIG" >&2
    exit 1
fi

if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate motus
else
    echo "Missing conda init script: $CONDA_SH" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY' "$BASE_CONFIG" "$RUNTIME_CONFIG"
import sys
import yaml

base_config, runtime_config = sys.argv[1:3]
with open(base_config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Only one physical GPU is exposed via CUDA_VISIBLE_DEVICES=4,
# so the converter must use logical device 0 inside the process.
config['cuda_devices'] = ['0']

with open(runtime_config, 'w', encoding='utf-8') as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(runtime_config)
PY

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using runtime config: $RUNTIME_CONFIG"
echo "Using PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

FREE_MEM_MB="$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
if [[ -n "$FREE_MEM_MB" ]] && [[ "$FREE_MEM_MB" -lt 45000 ]]; then
    echo "GPU ${CUDA_VISIBLE_DEVICES%%,*} free memory is only ${FREE_MEM_MB} MiB; need roughly 45 GiB free for T5 encoding." >&2
    echo "Free GPU 4 and rerun this script." >&2
    exit 1
fi

for requirement in easydict ftfy sentencepiece regex imageio; do
    if ! python -c "import ${requirement}" >/dev/null 2>&1; then
        echo "Installing missing dependency: ${requirement}"
        python -m pip install "$requirement"
    fi
done

MODE="$({ python - <<'PY' "$RUNTIME_CONFIG" "$FULL_CONVERT"
import sys
from pathlib import Path
import yaml

runtime_config, full_convert = sys.argv[1:3]
with open(runtime_config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

root = Path(config['target_root'])
meta_count = len(list(root.glob('**/metas/*.txt'))) if root.exists() else 0
t5_count = len(list(root.glob('**/umt5_wan/*.pt'))) if root.exists() else 0

if full_convert == '1' or meta_count == 0:
    print('full')
else:
    print('t5-only')
PY
})"

echo "Conversion mode: $MODE"

if [[ "$MODE" == "full" ]]; then
    python "$SCRIPT_DIR/robotwin_converter.py" --config "$RUNTIME_CONFIG" --verbose
else
    python - <<'PY' "$SCRIPT_DIR" "$RUNTIME_CONFIG"
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

script_dir, runtime_config = sys.argv[1:3]
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from robotwin_converter import T5EmbeddingProcessor

with open(runtime_config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

root = Path(config['target_root'])
wan_repo_path = config['wan_repo_path']
t5_max_length = int(config.get('t5_max_length', 512))

pending = []
for task_dir in root.iterdir():
    if not task_dir.is_dir() or task_dir.name in {'clean', 'randomized', '.cache'}:
        continue
    for subset_dir in task_dir.iterdir():
        if not subset_dir.is_dir():
            continue
        metas_dir = subset_dir / 'metas'
        umt5_dir = subset_dir / 'umt5_wan'
        if not metas_dir.exists():
            continue
        umt5_dir.mkdir(exist_ok=True)
        for meta_file in sorted(metas_dir.glob('*.txt')):
            t5_file = umt5_dir / f'{meta_file.stem}.pt'
            if not t5_file.exists():
                pending.append((str(meta_file), str(t5_file)))

print(f'Pending T5 files: {len(pending)}')
if not pending:
    raise SystemExit(0)

processor = T5EmbeddingProcessor(wan_repo_path, t5_max_length, device='cuda:0')
success = 0
for meta_path, t5_path in tqdm(pending, desc='Processing missing T5'):
    if processor.process_meta_file(meta_path, t5_path):
        success += 1

print(f'T5 embeddings completed: {success}/{len(pending)} successful')
PY
fi

python - <<'PY' "$RUNTIME_CONFIG"
import sys
from pathlib import Path
import yaml

runtime_config = sys.argv[1]
with open(runtime_config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

root = Path(config['target_root'])
(root / 'clean').mkdir(exist_ok=True)
(root / 'randomized').mkdir(exist_ok=True)

for task_dir in root.iterdir():
    if not task_dir.is_dir() or task_dir.name in {'clean', 'randomized', '.cache'}:
        continue

    clean_src = task_dir / 'aloha-agilex_clean_50'
    randomized_src = task_dir / 'aloha-agilex_randomized_500'

    clean_dst = root / 'clean' / task_dir.name
    randomized_dst = root / 'randomized' / task_dir.name

    if clean_src.exists() and not clean_dst.exists():
        clean_dst.symlink_to(clean_src)

    if randomized_src.exists() and not randomized_dst.exists():
        randomized_dst.symlink_to(randomized_src)

meta_count = len(list(root.glob('**/metas/*.txt')))
t5_count = len(list(root.glob('**/umt5_wan/*.pt')))
print(f'meta={meta_count}')
print(f'umt5={t5_count}')
PY