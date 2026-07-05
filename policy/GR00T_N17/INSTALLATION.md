# GR00T_N17 Installation

`install.sh` is the recommended path, but this document is kept because GR00T_N17 has system dependencies, LeRobot conversion requirements, modality metadata, and optional CUDA fixes that are not obvious from the one-line install command.

## 1. System Dependencies

`ffmpeg` is required for video IO, and `git-lfs` is recommended for model downloads:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs
git lfs install
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

## 2. Install GR00T and XPolicyLab

On x86_64 GPU hosts, the upstream `pyproject.toml` may try to resolve aarch64-only wheels. Use the local wrapper, which creates `gr00t_n17/.venv` and installs the needed packages directly:

```bash
cd XPolicyLab/policy/GR00T_N17
bash install.sh
source gr00t_n17/.venv/bin/activate
python -c "import gr00t; print('GR00T installed successfully')"
```

During evaluation, pass `uv` as `policy_conda_env`; the startup script activates `gr00t_n17/.venv`.

If `CUDA_HOME is unset`, run:

```bash
cd XPolicyLab/policy/GR00T_N17/gr00t_n17
uv run bash scripts/deployment/dgpu/install_deps.sh
```

## 3. RoboDojo Data Preparation

GR00T expects GR00T-flavored LeRobot v2.1 data plus `meta/modality.json`. Convert from LeRobot v3.0 with:

```bash
export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$HF_LEROBOT_HOME}"
export DATA_ROOT="${LEROBOT_DATA_ROOT}"
export SRC_DATASET="${GR00T_SRC_DATASET:-RoboDojo_sim_arx-x5_v30}"
export GR00T_DATASET="${GR00T_DATASET:-RoboDojo_sim_arx-x5_gr00t}"

cp -a "${DATA_ROOT}/${SRC_DATASET}" "${DATA_ROOT}/${GR00T_DATASET}"

cd XPolicyLab/policy/GR00T_N17/gr00t_n17
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --root "${DATA_ROOT}" \
  --repo-id "${GR00T_DATASET}"
```

Or use the adapter wrapper:

```bash
cd XPolicyLab/policy/GR00T_N17
bash process_data.sh RoboDojo cotrain arx_x5 joint
```

## 4. Add `meta/modality.json`

For RoboDojo `arx_x5`, create 14-D state/action metadata:

```bash
cat > "${DATA_ROOT}/${GR00T_DATASET}/meta/modality.json" <<'EOF'
{
  "state": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "action": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "video": {
    "front": { "original_key": "observation.images.cam_high" },
    "left_wrist": { "original_key": "observation.images.cam_left_wrist" },
    "right_wrist": { "original_key": "observation.images.cam_right_wrist" }
  },
  "annotation": {
    "human.task_description": { "original_key": "task_index" }
  }
}
EOF
```

## 5. Useful Paths and Variables

| Variable | Meaning |
| --- | --- |
| `LEROBOT_DATA_ROOT` | LeRobot dataset root; defaults to `$HF_LEROBOT_HOME`. |
| `GR00T_SRC_DATASET` | Source v3.0 dataset repo id. |
| `GR00T_DATASET` | Converted GR00T dataset repo id. |
| `GR00T_BASE_MODEL_PATH` | GR00T-N1.7-3B local path or Hugging Face id. |
| `GR00T_COSMOS_MODEL_PATH` | Cosmos-Reason2-2B path used by deployment. |

## 6. Smoke Checks

```bash
cd XPolicyLab/policy/GR00T_N17/gr00t_n17
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python gr00t/experiment/launch_finetune.py --help
PYTHONPATH=../../.. uv run python -c "import XPolicyLab; print('XPolicyLab ok')"
```
