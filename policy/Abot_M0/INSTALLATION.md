# Abot_M0 Installation

The top-level `install.sh` only installs the XPolicyLab-side dependencies inside the ABot runtime environment. This document is kept because ABot-M0 still requires upstream ABot setup, dataset preparation, and checkpoint path wiring that are not fully covered by the one-command installer.

## 1. Install the Upstream ABot Environment

Follow `abot_m0/INSTALLATION.md` first. That upstream guide creates the `ABot` conda environment and installs the ABot-Manipulation and VGGT dependencies.

After the upstream environment exists, install the XPolicyLab adapter dependencies:

```bash
cd XPolicyLab/policy/Abot_M0
conda activate ABot
bash install.sh
```

`install.sh` uses `ABOT_CONDA_ENV` when set and installs XPolicyLab plus `h5py`, `opencv-python`, and `pyyaml`.

## 2. Manual XPolicyLab Integration

```bash
conda activate ABot
cd XPolicyLab
pip install -e .
pip install h5py opencv-python pyyaml
```

## 3. Model and Data Variables

| Variable | Meaning |
| --- | --- |
| `BASE_VLM` | Qwen3-VL-4B-Instruct-Action directory or Hugging Face id. |
| `PRETRAIN_CKPT` | ABot-M0 pretrained checkpoint path. |
| `RELOAD_MODULES` | Module subset to reload, for example `qwen_vl_interface`. |
| `HF_LEROBOT_HOME` | LeRobot dataset root used by ABot data preparation. |
| `ABOT_STATS_JSON` | Optional eval fallback stats file. Defaults to `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_v21_video_abot/meta/stats_gr00t.json`. |

## 4. RoboDojo Data Preparation

```bash
cd XPolicyLab/policy/Abot_M0/abot_m0
cp examples/Robotwin/train_files/modality.json \
  "${HF_LEROBOT_HOME}/<repo_id>/meta/modality.json"

python3 examples/RoboDojo/prepare_RoboDojo_abot.py \
  --dataset-dir "${HF_LEROBOT_HOME}/<repo_id>"
```

## 5. Evaluation Checkpoint Layout

For existing checkpoints, create a stable name under `policy/Abot_M0/checkpoints/` and pass that name as `ckpt_name` during evaluation:

```bash
cd XPolicyLab/policy/Abot_M0
mkdir -p checkpoints
ln -sfn <checkpoint_dir> checkpoints/<ckpt_name>
```

The model adapter also resolves the standard normalized run id, for example `RoboDojo-cotrain-arx_x5-joint-0`, when the corresponding tuple fields are passed by the eval scripts.

Use `eval.sh` for a same-machine run, or split `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` for debugging.
