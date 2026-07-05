# SmolVLA Installation

`install.sh` is the normal setup path, but this document is kept because SmolVLA needs system video dependencies and exposes several installer knobs that are useful on shared machines.

## 1. One-command Install

```bash
cd XPolicyLab/policy/SmolVLA
bash install.sh
conda activate smolvla
```

The installer creates a conda environment, clones `huggingface/lerobot` into `policy/SmolVLA/smovla/`, installs `.[smolvla]`, and installs XPolicyLab in editable mode.

## 2. Optional Installer Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `SMOVLA_CONDA_ENV` | `smolvla` | Conda environment name. |
| `SMOVLA_PYTHON_VERSION` | `3.10` | Python version used for env creation. |
| `LEROBOT_REF` | `v0.4.4` | LeRobot tag or branch to clone. |
| `LEROBOT_REPO` | `https://github.com/huggingface/lerobot.git` | LeRobot repository URL. |
| `SMOVLA_SKIP_CONDA_CREATE` | `0` | Set to `1` to reuse an existing environment. |
| `SMOVLA_UPDATE_LEROBOT` | `0` | Set to `1` to update an existing `smovla/` checkout. |
| `SMOVLA_TORCH_INDEX` | unset | Optional PyTorch wheel index, for example cu128. |

## 3. System Dependencies for Video IO

```bash
sudo apt-get update
sudo apt-get install -y \
  git ffmpeg cmake build-essential pkg-config python3-dev \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

## 4. Manual Install Equivalent

```bash
conda create -n smolvla python=3.10 -y
conda activate smolvla

git clone --branch v0.4.4 --depth 1 https://github.com/huggingface/lerobot.git smovla
cd smovla
pip install -e ".[smolvla]"
# Optional: pip install -e ".[smolvla,peft]"

cd ../../..
pip install -e .
pip install h5py
```

## 5. Model and Dataset Notes

`train.sh` can override `SMOVLA_REPO_ID`. The default pretrained base is resolved by LeRobot/Hugging Face as `lerobot/smolvla_base`.
