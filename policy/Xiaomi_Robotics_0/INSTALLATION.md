# Xiaomi_Robotics_0 Installation

`install.sh` is the recommended path, but this document is kept because Xiaomi-Robotics-0 requires pretrained weight conversion and checkpoint-link conventions that are not handled by a plain dependency install.

## 1. Prepare Demonstration Data

If RoboDojo raw data is not already available, prepare it through the main XPolicyLab README. The expected raw layout is:

```text
<RoboDojo data root>/sim_cloud/<task_name>/<env_cfg_type>/*.hdf5
```

## 2. One-command Install

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_0
bash install.sh
conda activate mibot
```

The installer creates the `mibot` conda environment, installs PyTorch 2.8, Flash Attention, XR-0 dependencies, and installs both `xiaomi_robotics_0/xr0` and XPolicyLab in editable mode.

## 3. Manual Install Equivalent

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0
conda create -n mibot python=3.12 -y
conda activate mibot

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y ninja && pip install ninja
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .

cd ../../../..
pip install -e .
pip install opencv-python-headless tqdm scipy
```

## 4. Prepare Pretrained Weights

Download [Xiaomi-Robotics-0-Pretrain](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain) and convert it to the PyTorch checkpoint expected by XR-0 training:

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0
python tools/weight_convert.py \
  --model_path hf_pretrain \
  --output_dir pretrained_ckpt \
  --output_filename xr0_pretrained.pt
```

Or point to an existing converted checkpoint:

```bash
export XR0_PRETRAINED_PATH=/path/to/xr0_pretrained.pt
```

Default converted path:

```text
policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0/pretrained_ckpt/xr0_pretrained.pt
```

## 5. Link Evaluation Checkpoints

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_0
bash scripts/link_checkpoint.sh <ckpt_name> /path/to/finetuned_ckpt
```

`ckpt_name` should match the value passed to `eval.sh`.

## 6. Smoke Checks

```bash
conda activate mibot
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import mibot; print('mibot ok')"
python -c "import XPolicyLab; print('XPolicyLab ok')"
```
