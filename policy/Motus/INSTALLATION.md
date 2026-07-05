# Motus Installation

`install.sh` covers the standard environment, but this document is kept because Motus has important model-root conventions and checkpoint lookup behavior that are needed for evaluation.

## 1. One-command Install

```bash
cd XPolicyLab/policy/Motus
bash install.sh
conda activate motus
```

## 2. Manual Install Equivalent

```bash
conda create -n motus python=3.10 -y
conda activate motus

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install packaging psutil ninja wheel
pip install flash-attn --no-build-isolation

cd XPolicyLab/policy/Motus/motus
pip install -r requirements.txt
pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt
pip install -e .

cd ../../..
pip install -e .
```

## 3. Model and Data Paths

| Variable or argument | Meaning |
| --- | --- |
| `WAN_PATH` / `--wan_path` | Root containing `Wan2.2-TI2V-5B`, `Qwen3-VL-2B-Instruct`, and `Motus/`. |
| `LEROBOT_DATA_ROOT` | Parent directory for LeRobot datasets. |
| `MOTUS_CHECKPOINT_PATH` | Direct path to a checkpoint directory, or a parent that contains one. |
| `MOTUS_CKPT_SETTING` | Name under `checkpoints/` used by evaluation. |

Typical pretrained components are `Motus/` Stage2, `Wan2.2-TI2V-5B/`, and `Qwen3-VL-2B-Instruct/`. See `motus/README.md` for upstream LeRobot training and T5-cache details.

## 4. Evaluation Checkpoint Lookup

`train.sh` writes checkpoints under:

```text
policy/Motus/checkpoints/<ckpt_setting>/.../checkpoint_step_<N>/pytorch_model/mp_rank_00_model_states.pt
```

During eval, pass `ckpt_name=<ckpt_setting>`. `model.py` recursively finds the latest `mp_rank_00_model_states.pt`. For externally stored weights, either set `MOTUS_CHECKPOINT_PATH` or symlink them:

```bash
cd XPolicyLab/policy/Motus
mkdir -p checkpoints
ln -sfn <checkpoint_root> checkpoints/<ckpt_setting>
```
