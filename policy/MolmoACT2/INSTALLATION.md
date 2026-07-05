# MolmoACT2 Installation

`install.sh` can create the required environments, but this document is kept because MolmoACT2 has two upstream Python environments and XPolicyLab should use only the LeRobot training/evaluation environment.

## 1. Environment Overview

| Environment | Directory | Purpose |
| --- | --- | --- |
| XPolicyLab training/evaluation | `molmoact2/lerobot/.venv` | `lerobot_train`, `eval.sh`, and adapter inference. |
| Upstream FastAPI server | `molmoact2/.venv` | Optional official DROID/YAM server examples. |

RoboDojo training and XPolicyLab evaluation both use `molmoact2/lerobot/.venv`. The upstream `molmoact2/` source is not tracked in git; run `bash install.sh` on first setup.

## 2. One-command Install

```bash
cd XPolicyLab/policy/MolmoACT2
bash install.sh          # LeRobot training/eval env + XPolicyLab
bash install.sh all      # Above plus upstream FastAPI inference env
bash install.sh infer    # Only upstream FastAPI inference env, not XPolicyLab eval
```

## 3. Manual XPolicyLab Environment

```bash
cd XPolicyLab/policy/MolmoACT2/molmoact2/lerobot
UV_LINK_MODE=copy uv pip install -e ".[molmoact2,training,scipy-dep]" --index-strategy unsafe-best-match
source .venv/bin/activate
cd ../../..
pip install -e .
pip install h5py opencv-python
```

## 4. Optional Upstream FastAPI Environment

Only needed for official upstream DROID/YAM server workflows:

```bash
cd XPolicyLab/policy/MolmoACT2/molmoact2
uv sync
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download allenai/MolmoAct2
```

## 5. Useful Variables

| Variable | Meaning |
| --- | --- |
| `MOLMOACT2_CHECKPOINT_PATH` | Training starting checkpoint; defaults to HF `allenai/MolmoAct2`. |
| `MOLMOACT2_DATASET_ROOT` | LeRobot v3.0 dataset root. |
| `MOLMOACT2_DATASET_REPO_ID` | LeRobot dataset repo id. |
| `MOLMOACT2_OUTPUT_ROOT` | Training output root. |
| `MOLMOACT2_LOCAL_CACHE_ROOT` | Local HF datasets cache for multi-host training. |
| `SKIP_XPOLICYLAB=1` | Skip XPolicyLab install during `install.sh`. |

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `get_policy_class('molmoact2')` fails | Use `molmoact2/lerobot/.venv`, not `molmoact2/.venv`. |
| `import XPolicyLab` fails | Install XPolicyLab inside `lerobot/.venv`. |
| transformers version conflict | Use `lerobot/.venv` for XPolicyLab eval/training; reserve `molmoact2/.venv` for FastAPI server only. |
| `torchcodec` version conflict | Use `--index-strategy unsafe-best-match` during install. |
| Slow multi-host dataloading | Put `MOLMOACT2_LOCAL_CACHE_ROOT` on local disk. |
