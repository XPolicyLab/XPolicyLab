# X_WAM Installation

`X_WAM` currently does not provide a top-level `install.sh`, so this document is required. X-WAM also depends on Wan2.2-TI2V-5B base weights and experiment directories that must be wired manually.

## 1. Create the Environment

```bash
conda create -n XWAM python=3.10 -y
conda activate XWAM
```

## 2. Install X-WAM

Validated core dependency versions:

- `python>=3.10`
- `torch>=2.4.0` (tested with 2.8.0+cu129)
- `numpy<2` (tested with 1.23.5)
- `diffusers>=0.31.0` (tested with 0.38.0)
- `transformers>=4.49.0,<=4.51.3` (tested with 4.51.3)
- `flash-attn` (tested with 2.8.3)

```bash
cd XPolicyLab/policy/X_WAM/X-WAM
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

cd ../../..
pip install -e .
```

## 3. Model and Checkpoint Paths

X-WAM checkpoints are experiment directories that contain `config.yaml` plus `checkpoints/<step>.ckpt/...`. Runtime also needs Wan2.2-TI2V-5B base weights.

| Variable | Meaning |
| --- | --- |
| `XWAM_EXP_PATH` | Experiment directory containing `config.yaml` and `checkpoints/`. |
| `XWAM_CKPT_ROOT` | Root used with `XWAM_EXP_SETTING` when `XWAM_EXP_PATH` is unset. |
| `XWAM_EXP_SETTING` | Experiment directory name. |
| `XWAM_STEPS` | Checkpoint step, defaults to `last`. |
| `XWAM_WAN_CHECKPOINT_DIR` | Wan2.2-TI2V-5B base weight directory. |
| `XWAM_ALLOW_DUMMY_POLICY` | Set to `true` to skip weight loading for protocol debugging. |

Base weights are documented in the official Wan2.2 repository: https://github.com/Wan-Video/Wan2.2. X-WAM checkpoints and datasets are hosted at https://huggingface.co/sharinka0715/X-WAM-checkpoints.

## 4. Evaluation Checkpoint Layout

```bash
cd XPolicyLab/policy/X_WAM
mkdir -p checkpoints
ln -sfn <experiment_dir> checkpoints/<ckpt_name>
```

Pass `<ckpt_name>` to `eval.sh` or the split server/client scripts.
