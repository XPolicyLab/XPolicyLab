# GalaxeaVLA - Installation

GalaxeaVLA (`galaxea_fm`) is integrated into XPolicyLab as a `uv`-managed,
isolated environment. Default model variant: **G0Plus_3B** (PaliGemma-3B
backbone, 14-dim dual-arm joint actions). G0Tiny / pi0 / pi0fast are selectable
via `deploy.yml` (`model_variant` + `task_config_name`) without code changes.

## 1. Environment

```bash
cd XPolicyLab/policy/GalaxeaVLA
bash install.sh
```

This runs `uv sync` (Python 3.10, torch 2.7.1 cu128) and
`uv pip install -e .[dev]` inside `GalaxeaVLA/.venv`. `import XPolicyLab` is
resolved by prepending the repo root to `PYTHONPATH` in the eval/train scripts
(no separate install needed).

## 2. Manual downloads (not performed by install.sh)

| Asset | Where | Command |
| --- | --- | --- |
| ffmpeg | system | `sudo apt install -y ffmpeg` |
| PaliGemma-3B backbone | `weights/paligemma-3b-pt-224` | `hf download google/paligemma-3b-pt-224 --local-dir policy/GalaxeaVLA/weights/paligemma-3b-pt-224` |
| G0Plus_3B_base ckpt | `checkpoints/G0Plus_3B_base` | `hf download OpenGalaxea/G0-VLA --include "G0Plus_3B_base/*" --local-dir policy/GalaxeaVLA/checkpoints` |

Notes:
- The released weight file is `model_state_dict.pt` (not `model.pt`). `model.py`
  loads either name; `dataset_stats.json` must live in the same `checkpoints/`
  directory as the weights.
- Set `GALAXEA_PALIGEMMA_PATH` (or `paligemma_path` in `deploy.yml`) to the
  backbone dir from step 2.

## 3. Variant mapping

| `model_variant` | `task_config_name` | backbone |
| --- | --- | --- |
| `g0plus` (default) | `real/g0plus_xpolicylab_finetune` | PaliGemma-3B |
| `g0tiny` | `real/g0tiny_xpolicylab_finetune` | SmolVLM2-250M |
| `pi0` | `real/pi0_r1lite_finetune` | openpi |
| `pi0fast` | `real/pi0fast_r1lite_finetune` | openpi |

## 4. Embodiment note

G0Plus_3B_base ships 14-dim dual-arm joint actions/proprio
(`left_arm(6)/left_gripper(1)/right_arm(6)/right_gripper(1)`), which is
dimension-for-dimension aligned with XPolicyLab `arx_x5` (`dual_x5`). Pretrained
action/proprio projection layers load directly (no padding). ARX X5 and Galaxea
R1Lite are still physically different arms (relative-joint actions, distinct
limits/gripper travel), so dimensional alignment is not physical alignment:
fine-tune on XPolicyLab data before expecting good closed-loop behavior.
