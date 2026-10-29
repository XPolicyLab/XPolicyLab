<div align="center">

<img src="https://raw.githubusercontent.com/THUSI-Lab/PrismWAM/main/assets/logo.png" alt="PrismWAM logo" width="88">

# PrismWAM

**Decoupling State Representations in World Action Models**

[🚀 Quick start](#quick-start) · [🔥 Training](#training) · [🎬 Evaluation & deployment](#evaluation-and-deployment) · [📦 Models](#model-zoo)

[🌐 **Project page**](https://prism-wam.github.io/) · [📄 **arXiv**](https://arxiv.org/abs/2610.26564) · [🤗 **Hugging Face**](https://huggingface.co/THUSI-Lab/PrismWAM)

</div>

## News

- **2026-10-19** · PrismWAM ranks 🥇 **#1** on **RoboTwin 2.0 Clean → Randomized** and 🥈 **#2** on **RoboDojo**
  (w/o Agent track), ahead of models pretrained on thousands of hours of robot actions.
- **2026-10-19** · Released code and training recipes, with **PrismWAM** checkpoints for RoboTwin 2.0, RoboDojo and LIBERO on [Hugging Face](https://huggingface.co/THUSI-Lab/PrismWAM).

## Overview

Video latents entangle *how the world looks* (texture, colour, lighting) with *how it is
physically configured* (geometry, pose, contact), yet actions change only the latter.
PrismWAM decouples the two along a **Semantic Planning → Physical Rollout → Executable Action** chain:

- 🎯 **Semantic planning with sparse RGB.** Predict a single future RGB frame, the goal at the
  end of the action chunk. It teaches *what* to achieve and is only a training target, so inference generates no future RGB.
- 🧊 **Physical rollout in dense depth.** Predict every future depth frame. Depth keeps geometry
  and motion but drops texture and lighting, so it is easier to predict.
- 🦾 **Executable action from the rollout.** An action expert reads the depth rollout through
  one-way RGB → Depth → Action attention and is denoised jointly with it.

![PrismWAM architecture: a sparse future RGB frame is predicted during training only; future depth and the action chunk are denoised together through RGB-to-depth-to-action chain attention.](https://raw.githubusercontent.com/THUSI-Lab/PrismWAM/main/assets/architecture.webp)

Results without any embodied action pretraining:

| Benchmark | Setting | PrismWAM | PrismWAM\* |
| --- | --- | ---: | ---: |
| RoboTwin 2.0 | Clean → Randomized | 70.57 | 83.67 |
| RoboDojo | Score / SR | 25.10 / 18.55 | 36.23 / 29.30 |

Success rates in %, except the RoboDojo score. Clean → Randomized trains on clean scenes and
tests on clean and randomized ones. Comparisons with other models are on the
[project page](https://prism-wam.github.io/).

The LIBERO model is also tested on LIBERO-Plus and LIBERO-Pro, perturbed versions of LIBERO,
without further training (success rates in %):

| Benchmark | Setting | PrismWAM |
| --- | --- | ---: |
| LIBERO | 4 suites | 99.35 |
| LIBERO-Plus | Trained on LIBERO | 79.44 |
| LIBERO-Pro | Trained on LIBERO | 54.48 |

## Quick start

Install (Linux, NVIDIA GPU, CUDA 12.8 toolkit; see [installation](docs/installation.md)):

```bash
git clone https://github.com/THUSI-Lab/PrismWAM.git
cd PrismWAM
conda create -n prismwam --override-channels -c conda-forge python=3.13 pip
conda activate prismwam
pip install -r environments/torch.txt
pip install --no-build-isolation -r requirements.txt -e .
```

Load a released policy. The model and the T5 text encoder that reads its instructions are
downloaded to `weights/` on first use:

```python
from prismwam import load_pretrained

policy = load_pretrained("robotwin")             # or "robodojo", "libero", "prismwam_star/robotwin", or a local model folder
observation["prompt"] = policy.format_prompt("beat the block with the hammer")
out = policy.predict(observation, seed=0)        # out["action"]: 32 x 14 joint actions
```

See the [observation and action format](docs/deployment.md#observations-and-actions).

## Model zoo

| Model | Benchmark | Trained on | Recipe | Download |
| --- | --- | --- | --- | --- |
| PrismWAM | RoboTwin 2.0 | 50 tasks, clean scenes | [`robotwin`](configs/robotwin) | [🤗 robotwin](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robotwin) |
| PrismWAM | RoboDojo | 35 tasks, ARX X5 | [`robodojo`](configs/robodojo) | [🤗 robodojo](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo) |
| PrismWAM | LIBERO | 4 suites, 40 tasks | [`libero`](configs/libero) | [🤗 libero](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/libero) |
| PrismWAM\* | RoboTwin 2.0 | 50 tasks, clean scenes | — | [🤗 PrismWAM-Star robotwin](https://huggingface.co/THUSI-Lab/PrismWAM-Star/tree/main/robotwin) |

See the [model guide](docs/models.md) for downloads and model layout.
Training initializers are available in [`init/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/init).

## Training

Two stages per benchmark, one YAML recipe and one script each:

| Stage | What it learns | Trained parts | Starts from |
| --- | --- | --- | --- |
| **1 · Depth warm-up** | Current depth + instruction → 8 future depth frames (next 32 steps; LIBERO: 4 / 16) | Visual transformer | Pretrained video model |
| **2 · Joint training** | RGB history, current depth, robot state, instruction → sparse RGB goal, dense depth rollout, 32-step action chunk (LIBERO: 16) | Shared RGB/depth transformer + action expert | Stage 1 + action-expert initializer |

```bash
bash scripts/libero/stage1_depth.sh
bash scripts/libero/stage2_joint.sh       # starts from outputs/libero/stage1_depth
```

Set `DATA_ROOT`, `WEIGHTS_ROOT` and `OUTPUT_ROOT` at the top of each script or inline, e.g.
`DATA_ROOT=/data bash scripts/libero/stage1_depth.sh`. `key=value` arguments override the YAML;
`--dry-run` checks the recipe without training.

| Benchmark | Stage 1 | Stage 2 | Guide |
| --- | --- | --- | --- |
| RoboTwin 2.0 | 15K steps · batch 256 | 15K steps · batch 256 | [RoboTwin](docs/benchmarks/robotwin.md) |
| RoboDojo | 15K steps · batch 1024 | 40K steps · batch 1024 | [RoboDojo](docs/benchmarks/robodojo.md) |
| LIBERO | 15K steps · batch 256 | 15K steps · batch 256 | [LIBERO](docs/benchmarks/libero.md) |
| Real robot | 15K steps · batch 256 | 15K steps · batch 256 | [Real robot](docs/benchmarks/realworld.md) |

Batch sizes are global; gradient accumulation keeps them fixed on any GPU count that divides them.
See the [training guide](docs/training.md) and [data preparation](docs/data.md).

## Evaluation and deployment

- **RoboTwin, RoboDojo:** [XPolicyLab adapter](docs/xpolicylab.md); full suites in the
  [RoboTwin](docs/benchmarks/robotwin.md#evaluate) and [RoboDojo](docs/benchmarks/robodojo.md#evaluate) guides.
- **LIBERO:** `scripts/libero/eval.py` ([LIBERO guide](docs/benchmarks/libero.md#evaluate)).
- **Your own robot:** export the model and serve it over HTTP ([deployment](docs/deployment.md)).

```bash
python scripts/serve.py --model robodojo --port 8000    # local HTTP policy server
```

## Repository layout

```
configs/
  base.yaml                 shared optimisation, logging and path settings
  stages/                   what Stage 1 and Stage 2 train (model + schedule)
  <benchmark>/              stage1_depth.yaml, stage2_joint.yaml: data, batch, steps
scripts/
  <benchmark>/              stage1_depth.sh, stage2_joint.sh: launch one stage
  libero/eval.py            LIBERO evaluation: model server + simulator client
  train.sh                  generic launcher: bash scripts/train.sh <config> [key=value ...]
  prepare_data.py           indexes, normalization statistics, text caches
  export_model.py           package a trained checkpoint as a model folder
  serve.py                  serve a policy over HTTP
src/prismwam/               models, datasets, trainer, deployment, Hugging Face loading
integrations/xpolicylab/    XPolicyLab policy adapter for RoboTwin and RoboDojo
patches/                    RoboDojo depth rendering for evaluation
docs/                       installation, models, data, training, deployment, benchmarks
```

## Citation

If you find PrismWAM useful, please cite:

```bibtex
@misc{prism,
  title         = {PrismWAM: Decoupling State Representations in World Action Models},
  author        = {Xin, Tianyu and Zhong, Wanqi and Hu, Bowen and Guo, Zhangchi and Wang, Lijie and Wang, Yiming and Liao, Haoran and Yu, Heyang and Tian, Xiaoyu and Yu, Zhiding and Li, Yiming},
  year          = {2026},
  eprint        = {2610.26564},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.26564}
}
```

## License and acknowledgements

Code is [MIT](LICENSE), with some Apache-2.0 components ([third-party notices](THIRD_PARTY_NOTICES.md)).
Model weights, datasets and simulators keep their own licenses.

PrismWAM builds on code from FastWAM, DiffSynth-Studio, Wan and XPolicyLab. Thanks to their authors.
