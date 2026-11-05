# Meituan Robotics 0

**Contributor:** Meituan Robotics | **Paper / arXiv:** Not released | **Original code:** inference source included here.

Evaluation-only RoboDojo policy using Qwen3.5-4B and MMDiT-Psi0, with episode-first RGB and three completed one-second CD-LAM memory buckets. It predicts 50 actions and executes 16 with continuous gripper widths.

The policy network is in `source_starvla/`; the frozen CD-LAM network is in `source_cdlam/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Activate a dedicated Python 3.11 environment on Linux x86_64, with a compatible CUDA 12.x toolkit (`nvcc`), NVIDIA driver and C++ compiler. Keep a CUDA GPU visible for the final FLA/Triton import check. From the `XPolicyLab/` root, install the pinned policy dependencies:

```bash
conda activate meituan_robotics_0
CUDA_HOME=/usr/local/cuda bash policy/Meituan_Robotics_0/install.sh
```

Adjust `CUDA_HOME` to your toolkit location. The script pins the tested CUDA 12.6 PyTorch build, model kernels, image processing and transport packages, including the public `safetensors==0.8.0rc0` prerelease. CUDA extensions may compile during installation. CD-LAM source and its inference attention adaptation are included in this policy; no external source checkout or site-packages patch is needed. The simulator uses its existing RoboDojo environment with XPolicyLab's dependencies from `pyproject.toml`; run this installer only in the policy environment.

## Data Processing

Not included in this eval-only adapter.

## Training

Not included in this eval-only adapter. Training release date: not announced.

## Model Assets

Download these assets from the model repository:

```text
Meituan_Robotics_0/model.ckpt
cdlam/model.pt
base_vlm/Qwen3.5-4B-Action/
```

All three are required. The base VLM directory includes weights, tokenizer and processor assets. The 200k training initialization is not needed for inference.

The repository ID and commit are pending author upload. Run all commands below from the `XPolicyLab/` root. After activating the existing policy environment, download once:

```bash
conda activate meituan_robotics_0
bash policy/Meituan_Robotics_0/download_checkpoint.sh owner/repository COMMIT_SHA
```

Use a full commit SHA. The optional third argument sets the destination; the default is `checkpoints/mr0_width_130k/`. For existing local assets, set `ROBODOJO_ASSET_ROOT=/path/to/assets`.

## Evaluation

Place this directory at `XPolicyLab/policy/Meituan_Robotics_0` inside a complete official RoboDojo workspace. Its `env_cfg/` and `scripts/eval_policy.sh` must be alongside `XPolicyLab/` in the parent directory; RoboDojo simulator Assets must already be installed. Use the official XPolicyLab shared layer with RGB-marked JPEG support (validated reference: `6baccd57ff958570d0db1588370289ee36c8bb70`). Pass existing policy and simulator environments to the launcher:

```bash
EVAL_ENV_TYPE=sim bash policy/Meituan_Robotics_0/eval.sh \
  RoboDojo stack_bowls mr0 arx_x5 joint 0 \
  0 1 meituan_robotics_0 robodojo
```

Supported settings: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`. Replace `meituan_robotics_0` and `robodojo` with the actual policy and simulator environments. The environment arguments accept a conda name, environment directory or Python executable; the example selects policy GPU 0 and simulator GPU 1, which can also be adjusted. Use `EVAL_ENV_TYPE=debug` for the official debug client.

Simulation uses the native task budget by default. Set `EVAL_EPISODES=1` for one episode; an explicit `EVAL_NUM` takes precedence, and `EVAL_NUM=native` restores the native budget.

## Configuration

- `ROBODOJO_ASSET_ROOT` / `asset_root`: directory containing the three model assets.
- `ROBODOJO_RUN_ROOT` / `run_root`: writable runtime caches, defaulting to `runs/` here.
- `config.yaml` and `dataset_statistics.json`: inference configuration and q99 statistics for this checkpoint.

Environment variables take precedence over YAML values. Images are 640×480 RGB, ordered head, left wrist, right wrist. State/action order is left arm 6, left gripper 1, right arm 6, right gripper 1.

`deploy.py` collects every 25 Hz observation between action requests to supply causal history. The model uses ten Euler steps, an outer BF16 context and an FP32 action head. Episode reset clears memory without reseeding. Preserve this behavior when integrating another client.

Upstream licenses are retained in `LICENSE`, `source_starvla/LICENSE` and `source_cdlam/LICENSE`.
