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
checkpoints/model.pt
cdlam/model.pt
base_vlm/
config.yaml
dataset_statistics.json
```

These assets are required. The complete VLA checkpoint includes the VLM parameters; `base_vlm/` contains only configuration, tokenizer and processor files. The VLM is restored directly from `checkpoints/model.pt` through the same Hugging Face loader before the complete policy state is strictly loaded. The 200k training initialization is not needed for inference.

Model repository: [MeituanRobotics/Meituan-Robotics-0-RoboDojo](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboDojo). Run all commands below from the `XPolicyLab/` root. After activating the existing policy environment, resolve a full commit and download that fixed revision:

```bash
conda activate meituan_robotics_0
HF_COMMIT="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("MeituanRobotics/Meituan-Robotics-0-RoboDojo").sha)')"
bash policy/Meituan_Robotics_0/download_checkpoint.sh \
  MeituanRobotics/Meituan-Robotics-0-RoboDojo "$HF_COMMIT"
```

Retain `HF_COMMIT` with evaluation results and reuse it for subsequent downloads. The optional third argument sets the destination; the default is `checkpoints/mr0_width_130k/`. For existing local assets, set `ROBODOJO_ASSET_ROOT=/path/to/assets`. The released `checkpoints/model.pt` has SHA256 `c847f6df3aa7954541d09960a12620dd5dbf5b796b3e5861db0c629744f9d496`.

## Evaluation

Place this directory at `XPolicyLab/policy/Meituan_Robotics_0` inside a complete official RoboDojo workspace. Its `env_cfg/` and `scripts/eval_policy.sh` must be alongside `XPolicyLab/` in the parent directory; RoboDojo simulator Assets must already be installed. Use the official XPolicyLab shared layer with RGB-marked JPEG support (validated reference: `6baccd57ff958570d0db1588370289ee36c8bb70`). Pass existing policy and simulator environments to the launcher:

```bash
EVAL_ENV_TYPE=sim bash policy/Meituan_Robotics_0/eval.sh \
  RoboDojo stack_bowls mr0 arx_x5 joint 0 \
  0 1 meituan_robotics_0 robodojo
```

Supported settings: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`. Replace `meituan_robotics_0` and `robodojo` with the actual policy and simulator environments. The environment arguments accept a conda name, environment directory or Python executable; the example selects policy GPU 0 and simulator GPU 1, which can also be adjusted. Use `EVAL_ENV_TYPE=debug` for the official debug client.

Simulation uses the native task budget by default. Set `EVAL_EPISODES=1` for one episode; an explicit `EVAL_NUM` takes precedence, and `EVAL_NUM=native` restores the native budget.

## Results

The released checkpoint achieves **14.6333% success rate** and **20.1475 Score**, averaged over two independent full RoboDojo evaluations. Both runs use native seed 0, continuous gripper widths, 50 predicted actions and 16 executed actions per request. Each run uses the official fixed budget of 2,100 episodes.

The 54 native task variants are aggregated into 42 official tasks. Generalization tasks combine their fixed and random layouts. Each category averages its tasks, and the overall result is the equally weighted mean of the five categories. All values below are means of the two runs.

| Category | Success rate (%) | Score |
| --- | ---: | ---: |
| Generalization | 12.0833 | 17.4250 |
| Precision | 15.0000 | 23.6813 |
| Long-Horizon | 14.3750 | 27.0125 |
| Memory | 28.3333 | 29.0250 |
| Open | 3.3750 | 3.5938 |
| **Overall** | **14.6333** | **20.1475** |

### Per-task results

#### Generalization

| Task | Success rate (%) | Score |
| --- | ---: | ---: |
| `stack_bowls` | 27.00 | 32.10 |
| `push_T` | 0.00 | 0.00 |
| `pack_objects_into_box` | 1.00 | 10.40 |
| `fold_clothes` | 42.00 | 45.00 |
| `hang_mugs` | 2.00 | 10.25 |
| `sweep_blocks` | 2.00 | 2.00 |
| `pour_liquid_into_cup` | 0.00 | 0.00 |
| `make_toast` | 0.00 | 10.25 |
| `arrange_largest_number` | 15.00 | 20.10 |
| `sort_nesting_dolls_by_size` | 14.00 | 14.00 |
| `store_laptop_and_headphones` | 18.00 | 36.80 |
| `stack_blocks` | 24.00 | 28.20 |

#### Precision

| Task | Success rate (%) | Score |
| --- | ---: | ---: |
| `fasten_screws` | 11.00 | 31.00 |
| `plug_in_charger` | 2.00 | 2.00 |
| `insert_tubes` | 52.00 | 65.40 |
| `pour_balls_into_vase` | 19.00 | 19.00 |
| `play_Xylophone` | 2.00 | 2.00 |
| `deposit_coin` | 9.00 | 14.00 |
| `insert_key` | 0.00 | 13.95 |
| `build_tower` | 25.00 | 42.10 |

#### Long-Horizon

| Task | Success rate (%) | Score |
| --- | ---: | ---: |
| `put_bottles_into_dustbin` | 68.00 | 77.80 |
| `fill_pen_holder` | 0.00 | 4.55 |
| `classify_objects` | 12.00 | 22.80 |
| `play_tic_tac_toe` | 26.00 | 53.90 |
| `fill_egg_holder` | 0.00 | 2.90 |
| `organize_table` | 6.00 | 50.75 |
| `make_kong` | 3.00 | 3.00 |
| `play_stacking_toy` | 0.00 | 0.40 |

#### Memory

| Task | Success rate (%) | Score |
| --- | ---: | ---: |
| `cover_blocks` | 85.00 | 87.95 |
| `match_and_pick_from_conveyor` | 75.00 | 75.00 |
| `swap_blocks` | 2.00 | 2.00 |
| `swap_T` | 0.00 | 0.00 |
| `press_by_number` | 7.00 | 7.00 |
| `imitate_sorting_sequence` | 1.00 | 2.20 |

#### Open

| Task | Success rate (%) | Score |
| --- | ---: | ---: |
| `align_blocks` | 0.00 | 0.00 |
| `general_pickup` | 27.00 | 27.00 |
| `stack_blocks_by_language` | 0.00 | 1.00 |
| `solve_equation` | 0.00 | 0.00 |
| `classify_objects_by_language` | 0.00 | 0.50 |
| `pick_from_conveyor_by_image` | 0.00 | 0.00 |
| `store_tools_in_toolbox` | 0.00 | 0.25 |
| `pour_by_language` | 0.00 | 0.00 |

## Configuration

- `ROBODOJO_ASSET_ROOT` / `asset_root`: directory containing the three model assets.
- `ROBODOJO_RUN_ROOT` / `run_root`: writable runtime caches, defaulting to `runs/` here.
- `config.yaml` and `dataset_statistics.json` in the asset root: inference configuration and q99 statistics for this checkpoint.

Environment variables take precedence over YAML values. Images are 640×480 RGB, ordered head, left wrist, right wrist. State/action order is left arm 6, left gripper 1, right arm 6, right gripper 1.

`deploy.py` collects every 25 Hz observation between action requests to supply causal history. The model uses ten Euler steps, an outer BF16 context and an FP32 action head. Episode reset clears memory without reseeding. Preserve this behavior when integrating another client.

Upstream licenses are retained in `LICENSE`, `source_starvla/LICENSE` and `source_cdlam/LICENSE`.
