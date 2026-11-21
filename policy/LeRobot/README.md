# LeRobot

**Contributor:** Lahrlahr | **Paper:** Not applicable (runtime adapter) | **arXiv:** Not applicable | **Original code:** [LeRobot](https://github.com/huggingface/lerobot)

Eval-only integration of LeRobot checkpoints and their saved processors. The initial target is DM05-MEM with RoboDojo `arx_x5`, `action_type=joint`, and one environment (`eval_batch: false`). Other policy/robot combinations require separate validation.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install RoboDojo, its simulator environment, and robot/object/layout assets using the [benchmark installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/). Until this adapter is merged, obtain it explicitly:

```bash
export ROBODOJO=/absolute/path/to/RoboDojo
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git "$ROBODOJO"
git -C "$ROBODOJO" checkout f57e1633ffed59761bc8e568cfe7530c4b33381c
git -C "$ROBODOJO" submodule update --init XPolicyLab
git -C "$ROBODOJO/XPolicyLab" fetch origin pull/230/head
git -C "$ROBODOJO/XPolicyLab" checkout FETCH_HEAD
```

Keep the policy in a separate Python 3.12 environment. For DM05-MEM, the LeRobot MEM implementation is currently an external dependency, pinned here rather than assumed to exist on upstream main:

```bash
conda create -n lerobot-policy python=3.12 -y
conda activate lerobot-policy
pip install uv
git clone https://github.com/Lahrlahr/lerobot.git /path/to/lerobot
cd /path/to/lerobot
git checkout 82d2b010b37618d62e0b931a12f31ffdc54be797
uv sync --active --locked --extra dm05 --extra training --extra dataset --no-dev
uv pip install --python "$CONDA_PREFIX/bin/python" -e "$ROBODOJO/XPolicyLab"
export PYTHONPATH="$ROBODOJO:$ROBODOJO/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
```

Use the same policy environment for the server and the tests (`python -m pytest policy/LeRobot/test_model.py` from the XPolicyLab root; install pytest to run tests). Install XPolicyLab in the simulator environment too, for its websocket client dependencies. Do not share Python-version-specific dependency directories through `PYTHONPATH` between the two environments.

## Data Processing

No adapter-specific data conversion. LeRobot v2.1/v3.0 datasets can be prepared with the official XPolicyLab `scripts/transform_lerobot_v21_format.py` / `scripts/transform_lerobot_v30_format.py` converters. The standard `observation.state`, `action`, and `observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist` feature names match this adapter's default camera mapping. Preserve the robot's joint/gripper packing order when preparing data.

## Training

Training remains in LeRobot; no duplicate training pipeline is provided here. Follow the pinned LeRobot checkout's `docs/source/dm05.mdx`. Evaluate the resulting checkpoint's `pretrained_model` directory, retaining its saved pre/postprocessors, normalization statistics and tokenizer assets.

## Evaluation

The public format-converted base checkpoint is available without another conversion:

```bash
hf download Dexmal/DM05-Lerobot-MEM --revision 24c79b61b21ccafcf9fe8bf1e5444c4ebb98b2c1 \
  --local-dir /path/to/dm05-mem
export SIMULATOR_ENV=robodojo  # The simulator conda environment installed above.
cd "$ROBODOJO/XPolicyLab/policy/LeRobot"
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks /path/to/dm05-mem \
  arx_x5 joint 0 0 0 lerobot-policy "$SIMULATOR_ENV"
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks /path/to/dm05-mem \
  arx_x5 joint 0 0 0 lerobot-policy "$SIMULATOR_ENV"
```

For simulation, use the `pretrained_model` directory produced by the LeRobot training recipe above:

```bash
export CHECKPOINT=/absolute/path/to/outputs/train/dm05-mem-cover/checkpoints/040000/pretrained_model
cd "$ROBODOJO/XPolicyLab/policy/LeRobot"
EVAL_NUM=1 EVAL_ENV_TYPE=sim bash eval.sh RoboDojo cover_blocks "$CHECKPOINT" \
  arx_x5 joint 0 0 0 lerobot-policy "$SIMULATOR_ENV"
# Full 50-layout evaluation.
EVAL_NUM=50 EVAL_ENV_TYPE=sim bash eval.sh RoboDojo cover_blocks "$CHECKPOINT" \
  arx_x5 joint 0 0 0 lerobot-policy "$SIMULATOR_ENV"
```

The base checkpoint can also exercise the loading/IO path, but is not a task success-rate claim. Results and videos are written below `$ROBODOJO/eval_result/RoboDojo/cover_blocks/LeRobot/`. Check the new run's `_result.json`: `details` must contain the requested number of distinct `layout_id` values (0--49 for 50 layouts). Report the number of successful layouts divided by the requested count; never treat an incomplete run as a full result.

```python
import json
from pathlib import Path

result = json.loads(Path("/absolute/path/to/the/new/run/_result.json").read_text())
details = list(result["details"].values())
expected_layouts = 50  # Use 1 for the one-layout smoke check.
assert len(details) == expected_layouts
assert sorted(row["layout_id"] for row in details) == list(range(expected_layouts))
successes = sum(row["success"] for row in details)
print(f"{successes}/{expected_layouts} ({successes / expected_layouts:.1%})")
```

`deploy.yml` exposes `device` (default `auto`: CUDA when available, otherwise CPU), optional `prompt` (fallback when no runtime instruction is supplied), and `camera_mapping` (checkpoint feature key → runtime RGB camera name). Map every checkpoint image feature explicitly; default head camera mapping is `cam_high` → `cam_head`. Joint states/actions are packed/unpacked with the shared robot metadata; images remain RGB. Checkpoint statistics, robot metadata and preprocessing are loaded unchanged.

The demo evaluation loop and launch scripts are reused, not copied. Each request returns **one executed action** through public `select_action()`. LeRobot still predicts/queues full action chunks internally. This ensures MEM sees every control frame, even while queued actions are being executed. Run the environment at the training dataset's control frequency (the default ARX X5 profile is 25 Hz); `memory_stride` is in control frames, not prediction requests. `reset()` clears history, action queues and processors between episodes. Parallel environments and end-effector actions are deliberately rejected in this initial integration.
