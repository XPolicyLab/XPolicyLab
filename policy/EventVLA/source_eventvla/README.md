# EventVLA

EventVLA is the public RoboTwin-Mem VLA training and evaluation codepath. This release keeps one clean model route:

- framework: `EventVLA`
- method: `pure_image_keyframe_memory`
- temporal anchors: first frame, `t-30`, `t-15`, current frame
- action chunk: 50 steps
- data mixture: `robotwin_mem`
- example entrypoint: `examples/RoboTwin-Mem`

There are no legacy import shims or registry aliases in this branch. Old checkpoint configs and old ablation modes are intentionally unsupported.

## Install

```bash
pip install -e .
```

Install the model, training, and simulator dependencies required by your environment before running training or evaluation.

## Data

Prepare RoboTwin-Mem data in LeRobot format and point `datasets.vla_data.data_root_dir` to the data root. The public mixture expects these task directories under that root:

```text
cover_blocks_hard/
put_back_block_hard/
rearrange_blocks_hard/
observe_and_pickup_hard/
find_seal_and_seal_stamp/
observe_and_pickup_object/
reproduct_route/
press_button_keyframe/
```

The mixture name is `robotwin_mem`, and the robot type / normalization key is also `robotwin_mem`.

## Train

Single-node entry:

```bash
bash examples/RoboTwin-Mem/train_files/run_eventvla_train.sh /path/to/robotwin_mem/lerobot_data
```

Multi-node SLURM entry:

```bash
sbatch examples/RoboTwin-Mem/train_files/run_eventvla_train_batch.sh /path/to/robotwin_mem/lerobot_data
```

The default config is:

```text
examples/RoboTwin-Mem/train_files/eventvla_robotwin_mem.yaml
```

It fixes `framework.memory_ablation_mode=pure_image_keyframe_memory`, `sampling_interval=50`, action horizon 50, and temporal image anchors `absolute_indices=[0]`, `delta_indices=[-30, -15, 0]`.

Training logs intentionally keep the resolved `pure_image_keyframe_memory` profile, temporal anchor settings, keyframe image memory settings, and keyframe/event debug metrics for debugging.

## Evaluate

Start a policy server with an EventVLA checkpoint:

```bash
bash examples/RoboTwin-Mem/eval_files/run_policy_server.sh /path/to/checkpoints/steps_xxxxx_pytorch_model.pt 0 5840
```

The server also accepts legacy starVLA `QwenOFT` checkpoints when their checkpoint
config is the raw-image `pure_image_keyframe_memory` profile. Token-memory and
other old ablation checkpoints remain unsupported. For legacy RoboTwin weights
whose `dataset_statistics.json` uses `new_embodiment`, pass
`UNNORM_KEY=new_embodiment` during evaluation.

Run RoboTwin-Mem evaluation from a RoboTwin-Mem checkout:

```bash
ROBOTWIN_MEM_ROOT=/path/to/RoboTwin-Mem \
POLICY_CKPT_PATH=/path/to/checkpoints/steps_xxxxx_pytorch_model.pt \
bash examples/RoboTwin-Mem/eval_files/eval.sh <task_name> <task_config>
```

The eval interface maintains raw keyframe image memory outside the model and feeds those images back through the Qwen multi-image input path.

## Key Files

- `eventvla/model/framework/EventVLA.py`
- `eventvla/model/memory_ablation.py`
- `eventvla/dataloader/gr00t_lerobot/mixtures.py`
- `examples/RoboTwin-Mem/train_files/eventvla_robotwin_mem.yaml`
- `examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py`
