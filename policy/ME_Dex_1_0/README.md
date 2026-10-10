# ME-Dex-1.0

**Contributor:** Li Auto ME-Dex Team | **Paper:** [Technical Report](https://machembodied.com/ME-Dex/ME-Dex1.0.html) | **Original code:** https://github.com/MachEmbodied/ME-Dex-1.0

`ME_Dex_1_0` is the XPolicyLab adapter for RoboTwin. It supports
`bench_name=RoboTwin`, `env_cfg_type=arx_x5`, and `action_type=joint`.

## Installation

```bash
conda activate <policy_env>
cd XPolicyLab/policy/ME_Dex_1_0
bash install.sh
```

Install RoboTwin separately in `<robotwin_env>` following its official instructions.

## Evaluation

Download the ME-Dex-1.0 checkpoint and the Wan2.2 assets:

```bash
CHECKPOINT_DIR=checkpoints/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard

hf download liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard \
  --local-dir "${CHECKPOINT_DIR}"

hf download Wan-AI/Wan2.2-TI2V-5B \
  config.json Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json \
  google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json \
  google/umt5-xxl/tokenizer_config.json \
  --local-dir "${CHECKPOINT_DIR}/wan"
```

Run the standard RoboTwin evaluation:

```bash
cd XPolicyLab/policy/ME_Dex_1_0
ROBOTWIN_TASK_CONFIG=demo_randomized \
ROBOTWIN_TEST_NUM=100 \
bash eval.sh RoboTwin adjust_bottle \
  ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard \
  arx_x5 joint 42 0 0 <policy_env> <robotwin_env>
```

Use `ROBOTWIN_TASK_CONFIG=demo_clean` for Clean evaluation. The checkpoint expects
`input_color_order: bgr`. RoboTwin has no tactile observations, so the current tactile
frame is set to zero while preserving the sensor support mask.

## Training

The reference Clean50 training code is in `training/`. It includes the data loader,
configuration, distributed entry point, and the RoboTwin tactile replay collector.

```bash
cd XPolicyLab/policy/ME_Dex_1_0
python -m pip install -r training/requirements.txt
torchrun --nnodes=2 --nproc_per_node=16 \
  -m training.train --config training/configs/clean50_uni.yaml
```

Set the dataset, T5 cache, initialization checkpoint, tactile checkpoint, and Wan2.2
paths in `training/configs/clean50_uni.yaml` before starting a run. Set `model.topology`
to `full_joint` or `h_bridge` for the corresponding attention topology.

`training/robotwin_tactile/tactile_force_field.py` is the simulator-side collector used
to replay the original Clean50 demonstrations and append aligned three-axis tactile
fields. Its integration and configuration are documented in
`training/robotwin_tactile/README.md`.

The released checkpoint reports 88.9% on Clean, 71.9% on Random, and 80.4% average on
the RoboTwin benchmark.
