# EventVLA

EventVLA integration for RoboDojo/XPolicyLab inference.

## Install

```bash
conda activate XPolicyLab
cd /mnt/workspace/yangganlin/code/RoboDojo/RoboDojo/XPolicyLab/policy/EventVLA
bash install.sh
```

If the environment already has matching torch or flash-attn builds:

```bash
SKIP_TORCH_INSTALL=1 SKIP_FLASH_ATTN_INSTALL=1 bash install.sh
```

## Eval

The default checkpoint is:

```text
/nav-oss/yangganlin/models/robodojo/20260617_robodojo_pure_image_keyframe_memory_teacher_qwenoft/checkpoints/steps_150000_pytorch_model.pt
```

Override it with `EVENTVLA_CKPT_PATH` when needed.

```bash
cd /mnt/workspace/yangganlin/code/RoboDojo/RoboDojo/XPolicyLab/policy/EventVLA
export EVENTVLA_CKPT_PATH=/nav-oss/yangganlin/models/robodojo/20260617_robodojo_pure_image_keyframe_memory_teacher_qwenoft/checkpoints/steps_150000_pytorch_model.pt
bash eval.sh RoboDojo <task_name> eventvla arx_x5 3500 joint 0 0 1 XPolicyLab XPolicyLab
```

The adapter starts two servers:

- EventVLA websocket server from `source_eventvla/deployment/model_server/server_policy.py`
- XPolicyLab policy server from `XPolicyLab/setup_policy_server.py`

The model currently targets `arx_x5` with `action_type=joint`.
