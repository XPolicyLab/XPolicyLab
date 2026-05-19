# InternVLA-A1 In XPolicyLab

This wrapper reuses the original `InternVLA-A1` codebase at:

`/mnt/pfs/pg4hw0/niantian/InternVLA-A1`

Recommended environment:

```bash
conda activate internvla_a1
```

Start only the XPolicyLab policy server:

```bash
bash deploy.sh <gpu_id> <policy_conda_env> <ckpt_path> [port] [device] [stats_key] [dtype]
```

Run full debug evaluation:

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> joint <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <ckpt_path> [stats_key] [dtype]
```

Notes:

- `action_type` is currently fixed to `joint`.
- This integration follows the same observation/history logic as the RoboTwin adaptation, but runs inside XPolicyLab's local policy server.
