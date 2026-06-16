# aha-wam RoboDojo Adapter

This policy adapts the locally trained aha-wam checkpoint for XPolicyLab RoboDojo evaluation.

Default artifacts:

- checkpoint: `/mnt/petrelfs/caijisong/XPolicyLab/checkpoint/step_002500.pt`
- dataset stats: `/mnt/petrelfs/caijisong/XPolicyLab/checkpoint/dataset_stats.json`
- elava code: `/mnt/petrelfs/caijisong/linglong/project/fastwam/elava-prior-only/elava`
- base model cache: `/mnt/petrelfs/caijisong/dualWAM/checkpoints`
- env cfg root: `/mnt/petrelfs/caijisong/env_cfg` (`AHA_WAM_ENV_CFG_ROOT`)

The model was trained with `configs/task/robodojo_local_history_updated_kv_prior_only_16.yaml`, `action_type=joint`, and 14-D qpos actions ordered as `[left_arm, left_ee, right_arm, right_ee]`.
During evaluation, the default replanning cadence is one video DiT forward followed by two action DiT forwards, then another video DiT forward. Override with `AHA_WAM_ACTION_FORWARDS_PER_VIDEO_REPLAN`.

Policy server example:

```bash
cd /mnt/petrelfs/caijisong/XPolicyLab/policy/AHA_WAM
bash setup_eval_policy_server.sh RoboDojo stack_bowls local_aha_wam arx_x5 3500 joint 0 0 wam 12345 localhost
```

Full debug flow:

```bash
bash eval.sh RoboDojo stack_bowls local_aha_wam arx_x5 3500 joint 0 0 0 wam wam
```

For simulator evaluation, set `eval_env: sim` in `deploy.yml` and run from a workspace that provides `scripts/eval_policy.sh` and the RoboDojo simulator environment.
