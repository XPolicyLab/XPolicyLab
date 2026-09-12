# RoboTwin2 native evaluation

This is the primary RoboTwin2 evaluation path. It deliberately delegates
episode construction, expert seed filtering, instruction sampling, video
recording, and success accounting to RoboTwin's own
`script/eval_policy.py`.

The manager creates a global task queue. Each GPU worker loads one WSP2 model,
keeps it resident, and calls `script.eval_policy.main(usr_args, model=model)`
for every assigned task. `wsp2_policy` implements RoboTwin's `get_model`,
`eval`, and `reset_model` policy contract.

The supported launcher is:

```bash
ROBOTWIN2_EVAL_MODEL_PATH=/path/to/checkpoint \
ROBOTWIN_ROOT=/path/to/RoboTwin \
ROBOTWIN_GPU_IDS='[0,1,2,3]' \
./recipes/eval/eval_robotwin2.sh
```

Single-task example:

```bash
CUDA_VISIBLE_DEVICES=0 \
WORLDSCAPE_CHECKPOINT=/path/to/checkpoint \
ROBOTWIN_ROOT=/path/to/RoboTwin \
ROBOTWIN_EPISODES_PER_TASK=10 \
PYTHONPATH=src:. \
python experiments/robotwin/run_robotwin_manager.py \
  EVALUATION.task_name=adjust_bottle \
  MULTIRUN.gpu_ids='[0]'
```

All-task, multi-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ROBOTWIN_GPU_IDS='[0,1,2,3]' \
WORLDSCAPE_CHECKPOINT=/path/to/checkpoint \
ROBOTWIN_ROOT=/path/to/RoboTwin \
PYTHONPATH=src:. \
python experiments/robotwin/run_robotwin_manager.py \
  MULTIRUN.eval_phases=clean
```

## XPolicyLab entry

The native entry above remains supported and unchanged. This checkout is
vendored inside XPolicyLab at `policy/WorldScape_Policy_2/worldscape-policy/`;
the XPolicyLab WebSocket policy adapter (`model.py`, `deploy.py`, `eval.sh`,
`setup_eval_policy_server.sh`, `setup_eval_env_client.sh`) lives one level up
in `policy/WorldScape_Policy_2/` and is documented in its README. XPolicyLab
batch evaluation remains disabled because each WSP2 server owns stateful
visual/event memory.
