# OrimaPI

**Contributor:** Orimaton Team | **Paper:** TBD | **arXiv:** TBD | **Original code:** [Orimaton/OrimaPI](https://github.com/Orimaton/OrimaPI)

[简体中文](README_zh.md)

OrimaPI runs the formal code skills from `orimapi/skills/code_skills/` on RoboDojo-Sim with `arx_x5` and joint-space actions (`action_type: joint`). The adapter uses `configs/robodojo.yaml` from the vendored `OrimaPI/` source with `code-policy` in `direct` mode and one action per observation; GPT-6-Astra selects and orchestrates skills using the task instruction and RGB observations. This adapter is evaluation-only.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use a Linux machine with an NVIDIA GPU, a working RoboDojo installation in the `RoboDojo` Conda environment, and this XPolicyLab checkout at `RoboDojo/XPolicyLab`. Run from the RoboDojo root.

```bash
conda create -n orimapi python=3.11 -y
conda activate orimapi
bash XPolicyLab/policy/OrimaPI/install.sh
python XPolicyLab/policy/OrimaPI/download_assets.py

export ORIMAPI_NATIVE_PYTHON="$(conda run -n RoboDojo python -c 'import sys; print(sys.executable)')"
```

`install.sh` installs XPolicyLab and the bundled [OrimaPI source](OrimaPI/) into the active policy environment. `download_assets.py` downloads and verifies Grounding DINO, SAM, and Depth Anything under `policy/OrimaPI/OrimaPI/checkpoint/`. The native planning workers use the RoboDojo interpreter exported above.

## Data Processing

Not applicable. This is an eval-only adapter that consumes runtime observations directly; `process_data.sh` is omitted.

## Training

Not supported. OrimaPI has no local training entry in this adapter; `train.sh` is omitted.

## Evaluation

Standard 10-argument entry:

```bash
bash XPolicyLab/policy/OrimaPI/eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>
```

Offline debug closed-loop check (no simulator required):

```bash
cd XPolicyLab/policy/OrimaPI
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls code-skills arx_x5 joint 0 0 0 orimapi RoboDojo
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls code-skills arx_x5 joint 0 0 0 orimapi RoboDojo
```

From the RoboDojo root, run `stack_bowls` in simulation on GPU 0:

```bash
EVAL_NUM=1 bash XPolicyLab/policy/OrimaPI/eval.sh \
  RoboDojo stack_bowls code-skills arx_x5 joint 0 0 0 orimapi RoboDojo
```

For all tasks on one seed at their native episode counts:

```bash
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/OrimaPI \
  --policy-env orimapi --ckpt code-skills --env-cfg arx_x5 --action-type joint \
  --seed 0 --eval-num native
bash scripts/robodojo.sh summarize
```

For split-machine or separate server/client deployment via `setup_eval_policy_server.sh` and `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Model Assets

`python XPolicyLab/policy/OrimaPI/download_assets.py` downloads and verifies the pinned Hugging Face vision weights (Grounding DINO, SAM, and Depth Anything V2 Metric Indoor Small) into `policy/OrimaPI/OrimaPI/checkpoint/`. No VLA checkpoint is required; `ckpt_name` must be set to `code-skills`.

## Configuration

Edit [OrimaPI/configs/robodojo.yaml](OrimaPI/configs/robodojo.yaml) on the policy host. Set `supervisor.model` to the model ID and `supervisor.endpoint` to the full Responses URL. The default is `gpt-6-astra` at `https://api.githubcopilot.com/responses`. Restart the policy server after editing the YAML.

With the default `supervisor.auth.type: copilot`, create `policy/OrimaPI/OrimaPI/configs/copilot.credentials.json` on the policy host. This file is ignored by Git:

```json
{
  "github_token": "<your GitHub Copilot credential>"
}
```

For a Responses-compatible provider that takes an API key, replace the entire `supervisor.auth` mapping with `auth: {type: env}` and set `supervisor.endpoint` and `supervisor.model` for that provider. Pass the key through the policy server's `OPENAI_API_KEY` environment variable. `supervisor.api_key_env` specifies the variable name. The configuration checks that the response's model ID matches the requested model.

Extra `deploy.yml` keys and environment variables:

| Key / Variable | Default | Description |
| --- | --- | --- |
| `request_timeout_s` | `3600.0` | Per-request WebSocket client timeout in seconds |
| `ORIMAPI_NATIVE_PYTHON` | `OrimaPI/.venv-robodojo/bin/python` | RoboDojo Python interpreter used by native IK/planning workers |
| `ROBODOJO_ROOT` | `<XPolicyLab>/..` | RoboDojo workspace root containing `Assets/` and `env_cfg/` |
