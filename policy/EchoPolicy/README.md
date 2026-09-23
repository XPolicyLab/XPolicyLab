# EchoPolicy

**Contributor:** EchoPolicy team | **Paper:** to be added | **Original code:** private EchoPolicy repository

EchoPolicy adds VLM subgoal planning around the XPolicyLab Pi05 adapter for RoboDojo's `arx_x5` joint-action environment. It is currently an **eval-only** adapter: model weights are downloaded separately and the VLM key is injected at runtime.

Shared conventions, split-machine deployment, and official results are documented in the [XPolicyLab README](../../README.md) and the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

This adapter reuses the upstream Pi05 OpenPI environment:

```bash
cd XPolicyLab/policy/EchoPolicy
bash install.sh
```

The command delegates to `policy/Pi_05/install.sh`; no credentials or checkpoints are stored in this repository.

## Model assets

XPolicyLab contains the Pi05 loader and model code, but Git does not contain binary
weights. Supply the RoboDojo checkpoint as a mounted directory (the default
public release is step 14077):

```text
policy/EchoPolicy/checkpoints/pi05_robodojo_59999/
```

```bash
export ECHO_POLICY_CHECKPOINT_PATH=/path/to/pi05_robodojo_14077
bash download_checkpoint.sh
```

The published JAX/Orbax checkpoint can be downloaded directly from Hugging Face:

```bash
bash download_checkpoint.sh
```

The default is `cjgogo/RoboDojo-pi05-checkpoints`, path
`checkpoints/sim-10task/14077`. Set `ECHO_POLICY_CHECKPOINT_STEP=15000` to use
the later published simulation checkpoint, or override
`ECHO_POLICY_CHECKPOINT_REPO` and `ECHO_POLICY_CHECKPOINT_REMOTE_PATH` for an
approved mirror.

## VLM configuration

Set these only in the runtime environment:

```bash
export VLM_API_KEY='...'
export VLM_MODEL=gemini-3.8-flash
export VLM_BASE_URL='https://<approved-endpoint>/v1beta'
export VLM_THINKING_LEVEL=low
```

If `VLM_API_KEY` is absent, the adapter logs a warning and falls back to the original instruction. The key is never read from a checked-in file.

## Evaluation

```bash
cd XPolicyLab/policy/EchoPolicy
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo cover_blocks pi05_robodojo_14077 arx_x5 joint 0 0 0 uv <robodojo_env>
```

For a split policy server, start the policy machine with:

```bash
bash setup_eval_policy_server.sh RoboDojo cover_blocks pi05_robodojo_14077 arx_x5 joint 0 0 uv 9001 0.0.0.0
```

Then point the RoboDojo environment client at the policy machine's reachable IP:

```bash
bash setup_eval_env_client.sh RoboDojo cover_blocks pi05_robodojo_14077 arx_x5 joint 0 0 <robodojo_env> \
  'ckpt_name=pi05_robodojo_14077,action_type=joint' 9001 <POLICY_SERVER_IP>
```

The public server must expose the standard XPolicyLab websocket protocol. Never put an API key, SSH credential, or private host path in the PR.

## Limitations

- Training and data conversion scripts are not included; this is an eval-only submission.
- Official leaderboard evaluation requires a public checkpoint repository (set in `ECHO_POLICY_CHECKPOINT_REPO`) and maintainer confirmation that the evaluation environment may inject the VLM secret or use an approved VLM endpoint.
