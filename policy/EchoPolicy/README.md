# EchoPolicy

**Contributor:** EchoPolicy team | **Paper:** to be added | **Original code:** private EchoPolicy repository

EchoPolicy runs the RoboDojo `arx_x5` joint-action Pi05 policy through the same in-process orchestration path used by the deployed policy. Each observation is decomposed into VLM subgoals, expanded into 16 noisy candidates, grouped by endpoint, selected with the VLM, and emitted from a 10-step action cache. It is an **eval-only** adapter: model weights are downloaded separately and the VLM key is injected at runtime.

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
weights. Supply the official RoboDojo checkpoint as a mounted directory (the
default public release is step 59999):

```text
policy/EchoPolicy/checkpoints/pi05_robodojo_59999/
```

```bash
export ECHO_POLICY_CHECKPOINT_PATH=/path/to/pi05_robodojo_59999
bash download_checkpoint.sh
```

The published JAX/Orbax checkpoint can be downloaded directly from Hugging Face:

```bash
bash download_checkpoint.sh
```

The default is the official `RoboDojo-Benchmark/RoboDojo` Hugging Face dataset,
path `ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999`. Set
`ECHO_POLICY_CHECKPOINT_STEP` to another step only when the corresponding
directory exists in that official repository, or override
`ECHO_POLICY_CHECKPOINT_REPO` and `ECHO_POLICY_CHECKPOINT_REMOTE_PATH` for an
approved mirror. The helper passes `--repo-type dataset` to the Hugging Face CLI
and downloads only inference artifacts (`params`, `assets`, and metadata), not
the much larger training `train_state`.

## VLM configuration

Set these only in the runtime environment:

```bash
export VLM_API_KEY='...'
export VLM_MODEL=gemini-3.8-flash
export VLM_BASE_URL='https://<approved-endpoint>/v1beta'
export VLM_THINKING_LEVEL=low
```

`VLM_API_KEY` is required for evaluation. For offline protocol tests only, set `ECHO_ALLOW_PASSTHROUGH=1`; the key is never read from a checked-in file.

## Evaluation

```bash
cd XPolicyLab/policy/EchoPolicy
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo cover_blocks pi05_robodojo_59999 arx_x5 joint 0 0 0 uv <robodojo_env>
```

For a split policy server, start the policy machine with:

```bash
bash setup_eval_policy_server.sh RoboDojo cover_blocks pi05_robodojo_59999 arx_x5 joint 0 0 uv 9001 0.0.0.0
```

Then point the RoboDojo environment client at the policy machine's reachable IP:

```bash
bash setup_eval_env_client.sh RoboDojo cover_blocks pi05_robodojo_59999 arx_x5 joint 0 0 <robodojo_env> \
  'ckpt_name=pi05_robodojo_59999,action_type=joint' 9001 <POLICY_SERVER_IP>
```

The public server must expose the standard XPolicyLab websocket protocol. Never put an API key, SSH credential, or private host path in the PR.

## Runtime defaults

The public adapter uses these deployment defaults: Gemini `gemini-3.8-flash`, low thinking level, 16 VLA candidates, image noise standard deviation 5.0, joint-state noise standard deviation 0.05 for far targets, 10-step action chunks, CFG disabled, unlimited VLM concurrency, and Pi05 inference padded to batch size **160**. Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` and the XLA memory fraction to `0.9` in the policy process.

## Limitations

- Training and data conversion scripts are not included; this is an eval-only submission.
- Official leaderboard evaluation requires a public checkpoint repository (set in `ECHO_POLICY_CHECKPOINT_REPO`) and maintainer confirmation that the evaluation environment may inject the VLM secret or use an approved VLM endpoint.

The standard eval loop may update observations after each executed action; these updates only replace stored observations. Candidate inference runs on `get_action` / `get_action_batch`, in the requested environment order. Set `ECHO_LOG_DIR` to retain per-environment planning and selection diagnostics. GPU selection in the launch script uses the runtime device visibility API.
