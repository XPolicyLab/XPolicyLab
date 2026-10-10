# Contributing a Policy to XPolicyLab

This page is the submission standard for `policy/<POLICY>/` adapters: what a complete adapter contains, how to test it, and what a PR must include. For repo-wide concepts and workflows, see the [README](README.md) ([中文](README_zh.md)). Root `README.md` and `README_zh.md` must stay in sync — update both in the same change.

Two bundled Agent Skills automate most of this: `xpolicylab-model-integration` builds an adapter, `xpolicylab-adapter-check` audits one before a PR. They live in `.agents/skills/`, which `.cursor/skills` and `.claude/skills` symlink to, so Cursor, Claude Code and Codex all pick them up. [AGENTS.md](AGENTS.md) distills this page into the always-on rules every agent must follow; `CLAUDE.md` just imports it. See [Using a coding agent](#using-a-coding-agent).

## Getting started

The fastest route is to copy the reference adapter, keep the XPolicyLab boundary small, and debug before touching a simulator:

1. **Read [policy/demo_policy](policy/demo_policy/README.md)** — `model.py`, `deploy.py`, `deploy.yml`, and the `eval.sh` / `setup_eval_policy_server.sh` / `setup_eval_env_client.sh` trio.
2. **Scaffold** with `bash scripts/create_policy.sh <POLICY_NAME>`, then fill in its README.
3. **Implement `model.py` first**, keeping `bench_name`, `task_name`, `ckpt_name`, `env_cfg_type`, `action_type`, and `seed` consistent across data, training, and eval (README, [Common Workflow](README.md#-common-workflow)).
4. **Put deployment defaults in `deploy.yml`** and keep `deploy.py` aligned with `demo_policy/deploy.py` unless the environment loop truly differs.
5. **Run [Testing](#testing)**, then move to `EVAL_ENV_TYPE=sim` or a split-machine deployment (README, [Deployment Flow](README.md#-deployment-flow)).

Eval-only submissions are accepted when training code cannot be open-sourced yet: say so in the PR, notify the maintainers ([Contact](README.md#-contact)), and share a timeline. For leaderboard evaluation, attach a checkpoint download script (Hugging Face or ModelScope preferred).

## Adapter Standard

### Files

Scaffold with `bash scripts/create_policy.sh <POLICY>` (copies `policy/demo_policy/`), then keep:

```text
policy/<POLICY>/
├── README.md                    # required: install / data / train / eval guide
├── __init__.py                  # required: keeps XPolicyLab.policy.<POLICY> importable
├── install.sh                   # policy environment setup (see exceptions below)
├── eval.sh                      # required: same-machine evaluation
├── setup_eval_policy_server.sh  # required: policy-side server
├── setup_eval_env_client.sh     # required: environment-side client
├── deploy.yml                   # required: deployment config, protocol: ws
├── deploy.py                    # required: deployment loop
├── model.py                     # required: Model adapter class
├── process_data.sh              # data conversion (see exceptions below)
├── train.sh                     # training entry (see exceptions below)
└── INSTALLATION.md              # optional: extra setup notes
```

Three scripts have a narrow exception, and each one has to be **declared**; everything else in the tree is unconditional.

- `process_data.sh` / `train.sh` may be omitted only for an agreed **eval-only** submission: state it in the PR, notify the maintainers ([Contact](README.md#-contact)), and give a timeline for open-sourcing training.
- `install.sh` may be omitted only when the environment comes entirely from an upstream project's own recipe. Say so under `Installation` in the policy README and give the full manual steps there or in `INSTALLATION.md`, as `policy/Dexora_1B` and `policy/X_WAM` do.

### `model.py`

Define `class Model(ModelTemplate)` (`from XPolicyLab.model_template import ModelTemplate`):

| Method | Contract |
| --- | --- |
| `__init__(model_cfg)` | `model_cfg` is `deploy.yml` merged with per-run overrides (`ckpt_name`, `action_type`, `env_cfg_type`, `seed`, ...). Load checkpoints and processors here. |
| `update_obs(obs)` / `update_obs_batch(obs_list)` | Store observation dict(s) for the next action call. |
| `get_action()` | Return one action chunk: `list[dict]` of numpy arrays. |
| `get_action_batch(env_idx_list=None)` | Batched chunks aligned with active env indices. |
| `reset()` | Clear model state between episodes. |

Action dictionaries use the standard keys (`left_arm_joint_state`, `right_ee_joint_state`, `ee_pose`, ...) with dimensions taken from `get_robot_action_dim_info(env_cfg_type)` in `XPolicyLab.utils.process_data` — never hard-coded, and never through a private re-implementation of the lookup. `env_cfg/` lives in the parent workspace, outside this checkout, so an adapter that assembles that path itself gets it wrong. Observation and trajectory formats: README, [Standard Data Formats](README.md#-standard-data-formats).

A new robot must be registered in **both** robot-info files, or training and evaluation will disagree about action dimensions:

| File | Read by | Keyed by |
| --- | --- | --- |
| `<parent>/env_cfg/robot/_robot_info.json` | `get_robot_action_dim_info()` and `get_action_dim()` in `XPolicyLab.utils.process_data` — runtime and offline conversion | robot name, from `config.robot` in `<parent>/env_cfg/<env_cfg_type>.yml` |
| `utils/robot/_robot_info.json` | `utils/get_action_dim.sh`, called by `train.sh` | `env_cfg_type` |

Two entry points share the name `get_action_dim`, and they do **not** read the same file. Runtime and conversion code imports `get_action_dim` (or `get_robot_action_dim_info`) from `XPolicyLab.utils.process_data`. The training path — `train.sh`, or a Python training entry it invokes — uses `utils/get_action_dim.sh` instead: it delivers the value as a shell variable, runs on stdlib `json` alone (importing the Python module pulls in numpy, cv2, h5py and yaml), and works in a standalone checkout that has no outer `env_cfg/` tree.

Two more shared entry points, so adapters do not re-derive them: the importable root in `policy/<POLICY>/model.py` is the parent of this checkout, `Path(__file__).resolve().parent.parents[2]` (equivalently `Path(__file__).resolve().parents[3]`; counted from the policy directory, `parents[1]` or `parents[3]` is a bug), and checkpoint directories resolve through `XPolicyLab.utils.checkpoint_resolver` (`resolve_checkpoint_root`, or `build_run_dir_name` / `candidate_checkpoint_roots` when the adapter adds its own naming layer).

`model.py` never decodes images. The policy server decodes every observation it forwards, so `obs["vision"][<camera>]["color"]` is always a plain image array. This holds for `update_obs` / `update_obs_batch` and for any custom RPC a policy exposes that carries an observation, so an adapter with its own deploy loop still must not decode.

In offline code — conversion scripts and training dataloaders that read trajectory files — **only `decode_image_bit` from `XPolicyLab.utils.process_data` is supported**, and image bits that get written back out must come from its inverse, `encode_image_bit`. Never hand-roll `cv2.imdecode` / `np.frombuffer` / PIL decoding, and never write stored buffers with a bare `cv2.imencode`. The README states the rule in [Standard Data Formats](README.md#decode-only-through-decode_image_bit).

The reason is that stored image bits come in **two byte formats**, and only these two functions know the difference:

- **legacy** — a JPEG written by handing an RGB array straight to `cv2.imencode`, which reads its input as BGR. The stored bytes are channel-reversed with respect to the JPEG standard, so `cv2.imdecode` reverses them a second time and returns the original RGB, while PIL, ffmpeg or a browser show red and blue swapped. Everything collected before the marker existed is this format, and it is never migrated: JPEG cannot swap channels losslessly.
- **standard** — a conforming RGB JPEG written by `encode_image_bit`, carrying a JPEG `COM` segment with the payload `XPL-RGB1`. Every decoder skips an unknown `COM`, so the marker is free; PIL even surfaces it as `Image.open(...).info["comment"]`. `cv2.imdecode` returns BGR for these bytes, so decoding them owes exactly one swap.

`decode_image_bit` reads the marker and returns RGB for both, which is why callers never swap. Images are RGB end to end, and a `cv2.cvtColor(decode_image_bit(...), COLOR_BGR2RGB)` is always a bug: the decoded pixels are indistinguishable to the eye — only the marker in the encoded buffer tells the formats apart — so a caller-side swap is right on at most one of them and silently wrong on the other. That trap is easy to fall into — a PIL-based loader tested against fresh data looks perfect and then quietly corrupts older episodes.

No channel conversion belongs in conversion, training, or eval code; only `utils/process_data.py`, which owns the format distinction, may convert. Two exceptions: medium adapters — `COLOR_RGB2BGR` immediately before `cv2.VideoWriter.write(...)` and `COLOR_BGR2RGB` immediately after `cv2.VideoCapture.read()` — and a deliberate RGB→BGR conversion for a checkpoint trained on BGR data, which must be opt-in through a documented `deploy.yml` key that defaults to RGB (see `policy/Dexora_1B`'s `input_color_order`).

### `deploy.yml`

`policy_name` must equal the directory name — the server imports `XPolicyLab.policy.<policy_name>.model`, and the setup scripts derive the name from the directory. Keep `protocol: ws` (`legacy_tcp` is for unmigrated legacy adapters only). Per-run fields are overridden by the setup scripts; put stable defaults here. Reference (`policy/demo_policy/deploy.yml`):

```yaml
policy_name: demo_policy
protocol: ws
host: localhost
port: null
bench_name: null
task_name: null
ckpt_name: null
env_cfg_type: null
seed: null
action_type: null
gpu_id: null
eval_batch: false
```

All of these are required except `ckpt_name` and `gpu_id`, which the setup scripts supply per run. Keep a key even when a script already defaults it, and document model-specific extra keys in the policy README.

### Scripts

All adapters share the same entry-point conventions (argument meanings: README, [Common Workflow](README.md#-common-workflow)):

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

`eval.sh` starts the policy server, waits for it, runs the environment client, and cleans up — keep it aligned with `policy/demo_policy/eval.sh`; document any extra arguments in the policy README. Checkpoints resolve to `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` unless the policy README documents another layout.

### Policy README

All policy READMEs share one template — [policy/demo_policy/README.md](policy/demo_policy/README.md) is the minimal reference, [policy/AHA_WAM/README.md](policy/AHA_WAM/README.md) shows a complex adapter (model assets, custom environment variables):

1. **Header**: `**Contributor:** ... | **Paper:** ... | **arXiv:** ... | **Original code:** ...`, then 1–3 sentences on the model and what the adapter supports; mention vendored upstream directories here.
2. **Pointer paragraph** (verbatim): shared conventions link to the root README, official results link to the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).
3. **Sections in order**: `Installation`, `Data Processing`, `Training`, `Evaluation`, plus `Model Assets` / `Configuration` / `Notes` only when the adapter needs them. State explicitly when a stage is unsupported (eval-only, upstream-native data, ...).
4. **Policy-specific content only**: one command template plus one runnable example per stage, extra arguments, required environment variables, `deploy.yml` keys, checkpoint-layout deviations. Do not restate shared argument tables or the split-machine flow — link to the root README instead.
5. **Declare LeRobot data** when the policy trains on it. Under `Data Processing`, give the dataset version and say whether the keys match the official converters (README, [Official LeRobot conversion](README.md#official-lerobot-conversion)). If they do, name the converter or the prepared export you consume, and say whether `process_data.sh` is absent or only links and normalizes the dataset. If they do not, state the deviation and how to produce that layout — otherwise the next user feeds official output to a trainer that cannot read it.

## Testing

Run these in order before opening a PR.

**1. Static checks** (repo root):

```bash
git diff --check
bash -n policy/<POLICY>/*.sh
python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py
```

XPolicyLab data supports only `decode_image_bit` / `encode_image_bit` (README, [Standard Data Formats](README.md#decode-only-through-decode_image_bit)). The first grep must return nothing on adapter-owned conversion and training code; vendor files that never see XPolicyLab trajectories can be skipped, but a `cv2.imdecode` on those trajectories is a fail. The second grep catches image bits written without the format marker — a `cv2.imencode` whose output is stored or published is a fail, one whose output is decoded again in the same process is not. The third surfaces channel swaps that need judging (medium adapters and a documented `input_color_order` are the only allowed hits):

```bash
# only decode_image_bit is supported
grep -rnE 'cv2\.imdecode|np\.frombuffer|Image\.open' policy/<POLICY>/
# stored buffers must come from encode_image_bit — judge each hit
grep -rn 'cv2\.imencode' policy/<POLICY>/
# channel swaps — judge each hit
grep -rnE 'COLOR_BGR2RGB|COLOR_RGB2BGR|\.\.\., ::-1' policy/<POLICY>/
```

**2. Debug closed loop** — no simulator needed; verifies imports, server startup, observation serialization, action keys and dimensions, and batch logic:

```bash
cd policy/<POLICY>
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 <policy_env> base
```

The run must reach `[MAIN] eval finished` with no tracebacks. The debug client sends plain image arrays by default; re-run with `DEBUG_OBS_ENCODED=1` to make it send encoded camera colors instead — a JPEG buffer, raw bytes, and a plain array across the three cameras — which exercises the server-side decode path that real environment clients rely on. For a quick smoke test, `policy/demo_policy` accepts placeholder env names such as `base`.

**3. Simulator evaluation** — recommended for every PR and required before a leaderboard entry is published: run the same `eval.sh` with `EVAL_ENV_TYPE=sim` (or unset) inside a RoboDojo / RoboTwin workspace and record task success rates.

## Using a coding agent

`xpolicylab-model-integration` builds an adapter (a prompt like `Integrate <POLICY_NAME> into XPolicyLab` is enough). `xpolicylab-adapter-check` audits one against this page before a PR (`Check policy/<POLICY_NAME>`). For an agent that supports none of these, paste this checklist:

```text
Integrate <POLICY_NAME> into XPolicyLab.

Use policy/demo_policy as the reference.
1. Inspect the upstream model's inference API and dependencies.
2. Create or update policy/<POLICY_NAME>/README.md with install, checkpoint, train, and eval commands.
3. Implement install.sh and, if needed, process_data.sh and train.sh. Omitting any of the three requires a declared exception in the README.
4. Implement model.py with Model.__init__, update_obs, get_action, reset, and batch methods. model.py never decodes.
5. Offline conversion/training decodes XPolicyLab images only via decode_image_bit and writes them only via encode_image_bit. Stored bits come in two byte formats; only these functions tell them apart, and both give you RGB. Never swap channels yourself.
6. Keep deploy.py aligned with policy/demo_policy/deploy.py.
7. Put deployment defaults in deploy.yml, keeping the standard key set (protocol: ws, host, port, ...).
8. Run EVAL_ENV_TYPE=debug eval.sh and fix shape/action-key/server errors.
9. Summarize supported action_type, env_cfg_type, checkpoint layout, and remaining limitations.
```

## PR Standard

Title: `[policy] <POLICY>: <short summary>`, e.g. `[policy] FastWAM: add RoboDojo adapter`.

Every submission PR must include the adapter following the standard above, a working policy README, and test evidence. For **official leaderboard evaluation** ([RoboDojo](https://robodojo-benchmark.com/LeaderBoard), [RoboTwin](https://robotwin-platform.github.io/leaderboard)), the PR description must also carry a checkpoint download script — Hugging Face or ModelScope preferred — so we can reproduce your results; we evaluate and publish entries as soon as possible.

### PR description template

GitHub pre-fills this from [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md):

```markdown
## Policy
- Name / paper / upstream repo:
- Supported: bench_name=..., env_cfg_type=..., action_type=...
- Training support: full | eval-only (training release ETA: ...)

## Components
- [ ] install.sh (or upstream-native install, documented in the policy README)
- [ ] model.py (+ __init__.py)
- [ ] images: only decode_image_bit / encode_image_bit are supported (two byte formats → RGB), no channel swaps (see README)
- [ ] deploy.yml (standard key set incl. protocol: ws / host / port, policy_name matches the directory)
- [ ] deploy.py aligned with demo_policy (or divergence explained)
- [ ] eval.sh + setup_eval_policy_server.sh + setup_eval_env_client.sh
- [ ] process_data.sh / train.sh (or eval-only, declared above)
- [ ] policy README with install / data / train / eval commands

## Testing
- [ ] bash -n + py_compile pass
- [ ] decode/encode grep: only decode_image_bit and encode_image_bit on XPolicyLab data
- [ ] EVAL_ENV_TYPE=debug closed loop passes (paste the log tail)
- [ ] Simulator eval: task=..., success=... (if available)

## Checkpoint (required for leaderboard evaluation)
<download script, Hugging Face or ModelScope preferred>

## Limitations / notes
...
```
