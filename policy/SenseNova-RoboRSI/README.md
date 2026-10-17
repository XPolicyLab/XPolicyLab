# SenseNova-RoboRSI

**Contributor:** RoboRSI integration | **Paper:** not provided | **arXiv:** not provided | **Original code:** [RoboRSI](https://github.com/SenseTime-FVG/RoboRSI)

RoboDojo adapter for the original adaptive π0.5 hybrid. π0.5 proposes 50 actions;
Codex CLI reviews and approves 5–50 steps. The standard
`Model/reset/update_obs/get_action` interface retains the original robot tools,
ICL context, feedback and conversation recovery. Required RoboRSI, adaptive and
OpenPI sources are included under `vendor/`; model weights are downloaded.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Place `XPolicyLab/` inside the official RoboDojo workspace, beside `env_cfg/`.
The installation source ZIP preserves the complete [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
source tree and adds this policy. The PR changes only `policy/SenseNova-RoboRSI/`.
The official simulator image, Conda environment and Assets are
installed separately and keep their original dependencies.

```bash
cd XPolicyLab/policy/SenseNova-RoboRSI
bash install.sh
```

`install.sh` automatically builds Python 3.10.12 from fixed official source when
needed, obtains fixed uv, then installs the environments. A C compiler, `make` and Python build
libraries are required. See [INSTALL.md](INSTALL.md) for prerequisites.
The independent model-tool environment is built from a hash-locked package list
and the included OpenPI source. Its 77 package versions match the B-region
baseline, with the unused historical `heart-runtime` installation and five
test-only packages omitted. See [the dependency cleanup](INSTALL.md#b-region-alignment-and-dependency-cleanup)
for the exact runtime boundaries and intentional omissions.
No prebuilt environment archive is required.

| Runtime | Fixed identity |
| --- | --- |
| Policy host | Python 3.10.12; B controller's 72 versions, plus separately locked XPolicyLab/build dependencies |
| Model tools | Python 3.11.13; 77 B-region distribution versions, independent sandbox |
| VLA | Python 3.11.17; 142 B-region distribution versions and original 112 source overrides; historical B157 retained as provenance |
| Native CLI | `codex-cli 0.160.1`; authenticated executable and companion resources |

SDK/CLI-bin package versions remain `0.162.1` in the policy host and `0.159.3` in model tools; policy execution uses the separate CLI `0.160.1`.

Installation downloads the fixed CLI, all 27 ICL packages, π0.5 inference weights
and tokenizer. It checks pinned artifacts and actual CPU/sandbox capabilities,
but makes no model/API request. The source package contains no virtualenvs,
weights, credentials or installed environments. Allow at least 47 GB plus build
and simulator space. Required Linux namespaces and `memfd_create` must work;
there is no unsandboxed fallback.

## Data Processing

No conversion is required. Evaluation uses sealed public-training ICL packages
and released π0.5 inference weights.

## Training

This is an eval-only adapter; no additional policy training is provided.
A training-code release is not scheduled; no release date has been provided.

## Evaluation

Use the standard ten-argument evaluation entry:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

Set the runtime key privately, then run this RoboDojo example:

```bash
read -rsp 'CODEX_API_KEY: ' CODEX_API_KEY; export CODEX_API_KEY
EVAL_ENV_TYPE=sim EVAL_NUM=1 bash eval.sh \
  RoboDojo align_blocks roborsi-codex-v1 arx_x5 joint 0 0 0 uv RoboDojo
```

The final arguments select the policy `.venv` (`uv`) and official simulator Conda
environment (`RoboDojo`). Supported here: `arx_x5`, `joint`, one environment.
The benchmark owns physics, layouts and scoring. RGB decoding and action
sizes use XPolicyLab's shared helpers. Native task performance is not asserted
by installation or synthetic tests.

## Model Assets

The Agent is fixed to `gpt-6-astra/azure_L/qwb`, reasoning `max`, using
the configured provider. Installation does not select another endpoint or model.
The official π0.5 checkpoint is revision
`35efbc7dedfdbeeb6e95fb749bd885d73d483e41` of `RoboDojo-Benchmark/RoboDojo`,
step 59999, configuration `pi05_base_aloha_full_sim_arx-x5_seed_0`.
Its 18 inference files total 12,440,992,402 bytes; `pi05/checkpoint.json`
and `pi05/tokenizer.json` pin the exact inputs. `icl/artifacts.json` pins ICL.
Do not add these downloaded resources or `runtime/` to a source release.

## Configuration

The adapter-specific `deploy.yml` settings are:

| Key | Default | Meaning |
| --- | --- | --- |
| `ws_ping_interval_s` | `20.0` | WebSocket heartbeat interval in seconds. |
| `ws_ping_timeout_s` | `20.0` | WebSocket heartbeat timeout in seconds. |
| `checkpoint_path` | `checkpoints/pi05-arx-x5-seed0-step59999` | Explicit π0.5 checkpoint directory, relative to this policy directory. |
| `request_timeout_s` | `3600.0` | Client request timeout in seconds for the hybrid evaluation loop. |

The default weights live in
`policy/SenseNova-RoboRSI/checkpoints/pi05-arx-x5-seed0-step59999/`,
resolved through XPolicyLab's shared checkpoint resolver. This explicit
`checkpoint_path` overrides the standard run-derived checkpoint directory.
`ckpt_name=roborsi-codex-v1` identifies the pinned Agent preset.
