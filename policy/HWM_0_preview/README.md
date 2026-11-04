# HWM-0-Preview

**Contributor:** Horizon Robotics | **Paper:** Not announced | **arXiv:** Not announced | **Original code:** Inference implementation included here

HWM-0-Preview combines a vision-language subtask planner, visual and semantic memory,
and joint video/action flow matching. This eval-only adapter supports ARX X5
dual-arm joint control with head and wrist RGB observations.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Linux, Python 3.10, a CUDA-capable GPU, the CUDA toolkit (`nvcc`) and a C++
compiler for the attention extensions. Weights alone occupy approximately
32 GiB; allow additional memory for inference. Install the policy dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The installer checks extension imports and rebuilds incompatible wheels locally.

Set up RoboDojo in a separate evaluation environment following the official
instructions. The policy environment can be a Conda environment name or a Python
environment directory; the evaluation environment must be a Conda environment.

## Data Processing

Not included; this is an eval-only adapter.

## Training

Not included; this is an eval-only adapter.

## Evaluation

Evaluation follows XPolicyLab's policy-server / environment-client structure:

```text
RoboDojo environment client  ←── WebSocket ──→  HWM-0-Preview policy server
  Observations and history  ────────────────→  Subtask planning and action prediction
  Execute predicted actions ←────────────────  Action chunks and updated memory
```

The client runs the simulator and maintains each episode's visual history and
semantic memory. The policy server loads the checkpoint and predicts actions
from observations, task instructions and episode context.

By default, the model predicts 48 actions and the client executes 30 before
requesting the next prediction. Use `eval.sh` to launch both on one machine,
or start the server and client separately on different machines.

Command template, run from this directory:

```text
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

Example after downloading the bundle:

```bash
EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo stack_bowls ./HWM-0-Preview-RoboDojo arx_x5 joint 0 \
  0 0 .venv base
```

Replace the task, GPU IDs and environment names as needed. `debug` checks the
interface without a simulator; use `EVAL_ENV_TYPE=sim` with the installed RoboDojo
simulator for task evaluation. For split-machine deployment, follow the
[XPolicyLab guide](../../README.md).

### Deployment options

Set these options in [deploy.yml](deploy.yml). `null` uses the bundle's default.

| Option | Default | Description |
| --- | --- | --- |
| `num_inference_steps` | `null` → 10 | Denoising steps per prediction. |
| `flow_shift` | `null` → 5.0 | Flow-matching sampling schedule shift. |
| `replan_steps` | `null` → 30 | Actions executed before replanning (1–48). |
| `inference_batch_size` | 4 | Maximum environments per model batch; reduce for lower GPU memory use. |

Set these environment variables before starting the corresponding process:

| Variable | Process | Default | Description |
| --- | --- | --- | --- |
| `HWM_CLIENT_CONTEXT` | Client | `1` | Keep episode history and semantic memory on the client; use atomic inference requests. |
| `HWM_DYNAMIC_BATCH` | Server | `1` | Batch requests across clients; requires client context. |
| `HWM_INTERMEDIATE_HEAD_ONLY` | Client | `0` | Read only head RGB between action-chunk boundaries. |
| `HWM_SPARSE_HEAD_CAPTURE` | Client | `0` | Capture intermediate head frames only when needed for history; requires client context and takes precedence over head-only capture. |

Set both `HWM_CLIENT_CONTEXT=0` and `HWM_DYNAMIC_BATCH=0` for the legacy
single-client path. The two optional capture modes require RoboDojo capture APIs
and retain render synchronization, but change recording density; leave them
disabled for full-rate recording.

## Model Assets

Download the complete 60K EMA bundle from
[HorizonRobotics/HWM-0-Preview-RoboDojo](https://huggingface.co/HorizonRobotics/HWM-0-Preview-RoboDojo)
using [download_checkpoint.sh](download_checkpoint.sh):

```bash
bash download_checkpoint.sh
```

The script requires the `hf` CLI (`python -m pip install huggingface_hub` if needed).
The repository defaults to `HorizonRobotics/HWM-0-Preview-RoboDojo`. An optional
first argument overrides it; a second selects a revision (default: `main`).
The bundle is saved beside the script:

```text
HWM-0-Preview-RoboDojo/
├── config.yaml
├── model.pt
├── manifest.json
├── dataset_statistics.json
├── processor/
└── tokenizer/
```

Keep all files together and pass `./HWM-0-Preview-RoboDojo` as `ckpt_name`.
All evaluation seeds can share this directory.
