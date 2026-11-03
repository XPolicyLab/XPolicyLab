# SenseNova-RoboRSI

Simate beta + adaptive execution adapter (default 5–50 steps, configurable minimum). GPU evaluation acceptance is tracked separately; this source update does not claim a completed concurrency evaluation.

**Contributor:** RoboRSI integration | **Original code:** [RoboRSI](https://github.com/SenseTime-FVG/RoboRSI)

RoboDojo adapter for the original adaptive hybrid, with Simate beta selected
by the submitted `deploy.yml`. Simate beta proposes 50 actions;
Codex CLI reviews and approves 5–50 steps by default. Set
`ROBORSI_ADAPTIVE_MIN_STEPS` to an integer from 2 to 50 to change the minimum;
the maximum remains 50. The standard `Model/reset/update_obs/get_action` interface retains the
original robot tools, ICL context, feedback and conversation recovery. Required RoboRSI, adaptive and
OpenPI sources are included under `vendor/`; model weights are downloaded.

All 42 task templates use the same public interface,
[`BENCHMARK.md`](vendor/robodojo_adaptive/resources/interfaces/BENCHMARK.md).
It is copied unchanged to the model's read-only `/tools/BENCHMARK.md` at runtime.
The interface describes one episode without fixing its seed or layout; the evaluator
still selects both. Per-task configuration templates retain their existing action
budgets and materials. Their authenticated `adaptive_2_50` settings remain the
source baseline; runtime configuration overlays and records the selected limits.

## Installation

Place `XPolicyLab/` inside the official RoboDojo workspace, beside `env_cfg/`.
This adapter lives in the [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
repository at `policy/SenseNova-RoboRSI/`. Preserve the
`XPolicyLab/policy/SenseNova-RoboRSI` layout: installation also installs the
framework from `../..`, and evaluation uses its shared server and utilities.
The adapter and its tests are contained in that policy directory.
The official simulator image, Conda environment and Assets are
installed separately and keep their original dependencies.

```bash
cd XPolicyLab/policy/SenseNova-RoboRSI
bash install.sh
```

The default Simate installation includes pinned model dependencies, source
checks and checkpoint downloads. If the base RoboRSI runtime is already
installed, run `bash install_simate.sh` to add Simate only. See
[INSTALL.md](INSTALL.md#simate-backend-and-cached-assets) for existing-environment and cached-asset options.


`install.sh` automatically builds Python 3.10.12 from fixed official source when
needed, obtains fixed uv, then installs the environments. A C compiler, `make` and Python build
libraries are required. See [INSTALL.md](INSTALL.md) for prerequisites.
The independent model-tool environment is built from a hash-locked package list
and the included OpenPI source. Its 82 package versions match the pinned dependency
baseline, with the unused historical `heart-runtime` installation omitted.
No prebuilt environment archive is required. Install at the final physical path
and generate fresh installation receipts there. This source bundle supplies no
installed environments, weights, caches or evaluation state. The simulator, its
activation hooks, complete Assets and matching Kit extensions are prerequisites.
The optional `environment/run.py` namespace launcher requires an explicitly
selected and verified current plan via `--plan`; this PR includes no deployment
plan. The direct `install.sh` / `eval.sh` workflow uses the official workspace
layout described above.

| Runtime | Fixed identity |
| --- | --- |
| Policy host | Python 3.10.12; 72 pinned controller versions, plus separately locked XPolicyLab/build dependencies |
| Model tools | Python 3.11.13; 82 pinned distribution versions, independent sandbox |
| VLA | Python 3.11.17; 157 distributions and original 112 source overrides |
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

This is an evaluation adapter; no additional policy training is provided.

## Evaluation

Set the adaptive minimum on the policy-server process before evaluation:

```bash
export ROBORSI_ADAPTIVE_MIN_STEPS=5  # optional; unset also selects 5–50
# export ROBORSI_ADAPTIVE_MIN_STEPS=2  # reproduce the earlier 2–50 range
```

Empty, noninteger or out-of-range values fail before model startup. Omitting
`approved_steps` uses `max(5, minimum)`; an explicit value below the minimum
is rejected without movement. VLA prediction horizon remains 50. The resolved
limits are fixed per Model instance and recorded in controller `config.json`,
the evaluation prompt and each proposal. Resume requires the same limits as
the interrupted episode; use `ROBORSI_ADAPTIVE_MIN_STEPS=2` for a 2–50 source.
The standalone launcher uses the same environment setting. Parallel orchestration
is maintained separately and is not included in this submission.
After updating installed source, rerun `bash install_simate.sh` to regenerate
the verified installation receipt before evaluation; do not rewrite
historical experiment receipts.

Set the service URL and model explicitly, and enter the runtime key privately.
Replace the placeholder URL and model below with your intended Responses-compatible
service and model before using the original ten-argument entry:

```bash
export ROBORSI_CODEX_BASE_URL='https://example.invalid/v1'
export ROBORSI_CODEX_MODEL='your-model'
read -rsp 'CODEX_API_KEY: ' CODEX_API_KEY; export CODEX_API_KEY
EVAL_ENV_TYPE=sim EVAL_NUM=1 bash eval.sh \
  RoboDojo align_blocks roborsi-codex-v1 arx_x5 joint 0 0 0 uv RoboDojo
```

`ROBORSI_CODEX_BASE_URL` and `ROBORSI_CODEX_MODEL` are required environment
variables with no defaults; evaluation fails early when either is missing or
empty. Installation does not need these values. Choosing a different model
changes experimental conditions.

The final arguments select the policy `.venv` (`uv`) and official simulator Conda
environment (`RoboDojo`). Supported here: `arx_x5`, `joint`, one environment.
The benchmark owns physics, layouts and scoring. RGB decoding and action
sizes use XPolicyLab's shared helpers. Native task performance is not asserted
by installation or synthetic tests.

## Trajectories

Every episode writes its controller records and all three lossless RGB camera
streams directly to `outputs/trajectories/<task>/seed-<seed>/<episode-id>/` under
this policy. `close`, `reset` and server shutdown retain these files. Each episode
gets a new ID, so parallel evaluations and repeated tasks do not overwrite data.

Set `trajectory_dir` in `deploy.yml` to choose another directory. Relative paths
are resolved from this policy directory. The standard evaluation scripts also
accept `ROBORSI_TRAJECTORY_DIR`, which overrides that model configuration value:

```bash
ROBORSI_TRAJECTORY_DIR=/shared/evaluation/trajectories \
EVAL_ENV_TYPE=sim EVAL_NUM=1 bash eval.sh \
  RoboDojo align_blocks roborsi-codex-v1 arx_x5 joint 0 0 0 uv RoboDojo
```

The episode's `controller/` contains actions, executed feedback, VLA proposals
and reviews, requests, responses, episode results, camera PNGs and provenance.
The complete private runtime, workspace, CLI sessions and model logs are retained
under the episode's `runtime-*/` directory; they are not deleted on close. Keep
it together with `controller/` so retained links remain valid. Optional trace
archives exclude credential filenames and preserve symlink text. Native scores and videos remain in the
official benchmark's evaluation output.

## Layout selection and recovery

`ROBORSI_NATIVE_LAYOUT_ID` selects an actual pending native layout through the
official reset, with task/eval-seed/run/scene-SHA verification. The evaluator owns
scores and existing progress. `hybrid_resume.py` prepares an authenticated stopped
own-episode request; `resume_eval.py` validates it and launches standard `eval.sh`.
Original joint replay and checkpoint validation precede the original recovery
prompt and fresh session; cumulative time and native step limits are retained.
See [RUN.md](RUN.md) for runnable commands,
trajectory layout and limits. Legacy receipt-only manifests are rejected.

Native observation RPC already transports RGB arrays losslessly; see
[OBSERVATION_TRANSPORT.md](OBSERVATION_TRANSPORT.md). `ROBORSI_SHARED_VLA_PORT`
selects an existing local VLA service without taking its ownership. Runtime pins,
ICL, original adaptive proposal/review/FK/DLS and fresh prompt remain unchanged.
Real simulator evaluation is still required to establish physical reproducibility.

## Parallel evaluation

By default, each evaluation owns its policy server and VLA process. The VLA binds port `0`
and returns the actual bound port to its controller. Give each case a private
state directory, temporary directory and benchmark run ID. `eval.sh` allocates a
private state directory when `ROBORSI_RUNTIME_STATE_DIR` is unset. Share the installed
code and weights; finish installation before launching parallel evaluations.

For shared VLA evaluation, start one matching local VLA service per GPU and give
its actual port to the corresponding cases with `ROBORSI_SHARED_VLA_PORT`. Each
case still owns a separate policy server and isolated state and trajectory paths.
One shared VLA with ten concurrent cases per GPU remains pending real simulator
acceptance. This PR includes no ten-case orchestration harness or concurrency
validation results. Validate the selected service, simulator launch and GPU
assignment in the target workspace before claiming concurrent evaluation support
or task success rates.

The following independent-VLA example requires two available GPUs. Each VLA reserves 60% of its
selected GPU memory, so separate ports alone do not make two VLA processes fit
on one GPU.

```bash
case_root="$(mktemp -d)"
run_case() {
  local case_task=$1 case_gpu=$2
  local case_state="${case_root}/${case_task}-${case_gpu}"
  mkdir -p "${case_state}/tmp"
  ROBORSI_RUNTIME_STATE_DIR="${case_state}" TMPDIR="${case_state}/tmp" \
  ROBODOJO_RUN_ID="$(basename "${case_root}")-${case_task}-${case_gpu}" \
  EVAL_ENV_TYPE=sim EVAL_NUM=1 bash eval.sh \
    RoboDojo "${case_task}" roborsi-codex-v1 arx_x5 joint 0 \
    "${case_gpu}" "${case_gpu}" uv RoboDojo
}
run_case align_blocks 0 & case_pid_a=$!
run_case stack_bowls 1 & case_pid_b=$!
case_exit=0
wait "${case_pid_a}" || case_exit=$?
wait "${case_pid_b}" || case_exit=$?
test "${case_exit}" -eq 0
```

## Native diagnostics

Periodic native-client thread dumps are disabled by default, including when a
launcher sets `ROBORSI_NATIVE_DIAGNOSTICS_DIR`. Setting an output directory alone
does not start the 120-second watchdog. The official client's fatal-signal
handler and stderr logging remain in effect.

For an isolated diagnostic run, explicitly set
`ROBORSI_NATIVE_PERIODIC_TRACEBACK=1` together with
`ROBORSI_RUNTIME_STATE_DIR=RUN_ROOT/state` and
`ROBORSI_NATIVE_DIAGNOSTICS_DIR=RUN_ROOT/state/native-infrastructure`.
Use an absolute, owned run directory and create both directories before launch.
Unset the switch or set it to `0` for ordinary evaluation; other switch values
are rejected. The watchdog traverses live thread frames and has been implicated
in native-client SIGSEGVs, so enabling it is reserved for diagnosis.

## VLA diagnostics

Each policy-owned VLA launch keeps its logs under this policy's
`outputs/vla/run-<random>/server.stdout.log` and `server.stderr.log`. The run
directory has mode `0700`. Failed startup and normal shutdown both retain these
files; startup exceptions report the absolute `server.stderr.log` path. Review
the logs before retrying and remove old run directories when no longer needed.
For a service selected through `ROBORSI_SHARED_VLA_PORT`, its external owner
manages the service logs and lifetime.

## Model Assets

The Agent reads its service URL from `ROBORSI_CODEX_BASE_URL` and its model
from `ROBORSI_CODEX_MODEL`, with reasoning `max` unchanged. Both values must
be supplied at runtime; no service address or model is bundled as a default.
The official π0.5 checkpoint is revision
`35efbc7dedfdbeeb6e95fb749bd885d73d483e41` of `RoboDojo-Benchmark/RoboDojo`,
step 59999, configuration `pi05_base_aloha_full_sim_arx-x5_seed_0`.
Its 18 inference files total 12,440,992,402 bytes; `pi05/checkpoint.json`
and `pi05/tokenizer.json` pin the exact inputs. `icl/artifacts.json` pins ICL.
Do not add these downloaded resources or `runtime/` to a source release.
