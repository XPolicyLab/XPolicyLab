# WorldScape_RSI (R²SI)

**Contributor:** WorldScape RSI team | **Paper:** Pending | **arXiv:** Pending | **Original code:** team-provided WorldScape RSI Harness; provenance in [SOURCE_MAP.json](harness/SOURCE_MAP.json).

This release runs one GPT-6 Astra High inner Agent for every task. Its episode
conversation memory, task/generic Skills and reviewed tools consume official
RGB, robot state, the public instruction and execution feedback. The Agent can
choose native motion tools or either of two checkpoint-backed WAM Skills:
`worldscape-wam-memory` and `worldscape-wam-nomemory`. Both WSP versions operate
inside the inner Agent. Every completed or interrupted action chunk returns
control to that Agent. No task-to-executor routing table or JEV model is used. Task IDs scope text Skill loading and episode metadata;
they do not select a model.

WAM is part of the Skill layer. Both Skills call the adjacent
`XPolicyLab.policy.WorldScape_Policy_2.model.Model` in-process.
Checkpoint transforms and model implementations remain owned by that adapter.
WAM predictions are validated before execution, committed only after all native
actions have observed execution evidence, and discarded after partial or
uncertain execution. Consecutive calls to the memory WAM retain context; the no-memory WAM uses
only the current observation and resets model context before each prediction; native
motion or switching variants rebases WAM state at the current observation.
Episode end releases both WAM states. There is no automatic inference replay.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

The private Agent policies and compatible planner sources are retained under `harness/vendor/`; their original notices and provenance are documented in [THIRD_PARTY.md](harness/THIRD_PARTY.md).

## Installation

Install this update in XPolicyLab at the documented baseline, retaining the
adjacent RoboDojo-capable `WorldScape_Policy_2` adapter. The outer `env_cfg/`
directory belongs to the parent workspace. RGB decoding uses XPolicyLab's
shared server helpers; this Harness does not decode or swap image channels.

```bash
cd XPolicyLab/policy/WorldScape_RSI
bash install.sh
python harness/release.py --verify
```

The installer invokes the adjacent WSP dependency installer. Use a dedicated Linux
CUDA policy environment; the standard launcher takes a conda environment name.
The compatible native planner and CuRobo Python sources are bundled under
`harness/vendor/runtime_support/` and selected by the Harness. No delivery-folder
`PYTHONPATH` is needed. Prepare the matching RoboDojo robot assets and the
non-Python CuRobo/CUDA dependencies separately; see [INSTALLATION.md](INSTALLATION.md).

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
```

## Data Processing

Not implemented in this eval-only adapter. `process_data.sh` is intentionally
absent; it does not convert trajectories, create ICL demonstrations or train on
LeRobot data. Supply the checkpoint assets described below.

## Training

Not implemented. `train.sh` is intentionally absent; WAM training remains owned
by the upstream model, and this adapter does not train Astra or a router.
Eval-only maintainer agreement and a training-release timeline are pending.
Skill revisions are imported offline into a new frozen release.

## Evaluation

```bash
bash eval.sh RoboDojo stack_bowls inner-agent-wam-skills arx_x5 joint 0 0 0 policy-env sim-env
```

`ckpt_name` labels the run; the two actual checkpoint paths come from the
explicit bindings below. `release_path` defaults to `harness/release.json`;
`release_sha256` may pin its digest for a campaign. Neither changes by task.
Unlike the demo adapter, the Harness keeps the action queue on the server and
returns one native action per request, so each action receives execution
feedback. The standard client sends `TRIAL_END` for cleanup.

For the required real-resource debug checks (requires the same model/runtime
assets as simulation):

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls inner-agent-wam-skills arx_x5 joint 0 0 0 policy-env sim-env
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls inner-agent-wam-skills arx_x5 joint 0 0 0 policy-env sim-env
```

These commands were not run in the CPU migration environment. The transport
fixture below is a separate check and does not substitute for them.

Supported: `RoboDojo`, `arx_x5`, `joint`, `eval_batch=false`. One server process
owns one active trial. Unknown public task IDs can use generic Skills; the
same Agent and two WAM Skills remain available without a task route lookup.
The official evaluator retains counting, scoring, step budgets and termination.
Only public RGB and robot state enter the Harness. No depth, calibration,
hidden object state or reward/score features reach the Agent or WAM Skills.

Code, Skills, Agent settings and both checkpoint manifests are frozen in
`harness/release.json`; verification rejects modified release assets and
per-task deployment overrides. Changing the installed release requires a new
candidate and restart. Removing task routing does not itself establish
leaderboard eligibility or task success; protocol audit and real-checkpoint
GPU/simulator validation remain **Pending**.

## Model Assets and Configuration

Both checkpoint identities are included in `harness/configs/wam_skills.json`.
Each has a 16-file SHA-256 manifest, including six safetensors shards (about
25.27 GB per checkpoint). **The weight bytes are not bundled in this delivery.**
Use the matching existing checkpoints; public download URLs remain **Pending**.
The checkpoints are `mem20000` and `checkpoint30000`. The memory Skill enables
rolling observation history and causal memory. The no-memory Skill disables
both, uses only the current observation, and resets model context before each
prediction through the adapter's independent-block mode. These are inference
settings; checkpoint weights and transforms are unchanged. Task-conditioned
history-window selection is disabled for both variants.

```bash
export WRSI_WAM_MEMORY_CHECKPOINT=/models/mem20000
export WRSI_WAM_NOMEM_CHECKPOINT=/models/checkpoint30000
export WRSI_WORLDSCAPE_ROOT=/path/to/worldscape-policy
# Device names refer to the GPU visibility of the policy-server process.
export WRSI_WAM_MEMORY_DEVICE=cuda:0
export WRSI_WAM_NOMEM_DEVICE=cuda:0
python harness/verify_checkpoints.py --output /path/to/checkpoint-verification.json
```

The legacy `WRSI_WAM_CHECKPOINT` variable is accepted for the memory variant
only, still subject to its exact checkpoint hashes. The two variants have
separate bindings and are exposed to the Agent on every task. Models load
lazily when their Skill is invoked. Both instances can remain loaded within an
episode; switching resets context, not weights. Allow sufficient GPU memory,
or bind the variants to separate visible GPUs. Missing/mismatched resources fail explicitly.
Only the pinned adjacent WSP adapter can load these checkpoints.

Agent provider configuration uses either:

```bash
export R2SI_API_BASE=https://your-responses-endpoint/v1
export R2SI_API_KEY=your-private-key
```

or the original PR's external profile conventions:

```bash
export DEXBOTIC_SECRETS=/secure/path/secrets.json
export VLM_AGENT_PROFILE=astra
```

The legacy JSON profile supports `vlm` objects, named mappings or lists with
`name`, `model`, `base_url` and `api_key`. The model must equal the frozen
`gpt-6-astra` identity. The endpoint must support Responses requests.
Credentials stay outside the repository. The profile supplies transport
credentials; it cannot override the frozen High effort, tools or model.

## Development and validation

The outer recursion workflow is the team's overall development design.
`harness/development.py` only imports evidenced offline revisions of existing
Skill assets into a new frozen candidate. It does not implement automatic
failure diagnosis, experiments, model training or the complete outer Agent.
Historical task results are not validation of this updated implementation.

Run CPU fixtures from the XPolicyLab root using a separate test environment
with `requirements-test.txt` installed:

```bash
python -m pip install -r policy/WorldScape_RSI/requirements-test.txt
R2SI_TEST_XPL_ROOT=/path/to/XPolicyLab python -m unittest discover -s policy/WorldScape_RSI/harness/tests -v
R2SI_TEST_XPL_ROOT=/path/to/XPolicyLab python policy/WorldScape_RSI/harness/tests/protocol_smoke.py
```

Fixtures test Skill dispatch, WAM lifecycle, episode isolation, frozen assets
and RGB transport. Their exact results are recorded in `VALIDATION.json`;
they do not measure real WSP inference, native success or leaderboard scores.
The library contains 81 task Skill documents, 10 existing generic documents
and two new checkpoint-backed WAM Tool/Skill documents.

## Submission status

This is a draft integration, not a verified leaderboard release. Real-checkpoint
debug evaluation, CUDA motion planning, simulator success, public checkpoint
access, eval-only agreement and the technical-report URL remain pending.
The supplied Skill text is task-specific; the executor implementation and WAM
tool set are common. Two independent WAM checkpoints are selected online by the
Agent. This is not a claim of a single checkpoint, a pure memory ablation, or
verified cross-task routing generalization.
