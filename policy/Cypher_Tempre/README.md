# Cypher_Tempre

**Contributor:** CPHY / Cyberphysics SI | **Paper:** Technical report in preparation | **arXiv:** not assigned | **Original code:** https://github.com/cyberphysicsai/cypher-tempre-genesis

A single continuing model uses the existing Cypher Tempre Timechain, registries and generated code to control an ARX X5 robot from RGB and proprioception. The agent retains experience and grows faculties across episodes. The initial state receipt documents provenance; it does not reinitialize, freeze or replace the live skill. This directory reuses XPolicyLab's existing GPT_6_Astra_Direct_EEF Cartesian executor and contributes observation transport and a persistent controller. It contains no VLA or task-specific policy router.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use an existing XPolicyLab policy environment and RoboDojo simulator environment. Install the upstream EEF client's pinned provider/image dependencies:

```bash
bash policy/Cypher_Tempre/install.sh /path/to/robodojo/.venv/bin/python
```

The continuing-agent host needs an authenticated Codex executable with the selected model available, and the existing skill tree, including `chain/`, `registry/`, `active_ops.py` and its supporting modules. Authentication stays on that host. The integration does not create, download or substitute an empty memory. Expected model for this submission is `gpt-6.1-sol`, reasoning effort `xhigh`; the provider's immutable model snapshot is not exposed. Record the actual CLI executable hash/version with each run.

## Data Processing

No trajectory conversion is performed. Existing experience is consumed in its original permitted observational form. This is evaluation-only; `process_data.sh` is omitted. RGB observations are encoded/decoded by the upstream XPolicyLab interface. This adapter never queries live hidden object poses, depth, segmentation, contacts or planner paths.

## Training

No foundation-model weights are trained or redistributed. Learning consists of persistent experience, explicit model-scored comparisons and generated executable faculties in the continuing skill. `train.sh` is omitted for this eval-only agent. The exact initial artifact, growth procedure and subsequent chain/code changes must be available to evaluators; publication review of the accumulated private artifact is still pending.

## Evaluation

Start exactly one controller on the agent host. Use the existing populated skill and experience paths:

```bash
python policy/Cypher_Tempre/run_agent.py \
  --codex /path/to/codex \
  --skill-root /path/to/existing/cypher-tempre-self-model \
  --experience-root /path/to/existing/benchmark-experience \
  --stream-root /path/to/continuing-stream
```

For a colocated evaluation, set `CYPHER_TEMPRE_SKILL_ROOT` and `CYPHER_TEMPRE_STREAM_ROOT` to those same paths in the environment client. Reuse both paths for every episode and seed. Robot-scene resets create observation sessions while the same controller, Timechain and generated faculties continue.

For an official remote client, run the broker on the agent host and put a trusted HTTPS reverse proxy in front of its loopback port:

```bash
export CYPHER_TEMPRE_SKILL_ROOT=/path/to/existing/cypher-tempre-self-model
export CYPHER_TEMPRE_STREAM_ROOT=/path/to/continuing-stream
# Provision a random token privately on the host and evaluator; never put it in Git.
export CYPHER_TEMPRE_BROKER_TOKEN=YOUR_PRIVATE_TOKEN
python -m XPolicyLab.policy.Cypher_Tempre.broker --host 127.0.0.1 --port 8767
```

The environment client uses `CYPHER_TEMPRE_BROKER_URL=https://YOUR_APPROVED_AGENT_HOST` and the same token. The broker accepts only observation/action sessions; it exposes no filesystem or shell RPC. Only one session and inference operation can be active at a time.

Standard entry, with the usual XPolicyLab environments:

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
bash policy/Cypher_Tempre/eval.sh \
  RoboDojo stack_bowls continual-state arx_x5 joint 0 0 0 uv /path/to/eval/python
```

`action_type=joint` is the benchmark protocol output; the model selects EEF targets. `ckpt_name=continual-state` is a run label, not an alternative model checkpoint. The upstream EEF defaults and native task limits remain in force unless an evaluator-approved manifest explicitly declares a different budget. Batch/multiple-controller evaluation is unsupported.

The generic `EVAL_ENV_TYPE=debug` client currently lacks the RoboDojo `robot_manager` and Cartesian planner required by the inherited EEF executor, so its plain/encoded closed loops are not claimed as passed. Use a native RoboDojo environment for that executor. The included transport tests separately exercise observation/reply transport and memory retention across sessions; they do not measure robot success.

## Model Assets

Skill source revision: `ad1afd50c89be5088df6b2ad73016d9950c54920` (Genesis3.30.08). Existing learned state remains separate from the source revision. `state_manifest.py` records its starting chain tip, registry and source hashes, and can make a private archival copy without changing the live skill. Every episode boundary records before/after tips. The checkpoint-download/publication path is not yet available: the accumulated artifact requires provenance/privacy review, and the controller host requires privately provisioned model access. Do not describe this integration as submission-ready until that is resolved.

## Notes

- Prior exploratory best-per-task results and the incomplete cold-start comparison are separate experiments, not official results for this adapter.
- No new official result or leaderboard acceptance is claimed. RoboDojo's online evaluation, three seeds, hidden-layout verification and code review remain required.
- The proposed submission is a continuously learning agent with one model and one inference procedure across tasks. Cross-episode and cross-seed persistence must be explicitly accepted and ordered by the evaluators; it is not silently reported as independent zero-shot trials.
- Contact: CPHY, Cyberphysics SI, cyphertempre@cyberphysics.ai. Phone and WeChat unavailable.
