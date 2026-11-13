# Proposed continual-agent evaluation

This is a method specification and implementation report, not an official result or a measured claim of improvement.

## System

One model (`gpt-6.1-sol`, `xhigh`) controls every task through the same observation and action procedure. Cypher Tempre source revision is `ad1afd50c89be5088df6b2ad73016d9950c54920`. The model provider does not expose an immutable model snapshot; record the CLI version, executable hash and requested model identifier. No model weights are updated by this adapter.

The model reads the existing live Timechain and faculty registries, recalls relevant experience, examines its own RGB/proprioceptive observations, computes relevant faculty signals, compares candidate strategies and appends its evidence and decision. The controller may generate and execute new faculties under the operator's standing authorization. Those faculty changes and outcomes remain available for subsequent decisions. Model-supplied perspective/PoQ scores are judgments, not calibrated success probabilities or independent model votes.

The same populated state and controller continue across episodes. A robot-scene reset does not reset memory, registries, generated operations, or experience. A starting-state receipt documents provenance only. It never substitutes a fresh chain or rolls the live chain back between tasks. `run_agent.py` starts one sequential controller for the entire stream; alternatively, one existing interactive controller can use `agent_io.py` directly. No other model worker or task-specific checkpoint router is involved.

## Observation and action boundary

The existing `GPT_6_Astra_Direct_EEF` executor supplies RGB and named robot state, and converts model-selected Cartesian targets into joint motions using RoboDojo's existing planner. The adapter transports those observations and one model-authored reply. It does not inspect live object poses, depth, segmentation, contacts or hidden layouts. Default native action/call limits apply. Record any approved budget or rendering changes; the local Windows wiring diagnostic used a compatibility renderer and is not an official benchmark run.

## Proposed official protocol

Before a scored run, evaluators must accept cross-episode/cross-seed learning, specify the task/episode/seed order and decide how hidden verification interacts with persistent experience. Use one fixed model and learning procedure throughout that order. Publish all attempted episodes and protocol changes, not selected best retries. Report the benchmark's score and success metrics with its required seeds and aggregation, alongside wall time, inference usage when available, and memory/code growth. Disclose prior public-task exposure in the initial state. Do not imply independent zero-shot trials.

The evaluated initial artifact, permitted experience dependencies, code and run configuration must be reproducible. The existing accumulated state is currently private and has not been uploaded; a reviewed artifact or evaluator-approved remote-access arrangement is still needed. This draft does not promise a release date or imply that the benchmark has approved this method.

## Comparison question

An official skill-enabled score alone cannot establish the skill's causal benefit. A separate matched study would need equal model, budgets, task exposure and seed ordering while allowing each condition to retain its own permitted learning. A cold-start experiment that empties the skill state each episode evaluates within-episode skill use; it omits accumulated cross-episode learning. Preserve such negative results, label their scope, and measure the persistent condition rather than assuming it improves scores.

## Evidence available in this draft

See `VALIDATION.md`. Transport checks and a zero-call native wiring diagnostic passed. No official online evaluation, hidden-layout verification, skill-lift estimate or full task score is reported for this adapter.
