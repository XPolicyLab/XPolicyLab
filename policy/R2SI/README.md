# r²si (R2SI)

**Status:** Collaborative experiment infrastructure implemented; XPolicyLab integration and evaluation evidence Pending.

**Planned benchmark:** RoboDojo.

r²si connects embodied task execution, feedback, and reusable Skill accumulation. Its experiment infrastructure combines cloud management and sharing, participant-local agent clients, and isolated remote GPU execution, preserving Skill versions and execution evidence across collaborative experiments.

- **Inner execution agent:** Uses multi-view images, robot state, environment feedback, and general and task Skills to observe, decide, act, and reassess.
- **Outer iteration agent:** Reviews execution traces and accumulated experience, improves the inner agent's harness, and creates, refines, or replaces reusable Skills, including partial procedures and failure recovery strategies.

Human guidance supports initial bootstrapping. Subsequent development iterations can be driven autonomously by the outer agent, forming an execution–feedback–experience–execution loop. Skills retain their provenance and supporting execution evidence for reuse and review.

Harness and skill updates occur during development. Official evaluation will use one frozen release with fixed model versions, harness, prompts, skills, configuration, and inference procedure across tasks.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Pending.

## Data Processing

Pending.

## Training

Pending.

## Evaluation

Pending.
