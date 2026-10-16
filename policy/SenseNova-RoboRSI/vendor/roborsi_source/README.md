<h1 align="center">SenseNova-RoboRSI: Recursive Self-Improvement for Embodied Agents</h1>

<p align="center"><strong>Build, evaluate, and improve embodied agents with a shared runtime.</strong></p>

<p align="center">
  <img src="assets/badges/python.svg" alt="Python 3.10–3.12" />
  <img src="assets/badges/license.svg" alt="License: Apache 2.0" />
  <img src="assets/badges/status.svg" alt="Status: Pre-Alpha" />
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a> ·
  <a href="#results">Results</a> · <a href="#release-plan--roadmap">Release plan</a>
</p>

## News

- **2026-10-09** — SenseNova-RoboRSI achieves an **average score of 56.83** and a **success rate of 50.83%** on RoboDojo, a gain of **+27.86 points** over the GPT-6 Astra baseline (**28.97**). See [Results](#results) for benchmark comparisons.

## Overview

Connect your model, skills, and robot tools through a shared runtime, customize
prompts and context, and evaluate policies in environments that retain their
native execution and scoring logic.

On top of this runtime, **recursive self-improvement (RSI)** uses execution
traces and task feedback to improve the Harness while keeping model parameters
fixed. Candidate changes are retested, and validated improvements are carried
into the next agent version, connecting task execution with continuous improvement.

<p align="center">
  <img src="assets/sensenova-roborsi-overview-en.webp" alt="SenseNova-RoboRSI agent interfaces, execution loop, benchmark integration, and RSI workflow" width="75%" />
</p>

### Core capabilities

#### Harness: A shared runtime

- **Flexible agent interfaces.** Connect models, prompts, skills, and tools/MCP;
  configure context and caching to suit each task.
- **Reusable VLA and ICL components.** Predict actions with
  [VLA](docs/source-en/rst_source/usage/vla.rst), and manage reference packages
  and writable copies with [ICL](docs/source-en/rst_source/usage/icl.rst).
  Use them independently or with an Agent. The
  [composition preset](configs/agents/components/icl_vla.yaml) combines
  a synthetic provider with a small reference package.
- **Unified execution and evaluation.** Run VLA skills, scripted actions, or
  end-effector tools across benchmarks with native feedback and scoring.
- **Experiment records and training data.** Record full traces, export training
  episodes, and run batch evaluations with concurrency and recovery.

#### RSI: Continuous Harness improvement

- **Learn from execution feedback.** Refine action prediction, planning,
  feedback, and memory organization while keeping model parameters fixed.
- **Retest and retain improvements.** Track agent versions, evaluate candidate
  changes under the same task conditions, and carry validated improvements forward.

## Results

Each table shows SenseNova-RoboRSI and four reference results. Higher is better.

### [RoboDojo](https://robodojo-benchmark.com/)

| Method | Average score ↑ | Success rate ↑ |
| --- | ---: | ---: |
| SenseNova-RoboRSI / GPT-6 Astra (ours) | **56.83** | **50.83%** |
| VPP2 | 39.26 | 32.26% |
| PhysicalRSI | 36.27 | 31.38% |
| Awomo-0.5 | 35.34 | 29.64% |
| Simate-beta | 33.95 | 27.96% |

Reference results: [RoboDojo simulation leaderboard](https://robodojo-benchmark.com/leaderboard).

### [LIBERO-PRO](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/)

| Method | Success rate ↑ |
| --- | ---: |
| SenseNova-RoboRSI / GPT-6 Astra (ours) | **94.50%** |
| RPent / GPT-6 Astra | 92.63% |
| RPent / Opus-4.7 | 82.40% |
| RPent / GPT-5.5 | 75.13% |
| RPent Flash Mode / Molmo2-8B | 72.63% |

The reference scores above, including GPT-6 Astra's 92.63%, are reported by [RPent](https://rpent.readthedocs.io/zh-cn/latest/rst_source/leaderboard/performance.html) on Task/Swap suites. The [official LIBERO-PRO leaderboard](https://github.com/Zxy-MLlab/LIBERO-PRO#-libero-pro-model-leaderboard) reports a different aggregate across five perturbation types.

### [RoboCasa365](https://robocasa.ai/)

| Method | Evaluation setting | Success rate ↑ |
| --- | --- | ---: |
| SenseNova-RoboRSI / GPT-6 Astra (ours) | Target kitchens + task memory | **64.60%** |
| Paimon-0 | Pretraining kitchens | 58.10% |
| Xiaomi-Robotics-1 | Pretraining kitchens | 57.40% |
| Phasor-m7 | Pretraining kitchens | 56.80% |
| ProWAM | Pretraining kitchens | 48.10% |

Reference results: [official RoboCasa leaderboard](https://robocasa.ai/leaderboard.html). SenseNova-RoboRSI uses the Target50 setting; the official entries use pretraining kitchens. These results provide context across different evaluation settings and do not establish a ranking under the same protocol.

## Release Plan & Roadmap

### Release plan

- [x] Project introduction
- [ ] Code release
- [ ] Usage documentation
- [ ] Demo

### Research roadmap

- **Experience retrieval.** We plan to organize successful strategies, failure
  cases, and execution traces into a searchable experience library.
- **Transfer across tasks and environments.** We will investigate whether tool
  combinations, planning strategies, and feedback mechanisms validated in one
  setting can provide useful starting points in another.
- **More robot platforms and real-world scenarios.** We aim to extend integration
  and evaluation to more robot embodiments and physical settings, and examine the
  reliability of feedback-driven improvement in those environments.

## Community & Acknowledgements

We welcome collaboration with teams working on foundation models,
VLA policies, simulation platforms, and real robots.

- **Contributions:** We welcome new integrations, evaluation tasks,
  reproducible failure cases, and suggestions for Harness design.
- **Questions & collaboration:** Open an
  [issue](https://github.com/OpenSenseNova/SenseNova-RoboRSI/issues)
  describing your platform, use case, and proposed contribution.
- **Release updates:** Watch the
  [SenseNova-RoboRSI repository](https://github.com/OpenSenseNova/SenseNova-RoboRSI).
  Contribution instructions will accompany the code release.
- **Acknowledgements:** We thank
  [RoboDojo](https://robodojo-benchmark.com/),
  [LIBERO-PRO](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/),
  [RoboCasa](https://robocasa.ai/), and the broader open-source
  community supporting embodied-agent research.
- **Licenses:** External models, simulators, and datasets retain
  their respective licenses.

**This achievement reflects the joint team's continued exploration and close collaboration in embodied AI and agent systems. We will continue to advance and validate these technologies, and release a full technical report soon covering the core methods, experimental design, and evaluation results. Stay tuned.**
