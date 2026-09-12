# WorldScape Policy 2.0 model card

## Model summary

WorldScape Policy is an alpha-stage world-action policy for robotics research.
It combines visual context, language or planning features, bounded event
memory, and a video/action world model. Auto mode uses shared VLM
perception/planning features and event memory; Interactive mode uses direct T5
instruction conditioning without event memory.

The software can load native WorldScape checkpoints and explicitly supported
DreamZero-compatible checkpoints. This repository does not bundle weights.

## Intended use

- robotics and embodied-AI research;
- offline evaluation and controlled simulator experiments;
- supervised operation on research robots by qualified teams;
- checkpoint migration and numerical compatibility studies.

It is not intended for safety-critical, medical, public-facing autonomous, or
unsupervised real-world deployment.

## Inputs and outputs

Inputs may include multi-camera observations, robot state, language
instructions, goal images, and video demonstrations. Outputs include predicted
action chunks and optional generated video/world-model state. Exact modalities,
normalization, action dimensions, and camera ordering are embodiment- and
checkpoint-specific.

## Training data and evaluation

The upstream DreamZero-DROID release was trained from DROID-derived data; other
checkpoints may use different datasets or post-training data. Consult the
checkpoint publisher and its manifest rather than assuming a checkpoint's
lineage from this software package.

Reported upstream results do not establish performance for every WorldScape
configuration, embodiment, simulator, or converted checkpoint. Validate each
artifact with task-specific metrics and safety constraints before use.

## Limitations and risks

- Predictions can be incorrect, unstable, delayed, or physically unsafe.
- Distribution shift in scenes, cameras, language, dynamics, or embodiments
  can cause silent performance degradation.
- Generated video is not a guarantee that an action is feasible or safe.
- Dataset and pretrained-model biases may propagate into behavior.
- Legacy compatibility paths preserve historical behavior, including its
  limitations; conversion does not improve a model.
- Real-robot execution requires independent collision avoidance, workspace
  limits, emergency stops, human supervision, and command validation.

## Responsible use

Use least-privilege robot access, test first in replay and simulation, retain
human override, log inputs and commands, and stop on anomalous behavior.
Respect privacy and consent when collecting camera or operator data.

See [provenance](docs/provenance.md), the
[P2 migration guide](docs/P2_MIGRATION.md), and `LICENSE`.
