# Riemann-RSI-Preview

**Contributor:** [bolwang](https://huggingface.co/bolwang) | **Paper:** None | **arXiv:** None | **Original code:** This directory.

Riemann-RSI-Preview runs RoboDojo tasks through the standard XPolicyLab interface.
It includes task skills, workflows and
[Riemann1.0](https://huggingface.co/bolwang/Riemann-RSI-Preview_robodojo) inference.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Place this directory at `XPolicyLab/policy/Riemann-RSI-Preview/`.

Requirements:

- Linux and Python **3.11.15**.
- RoboDojo `f57e1633ffed59761bc8e568cfe7530c4b33381c`, with simulator and assets installed using the [official guide](https://robodojo-benchmark.com/doc/usage/install-and-download/).
- XPolicyLab protocol baseline `aa46382aa12dc46b360e2fdc63d0d3d39215b3e9`, or a descendant with the same protocol implementation. Keep this baseline in the local Git history.
- One physical GPU for task workflows, or two distinct physical GPUs for Riemann1.0 inference. Tested on RTX 5090; minimum VRAM has not been established.

Set your paths, download the models and resources, then create the policy environment:

```bash
cd XPolicyLab/policy/Riemann-RSI-Preview
export RIEMANN_OFFICIAL_ROOT=/absolute/path/to/RoboDojo-official
export RIEMANN_EVAL_PYTHON=/absolute/path/to/robodojo-official/bin/python
export RIEMANN_RESOURCES_BASE="$HOME/.cache/riemann-rsi/resources"
export RIEMANN_POLICY_ENV="$HOME/.venvs/riemann-rsi"

bash download_checkpoints.sh --destination "$RIEMANN_RESOURCES_BASE" --prepare \
  --official-assets "$RIEMANN_OFFICIAL_ROOT/Assets"

RIEMANN_INSTALL_PYTHON="$RIEMANN_EVAL_PYTHON" bash install.sh \
  --robodojo "$RIEMANN_OFFICIAL_ROOT" --eval-python "$RIEMANN_EVAL_PYTHON" \
  --create-policy-env "$RIEMANN_POLICY_ENV" \
  --resources-base "$RIEMANN_RESOURCES_BASE" \
  --wam-checkpoint "$RIEMANN_RESOURCES_BASE/Riemann1.0" \
  --wam-encoder "$RIEMANN_RESOURCES_BASE/encoder"
```

Use `--policy-python /existing/env/bin/python` instead of `--create-policy-env` to
use an existing compatible environment. Add `--plan` to preview the installation.
Resource directories can be set individually with `--resource-root NAME=/absolute/path`;
available names are in `configs/resource-roots.json`.

After installation, check the configuration with:

```bash
RIEMANN_INSTALL_PYTHON="$RIEMANN_EVAL_PYTHON" bash install.sh --check-only
```

For runtimes with Vulkan/X11 libraries outside the system loader path, pass
`--native-library-dir /absolute/path/to/native/lib` during installation or set
`RIEMANN_NATIVE_LIBRARY_DIRS` to a path-separated list of directories.

## Data Processing

This evaluation-only release does not include a data-processing pipeline or
training dataset.

```bash
bash process_data.sh
```

The script reports that data processing is unsupported and exits with status 2.

## Training

This release provides inference and evaluation only. Training code is not
included, and no training-code release date has been announced.

```bash
bash train.sh
```

The script reports that training is unsupported and exits with status 2.

## Evaluation

After installation, use the standard ten-argument interface:

```bash
bash eval.sh <bench> <task> <checkpoint> <robot> <action> <seed> \
  <policy_gpu> <env_gpu> <policy_python_or_conda_env> <official_eval_python>

# Rule example: one physical GPU, joint actions.
bash eval.sh RoboDojo insert_key rule arx_x5 joint 0 \
  0 0 "$RIEMANN_EVAL_PYTHON" "$RIEMANN_EVAL_PYTHON"

# Riemann1.0 example: two physical GPUs, EE actions.
bash eval.sh RoboDojo stack_bowls "$RIEMANN_RESOURCES_BASE/Riemann1.0" arx_x5 ee 0 \
  0 1 "$RIEMANN_POLICY_ENV/bin/python" "$RIEMANN_EVAL_PYTHON"
```

The environment arguments accept an interpreter path, environment directory or
Conda name. Evaluation uses the SDK's default episode counts: 25 per standard or
random mode in a generalization pair, and 50 for other modes.

Use one active environment per policy process; `eval_batch` defaults to `false`.
For split-machine deployment, follow the shared XPolicyLab guide. The Riemann1.0
policy server's GPU argument takes two physical GPU indices, such as `5,6`;
the environment client takes the single simulator GPU index.

### Results

Local RoboDojo evaluation, summarized with `robodojo.sh summarize`. Avg SR and
Avg Score use equal weight for the five capability dimensions; score is shown
on a 0–100 scale.

| Success rate | Score / 100 |
|---:|---:|
| 65.86% | 71.57 |

Full official per-task summary: [summary.md](summary.md).

## Model Assets

| Component | Contents |
|---|---|
| Riemann1.0 | Policy checkpoint and normalization files under `Riemann1.0/`. |
| Common visual models | SAM3, CLIP ViT-B/32, GroundingDINO Tiny, and Wan2.2 VAE/T5/tokenizer. |
| Task detectors | Classifiers used by conveyor and imitation tasks. |
| Templates | Visual and geometric templates used by task skills. |
| Robot geometry | Prepared from the official SDK's `Assets/` directory. |
| Runtime dependencies | SAM3 runtime components and WebSocket protocol dependencies. |

Model and template downloads use the
[Hugging Face repository](https://huggingface.co/bolwang/Riemann-RSI-Preview_robodojo/tree/e0fa3f2a920eb74ae105194829e13cb07b861c08)
at revision `e0fa3f2a920eb74ae105194829e13cb07b861c08`.
The downloader checks file sizes and SHA256 values and reuses matching files
when run again. Downloads are stored under `RIEMANN_RESOURCES_BASE`.

## Configuration

`deploy.yml` contains the policy name, WebSocket host/port and evaluation defaults.
Optional Riemann1.0 overrides are `wam_checkpoint`, `encoder_path`, `device` and
`t5_device`.

Use the installation arguments above to configure SDK, Python and resource paths.
The evaluation command selects the task, checkpoint, action type, seed and GPUs
for each run.
