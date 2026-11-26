# MotuBrain2.0-Robodojo evaluation

Standalone inference source for the four-step MotuBrain2.0 RoboDojo student.
Evaluators download the checkpoint and frozen encoders, then run the policy on
their own GPU. There is no team-hosted inference endpoint or cluster dependency.
The included XPolicyLab adapter uses a localhost connection between the policy
and the simulator's separate Python environments.

## Supported policy

- Benchmark: `RoboDojo`; robot: `arx_x5`; actions: `joint`.
- Three RGB inputs: head, left wrist, right wrist. The adapter never swaps channels.
- Absolute joint14 output: left joints 6, left gripper 1, right joints 6, right gripper 1.
- Execute all **32 actions** before the next query; capture an observation at every
  native action, including actions between queries.
- History: episode-zero anchor plus eight past frames, spaced by **16 native
  control steps**, compressed as `causal_9_to_3`. Current RGB is encoded separately.
- Four CMSolver steps, BF16, query seeds `42 + query_index`, reset each episode.
- One environment and one GPU per policy process. No quantization, RTC or action smoothing.

## Installation

Use Linux and a CUDA GPU with sufficient memory for the policy and frozen
encoders. Keep the policy and RoboDojo simulator in separate Python environments.

```bash
cd /path/to/MotuBrain2.0-Robodojo-eval
conda env create -f environment-policy.yml
conda activate motubrain-robodojo
python -m pip install --no-deps -e .
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/Depth-Anything-3
git -C third_party/Depth-Anything-3 checkout --detach 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
export MOTUBRAIN_SOURCE_ROOT="$PWD"
export PYTHONPATH="$PWD/third_party/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
```

Use Python 3.11 with the pinned NumPy 1.26.4 and PyTorch 2.9.1.
Install the PyTorch CUDA wheel matching your driver if the default wheel is unsuitable.

DA3 source is pinned to `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
The commands use Git directly and do not install the DA3 web/demo stack.

## Download weights

The selected bundle contains the published MotuBrain2.0 checkpoint,
normal BF16 weights with four CM denoising steps, published at immutable HF
revision `26c53046384721b21ef2b6d39b47b339cbc4bba8`.

```text
checkpoints/motubrain/
  step-5000.safetensors
  config.yaml
  action_stats.json
```

Only these three inference assets are needed. From the XPolicyLab adapter directory,
run `bash download_checkpoint.sh /path/to/checkpoints/motubrain`.
It calls `hf download` at the pinned revision; all three files are at the bundle root.
The loader checks that local files exist; use matching configuration and statistics.

[Checkpoint](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo/resolve/26c53046384721b21ef2b6d39b47b339cbc4bba8/step-5000.safetensors), [configuration](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo/resolve/26c53046384721b21ef2b6d39b47b339cbc4bba8/config.yaml), [action statistics](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo/resolve/26c53046384721b21ef2b6d39b47b339cbc4bba8/action_stats.json).

Frozen encoders are downloaded separately under their upstream terms:

```bash
hf auth login  # needed after accepting the gated T5Gemma model's terms
hf download google/t5gemma-9b-9b-prefixlm-it --local-dir /path/to/t5gemma
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir /path/to/wan
hf download depth-anything/DA3-LARGE --local-dir /path/to/da3

export SOMA_TEXT_ENCODER_PATH=/path/to/t5gemma
export SOMA_VAE_PATH=/path/to/wan/Wan2.2_VAE.pth
export DA3_CONFIG_PATH=/path/to/da3/config.json
export DA3_CHECKPOINT_PATH=/path/to/da3/model.safetensors
export SOMA_ARTIFACT_CACHE=0
```

Use `prefixlm-it`, not `ul2`, and original `DA3-LARGE`, not `DA3-LARGE-1.1`.
Encoder downloads use upstream default revisions. Record the downloaded revision
IDs alongside your evaluation configuration.

## Evaluate with RoboDojo / XPolicyLab

Install RoboDojo and its assets using the benchmark's own instructions. Keep the
simulator in its own environment. The expected workspace layout is:

```text
benchmark-workspace/
  env_cfg/
  RoboDojo/
  XPolicyLab/
```

The XPolicyLab package contains the adapter; the standalone source ships the
`motubrain_robodojo` package and its dependencies, installed by `install.sh`.
Run the evaluation from the adapter inside the XPolicyLab checkout:

```bash
cd /path/to/benchmark-workspace/XPolicyLab/policy/MotuBrain2_0
bash eval.sh RoboDojo stack_bowls /path/to/checkpoints/motubrain \
  arx_x5 joint 0 0 1 motubrain-robodojo <robodojo-simulator-env>
```

The ten arguments are benchmark, task, checkpoint **bundle directory**, robot,
action type, simulator seed, policy GPU, simulator GPU, policy environment and
simulator environment. Replace the paths, GPU IDs and environment name for your
machine. The shell starts the model locally, then the benchmark client, and
stops its own model process on exit. No server URL or access to our cluster is needed.

`deploy.py` calls XPolicyLab's demo episode loop, which already reads an
observation between every native action. A small environment wrapper adds only
the native action count. `eval_batch=false` is required; start separate processes
for independent environments. Checkpoint paths, robot dimensions and state/action
packing use XPolicyLab's shared helpers.

With XPolicyLab importable, `motubrain_robodojo.load_policy(bundle)` returns
an object with `reset()`, `update_obs(observation)`, `get_action()` and `close()`.
Observations must carry already decoded RGB, joint states, `instruction` and a
contiguous integer `observation_step` starting at zero. Each returned list
contains the complete native action chunk (32 actions for the published checkpoint).
The direct API also uses XPolicyLab's state/action helpers; it does not load a
simulator. Install XPolicyLab following its instructions, or put the parent of
its checkout on `PYTHONPATH`.

## Switching compatible checkpoints

Within RoboDojo / `arx_x5` / absolute joint14 / `causal_9_to_3`, switching
checkpoints only requires downloading the assets and updating configuration.
Use `hf download` for the chosen repository, revision and files, then set the
paths in the adapter's `deploy.yml`:

```yaml
checkpoint_file: other.safetensors
model_config_file: config.yaml
action_stats_file: action_stats.json
policy_config: /path/to/other-policy.py
query_seed_base: 42
```

The three asset paths are relative to the checkpoint bundle selected by
`ckpt_name`; absolute paths also work. `policy_config: null` uses the bundled
policy config, and a relative policy-config path resolves from the adapter.
Copy `motubrain_robodojo/config.py` when changing inference steps, solver,
compile settings, preprocessing or prompts. The selected model YAML supplies
architecture, sampling window and history stride; the action horizon is derived
from its prediction frames and action frequency. The selected statistics replace
the profile's initial statistics. `query_seed_base: null` disables fixed inference
seeds; `42` preserves the default per-query sequence.

The standalone `load_policy(bundle, ...)` API accepts the same configuration
keywords. Its default asset paths match the flat published bundle above.
Different robot protocols, history encodings, or architectures requiring unavailable model
implementations remain outside this adapter's scope.

## Data processing and training

This is **evaluation-only**. It does not include dataset conversion, training,
optimizer state or a training-resume checkpoint. Training release date: not
announced. No `process_data.sh` or `train.sh` is required for this contribution.

## Third-party components

See [NOTICE](NOTICE) for copyright notices and third-party component terms.
