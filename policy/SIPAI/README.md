# SIPAI

**Contributor:** SIPAILab | **Paper:** Not provided | **arXiv:** Not provided | **Original code:** Inference implementation in this directory

Self-contained, **eval-only** SIPAI AE memory policy for XPolicyLab. This directory
contains the inference implementation; no separate SIPAI repository or runtime
is required. Dependencies are PyTorch, standard public Python packages and the
shared XPolicyLab server/client utilities.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

On Linux with Conda (Miniconda/Miniforge) and a compatible NVIDIA driver,
run from this directory:

```bash
bash install.sh
conda activate sipai-eval
```

The script creates a Python 3.11 environment when needed, reuses installed CUDA
PyTorch 2.6/2.7/2.8, installs the inference packages and XPolicyLab, and checks
dependencies and model/server imports. When PyTorch is absent, it defaults to
PyTorch 2.7.0 with CUDA 12.8 wheels. An existing unsupported or broken PyTorch
installation reports an error instead of being automatically replaced. No preinstalled PyTorch or XPolicyLab is
required. An existing environment must use Python 3.11. Matching installed
packages are reused without forced upgrades. Pip uses its configured cache;
set `PIP_CACHE_DIR=/path/to/pip-cache` to reuse a shared cache.

For a fully offline installation, provide a directory of compatible wheels with
`PIP_NO_INDEX=1 PIP_FIND_LINKS=/path/to/wheels` and set `CONDA_OFFLINE=true`.
Conda must already have the Python environment or its packages cached. Missing
packages cause an error instead of a download; pip's HTTP cache alone is not an
offline wheel directory.

To choose an environment name or path:

```bash
bash install.sh my-sipai-env
# Or: bash install.sh /path/to/sipai-env
```

Use that same name/path as the policy environment argument to `eval.sh`.
If Conda is not on PATH, set `CONDA_EXE` to its executable. This installs the
policy environment only; install the RoboDojo simulator environment separately.
See the [XPolicyLab README](../../README.md) for shared deployment conventions.

## Data Processing

Not supported by this eval-only adapter; no training dataset is required.

## Training

Not included. This is an inference-only submission and needs no training data,
training configuration, optimizer state or access to a training repository.

## Evaluation

Supported target: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`,
`action_type=joint`. The checkpoint's two six-joint arms and scalar grippers are
validated against XPolicyLab's shared robot configuration.

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls sipai-robodojo-eval arx_x5 joint 0 0 1 sipai-eval RoboDojo
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls sipai-robodojo-eval arx_x5 joint 0 0 1 sipai-eval RoboDojo
```

The ten positional arguments follow the standard XPolicyLab convention. For
split-machine evaluation, use the shared policy server and simulator client.

`deploy.py` updates observations once per action step, including steps inside a
chunk. For N history frames at interval K, inference at step t uses exactly
`t-N*K, ..., t-K`, oldest first. Missing pre-episode frames are masked; three
current camera views are separate. History is isolated by evaluation and
environment ID and cleared between episodes, including when evaluation fails.
Each valid history image passes through SigLIP and the AE encoder, reducing its
16 by 16 token grid to 4 by 4. Current camera images retain their full token grids.

Batch calls execute inference sequentially and preserve requested environment
order, following other XPolicyLab adapters. Finished environments leave the active
batch. Actions are relative joint predictions decoded against the current joint
state; grippers predict absolute openness and receive no added compensation.

## Model Assets

The portable checkpoint directory requires these two files:

- `model.safetensors`: complete FP32 AE policy weights, including SigLIP, the
  history encoder and the action model. An encoder-only file is insufficient.
- `tokenizer.model`: SentencePiece tokenizer.

The download script reads the current weights from
[SIPAILab/sipai-robodojo-eval](https://huggingface.co/SIPAILab/sipai-robodojo-eval):

```bash
# Run in the activated policy environment.
bash download_checkpoint.sh
# Or choose a checkpoint directory:
bash download_checkpoint.sh /path/to/checkpoint
```

The default destination is `checkpoints/sipai-robodojo-eval` under this adapter,
independent of the working directory. Pass `sipai-robodojo-eval` as `ckpt_name`
to `eval.sh` for that default, or pass the absolute path of a custom destination.
Checkpoint lookup uses XPolicyLab's shared resolver.

This adapter requires an AE policy export. Pooling checkpoints are incompatible
and are rejected by both the loader and download script. Until compatible AE
weights are published, use a local AE checkpoint directory containing the two
files above and pass its absolute path as `ckpt_name`.

Each run checks the current `main` branch. Matching local weights are reused;
otherwise the script uses the Hugging Face cache or downloads the current weights.
It verifies the model's SHA-256 against the remote metadata before replacing
an existing file. Cache files are copied into the destination, so reserve space
for both copies when the cache is on the same disk. Checking for updates requires
network access.

The script fetches only `model.safetensors` from the SIPAI repository and
`tokenizer.model` from
[Google's official PaliGemma repository](https://huggingface.co/google/paligemma-3b-pt-224).
A compatible existing tokenizer is reused, with its checksum verified.
For an uncached tokenizer, first accept the Google repository's access terms
and authenticate with `hf auth login` or `HF_TOKEN`.

Once the SIPAI repository is public, its weights can be downloaded without
logging in. While it is private, authentication with read access is required.
Standard proxy and Hugging Face cache settings are honored; credentials are
not stored in this script. No PaliGemma weights or extra inference JSON files
are downloaded.

## Configuration

`deploy.yml` contains all inference settings in addition to the standard
XPolicyLab fields:

```yaml
device: cuda:0
exec_chunk_size: 10
architecture: sipai_memory_ae_joint_v1
dtype: float32
history_frames: 25
history_interval: 20
history_grid_size: 4
history_compressor_hidden_dim: 512
max_token_len: 200
num_inference_steps: 10
```

This selects 25 primary-camera history frames spaced 20 action steps apart, a
4 by 4 AE history token grid with a 512-channel encoder, FP32 compute, and
10 flow integration steps. Only the encoder is needed for inference; no decoder
or separate AE checkpoint is loaded.
`exec_chunk_size` must be between 1 and 50. Keep these settings aligned with the
checkpoint when changing weights. The release's fixed state/action q01 and q99
statistics are embedded in `processing.py` and used for state tokenization and
action decoding. Checkpoints used with this adapter must share those statistics.
Neither `inference.json` nor `normalization.json` is read or required.

RGB decoding belongs to the shared server. This adapter preserves RGB ordering
and applies the checkpoint's aspect-preserving image resize and padding.
