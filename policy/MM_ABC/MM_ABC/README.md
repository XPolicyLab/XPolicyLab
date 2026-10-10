# MM-ABC source package

MM-ABC combines a Qwen3-VL backbone with a rectified-flow action expert for
mobile manipulation. The XPolicyLab adapter is in the parent directory; see
[its README](../README.md) for installation and closed-loop evaluation.

The default is **clean-action prediction (`pred_type: x`) with velocity-space
loss (`loss_type: v`)** in every shipped model configuration. An explicit
`pred_type: v` remains available for checkpoints trained with velocity prediction.
Use the prediction type saved with a checkpoint when loading existing weights.

## Installation and paths

Use the parent adapter's installation script in an existing CUDA environment.
The package can then be installed from this directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Python 3.10 or later is required. Training uses CUDA, PyTorch distributed/FSDP2,
and a compatible FlashAttention installation. `MMABC_PYTHON` selects the Python
interpreter used by launch scripts; it defaults to `python` on `PATH`.

Configuration paths resolve relative to this directory. Asset paths such as
`data/`, `pretrained_ckpt/`, and `checkpoints/` resolve relative to the parent
adapter directory, or `MMABC_HOME` when set. Absolute paths are accepted.
Model weights, datasets, videos, logs and checkpoints are external assets.

## Model configurations

| Setting | 75-d mobile layout | 80-d compatible layout |
| --- | --- | --- |
| Training config | `configs/train/mobile.yaml` | `configs/train/mobile_pretrain.yaml` |
| Model config | `configs/model/mmabc_4b_mobile.yaml` | `configs/model/mmabc_4b_mobile_pretrain.yaml` |
| Initialisation | Pretrained Qwen3-VL backbone, fresh expert | Compatible MM-ABC weights via `init_from` |
| Action targets | Absolute commands, min-max normalised | Layout-defined increments and absolute commands |
| Flow | x-pred, v-loss | x-pred, v-loss |
| Time weights | Unnormalised | Normalised to unit batch mean |
| Prompt | Task instruction | Embodiment/control header and task instruction |

Both mobile configurations use 32-step action chunks, execute 16 steps before
replanning, and use 10 Euler integration steps. The backbone provides four
context taps to the action expert. Optional frozen visual teachers supervise
future-frame representations; action tokens cannot attend to future tokens.

The flow path is `x_t = (1 - t) * noise + t * action`. For clean-action
prediction, the velocity-space squared error weights the clean-action squared
error by `1 / max(1 - t, sigma_min)^2`. `normalize_weight` optionally rescales
these weights to a unit batch mean.

The included embodiment is the `m92uw` mobile platform: two 7-DoF arms, two
12-DoF hands, waist, leg, head pose and base velocity. The mapping is in
`mmabc/embodiments/mobile.py`. It accepts singular runtime keys and plural
trajectory keys. The 75-d layout stores poses as position plus rotation-6D.
The 80-d layout preserves pretraining slots, stores planar base twist only,
and places head pose in the reserved block. Its inverse sets `vz`, `wx` and
`wy` to zero. Other robot layouts require matching configuration and conversion.

## Data processing

The converter reads XPolicyLab HDF5 episodes from
`<source>/<task>/<episode-subdir>/episode_*.hdf5`; `episode-subdir` defaults to
`data` and is configurable. Episodes must contain the state/action keys from
`mmabc/embodiments/mobile.py`, `instruction`, and encoded camera frames under
`vision/<camera>/colors`. The supported cameras are `cam_head`,
`cam_left_wrist` and `cam_right_wrist`.

```bash
python scripts/convert_mobile.py \
  --src /path/to/trajectories \
  --episode-subdir data \
  --out ../data/mobile_mmabc \
  --workers 8

# Optional: derive the compatible 80-d profiles after the first conversion.
python scripts/build_mobile_c80.py
```

Image bytes are decoded with `XPolicyLab.utils.process_data.decode_image_bit`,
which handles both legacy and marked standard RGB encodings. Frames are
resized to 224×224 with the same helper used during inference and written as
RGB-derived H.264 videos. No caller-side channel swap is applied.

The output uses LeRobot v3.0 parquet/video organization with
`observation.state`, `action`, and `observation.images.<camera>` keys. This is a
custom converter because the full-body vectors and additional
`meta/embodiment.json`, `meta/modality.json`, and layout metadata are required
by MM-ABC; the keys and vector layout differ from the official joint-only
conversion recipe. Do not substitute a standard export without adapting its
layout metadata.

Conversion tracks completed episodes in `.done/`, then regenerates the
embodiment configurations, task mixture and normalisation statistics under
`configs/`. Reruns skip completed episodes. Regenerate these files for your
own data; bundled statistics describe only the accompanying example profiles.
`--max-episodes` and `--tasks` can restrict a trial conversion. The 80-d builder
rewrites state/action parquet files and links the source videos, so keep the
source profiles available.

## Training

Place the backbone and optional teacher assets at the paths in the selected
model configuration before launching. A compatible weights-only checkpoint is
also required for `mobile_pretrain.yaml`:

```bash
python scripts/export_pretrain_weights.py \
  --src /path/to/resume/step_00200000 \
  --out ../pretrained_ckpt/mmabc_pretrain_200k

bash scripts/train_single_node.sh configs/train/mobile_smoke.yaml
bash scripts/train_single_node.sh configs/train/mobile.yaml
bash scripts/train_single_node.sh configs/train/mobile_pretrain.yaml
```

Training settings accept dotlist overrides:

```bash
NPROC=1 bash scripts/train_single_node.sh configs/train/mobile.yaml \
  micro_batch_size=1 max_steps=100 \
  model_overrides.backbone.gradient_checkpointing=true
```

The example configuration uses eight samples per rank; choose a batch size
appropriate for available GPU memory. `checkpoint.min_free_gb` checks available
disk space before saving. Resume checkpoints contain optimizer state and are
kept under `resume/step_<number>`; inference weights go under
`milestones/step_<number>`. Keep the sibling `run_meta/` directory, which saves
model configuration, layout, normalisation statistics and the embodiment
contract needed for reproducible inference.

Set `WANDB_API_KEY`, or point `MMABC_WANDB_ENV` at a local environment file, to
enable W&B logging. Without credentials the trainer logs to the console.
Never commit the environment file.

### Multiple nodes

Launch scripts require Linux and Bash 4.4 or later. Copy the commented examples
in `configs/cluster/hostfile` to a local hostfile and supply your SSH hosts,
rendezvous IPs and key paths. The first active row is the master. All nodes
must share identical source, asset and Python-environment paths. Set `NPROC`
to the number of GPUs per node and configure SSH host keys before launching.

`configs/cluster/nccl.env` leaves network-device selection to NCCL. Configure
fabric-specific variables for your cluster; `FORWARD_ENV` forwards named
variables to remote nodes.

```bash
HOSTFILE=/path/to/hostfile NPROC=8 \
  bash scripts/train_multinode.sh configs/train/mobile.yaml

HOSTFILE=/path/to/hostfile NPROC=8 \
  bash scripts/run_multinode.sh scripts/nccl_verify.py
```

`train_detached.sh` detaches a single-node run. `train_supervised.sh` detaches a
multi-node supervisor, cleans up stale processes associated with this source
tree before each attempt, and resumes after failures. Do not run independent
jobs from the same source tree while using that supervisor. It prints the PID,
log location and a `STOP` file path that prevents further retries.

## Evaluation and tests

Open-loop evaluation reads the training mixture and checkpoint contract:

```bash
python scripts/open_loop_eval.py \
  --checkpoint ../checkpoints/mobile-4b/milestones/step_00060000
```

Closed-loop evaluation uses the parent XPolicyLab adapter. The runtime adapter
expects already-decoded RGB arrays from the policy server.

```python
from mmabc.eval.adapters.mobile import MobileAdapter
from mmabc.eval.policy import MMABCInferencePolicy

policy = MMABCInferencePolicy.from_run("/path/to/milestones/step_00060000")
adapter = MobileAdapter(variant="m75")  # use "c80" with the 80-d layout
observation = adapter.observe(raw_observation, instruction)
actions = adapter.from_canonical_action(policy.act(observation)["action_canonical"])
```

Run unit tests without downloading model weights:

```bash
python -m pytest tests -q
```

Tests that require converted datasets skip when those assets are absent. Optional
cross-embodiment tests use `MMABC_CORPUS_ROOT` with `mobile_platform`,
`single_arm`, and `dual_arm` profile directories. A complete training or
closed-loop run additionally requires the configured assets and runtime
environment; static checks and unit tests do not validate model performance.
