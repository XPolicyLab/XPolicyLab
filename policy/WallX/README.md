# Wall-X

**Contributor:** X Square Robot | **Original code:** https://github.com/X-Square-Robot/wall-x

`WallX` adapts [Wall-X](https://github.com/X-Square-Robot/wall-x), a
Qwen3.5-based vision-language-action policy, to XPolicyLab/RoboDojo. The
delivered checkpoint predicts a 32-step native EE-pose action chunk, polishes it
with a residual refiner (four chained passes), and keeps a **keyframe memory**:
it re-reads a sliding window of recent head-camera frames and decides which ones
to retain, which is what makes the long-horizon and memory-dependent RoboDojo
tasks reachable.

Shared conventions — argument meanings, checkpoint naming, split-machine
deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md).
Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

The adapter, the deployment loop, the checkpoint export, the training recipe and
the base model our checkpoint was fine-tuned from are all published — here, in
the wall-x repository, and as
[`x-square-robot/wall-oss-1.0-lite-robodojo-base`](https://huggingface.co/x-square-robot/wall-oss-1.0-lite-robodojo-base)
— a preview of Wall-OSS-1.0-lite, published ahead of that release and trained
for a different number of iterations.

## Installation

The Qwen3.5 release and the serving changes this adapter needs are under review
in X-Square-Robot/wall-x#128. Until that is merged, install wall-x from its
branch, pinned to the commit the published numbers were produced with:

```bash
git clone -b starrick/robodojo-serving https://github.com/StarrickLiu/wall-x.git <bench_root>/wall-x
git -C <bench_root>/wall-x checkout 39fa65b9c9e95b0316db0fc9dbe41211e3a0d4cc
cd XPolicyLab/policy/WallX
WALLX_REPO=<bench_root>/wall-x bash install.sh
conda activate <policy_env>
```

That commit is the serving checkout the published numbers were produced with.
The training code is on the same fork's `starrick/oss-training` branch, which
carries the RoboDojo data pipeline, the two example configs and the calibration
script the [Training](#training) section uses; check that branch out instead if
you intend to train.

`install.sh` installs XPolicyLab (editable) and, when `WALLX_REPO` is set, the
wall-x serving package from that checkout (default `<bench_root>/wall-x`, a
sibling of the XPolicyLab checkout). Install the wall-x dependencies from its
`requirements.txt`; the published numbers were produced with:

| Package | Version |
| --- | --- |
| wall-x | `39fa65b` (StarrickLiu/wall-x `starrick/robodojo-serving`, X-Square-Robot/wall-x#128) |
| torch | 2.10.0 (CUDA 12.8) |
| transformers | 5.2.0 |
| flash-linear-attention | **0.4.2** — 0.5.2 changes the gated-delta kernel numerics |
| flash-attn | 2.8.1 |

Install wall-x with its CUDA operators compiled (`MAX_JOBS=8 pip install
--no-build-isolation -e .` in the wall-x checkout). Without them it runs PyTorch
fallbacks, which agree within floating-point tolerance but are not bit-identical
to the run behind the published numbers.

## Training data

Both stages read exactly two inputs, and the code refuses to start if either is
not what it expects:

* **`data/RoboDojo_ee_lerobot_v30_video`** — the official end-effector export,
  read as published. Every observation and every action in training comes from
  decoding it. There is no conversion step, no dataset mixing, and no second
  corpus; an unrecognised key under `data.robodojo` is rejected rather than
  ignored.
* **[`x-square-robot/robodojo-memory-annotations`](https://huggingface.co/datasets/x-square-robot/robodojo-memory-annotations)**
  — frame-level labels we produced over those same episodes, and publish: which
  frames of a memory task made visible progress, an optional box around the
  object the progress was about, and the interval of frames in which recording
  that keyframe counts as correct. 600 files, one per episode, covering all 100
  episodes of each of the six memory tasks. They carry no pixels, no states and
  no actions. Training binds each one to its episode by task instruction and
  then checks the episode lengths agree, and stops if they do not.

No other demonstrations, teleoperation or simulator-generated trajectories were
used in either stage.

## Data Processing

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
```

This fetches those two things and checks that they agree.

**The episodes.** RoboDojo publishes an end-effector LeRobot export,
`data/RoboDojo_ee_lerobot_v30_video`, alongside the joint-space one. It already
carries what this policy acts in — a position, a `wxyz` quaternion and a
gripper value per arm — so it is read as published and there is no conversion
step:

```bash
bash ../../scripts/RoboDojo/download_robodojo_data.sh lerobot_ee
```

Its videos are lossy, so a decoded frame differs slightly from what the
simulator rendered. That is what everyone trains on and the policy is trained
through it, but pixel-exact comparisons against a lossless source will not
match.

**The memory annotations.** The memory-dependent tasks turn on earlier frames:
a block whose colour was seen before it was covered, a slot that was already
filled. Which frames those are is not in the demonstration, so we annotated it,
and publish the annotations as
[`x-square-robot/robodojo-memory-annotations`](https://huggingface.co/datasets/x-square-robot/robodojo-memory-annotations).
Per episode they record

* the **keyframes** — the frames at which the task made visible progress, which
  are the frames the policy is trained to have in its memory;
* an optional **box** around the object that progress was about, so a recalled
  frame can be cropped to it and the memory carries the object rather than the
  whole scene;
* an acceptance **window** per keyframe. A keyframe is a moment, and several
  consecutive frames show that same moment equally well, so the selector is
  trained against the interval of frames that would be a correct answer rather
  than against one index. Scoring a single index would punish a right answer
  for being one frame early.

They annotate the official episodes rather than replacing them: every pixel and
every action still comes from the export above.

**The check.** An episode bound to the wrong annotation trains the policy to
recall the wrong frames, and nothing about the loss curve says so. The script
therefore binds every annotation to its episode and compares the lengths before
it exits — the one historical numbering drift in this dataset showed up exactly
as an episode length differing by one.

**Normalisation statistics** are not measured: they come with the weights. The
base model ships the pair it was pretrained with, and stage 2 inherits them
from the stage-1 checkpoint it refines. The model learned robot control reading
those numbers, and measuring new ones over RoboDojo alone would
shift the inputs every pretrained layer sees — with nothing in the loss to say
so.

To check the wiring before downloading 200 GB, `scripts/robodojo_smoke.py`
writes a two-episode stand-in of the whole setup — export, annotations, a
normalizer, both configs — and pushes a few batches through the real processor.

| Variable | Default | Meaning |
| --- | --- | --- |
| `WALLX_ANNOTATIONS` | `<bench_root>/data/robodojo_memory_annotations` | where the annotations are unpacked |
| `WALLX_ANNOTATION_REPO` | `x-square-robot/robodojo-memory-annotations` | the dataset to fetch them from |

## Model Assets

The checkpoint is published on Hugging Face as
[`x-square-robot/wall-oss-1.0-lite-robodojo`](https://huggingface.co/x-square-robot/wall-oss-1.0-lite-robodojo)
(F32 weights, no training-side state, ~24 GB on disk):

```bash
hf download x-square-robot/wall-oss-1.0-lite-robodojo --local-dir <bench_root>/wall-oss-1.0-lite-robodojo
```

```text
wall-oss-1.0-lite-robodojo/
├── checkpoint/            # config.json, config.yml, model.safetensors, normalisers
└── assets/                # tokenizer and processor files
```

`deploy.yml` resolves it in this order: `WALLX_CHECKPOINT` (passed through
`setup_eval_policy_server.sh`), the `checkpoint_path` key, then
`XPolicyLab.utils.checkpoint_resolver` under `policy/WallX/checkpoints/`.

The refiner settings (`refine_loops: 4`, `refine_step_scale: 0.25`) and the
keyframe-memory geometry are read from the checkpoint config. At load time the
server casts everything except the normalisation layers to bf16; one server
process uses about 19 GB of GPU memory with ten environments batched.

## Training

Two stages, because the second needs the first to exist:

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
WALLX_STAGE=refiner bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

**Stage 1 — keyframe memory.** Trains the policy that reads a memory of
recalled keyframes, and the selector that decides which recent frames to
record, on the same episodes: most samples are action samples that read a
memory built from the annotation, and interleaved among them are selection
samples that ask which of sixteen candidate frames to keep. Both are needed.
A policy trained only to read a memory depends at inference on a memory nobody
writes.

**Stage 2 — residual refiner.** A flow-matching policy is near a fixed point
of its own denoiser: feeding its output back in returns roughly the same thing,
so it cannot correct itself, and spending more integration steps does not help.
What it never sees while training is the mapping it actually needs — from *its
own output* to the demonstration. Stage 2 adds a second copy of the action
branch that learns exactly that, over a frozen stage 1, applying a quarter of
its residual on each of four chained passes. Its output head starts at zero, so
an untrained refiner changes nothing — the `correction` the run logs starts at
exactly 0 and grows.

A step samples a candidate from the frozen policy first, through the same
integration budget and the same cached observation that serving uses, so the
residual is fitted to the distribution that is actually served. Of the four
chained passes only the last is trained; the earlier ones build the estimate it
starts from and run without gradients, as they do at inference.

Stage 2 is only scored on candidates that are close enough to the demonstration
to be worth correcting; a sample that chose a different reasonable way to do the
task would otherwise teach the refiner to undo a decision. "Close enough" is a
property of the policy being refined, not of the task, so it is measured rather
than shipped:

```bash
cd <bench_root>/wall-x
python scripts/calibrate_refiner_margins.py \
  --train-config <stage-2 run config> --out <policy/WallX>/margins.json
```

`train.sh` writes that run config next to the checkpoint, with this run's seed,
output path and — for stage 2 — the stage-1 checkpoint it is refining written
in, so the config beside a checkpoint is the one that produced it. It refuses to
start stage 2 without a stage-1 checkpoint or without the margins, and says
which is missing.

### What these configs are, and what they are not

They are the recipe. They are not a bit-exact reproduction of the leaderboard
run, and the differences are these — all of them, as far as we know:

1. **`data.memory.short_term`** is `0` here. The published run enabled recent
   head-camera frames for the conveyor task only; the configs do not carry
   per-task overrides.
2. **Attention kernel.** The published run used a masked-attention kernel that
   is not one of this package's dependencies. These configs use `sdpa`, which
   is numerically equivalent but slower.
3. **How the two objectives meet.** The published run packed samples of both
   kinds into one sequence, so every step carried both. Packing needs that
   kernel, so these configs reach the same place by accumulating four
   micro-batches per update, with the sampler dealing the kinds at their own
   spacing. One consequence: a packed step weights the two objectives
   `ar_loss_weight` to one whatever the mix, while accumulation weights them by
   their share of the samples. Multiply `ar_loss_weight` by the ratio of action
   samples to selection samples -- both printed at startup -- to train with the
   published balance.

The training data is the same in both: the official end-effector export plus the
published memory annotations, and nothing else.

| Variable | Default | Meaning |
| --- | --- | --- |
| `WALLX_STAGE` | `memory` | `memory` or `refiner` |
| `WALLX_REPO` | `<bench_root>/wall-x` | the wall-x checkout holding the trainer |
| `WALLX_CONFIG` | the stage's example config | a config of your own |
| `WALLX_STAGE1_CKPT` | the stage-1 run of the same setting | which policy stage 2 refines |
| `WALLX_MARGINS` | `policy/WallX/margins.json` | the calibrated margins |
| `WALLX_NPROC` | the number of GPUs in `<gpu_id>` | processes per node |

## Evaluation

```bash
cd XPolicyLab/policy/WallX
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
             <seed> <policy_gpu_id> <env_gpu_id> \
             <policy_conda_env> <eval_env_conda_env>

# Example
WALLX_CHECKPOINT=<bench_root>/wall-oss-1.0-lite-robodojo/checkpoint \
  bash eval.sh RoboDojo swap_blocks wall-oss-1.0-lite-robodojo arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it
unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine
deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow
the [Deployment Flow](../../README.md#-deployment-flow).

A full benchmark run uses RoboDojo's own client against one long-lived server:
start the server once, then point `bash scripts/robodojo.sh client` (single task
or batch mode) at it. One server on one GPU kept up with six simulator
processes at `num_envs: 10` in our run; the simulator, not the policy, is the
bottleneck.

Supported `env_cfg_type`: `arx_x5` (dual arm, EE-pose robot). Supported
`action_type`: `ee` — the checkpoint predicts native EE poses and gripper states,
so a `joint` run is rejected at startup.

## Configuration

`deploy.yml` keys beyond the standard set:

| Key | Default | Meaning |
| --- | --- | --- |
| `checkpoint_path` | `null` | bundle `checkpoint/` dir; falls back to the resolver |
| `device` | `cuda:0` | server device |
| `norm_key` | `ex_normal` | normaliser set used by the export |
| `image_resize` | `serving` | `serving` reproduces the two-stage resize the checkpoint was evaluated with; `training` is the single-resize data-pipeline path |
| `action_chunk_length` | `32` | actions returned per request |
| `replan_steps` | `16` | actions the simulator executes before the next request |
| `capacity` / `num_candidates` / `select_stride` | `22` / `16` / `48` | keyframe memory geometry (`null` keeps the checkpoint's value) |

Both the server and the environment-side loop read `action_chunk_length` and
`replan_steps` from this file, so change them here rather than with launch-time
overrides.

`eval_batch` is `true`: the simulator runs its environments together (the
RoboDojo default `num_envs: 10`) and the requests due at a step go to the server
as one `infer_batch` call. Each environment keeps its own memory state on the
client, so batching does not mix episodes. With `eval_batch: false` the same
policy runs one environment at a time.

## Notes

### Deploy loop

Unlike `demo_policy`, `deploy.py` runs **chunked**: it reads an observation every
simulator step, sends a request every `replan_steps` steps, and replays the
returned chunk until the next request.

The memory bookkeeping lives on the client (`memory_client.KFMemClientState`),
because the selection pass needs the raw pixels of the frames it chooses between
and the server must not accumulate per-episode video across a parallel
evaluation. Each request therefore carries `kfmem_client_state_v1`: the frame ids
retained so far, the selection counters, the CUDA RNG state and the raw head frames
the server will read. The first request of an episode sends the whole sliding
window (60 frames) plus `kfmem_history`; its response includes a `ring_plan`, after
which the client sends only the current frame and, every `select_stride` frames,
the selection candidates (about 1 frame per request instead of ~77). The server
recomputes which frames it reads and rejects a request missing any of them, so the
model inputs are the same as with the full window.

The server answers with the action chunk plus the updated memory state, which the
client adopts before the next request. Both sides validate causality (memory ids
strictly older than the current frame), so a mismatch fails loudly instead of
silently degrading the rollout. Each request's flow noise is drawn from that
environment's own RNG state, so a batched request gets the same noise it would
get alone.

### Known limitations

* Selection passes still run one request at a time inside a batch; only the
  flow-action pass is batched (requests with the same prompt layout share a forward).
* Batched and single requests agree only to bf16 batch-reduction precision
  (≈1e-3 in action space), so `num_envs` changes individual rollouts, not the
  policy.
* EE-pose actions only; joint-space checkpoints would need a different export.
* No quantised bundle is published.
