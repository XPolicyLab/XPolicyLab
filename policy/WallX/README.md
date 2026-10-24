# Wall-X

**Contributor:** Starrick Liu | **Original code:** https://github.com/X-Square-Robot/wall-x

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

This is an **eval-only** submission: the adapter, deployment loop and checkpoint
export are open source here and in the wall-x repository; the in-house training
pipeline and data converter are being prepared for release (tracked in the
PR), so `process_data.sh` and `train.sh` are stubs for now.

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

## Data Processing

Not released yet. The checkpoint was trained on RoboDojo demonstrations
converted by the in-house pipeline into the same `vision` / `state` / native EE
action layout seen at evaluation; the converter ships with the training code.
`process_data.sh` is a stub until then: once a wall-x checkout provides
`tools/convert_robodojo.py` it forwards to it, otherwise it only prints the
target directory.

```bash
WALLX_REPO=/path/to/wall-x bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
```

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

Not released yet (eval-only submission; the release is tracked in the PR).
`train.sh` is a stub with the standard arguments: once a wall-x checkout
provides `tools/train.py` it forwards to it, otherwise it only prints the
checkpoint directory it would write.

```bash
WALLX_REPO=/path/to/wall-x bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

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
