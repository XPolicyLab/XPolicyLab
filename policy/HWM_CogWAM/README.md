# HWM_CogWAM

**Contributor:** Horizon Robotic Team | **Paper:** CogWAM (arXiv link pending) | **arXiv:** not yet public | **Original code:** CogWAM — intended public home `https://github.com/CogWAM/CogWAM` (declared in `pyproject.toml`; source release in internal open-source review, see Notes) | **Checkpoint:** [`HorizonRobotics/CogWAM`](https://huggingface.co/HorizonRobotics/CogWAM)

`HWM_CogWAM` wires [CogWAM](https://github.com/CogWAM/CogWAM) into XPolicyLab
as an **eval-only** policy for the RoboDojo leaderboard. It is a thin
adapter: every piece of CogWAM's own inference logic — checkpoint loading,
tri-view camera compositing, state normalization, flow-matching action
denoising, chunk/replan bookkeeping, and event-memory semantic-state
tracking — runs unmodified inside CogWAM's `cogwam.eval.robodojo_policy.Model`,
which this policy's `model.py` re-exports. Nothing from CogWAM is copied or
reimplemented here.

Shared conventions — argument meanings, checkpoint naming, split-machine
deployment, `EVAL_ENV_TYPE` — are documented in the
[XPolicyLab README](../../README.md). Official results:
[RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Architecture: two servers, one policy

```
RoboDojo sim  <--XPolicyLab ws-->  XPolicyLab policy server   <--CogWAM ws-->  cogwam.serve.policy_server
                                    (hosts model.Model =                       (loads the checkpoint,
                                     cogwam.eval.robodojo_policy.Model;         holds the GPU, runs the
                                     no GPU work, pure translation +            flow-matching denoiser)
                                     chunk/replan/event-memory state)
```

`setup_eval_policy_server.sh` starts the CogWAM backend
(`cogwam.serve.policy_server`) first, waits for its WebSocket handshake, and
only then starts XPolicyLab's own unmodified `setup_policy_server.py`
pointed at it via the `policy_server_host`/`policy_server_port` overrides.
`Model.__init__` cross-checks the backend's handshake metadata (checkpoint
path, action chunk size, image layout, action/state key order,
event-memory contract) against `deploy.yml`'s expectations before serving a
single observation — see CogWAM's `cogwam/eval/robodojo_policy.py` for the
checks.

## Installation

```bash
cd XPolicyLab/policy/HWM_CogWAM
export COGWAM_ROOT=/path/to/CogWAM   # CogWAM checkout — supplied out-of-band until the public release lands, see Notes
bash install.sh
conda activate <policy_env>          # hosts BOTH the bridge and the CogWAM backend
```

There is no separate environment for the backend: `cogwam.eval.robodojo_policy`
only needs numpy/opencv/torch/torchvision/websockets/msgpack, all of which
are already inside CogWAM's own `requirements.txt`, so one conda env covers
both halves of the two-hop server above.

## Data Processing / Training

Not part of this PR — **eval-only submission**. `process_data.sh` and
`train.sh` are intentionally omitted (CONTRIBUTING.md's declared exception
for eval-only PRs). CogWAM's own training pipeline (`scripts/train_multi_node.sh`,
`cogwam/training/`) is untouched and out of scope here.

## Evaluation

```bash
cd XPolicyLab/policy/HWM_CogWAM

# One-time: fetch the checkpoint and both backbones (see Model Assets).
# DINOv3 is gated — `hf auth login` with an approved account first.
python download_checkpoint.py --dest /path/to/assets

export COGWAM_ARTIFACT_DIR=/path/to/assets/cogwam-robodojo-h25-50k  # released artifact dir, never copied into this repo
export COGWAM_BASE_VLM=/path/to/assets/rynnbrain1.1-2B              # backbone weights, not redistributed with the checkpoint
export COGWAM_DINO_MODEL=/path/to/assets/dinov3-vitb16              # backbone weights, not redistributed with the checkpoint

bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: offline shape/IO smoke check (no simulator)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls cogwam-robodojo-h25-50k arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`COGWAM_BASE_VLM` and `COGWAM_DINO_MODEL` must be **local snapshot directories**,
not Hugging Face repo ids: CogWAM only sets `local_files_only=True` when the
value is a directory, so a repo id turns into a hub download — and DINOv3's is
gated. `setup_eval_policy_server.sh` checks this up front and fails with a
pointer to `download_checkpoint.py` rather than dying inside model
construction.

`ckpt_name` is a checkpoint *label*, not itself a path: it either matches a
directory under `checkpoints/` (create with
`mkdir -p checkpoints && ln -sfn <artifact_dir> checkpoints/<ckpt_name>`) or is
ignored entirely when `COGWAM_ARTIFACT_DIR` is set explicitly, which
`setup_eval_policy_server.sh` prefers when present.

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it
unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine
deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`,
follow the [Deployment Flow](../../README.md#-deployment-flow) — note that in
that mode the *CogWAM backend* also needs `COGWAM_BASE_VLM`/`COGWAM_DINO_MODEL`
reachable from the policy-server machine, not the env-client machine.

## Configuration

`deploy.yml` keys to check before evaluation: `unnorm_key`, `use_ddim`,
`num_ddim_steps`, `replan_interval`, `rtc_enabled` and the `rtc_*` fields,
`image_size`, `expected_image_layout`, `expected_composite_view_key`,
`expected_action_chunk_size`, `expected_action_dim`. These mirror CogWAM's
own `configs/robodojo_deploy.yml` and are hard ABI checks against the
checkpoint, not tuning knobs — see that file's comments in the CogWAM
checkout for what each one means. `policy_server_host` / `policy_server_port`
/ `expected_checkpoint_path` are filled automatically by
`setup_eval_policy_server.sh` at launch and should not be hand-edited.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `COGWAM_ROOT` | CogWAM checkout used by `install.sh` (`pip install -e`) and, only when CogWAM isn't already installed in the active env, as a `PYTHONPATH` fallback in `setup_eval_policy_server.sh`. |
| `COGWAM_ARTIFACT_DIR` | Released artifact directory (`model.safetensors`, `dataset_statistics.json`, `inference_config.yaml`, ...) or a raw training checkpoint directory. Takes precedence over `checkpoints/<ckpt_name>` when set. |
| `COGWAM_BASE_VLM` | RynnBrain1.1-2B backbone weights (Apache-2.0); not redistributed with the checkpoint. Must be a local directory. |
| `COGWAM_DINO_MODEL` | DINOv3 ViT-B/16 backbone weights (Meta `dinov3-license`, gated); not redistributed with the checkpoint. Must be a local directory. Required at inference, not just training — see Model Assets. |
| `COGWAM_RUN_ROOT` / `COGWAM_DATA_ROOT` | Training-only config interpolations that OmegaConf still resolves eagerly when loading `inference_config.yaml`. `setup_eval_policy_server.sh` defaults both to a scratch directory; nothing at inference reads them. |
| `COGWAM_QWEN35_DISABLE_CAUSAL_CONV1D` / `COGWAM_QWEN35_DISABLE_FLA` / `COGWAM_QWEN35_ATTN_IMPLEMENTATION` | RynnBrain fast-kernel switches. `setup_eval_policy_server.sh` defaults to the `torch_safe` fallback (`1`/`1`/`sdpa`); set the first two to `0` once `flash-linear-attention==0.3.2` and `causal_conv1d==1.5.0.post8` are actually built in the env. |
| `EVAL_ENV_TYPE` | Evaluation client mode: unset/`sim`, `debug`, or `real`. |

## Model Assets

The released CogWAM checkpoint is published at
**[`HorizonRobotics/CogWAM`](https://huggingface.co/HorizonRobotics/CogWAM)**
(public, no token required). The repository root *is* the artifact-directory
layout the policy server expects, so the download target can be handed to
`COGWAM_ARTIFACT_DIR` verbatim:

```bash
python download_checkpoint.py --dest /path/to/assets
```

The script pins the evaluated revision
(`017926b231a681944dc90ccacc77862847b10140`) so a later push to the model repo
cannot silently change what gets evaluated.

Six files make up the artifact; nothing else in the checkpoint's parent tree is
needed and none of it should be re-packaged wholesale:

| File | Purpose | sha256 (from the artifact's own `artifact_manifest.json`) |
|---|---|---|
| `model.safetensors` | CogWAM weights (2272 BF16 tensors, 7.8 GiB) | `b467d71bc267e89e0e410935e19d223239f7d92d3efa4ead92a4b65c38c0b20a` |
| `dataset_statistics.json` | action/state normalization — required to un-normalize actions | `50c79872b1984c6cc7269aba632e1f12c192b7a2fd5424314b04acd738103d52` |
| `checkpoint_keys.json` | key→(dtype, shape) inventory, for validating without loading weights | `af5f987635366e885614f12cbb98fbcabf8cd9b64b0e8a12ff3668130885c683` |
| `inference_config.yaml` | resolved training-time config this checkpoint was built from | `efcba6fb6d95cdbe5b68f06d13d4c75869b2dbbe472e97828af7871108899e1e` |
| `artifact_manifest.json` | self-hashed manifest tying the four files above together | manifest self-hash `e2b3601d32a68e3da922d1bc86ca6c272bbaaa6d75e82c2e8dd01a21782e00a0` |
| `README.md` | the artifact's own release notes | — |

`cogwam.serve.policy_server` verifies these hashes itself on every load
(`PolicyServerWrapper`); a mismatch fails fast instead of silently loading a
different checkpoint.

Two backbones are referenced by the checkpoint rather than bundled with it:

| Backbone | Repo | License | Access |
|---|---|---|---|
| RynnBrain1.1-2B | [`Alibaba-DAMO-Academy/RynnBrain1.1-2B`](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-2B) (also on [ModelScope](https://modelscope.cn/models/DAMO_Academy/RynnBrain1.1-2B)) | Apache-2.0 | public |
| DINOv3 ViT-B/16 | [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | Meta `dinov3-license` | **gated, manual approval** |

> **DINOv3 access is a hard prerequisite for evaluating this policy.** The repo
> is gated with *manual* approval (anonymous requests return `401`), and DINOv3
> is not a training-only teacher: the released `inference_config.yaml` pins
> `framework.dino.load_live_backbone: true`, so the backbone is instantiated
> during model construction, and `CogWAM.predict_action` encodes every current
> observation with it on every control step. There is no code path that skips
> it and no substitute weight set. Whoever runs the evaluation must request
> access at the link above with their own Hugging Face account, wait for Meta
> to grant it, and be logged in (`hf auth login`) before
> `download_checkpoint.py` will succeed. Meta's license does not permit us to
> redistribute these weights alongside the checkpoint.

## Notes

- The checkpoint currently targeted for leaderboard submission is
  `cogwam_robodojo_causal_dino_mot_h25_eventmem_dino_multilayer_50k_v1`
  (step 50,000), released as the artifact directory
  `cogwam-robodojo-h25-50k`. This is the only checkpoint present in this
  release and the one all evaluation evidence in the PR was produced with.
- CogWAM's checkpoint README documents evaluation against RoboDojo 0.2.0 at
  commit `9b4cc885e8f530ed3ab14a30a312ae70242771c4`; use a matching RoboDojo
  checkout for comparable results.
- Camera order, image compositing, state/action layout, chunk size, and
  replan cadence are all hard-asserted by `Model.__init__` against the
  backend's handshake metadata — a mismatch fails fast instead of silently
  evaluating the wrong contract.
- **Training support: eval-only for this PR.** `process_data.sh` / `train.sh`
  are intentionally omitted per CONTRIBUTING.md's declared exception; a
  training-release timeline has not been set — see the PR description.
- **CogWAM's source release is in internal open-source review.**
  `pyproject.toml` declares `https://github.com/CogWAM/CogWAM` as its intended
  home, but that URL is not live yet, so `install.sh` needs a CogWAM checkout
  supplied out-of-band. For official leaderboard evaluation we coordinate the
  inference runtime with the maintainers directly — see the PR description.
  The checkpoint itself is already public and unrestricted.
- CogWAM's checkpoint README points at `docs/environment.md` and
  `docs/evaluation.md`; those files are not present in the current CogWAM tree
  and need to ship with the public release.
