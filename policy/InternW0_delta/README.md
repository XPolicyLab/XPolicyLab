# InternW0_delta

**Contributor:** Xingyu Miao | **Paper:** Not released | **arXiv:** Not released | **Original code:** evaluation-only inference closure in `wam_runtime/`

This adapter reproduces the RoboDojo evaluation of InternW0_delta for `env_cfg_type=arx_x5` with `action_type=joint`, using a 32-step action horizon, replanning after 10 executed actions, and 10 denoising steps. Training code and training datasets are intentionally not included; the vendored `wam_runtime/` contains only the model and online-inference modules required by this evaluation.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd policy/InternW0_delta
bash install.sh internw0-delta
conda activate internw0-delta
```

`install.sh` also installs a local C/C++ toolchain because RynnBrain's
FLA/Triton kernels compile a small runtime helper during their first forward
pass. No system-wide compiler installation is required. The tested policy
runtime is Python 3.11, PyTorch 2.10.0 with CUDA 12.8, Transformers 5.13.0,
and BF16 inference; the install script pins these model-sensitive packages.

## Data Processing

Not applicable. This is an evaluation-only submission.

## Training

Not included in this submission. This package contains the inference runtime
and the configuration required to reproduce the reported checkpoint evaluation.
Training code is scheduled for release before the end of October 2026.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate checkpoints/robodojo.pt on stack_bowls
bash eval.sh RoboDojo stack_bowls robodojo arx_x5 joint 0 0 0 internw0-delta <eval_env_conda_env>
```

The checkpoint comes from `checkpoint_path` in `deploy.yml` (or
`WAM_CHECKPOINT_PATH`), so `ckpt_name` only labels the run. For the offline
wiring check without model weights, run
`EVAL_ENV_TYPE=debug WAM_ALLOW_DUMMY_POLICY=true bash eval.sh ...` with the
same arguments; leave `EVAL_ENV_TYPE` unset or set `EVAL_ENV_TYPE=sim` for
RoboDojo simulation. For split-machine deployment via
`setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the
[Deployment Flow](../../README.md#-deployment-flow).

## Model Assets

Model weights are deliberately excluded from Git and from source archives.
The evaluator must prepare these three artifacts:

| Artifact | Upstream ID / source | Purpose | Local path relative to this directory |
| --- | --- | --- | --- |
| Wan2.2 | `Wan-AI/Wan2.2-TI2V-5B` | WAM video backbone, VAE and tokenizer resources | `assets/Wan-AI/Wan2.2-TI2V-5B/` |
| RynnBrain | `Alibaba-DAMO-Academy/RynnBrain1.1-2B` | visual-language understanding encoder | `assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B/` |
| InternW0_delta checkpoint | [`InternRobotics/InternW0-Delta-RoboDojo`](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo) | RoboDojo evaluation weights, file `robodojo.pt` | `checkpoints/robodojo.pt` |

Download the two public base models from ModelScope:

```bash
bash download_assets.sh
```

For Wan2.2 the script downloads only the VAE, UMT5 encoder and tokenizer files
used by evaluation; the full Video-DiT snapshot is unnecessary because
`robodojo.pt` already contains that expert. RynnBrain is downloaded as its full
ModelScope snapshot because Transformers loads its processor, tokenizer,
configuration, chat template, and model weights from that local directory.

This command writes only below `assets/`. Review and comply with the upstream
model licenses before downloading. To use an already populated model store,
pass its destination root instead:

```bash
bash download_assets.sh /path/to/local/assets
```

Download the evaluation checkpoint from Hugging Face and verify its hash:

```bash
bash download_checkpoint.sh
```

A local file or HTTPS URL can be passed instead; it is copied to
`checkpoints/robodojo.pt` and checked against the same SHA256.

The checked-in z-score statistics are in `config/dataset_stats.json`. The
exact expected artifact paths and hashes are recorded in
`config/artifacts.lock.json`.

## Configuration

All model locations are resolved from this policy directory by default.
Adapter-specific `deploy.yml` keys:

- Artifact paths: `checkpoint_path`, `base_model_dir`, `vlm_model_path`,
  `dataset_stats_path`, `train_config_path`.
- Inference contract of the checkpoint (keep the defaults to reproduce the
  reported result): `device`, `mixed_precision`, `action_horizon`,
  `replan_steps`, `num_inference_steps`, `action_hz`, `text_cfg_scale`,
  `negative_prompt`, `rand_device`, `tiled`.
- Diagnostics: `timing_enabled`, `default_instruction` (used when an
  observation carries no instruction), `allow_dummy_policy`.

`setup_eval_policy_server.sh` reads these environment variables:

| Environment variable | Override |
| --- | --- |
| `WAM_CHECKPOINT_PATH` | post-trained `robodojo.pt` file |
| `WAM_DATASET_STATS_PATH` | z-score statistics JSON |
| `WAM_EVAL_CONFIG_PATH` | evaluation model configuration |
| `WAM_WAN22_PATH` | local Wan2.2 directory |
| `WAM_RYNNBRAIN_PATH` | local RynnBrain directory |
| `WAM_ALLOW_DUMMY_POLICY=true` | debug wiring only; skips all real weights |

Evaluation uses the checkpoint's z-score `global_mean` / `global_std`
statistics, RGB input without training-time color jitter, discrete Action
RoPE, physical-time RoPE disabled, fan-in calibration disabled, and the
recent-KV-cache mask fix.

## Notes

- Batched evaluation (`eval_batch: true`): the policy is stateful, so the
  adapter keeps one WAM session (memory frames, pending actions, step counter)
  per `env_idx`, and every environment replans through the same single-sample
  inference path as `eval_batch: false`; GPU inference runs one environment
  at a time. Batched observations must carry `env_idx`, as the RoboDojo and
  debug environment clients do.
- Reference result: a self-run, single-environment evaluation over 54 tasks
  and 6,300 episodes completed 1,444 successful episodes — 22.92% success rate
  and 30.35 mean score.
