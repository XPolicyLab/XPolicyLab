# Awomo-0.5

**Contributor:** Auwomo | **Paper:** TBD | **arXiv:** TBD | **Original code:** https://github.com/Awomo-WestlakeDI/Awomo-0.5

`Awomo05` adapts Awomo-0.5 to XPolicyLab/RoboDojo (`arx_x5`, absolute joint
control, batched inference). Adapter files live in this directory; the vendored
inference implementation lives in `awomo/`. This release provides inference only.
Local evaluation used RoboDojo
[`2184bf8`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/2184bf8844ea9d205382c4aefa3a694311418251)
with adjusted XPolicyLab submodule configuration.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use a Python >= 3.10 policy environment with CUDA-compatible PyTorch. From the
XPolicyLab repository root:

```bash
bash policy/Awomo05/install.sh
```

The requirements specify PyTorch 2.8.0, Transformers 5.5.0, and PEFT 0.5.0.
Install RoboDojo's simulator environment separately. The parent workspace must
provide `env_cfg/arx_x5.yml` and the robot configuration.

## Data Processing

Not supported. This is an **eval-only** submission: `process_data.sh` is omitted.

## Training

Not supported. `train.sh` is omitted; the eval-only scope must be agreed with the
maintainers before the submission is accepted.

## Model Assets

The checkpoint repository is
[Hugging Face Auwomo/Awomo-0.5-Robodojo](https://huggingface.co/Auwomo/Awomo-0.5-Robodojo).
Obtain authorized access, then place `model.pt`, `dataset_stats.json`,
`SHA256SUMS`, and `LICENSE.md` in `policy/Awomo05/weights/`.
Use the matching statistics and comply with the supplied evaluation-only license.

```bash
# Authenticate with an account authorized to access the private checkpoint.
hf auth login
# Download the checkpoint, verify SHA256SUMS, and download base weights.
bash policy/Awomo05/download_weights.sh
# Optional: download only the released checkpoint.
bash policy/Awomo05/download_weights.sh checkpoint
```

Authenticate with Hugging Face before accessing restricted repositories.
The base downloader retrieves FLUX.2-klein-base-4B, the FLUX.2-dev autoencoder,
and Qwen3-VL-4B-Instruct alongside the released fine-tuned checkpoint.
The checkpoint revision is pinned; the script verifies its supplied SHA256SUMS.
Base models remain subject to their own licenses and access requirements.

## Evaluation

From the XPolicyLab root, start the policy server with the downloaded assets:

```bash
export AWOMO05_WEIGHTS="$(pwd)/policy/Awomo05/weights"
export AWOMO05_FLUX2_BASE="$AWOMO05_WEIGHTS/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
export AWOMO05_FLUX2_AE="$AWOMO05_WEIGHTS/flux2-dev/ae.safetensors"
export AWOMO05_QWEN_VL="$AWOMO05_WEIGHTS/Qwen3-VL-4B-Instruct"

CUDA_VISIBLE_DEVICES=0 python setup_policy_server.py \
  --config_path policy/Awomo05/deploy.yml --host localhost --port 9000 \
  --overrides policy_name=Awomo05 bench_name=RoboDojo \
  task_name=press_by_number env_cfg_type=arx_x5 action_type=joint seed=0 \
  ckpt=model.pt input_color_order=bgr replan_steps=8
```

Choose a free GPU and port. Connect RoboDojo's environment client to that host
and port using policy `Awomo05`, robot `arx_x5`, and action type `joint`.
Use this adapter's `deploy.py` to preserve the 25 Hz observation history and
keep RoboDojo's simulator single-GPU mask.

For same-machine evaluation, use the standard entry point:

```bash
# Arguments: benchmark task checkpoint robot action seed policy-GPU sim-GPU policy-env sim-env
bash policy/Awomo05/eval.sh RoboDojo press_by_number model arx_x5 joint 0 0 1 awomo05 RoboDojo

# Debug the server/client interface with plain and encoded image inputs.
EVAL_ENV_TYPE=debug bash policy/Awomo05/eval.sh RoboDojo press_by_number model arx_x5 joint 0 0 1 awomo05 RoboDojo
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash policy/Awomo05/eval.sh RoboDojo press_by_number model arx_x5 joint 0 0 1 awomo05 RoboDojo
```

Set the base-weight environment variables above before running these commands.
Use separate policy and simulator environments. If the policy dependencies are
installed in the active system/virtualenv interpreter, pass `system` as the
policy environment; `AWOMO05_PYTHON` can select that interpreter explicitly.
The standard server script explicitly selects BGR for this checkpoint, and
`deploy.yml` also defaults to BGR. Direct server launches must pass
`input_color_order=bgr`. `ckpt_name=model` resolves to `weights_dir/model.pt`;
explicit checkpoint paths and standard XPolicyLab checkpoint directories are
also supported. For split-machine evaluation, use the supplied server and
client setup scripts with the standard argument order.

## Configuration

Keep the plugin directory and `policy_name` as `Awomo05`.
Check these `deploy.yml` settings before evaluation:

| Setting | Notes |
| --- | --- |
| `ckpt` / `ckpt_name` | Explicit `ckpt` takes precedence; standard scripts pass `ckpt_name`. |
| `input_color_order` | Defaults to **`bgr`** for this checkpoint; standard setup scripts also select it explicitly. |
| `action_type` | `joint`, with absolute joint targets. |
| `replan_steps` | `8` actions per inference. |
| `eval_batch` | `true` for batched inference. |
| `dataset_stats_path` | Defaults to `weights_dir/dataset_stats.json`. |
| `local_dir` | Local checkpoint staging directory; default `/tmp/Awomo05`. |

Base paths can be set in `deploy.yml` through `flux2_base_path`, `ae_path`, and
`qwen_vl_path`, or through the environment variables above. Preserve the supplied
`model_config.yaml` and history settings. Do not add client-side channel swaps.
