# Awomo-0.5

**Contributor:** Auwomo | **Paper:** TBD | **arXiv:** TBD | **Original code:** https://github.com/Awomo-WestlakeDI/Awomo-0.5 | **Model assets:** [Awomo-0.5I-Robodojo](https://huggingface.co/Auwomo/Awomo-0.5I-Robodojo)

`Awomo05` adapts Awomo-0.5 to XPolicyLab/RoboDojo (`arx_x5`, absolute joint
control, batched inference). Model, data and training code are bundled in `awomo/`.
Local evaluation used RoboDojo
[`2184bf8`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/2184bf8844ea9d205382c4aefa3a694311418251)
with adjusted XPolicyLab submodule configuration.

## Updates

- **2026-10-20**: Add RoboDojo SFT support.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Run from the repository root. `install.sh` creates or reuses the `awomo05` conda
environment (Python 3.11) and installs dependencies. `download_weights.sh` downloads
the RoboDojo SFT checkpoint, Qwen3-VL-4B-Instruct and FLUX.2 VAE to `policy/Awomo05/weights/`.

```bash
bash policy/Awomo05/install.sh
bash policy/Awomo05/download_weights.sh
```

## Data Processing

Prepare the HDF5 training index in the [training environment](training/README.md#installation):

```bash
conda activate awomo05-train
export AWOMO_DATA_ROOT=/datasets/RoboDojo
bash policy/Awomo05/process_data.sh RoboDojo sft arx_x5 joint
```

This writes `index.pt` and the supplied `dataset_stats.json` to
`policy/Awomo05/data/RoboDojo-sft-arx_x5-joint/` without rewriting images or recomputing statistics.
Evaluation with the published checkpoint does not require this step.

## Training

Data preparation and training are included in this adapter. See the
[RoboDojo SFT guide](training/README.md) for the pretrain model, data index and reference configuration.

## Evaluation

Choose a free GPU and port. Connect RoboDojo's environment client to that host
and port using policy `Awomo05`, robot `arx_x5`, and action type `joint`.
Use this adapter's `deploy.py` to preserve the 25 Hz observation history and
keep RoboDojo's simulator single-GPU mask.

For same-machine evaluation, use the standard entry point:

```bash
# Arguments: benchmark task checkpoint robot action seed policy-GPU sim-GPU policy-env sim-env
bash policy/Awomo05/eval.sh RoboDojo press_by_number policy/Awomo05/weights arx_x5 joint 0 0 1 awomo05 RoboDojo

# Debug the server/client interface with plain and encoded image inputs.
EVAL_ENV_TYPE=debug bash policy/Awomo05/eval.sh RoboDojo press_by_number policy/Awomo05/weights arx_x5 joint 0 0 1 awomo05 RoboDojo
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash policy/Awomo05/eval.sh RoboDojo press_by_number policy/Awomo05/weights arx_x5 joint 0 0 1 awomo05 RoboDojo
```


## Configuration

Keep the plugin directory and `policy_name` as `Awomo05`.
Check these `deploy.yml` settings before evaluation:

| Setting | Notes |
| --- | --- |
| `ckpt` / `ckpt_name` | Explicit `ckpt` takes precedence; standard scripts pass `ckpt_name`. |
| `input_color_order` | Set **`bgr`** explicitly for this checkpoint. |
| `action_type` | `joint`, with absolute joint targets. |
| `replan_steps` | `8` actions per inference. |
| `eval_batch` | `true` for batched inference. |
| `dataset_stats_path` | Defaults to `weights_dir/dataset_stats.json`. |
| `local_dir` | Local checkpoint staging directory; default `/tmp/Awomo05`. |

The standard server script uses the downloaded assets and BGR input automatically.
Override asset paths with `AWOMO05_QWEN_VL` / `AWOMO05_FLUX2_VAE` if needed.
Defaults: 16-step chunks, replan 8, 10 denoising steps, batched encoding and serial DiT.
