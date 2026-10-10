# GeoRefiner

**Contributor:** chenhaijier | **Paper:** Not public yet | **arXiv:** Not public yet | **Original code:** Not public yet

`GeoRefiner` uses X-VLA as its base policy and refines the nominal action chunk before execution. This eval-only adapter supports RoboDojo with `env_cfg_type=arx_x5` and `action_type=ee`; the minimal GeoRefiner inference code is included under `georefiner/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/GeoRefiner
bash install.sh
conda activate XVLA
```

Set `XVLA_CONDA_ENV` to use a different conda environment name.

## Data Processing

Not included. This is an eval-only submission.

## Training

Not included. GeoRefiner training code will be released separately; release ETA: coming soon.

## Model Assets

The reproducible configuration is:

- X-VLA checkpoint: [`RoboDojo-sim-arx_x5-ee-0/ckpt-100000`](https://huggingface.co/Hazel500am/X-VLA-GeoRefiner-RoboDojo/tree/44870539ffabd538a10849809c5bc34e3c93bbff/xvla/RoboDojo-sim-arx_x5-ee-0/ckpt-100000)
- GeoRefiner: [`georefiner_xvla_trained_fp32.pt`](https://huggingface.co/Hazel500am/X-VLA-GeoRefiner-RoboDojo/blob/44870539ffabd538a10849809c5bc34e3c93bbff/georefiner_xvla_trained_fp32.pt)

Download both checkpoints from the Hugging Face repository:

```bash
cd XPolicyLab/policy/GeoRefiner
bash download_checkpoint.sh
```

The script installs the X-VLA checkpoint under the sibling `X_VLA/checkpoints/` adapter directory and the GeoRefiner assets under this adapter's `checkpoints/` directory. The `ckpt_name` argument selects the X-VLA checkpoint. `GEOREFINER_CHECKPOINT` may override the GeoRefiner checkpoint path.

## Evaluation

```bash
cd XPolicyLab/policy/GeoRefiner
bash eval.sh <bench_name> <task_name> <xvla_ckpt_name> <env_cfg_type> ee <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

Example:

```bash
cd XPolicyLab/policy/GeoRefiner
EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo insert_key RoboDojo-sim-arx_x5-ee-0 arx_x5 ee 0 0 0 XVLA RoboDojo
```

Leave `EVAL_ENV_TYPE` unset for simulator evaluation. Set `DEBUG_OBS_ENCODED=1` for the encoded-observation debug check.

## Configuration

`GEOREFINER_MODE` accepts `refine` (default), `shadow`, or `disabled`. Local assets may be selected with `GEOREFINER_CHECKPOINT` and `GEOREFINER_ARTIFACT_DIR`.
