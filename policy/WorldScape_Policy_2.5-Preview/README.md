# WorldScape Policy 2.5 Preview

**Contributor:** WorldScape team | **Paper:** pending | **arXiv:** pending | **Original code:** Inference runtime included in `worldscape-policy/`.

Supports RoboDojo, `arx_x5`, absolute `joint` actions.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Prepare the RoboDojo workspace with `XPolicyLab/` and `env_cfg/` as siblings. Use a separate Python 3.11 environment for the policy and keep the simulator in its RoboDojo environment.

```bash
# Run inside the activated policy environment.
cd RoboDojo/XPolicyLab/policy/WorldScape_Policy_2.5-Preview
bash install.sh
bash download_checkpoint.sh
```

## Data Processing

Image preprocessing and text/visual feature extraction run online during evaluation.

## Training

This preview includes inference only. Training code and its release timeline have not been announced.

## Evaluation

Run one task and seed from this policy directory:

```bash
bash eval.sh RoboDojo stack_bowls \
  wsp_2_5_preview_posttrain_robodojo arx_x5 joint 0 \
  0 0 /path/to/policy-env RoboDojo
```

The final four arguments select the policy GPU, simulator GPU, policy environment and simulator environment. Replace `/path/to/policy-env` and `RoboDojo` with your environment paths or names. By default, the benchmark root is the parent of `XPolicyLab/`, matching the standard RoboDojo layout. Set `BENCH_ROOT=/path/to/RoboDojo` only for a separate benchmark workspace; the shared robot metadata must remain beside this XPolicyLab checkout. Simulation uses the standard XPolicyLab client launcher and the simulator's default runtime/cache settings.

`deploy.yml` is the deployment configuration. Keep the supplied settings and custom client, which records observations at every control step for visual memory. For a protocol smoke test without simulation, set `EVAL_ENV_TYPE=debug` and use the policy environment for both environment arguments; add `DEBUG_OBS_ENCODED=1` to check encoded images.

## Model Assets

Model repository: [manifoldai-research/WorldScape-Policy-2-5](https://huggingface.co/manifoldai-research/WorldScape-Policy-2-5).

`bash download_checkpoint.sh` downloads the `wsp_2_5_preview_posttrain_robodojo` subfolder into `checkpoints/wsp_2_5_preview_posttrain_robodojo/`. It does not overwrite an existing directory.

Keep all checkpoint shards, configuration, normalization, tokenizer and processor files together. T5 and VLM weights are included; text and visual features are computed online. To use a checkpoint elsewhere:

```bash
export WORLDSCAPE_CHECKPOINT=/path/to/wsp_2_5_preview_posttrain_robodojo
```

Observations contain three RGB camera views, a language instruction and robot state. Preserve the supplied `mosaic_T` layout and `vlm_input_color_order: bgr` setting: only the frozen VLM's private input is converted to match training; framework observations and VAE inputs remain RGB.
