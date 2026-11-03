# RoboRSI_RoboProbe

**Contributor:** RoboRSI | **Paper:** none | **arXiv:** none | **Original code:** [RoboProbe](https://github.com/RoboProbe/RoboProbe)

A non-learned Skill layer over the pinned RoboProbe L3 EEF policy, using its environment-side action generation. Sibling `RoboDojo_Agent_L3_Inspect` and `RoboDojo_Agent_L3_Inspect_EEF` directories contain retained upstream sources with integration adjustments. Shared conventions are in the [XPolicyLab README](../../../../../../README.md); official results belong on the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use the existing official RoboDojo, Isaac and CuRobo environment. Run `bash install.sh /path/to/environment/bin/python` to verify Python policy dependencies. There is no robot-policy checkpoint and no separate robot execution environment. Deployment details and the exact installed entry are in [the release README](../../../../README.md).

## Data Processing

No training dataset or trajectory converter is used. `process_data.sh` is omitted for this eval-only, non-learned adapter. Images are official RGB end to end; model.py performs no decoding. Encoded debug observations use the upstream shared decode helper.

## Training

No training stage, weights, download script or `train.sh` is applicable. Skills are reviewed text plus explicit in-process non-learned tool dispatch. No learned action service is supported.

## Evaluation

Standard argument order is supported:

```bash
bash eval.sh RoboDojo stack_blocks none arx_x5 joint 0 1 1 /opt/robodojo-tools/envs/RoboDojo /opt/robodojo-tools/envs/RoboDojo
```

Set `ROBORSI_PROVIDER_FILE` to the release's inference-only provider, or configure the official RoboProbe API backend. `ROBORSI_OUTPUT` selects a fresh output directory; otherwise the launcher creates one. `EVAL_NUM=native` is the default. In this upstream standalone entry point, policy GPU and checkpoint arguments are retained for interface compatibility; planning runs in the simulator environment. The environment GPU argument chooses the GPU.

For split deployment use `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` with standard XPolicyLab arguments. These instructions describe the upstream standalone entry points. For the integrated WorldScape_RSI Agent and WAM deployment, including debug checks, use the [adapter README](../../../../README.md).

## Configuration

`policy_name` is this directory name; `protocol=ws`, `eval_batch=false`, `action_type=joint`. `policy_uv_env_path`, `result_dir`, and `obs_transform_pipeline` retain upstream deployment compatibility. One native environment runs per conversation. Native evaluator termination, scores, task configuration and layout selection remain authoritative. No extra observation, stop/reset tool, episode deadline, per-motion cap, model-call cap, or repair cap is added.

## Notes

This is not an organizer-certified submission. The included diagnostics establish interface behavior, not task success across the benchmark. New episodes reset conversation, gripper targets and pending images. Every episode exposes the same shared capability catalog; the Agent selects Skills from the live instruction, observations and execution feedback.
