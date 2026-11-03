# Phoenix-MWA · RoboDojo

[Project](https://openphoenix-team.github.io/Phoenix-pages/) · [Original code](https://github.com/OpenPhoenix-Team/Phoenix-MWA) · [Weights](https://www.modelscope.cn/models/ssuuuuu/phoenix-mwa/files)

Supports `RoboDojo / arx_x5, piper, piper_x / joint`. This release provides one **30K-step checkpoint per robot**, initialized from step **256013**. Model configuration, tokenizer, normalization statistics and instruction features are included; training data is not required for inference.

Shared conventions are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux, CUDA and BF16 support are required.

```bash
cd XPolicyLab/policy/Phoenix_WAM
bash install.sh phoenix-mwa
conda activate phoenix-mwa
bash download_checkpoint.sh checkpoints/phoenix-mwa-h5-nocrop-init256013-step30000-20261101
```

The downloader retrieves all three policy checkpoints and shared model resources. Add `--robot-type piper` (or `arx_x5` / `piper_x`) to download one robot only. Filenames, sizes and SHA256 checksums are pinned in `phoenix_release.json` and verified before loading.

## Data Processing

The models were trained on RoboDojo HDF5 observations and native joint actions. This evaluation release does not include data conversion; `process_data.sh` reports that it is unsupported.

## Training

Evaluation-only. `train.sh` reports that training is unsupported; a public training release date has not been announced.

## Evaluation

Start the policy server for the official RoboDojo hardware client:

```bash
bash setup_eval_policy_server.sh \
  RoboDojo fill_pen_holder phoenix-mwa-h5-nocrop-init256013-step30000-20261101 \
  piper joint 0 0 phoenix-mwa 9000 0.0.0.0
```

For ARX X5 or Piper X, replace `piper` and the task with the matching robot/task. RoboDojo supplies the client and parent-workspace `env_cfg/` definitions. The adapter selects the matching checkpoint and robot profile automatically.

Default instructions are listed in [instructions.json](phoenix_source/instructions.json); use `WAM_TEXT_MODE=online_t5` for arbitrary instructions. Set `WAM_ACTION_HZ` to the actual controller rate: native rates are 30 Hz for ARX X5/Piper and 25 Hz for Piper X. The client must acknowledge each executed action.

## Validation

62 CPU tests passed (15 historical tests skipped). All three checkpoints passed GPU inference checks, and both plain-RGB and encoded-RGB WebSocket debug loops passed. Real-robot task-success evaluation is pending. See [VALIDATION.md](VALIDATION.md) for the validation record.
