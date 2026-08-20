# OpenDM

**Contributor:** XPolicyLab Team | **Paper:** DM0.5 technical report | **arXiv:** TBD | **Original code:** [Dexmal OpenDM](https://gitlab.dexmal.com/robotics/opendm)

This eval-only adapter runs `mm-robodojo` with the XPolicyLab WebSocket policy server. This release supports RoboDojo simulation only: Dual ARX5 (`arx_x5`), joint control, and absolute actions. The pinned upstream OpenDM source is vendored under `opendm/`; see [opendm/UPSTREAM.md](opendm/UPSTREAM.md).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The default environment uses Python 3.10, PyTorch 2.11.0, torchvision 0.26.0, and CUDA 12.8 wheels:

```bash
cd XPolicyLab/policy/OpenDM
bash install.sh
conda activate opendm
```

PyTorch SDPA is used by default, so `flash-attn` is not required. Set `OPENDM_INSTALL_FLASH_ATTN=1` before running `install.sh` only if you explicitly switch the vision attention implementation.

## Data Processing

Not included in this eval-only release. RoboDojo simulation observations are consumed online through XPolicyLab; no dataset conversion is needed for evaluation.

## Training

Training is not included in this eval-only release.

## Model Assets

| Item | Value |
| --- | --- |
| Model name | `mm-robodojo` |
| Version | `v1` |
| Repository | [TalentBoy/mm-robodojo](https://huggingface.co/TalentBoy/mm-robodojo) |

Download `v1` from Hugging Face into the local checkpoint directory:

```bash
huggingface-cli download TalentBoy/mm-robodojo \
  --revision v1 \
  --local-dir ./checkpoints/mm-robodojo
```

The adapter expects the complete Hugging Face-style directory at:

```text
policy/OpenDM/checkpoints/mm-robodojo/
```

It must include model/config, processor/tokenizer files, and the matching `norm_stats.json`. The adapter also accepts a checkpoint directory as `ckpt_name` or through `MODEL_PATH`.

Model contract:

| Item | Value |
| --- | --- |
| Benchmark | RoboDojo simulation |
| Robot | Dual ARX5 (`arx_x5`) |
| Cameras | head, left wrist, right wrist RGB |
| State/action | 14-D: left arm 6, left gripper 1, right arm 6, right gripper 1 |
| Action semantics | absolute joint position |

The vendored OpenDM code remains under Apache-2.0. The `mm-robodojo` weights are distributed under the repository's MM-RoboDojo Community License, together with the upstream DM05/Gemma terms identified in its `NOTICE` and `THIRD_PARTY_NOTICES.md` files.

## Evaluation

Run one RoboDojo simulator task from this policy directory:

```bash
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo <task_name> mm-robodojo \
  arx_x5 joint 0 <policy_gpu_id> <env_gpu_id> opendm <robodojo_env>
```

For example, using policy GPU 0 and simulator GPU 1:

```bash
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo stack_bowls mm-robodojo \
  arx_x5 joint 0 0 1 opendm RoboDojo
```

The policy server can also be started independently with `setup_eval_policy_server.sh`; use the standard split-machine workflow linked above.

Each OpenDM policy-server instance currently supports one active RoboDojo
evaluation client. Although observation and history state is keyed by
`(evaluation_id, env_idx)`, RoboDojo environment reset still invokes a
server-wide policy reset. Do not share one policy-server instance between
concurrent evaluation clients; start a separate instance for each evaluator.

## Configuration

The runtime settings required by `v1` are fixed in [deploy.yml](deploy.yml) and should not be changed. `MODEL_PATH` and `NORM_STATS_PATH` may be used to select local model assets.

## Limitations

- Simulation only; real-robot deployment and safety controls are out of scope.
- Only `arx_x5` with `action_type=joint` is supported.
- Batch API calls are processed serially by the model adapter.
- One active evaluation client is supported per policy-server instance; concurrent evaluators require separate server instances.
- The checkpoint checksum, GPU-memory requirement, and benchmark score are pending.
- Request logging, cloud deployment, and distributed evaluation are intentionally outside the public policy adapter.
