# G05 RoboDojo Real Evaluation

**Contributor:** OpenGalaxea | **Runtime provenance:** documented in [`SOURCE.md`](SOURCE.md)

This directory is the real-robot extension of the G05 XPolicyLab adapter. It is intentionally isolated from the RoboDojo simulation adapter in `policy/G05/`: it has its own deployment configuration, checkpoint layout, validation command, and server launcher. No file in this directory is used by simulation evaluation.

Shared XPolicyLab conventions are documented in the [repository README](../../../README.md). Official results are published on the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Scope

- Evaluation only; the real-robot client is operated by RoboDojo and is not included.
- Joint-position actions for `arx_x5`, `piper`, and `piper_x`.
- One checkpoint per physical embodiment. A bundle is rejected when its saved embodiment does not match `env_cfg_type`.
- Native 25 Hz observations and actions.
- The checkpoint predicts 32 actions; the server returns the first 16 actions before the official client replans.
- The released AR+FM checkpoint is deployed through its continuous FM head only.

Only the ARX-X5 checkpoint is currently released and validated. Piper and Piper-X must not be evaluated with the ARX-X5 checkpoint.

## Directory layout

```text
policy/G05/real/
├── README.md
├── SOURCE.md
├── deploy.yml
├── install.sh
├── model.py
├── setup_eval_policy_server.sh
├── validate_bundle.py
├── checkpoints/
│   └── arx_x5/
│       ├── config.yaml
│       ├── .hydra/config.yaml
│       ├── checkpoints/checkpoint.pt
│       ├── dataset_stats.json
│       ├── dataset_stats_robodojo_real_arx_x5_h32.json
│       ├── input_processor/
│       ├── action_tokenizer_hf/
│       └── inference_runtime/   # version-matched inference-only G05 code
└── configs/
```

## Model assets

The validated ARX-X5 real bundle is published at
[ModelScope: `ZhyRobert/g05-robodojo-real`](https://modelscope.cn/models/ZhyRobert/g05-robodojo-real/files).
Download only the robot-specific `real/arx_x5` directory and place its contents in
`policy/G05/real/checkpoints/arx_x5`:

```bash
cd XPolicyLab/policy/G05/real
python -m pip install -U modelscope

modelscope download --model ZhyRobert/g05-robodojo-real \
  --include 'real/arx_x5/**' \
  --local_dir checkpoints/modelscope

mkdir -p checkpoints/arx_x5
cp -a checkpoints/modelscope/real/arx_x5/. checkpoints/arx_x5/
```

The public weight name is `checkpoint.pt`. Its expected SHA-256 digest is:

```text
1b6d0003c8ba6ea200fdc29a34f94807f1c9c849e93d2f94c69974bdd98f9047  checkpoints/checkpoint.pt
```

The bundle also includes `SHA256SUMS`, which covers every distributed file. Verify all files from the bundle root:

```bash
cd checkpoints/arx_x5
sha256sum --check SHA256SUMS
cd ../..
```

Before starting a server, validate it:

```bash
"$G05_REAL_PYTHON" validate_bundle.py \
  --bundle checkpoints/arx_x5 \
  --embodiment robodojo_arx_x5
```

The old mixed-embodiment real archive is not compatible with this adapter.

## Installation

Download the model bundle first, then use Python 3.10 with a CUDA-enabled PyTorch environment:

```bash
cd XPolicyLab/policy/G05/real
export G05_REAL_PYTHON=/path/to/python
export G05_REAL_CHECKPOINT_PATH="$PWD/checkpoints/arx_x5"
bash install.sh
```

The runtime distributed with the robot-specific model bundle is authoritative. `G05_ROOT` and the older vendored runtime under `policy/G05/G05` are deliberately not consulted.

## Evaluation server

RoboDojo operates the real-robot client. The contributor starts only the policy server:

```bash
cd XPolicyLab/policy/G05/real
export G05_REAL_CHECKPOINT_PATH="$PWD/checkpoints/arx_x5"

bash setup_eval_policy_server.sh \
  RoboDojo put_objects_into_basket checkpoint \
  arx_x5 joint 0 0 "$G05_REAL_PYTHON" 6061 0.0.0.0
```

The server imports `XPolicyLab.policy.G05.real.model`, checks the bundle against `robodojo_arx_x5`, loads the FM head, and serves websocket actions on port 6061.

## Robot-specific bundles

| `env_cfg_type` | Required saved embodiment | Status |
| --- | --- | --- |
| `arx_x5` | `robodojo_arx_x5` | trained and locally validated |
| `piper` | `robodojo_piper` | configuration supported; checkpoint pending |
| `piper_x` | `robodojo_piper_x` | configuration supported; checkpoint pending |

Never point two robot names at one bundle. Normalization statistics, processor artifacts, and model weights are treated as one indivisible per-embodiment artifact.
