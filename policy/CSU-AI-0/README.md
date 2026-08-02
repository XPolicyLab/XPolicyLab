# CSU-AI-0

`CSU-AI-0` is an XPolicyLab policy with a standard
`Model(ModelTemplate)` entry point.  It combines the frozen `QRouter`
HistGradientBoosting router with three process-isolated experts:

- `Xiaomi_Robotics_1` (`ee` actions)
- `Spatial_Forcing` (`joint` actions)
- `Hy_Embodied_05_VLA` (`ee` actions)

The outer policy server always loads `XPolicyLab.policy.CSU-AI-0.model.Model`.
At construction time, the model verifies the frozen router, selects one expert
for the requested task, starts that expert's normal XPolicyLab server in its
own Python environment, and forwards the official reset/infer/case/trial
protocol over localhost WebSocket.  No expert dependency stack is imported
into the router process.

## Interface

`model.py` subclasses `XPolicyLab.model_template.ModelTemplate` and implements:

- `update_obs` / `get_action`
- `update_obs_batch` / `get_action_batch`
- `reset`
- `prepare_case`
- `on_trial_end`

`expert_proxy.py` owns the selected child process and a dedicated WebSocket
I/O thread.  The child listens only on `127.0.0.1`, inherits the selected GPU,
and receives only that expert's environment and checkpoint configuration.

## Frozen router

- name: `QRouter`
- family: `HistGradientBoostingClassifier`
- training rows: 4,122 from seeds 1 and 2
- seed 0 used for training: no
- inference overrides: none
- expected seed-0 routes: Xiaomi 2,000; Hunyuan 50; Spatial Forcing 50
- QRouter SHA256:
  `1600870c4999b7e183bb34cb528d8fcb8e9a1527b30c4e6034b4fe5dc9972a87`

The joblib and portable tree export live in `router/`; hashes are pinned in
`adapter_manifest.json` and checked before every model launch.

## Checkpoint bundle

All model assets are presented as one logical bundle under:

```text
policy/CSU-AI-0/checkpoints/CSU-AI-0-v1/
├── xiaomi/RoboDojo-sim-arx_x5-ee-0/
├── spatial_forcing/RoboDojo-sim-arx_x5-joint-0/
├── hunyuan/rd20/
└── shared/huggingface/
```

The three directories remain separate checkpoint namespaces; they are not
tensor-averaged because the architectures and action heads are incompatible.
`shared/huggingface` contains the Xiaomi base-model cache.  To install a
published bundle:

```bash
export CSU_CHECKPOINT_REPO_ID=<organization/repository>
bash XPolicyLab/policy/CSU-AI-0/download_checkpoints.sh
python XPolicyLab/policy/CSU-AI-0/verify_checkpoints.py
```

The final public repository ID and immutable revision must be set before the
leaderboard pull request is submitted.  Existing local assets can be used
without copying by setting `CSU_CHECKPOINT_ROOT`, or by setting the expert
overrides `CSU_XIAOMI_CHECKPOINT`, `CSU_SPATIAL_CHECKPOINT`, and
`CSU_HUNYUAN_CHECKPOINT`.  Xiaomi's base cache can be overridden with
`CSU_XIAOMI_HF_HOME`.

## Expert environments

Defaults are `mibot` for Xiaomi and `uv` for Spatial/Hunyuan.  Override them
without editing tracked files:

```bash
export CSU_XIAOMI_POLICY_ENV=/absolute/path/to/mibot
export CSU_SPATIAL_POLICY_ENV=/absolute/path/to/open_sf
export CSU_HUNYUAN_POLICY_ENV=/absolute/path/to/hy_embodied
```

The router environment needs XPolicyLab's normal WebSocket dependencies but
does not need torch/JAX/Hunyuan packages from all three experts.

## Install and evaluate

```bash
bash XPolicyLab/policy/CSU-AI-0/install.sh

bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/CSU-AI-0 \
  --task stack_bowls \
  --ckpt QRouter \
  --env-cfg arx_x5 \
  --action-type auto \
  --seed 0 \
  --policy-gpu 0 \
  --env-gpu 0 \
  --policy-env auto \
  --eval-env <robodojo-conda-env> \
  --eval-num 1
```

`install.sh` delegates to the three upstream expert installers, which create
separate environments.  It also installs the XPolicyLab WebSocket dependencies
into the Python selected by `CSU_ROUTER_PYTHON`.  For a code-only CI check on a
machine where those dependencies and expert environments already exist, set
`CSU_SKIP_ROUTER_INSTALL=1` and `CSU_SKIP_EXPERT_INSTALL=1`.

`eval.sh` runs the router once to supply the environment client's required
action type and audit metadata.  The launched outer policy server independently
verifies and repeats the same deterministic route before starting the expert.

## Tests

```bash
python -m unittest \
  XPolicyLab.policy.CSU-AI-0.tests.test_model_contract \
  XPolicyLab.policy.CSU-AI-0.tests.test_ws_worker
```

The first test proves the official inheritance and method contract while
exercising the frozen router.  The second proves that every inner WebSocket
operation remains on its owning I/O thread.

## Training status

This submission is evaluation-only.  `QRouter` is frozen and reproducible from
the retained training artifacts; the three action experts use their upstream
training implementations and immutable checkpoints.  No inference-time
fine-tuning or post-hoc routing override is performed.
