---
name: xpolicylab-model-integration
description: Integrate a robot policy into XPolicyLab as a policy/<POLICY>/ adapter. Use when adding or updating a policy adapter, implementing model.py or deploy.yml, wiring install/process_data/train/eval scripts, or debugging debug-mode evaluation in the XPolicyLab repo.
---

# XPolicyLab Model Integration

Add each policy as a self-contained `policy/<POLICY>/` adapter; keep upstream model code unchanged where possible. `policy/demo_policy/` is the minimal reference — mirror it unless the model truly needs more.

## Before starting

Confirm with the user, or state the assumption explicitly in your first reply:

- Where the upstream model code and checkpoints live (repo URL, local path, or "not available yet").
- Which `env_cfg_type` (robot) and `action_type` (`joint` / `ee`) the model supports.
- Whether this is a full integration or eval-only (no `process_data.sh` / `train.sh`).
- Which conda/uv environment to use for the policy side.

## Workflow

1. Read `policy/demo_policy/` (`model.py`, `deploy.py`, `deploy.yml`, `eval.sh`, `README.md`), then the upstream model's inference API, dependencies, and checkpoint layout.
2. Scaffold: `bash scripts/create_policy.sh <POLICY>` (copies `demo_policy`).
3. Implement `model.py` first (contract below). The policy server imports `XPolicyLab.policy.<POLICY>.model`, so keep the directory importable (`__init__.py`). Put environment setup in `install.sh`; add `process_data.sh` / `train.sh` only if the model supports them.
4. Keep `deploy.py` aligned with `policy/demo_policy/deploy.py` unless the environment loop truly differs. Put runtime defaults in `deploy.yml`, keeping the whole key set from `policy/demo_policy/deploy.yml` — `policy_name` (equal to the directory name), `protocol: ws`, `host`, `port`, plus the per-run fields the setup scripts override. Keep a key even where the scripts already default it.
5. Debug without a simulator, from `policy/<POLICY>/`:

   ```bash
   export EVAL_ENV_TYPE=debug
   bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 <policy_env> base
   ```

   `<policy_env>` (arg 9) follows the adapter's own convention — a conda env name, `uv`, or an environment path; `policy/Pi_05`, for example, requires `uv` (see its README). The trailing `base` is the eval-env conda env (arg 10).

   Fix import, server-startup, action-key, and shape errors until the loop completes. Run once more with `DEBUG_OBS_ENCODED=1` so the debug client sends encoded camera colors and the server-side decode path is exercised.
6. Static checks from the repo root: `bash -n policy/<POLICY>/*.sh` and `python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py`.
7. Write `policy/<POLICY>/README.md`: install, data, train, and eval commands, supported `action_type` / `env_cfg_type`, checkpoint layout, known limitations.

## Model contract (`model.py`)

Define `class Model(ModelTemplate)`, importing `ModelTemplate` from `XPolicyLab.model_template`:

| Method | Contract |
| --- | --- |
| `__init__(model_cfg)` | `model_cfg` is `deploy.yml` merged with CLI overrides (`ckpt_name`, `action_type`, `env_cfg_type`, `seed`, `gpu_id`, ...). Load checkpoints and processors here. |
| `update_obs(obs)` / `update_obs_batch(obs_list)` | Store observation dict(s) for the next action call. |
| `get_action()` | Return one action chunk: `list[dict]` of numpy arrays. |
| `get_action_batch(env_idx_list=None)` | Batched chunks aligned with active env indices; fall back to the config batch size when `None`. |
| `reset()` | Clear model state between episodes. |

Action dict keys: dual-arm uses `left_arm_joint_state` / `right_arm_joint_state` plus `left_ee_joint_state` / `right_ee_joint_state`; single-arm drops the `left_` / `right_` prefix; `action_type=ee` replaces `*_arm_joint_state` with `*_ee_pose` as `[x, y, z, qw, qx, qy, qz]`.

Take dimensions from `get_robot_action_dim_info(env_cfg_type)` in `XPolicyLab.utils.process_data`, and register every new robot in **both** robot-info files — `AGENTS.md` at the repo root has the table and explains why the two `get_action_dim` entry points read different files. A private loader is acceptable only when it must run without importing XPolicyLab, and then it needs a configurable root plus a fallback, as `policy/AHA_WAM` does with `env_cfg_root`.

## Conventions

`AGENTS.md` at the repo root states the image, path, and dimension rules that apply to every change; it is always loaded, so follow it rather than re-deriving anything. The three that adapters get wrong most often: `model.py` never decodes (the server already did), offline code decodes **only** via `decode_image_bit` (the single supported decoder — image bits carry some inconsistency from earlier data versions; see README [Standard Data Formats](../../../README.md#decode-only-through-decode_image_bit)), and everything is RGB end to end — a `COLOR_BGR2RGB` added after a decode is always a bug, and the only exceptions are medium adapters around `cv2.VideoWriter` / `cv2.VideoCapture` and a documented `deploy.yml` opt-in like `policy/Dexora_1B`'s `input_color_order`.

Adapter-specific conventions on top of those:

- `eval.sh` positional args, same for all adapters: `bench_name task_name ckpt_name env_cfg_type action_type seed policy_gpu_id env_gpu_id policy_env_or_uv_path eval_env_conda_env`.
- **The importable root is `Path(__file__).resolve().parents[2]`** — the parent of the XPolicyLab checkout, since the server imports `XPolicyLab.policy.<POLICY>.model`. `parents[3]` is especially damaging: it silently shadows modules when inserted at `sys.path[0]`.
- **Resolve checkpoints with `XPolicyLab.utils.checkpoint_resolver`**: `resolve_checkpoint_root(model_cfg, checkpoints_dir)` covers the shared precedence (explicit `deploy.yml` path key, `ckpt_name` as a path, the concatenated run-dir name, then `checkpoints/<ckpt_name>`). Adapters with an extra naming layer build on `build_run_dir_name` / `candidate_checkpoint_roots` instead of starting over. Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`.
- Observations carry the language prompt under `instruction` (string; fall back to `instructions`). Poses are `[x, y, z, qw, qx, qy, qz]`.
- Trajectory HDF5 files store a singular `instruction` string and camera extrinsics as `extrinsic_matrix`; runtime observations use `extrinsics_matrix`.
- Full observation/trajectory format trees live in the repo README under "Standard Data Formats".

## Debug-mode troubleshooting

| Symptom | Usual cause |
| --- | --- |
| `ModuleNotFoundError` for the policy | Missing `policy/<POLICY>/__init__.py`, or `install.sh` did not install into the env passed as `<policy_env>`. |
| Client retries connect, then gives up | The server crashed during model loading — read the policy-server log, not the client log; model tracebacks are only printed server-side. |
| Action rejected for wrong keys/dims | Action dict keys must match `action_type` and the arm count; take dims from `get_robot_action_dim_info(env_cfg_type)`. |
| `FileNotFoundError` on `env_cfg/<type>.yml` or `_robot_info.json` | The adapter built the `env_cfg` path itself and pointed it inside the repo, or the robot is registered in only one of the two robot-info files. |
| Works plain, fails with `DEBUG_OBS_ENCODED=1` | `model.py` is decoding images itself, or assumes writable arrays — decoded arrays are read-only views, so copy first. |
| Call times out | Inference is slower than `request_timeout_s` (default 120 s); raise it in `deploy.yml`. |

## Reporting back

Finish with: what was created or changed, the exact eval command you ran and its result, supported `action_type` / `env_cfg_type`, expected checkpoint layout, and anything still unverified (e.g. training untested, no real checkpoint available). Never claim the debug loop passed if you could not run it.
