# XPolicyLab Runtime Audit

Audit date: 2026-07-06

Scope: current `XPolicyLab` checkout inside RoboDojo, commit `aa4d9c1`, the
configured `Luminis-Sim/XPolicyLab` and `Luminis-Sim/RoboDojo` remotes, all
top-level policy adapters, and the official/vendored upstream repos referenced
by policy READMEs.

## Executive Summary

The intended current architecture is:

```text
scripts/robodojo.sh eval
  -> XPolicyLab/policy/<POLICY>/eval.sh
  -> setup_eval_policy_server.sh -> XPolicyLab/setup_policy_server.py
  -> setup_eval_env_client.sh -> utils/setup_env_client.sh
  -> sim: RoboDojo/scripts/eval_policy.sh -> src/eval_client/main.py
  -> src/eval_client/eval_env.py -> XPolicyLab.policy.<POLICY>.deploy
```

The default transport is websocket (`protocol: ws`) for every top-level
`deploy.yml`. `legacy_tcp` remains available only as an explicit compatibility
mode.

Before this audit, the policy server and debug client had moved to websocket,
but the RoboDojo sim client still constructed a legacy-style `ModelClient` from
a non-existent `XPolicyLab.client_server.model_client` module. Full sim eval was
therefore blocked even though `EVAL_ENV_TYPE=debug` could work.

## Critical Findings

| Severity | Area | Finding | Resolution |
| --- | --- | --- | --- |
| Critical | Sim communication | `src/eval_client/eval_env.py` imported `XPolicyLab.client_server.model_client` and called `ModelClient(host, port)`, but the current default client is `WsModelClient(url, evaluation_id, trial_id, ...)`. | Fixed by protocol-aware client construction. |
| Critical | Python path | Sim eval only added RoboDojo root to `PYTHONPATH`; websocket modules also need the XPolicyLab root because they import `client_server.*`. | Fixed in `scripts/eval_policy.sh`. |
| High | CLI args | `scripts/eval_policy.sh` parsed trailing `extra_args` and then reset them to an empty array, silently dropping them. | Fixed by removing the reset. |
| High | README runtime contract | Root README implied standalone XPolicyLab can run sim eval; sim eval requires the parent RoboDojo checkout, `env_cfg/`, `scripts/`, `src/eval_client/`, and tasks. | Fixed in root README. |
| High | Policy docs | Several uv-based policies documented conda envs for arg 9, causing users to pass the wrong runtime handle. | Fixed for highest-risk uv adapters; remaining lower-risk env-name drift is documented below. |

## Current Data And Eval Paths

Most policy adapters follow:

```text
process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> ...
  -> data/checkpoint layout under policy-specific folders
train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu>
  -> checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed>
  -> websocket policy server + RoboDojo env client
```

Known intentional exceptions:

| Policy | Exception |
| --- | --- |
| `Dexora_1B`, `Spatial_Forcing` | Eval-only wrappers; no standard top-level training flow. |
| `Hy_Embodied_05_VLA` | Uses upstream Hy training scripts and uv; default action type is `ee`. |
| `EventVLA`, `starVLA`, `LingBot_VA` | Proxy additional upstream inference services. |
| `Mem_0` | Optional planning server and M1/Mn task modes. |
| `RISE` | Multi-stage advantage/policy/all training. |
| `Pi_0`, `Pi_05`, `Pi_0_Fast`, `GalaxeaVLA`, `Spatial_Forcing` | uv-style policy runtime paths. |

## Difference From `aa4d9c1`

Current XPolicyLab is `a15adcd7` on `main`. Compared with `aa4d9c1`, the branch
made these contract-level changes:

- `dataset_name` was renamed to `bench_name` across scripts and configs.
- Eval routing moved from YAML fields to `EVAL_ENV_TYPE`.
- Transport moved from `robodojo_ws` / eval-station to `client_server/ws` with
  `protocol: ws`.
- eval-station daemon/web integration was removed.
- The standard eval script signature was unified to ten positional args.
- Several unfinished adapters were removed and `Spatial_Forcing` was added.

Parent RoboDojo is on `c554e137` and has a dirty working tree. Its recorded
XPolicyLab submodule pin lags behind the checked-out `a15adcd7` state, so
submodule pinning must be updated before release if this integration is meant
to be reproducible from a clean clone.

## Official-Origin Comparison

The upstream comparison shows that XPolicyLab is intentionally a wrapper layer:
it should not copy upstream train/eval commands verbatim, but its README must
accurately describe where it diverges.

| Policy group | Upstream observation | XPolicyLab implication |
| --- | --- | --- |
| OpenPI (`Pi_0`, `Pi_05`, `Pi_0_Fast`, `Spatial_Forcing`) | Official OpenPI uses project-local uv workflows. | XPolicyLab arg 9 should be `uv` or the OpenPI project path, not a conda env. |
| `Hy_Embodied_05_VLA` | Official repo uses `uv sync`, `scripts/train_robotwin_umi.sh`, released `norm_stats.pkl`, and relative EE action chunks. | XPolicyLab README must document uv, upstream training pass-through, `ee`, and `ckpt_path`. |
| `MolmoACT2` | Official repo uses LeRobot + uv and FastAPI-style upstream servers. | XPolicyLab may wrap it behind `ws`, but docs must mention `uv`/path support. |
| `Dexora_1B` | Official repo has multi-process ZMQ real-robot deployment and staged training. | XPolicyLab eval-only adapter and absolute checkpoint config are intentional but must be treated as local setup requirements. |
| `DreamZero` | Official repo already has a websocket inference server and distributed GPU inference. | XPolicyLab websocket wrapper is aligned; timeout notes remain relevant for slow first inference. |
| `AHA_WAM`, `X_WAM`, `RISE`, `Xiaomi_Robotics_0` | Official repos use their own config systems and benchmark-specific eval managers. | XPolicyLab wrappers are expected to be divergent; README examples should avoid implying upstream paths exist under standard checkpoint names unless wrapper scripts create them. |
| `ACT`, `DP` | Official repos are conda/pip baselines with their own sim envs. | XPolicyLab standard wrapper is acceptable as long as data conversion and checkpoint naming are clear. |
| Vendored-only adapters (`A1`, `Being_H05`, `LDA_1B`, `GO1`, `Mem_0`, `X_VLA`, `LingBot_*`, `Motus`, `InternVLA_A1`, `Dexbotic_DM0`, `GigaWorldPolicy`) | Local vendored READMEs are the practical source of truth. | No network-origin contradiction found in the outer wrapper; env-name examples still need periodic cleanup. |

## Policy-Specific Reproducibility Risks

These are not blockers for the shared websocket/sim eval plumbing, but they can
break first-run install, training, or evaluation for individual adapters.

| Severity | Policy | Risk |
| --- | --- | --- |
| High | `H_RDT` | `train.sh` hardcodes an internal RoboDojo dataset root instead of deriving it from the wrapper args or an env var. |
| High | `starVLA` | `train.sh` only passes `run_id`; actual dataset location remains in `xpolicy_oft_vla.yaml`, so `process_data.sh` output is not automatically wired into training. |
| High | `EventVLA` | `process_data.sh` downloads/uses a fixed RoboTwin-Mem-style dataset and ignores `expert_data_num`; `deploy.yml` also contains an absolute checkpoint path. |
| High | `AHA_WAM` | Training/eval scripts depend on site paths and an Apptainer image default; README does not fully state the portability requirement. |
| High | `Dexora_1B` | No vendored Dexora source is present even though the README points users to upstream docs; `deploy.yml` uses local absolute paths. |
| High | `LingBot_VA` | `deploy.yml` and backend startup use site-specific checkpoint/base-model paths; README omits the separate Wan-VA backend contract. |
| High | `Motus` | `deploy.yml` and `model.py` default to private `/mnt/xspark-data/...` model/cache paths. |
| High | `InternVLA_A1` | `deploy.yml` defaults to private model paths; README does not call out all required assets or the current `joint`-only adapter assumption. |
| High | `Dexbotic_DM0` | `process_data.sh` defaults `DM0_RAW_DATA_ROOT` to a private `/vepfs...` path. |
| Medium | `Spatial_Forcing` | README should describe the adapter as a Spatial-Forcing/OpenPI fork integration, not plain upstream OpenPI. |
| Medium | `LingBot_VLA` | `deploy.yml` has a non-importable default `policy_name: lingbot-vla`; wrapper scripts override it, but direct deploy config use violates the shared contract. |
| Medium | `Mem_0` | Split server/client docs mention `planning_gpu_ids`, but only `eval.sh` starts the planning server path. |
| Medium | `LDA_1B` | Top-level README has stale official-origin metadata; installation docs identify `jiangranlv/latent-dynamics-action` and arXiv `2602.12215`. |
| Medium | `GigaWorldPolicy` | Generic train README hides required upstream assets (`GIGAWORLD_PRETRAINED_PATH` or `WAN22_DIFFUSERS_PATH`) that are only in installation notes. |
| Medium | `Xiaomi_Robotics_0` | Generic examples say `joint`, while deploy defaults and upstream XR-0 action chunks are `ee`-style. |
| Low | `TinyVLA` | `train.sh` can prompt interactively for pretrained VLM weights; README should document that first-run prompt or add noninteractive env vars. |
| Low | `Abot_M0` | Official URL now lands on ABot-M0.5 docs; README should pin the ABot-M0 branch/origin for accurate comparison. |
| Low | `A1` | Upstream deployment is HTTP/API-server based; XPolicyLab websocket wrapping is intentional but should be called out explicitly. |

## Remaining Risks

- Many policy README examples still use generic checkpoint names such as
  `RoboDojo-cotrain-arx_x5-joint-0`. That is fine for standard wrappers but
  misleading for policies with released upstream checkpoints or proxy servers.
- Some `deploy.yml` files contain developer-specific absolute paths
  (`EventVLA`, `Dexora_1B`, `Spirit_v15`, `LingBot_VA`, `Motus`,
  `InternVLA_A1`, and others listed above). These cannot run on a clean machine
  until users edit paths or download the stated assets.
- `demo_policy/eval.sh` still relies on a short sleep before the client instead
  of `wait_for_policy_server.sh`, so it can race on slow machines.
- Full sim eval still depends on Isaac/conda/assets/checkpoints being present.
  A passing shell exit alone is not sufficient; verify `_result.json` has
  `eval_time >= 1`.

## Validation Checklist

Minimum pre-release checks:

```bash
git diff --check
bash -n scripts/eval_policy.sh
bash -n XPolicyLab/utils/setup_env_client.sh
bash -n XPolicyLab/utils/run_sim_env_client.sh
python -m py_compile src/eval_client/main.py src/eval_client/eval_env.py
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
EVAL_ENV_TYPE=debug bash XPolicyLab/policy/demo_policy/eval.sh RoboDojo stack_bowls ckpt arx_x5 ee 0 0 0 <policy_env> <eval_env_conda_env>
```
