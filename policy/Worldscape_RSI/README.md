# Worldscape_RSI

**Contributor:** WorldScape RSI team | **Paper:** in preparation | **arXiv:** not available | **Original code:** included in this adapter.

Draft inference adapter for an episode-level learned selector between WorldScape WAM and an Astra visual controller. Our existing RoboDojo motion harness, Open task branches, task skills and ICL asset manifests are included. The classifier architecture is implemented; trained routing weights are not included. This draft is not a request to run the official leaderboard evaluation yet.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux / Python 3.11, using the adjacent WorldScape_Policy_2 GPU environment (see its installation guide). The policy server loads WAM lazily in-process. The separate RoboDojo simulation environment retains Isaac Sim / cuRobo. This adapter depends on the company dev branch WorldScape_Policy_2 adapter, including deploy_robodojo.yml and xpolicylab_adapter/.

```bash
cd XPolicyLab/policy/Worldscape_RSI
bash install.sh
```

No secrets, WAM weights or trained JEV weights are bundled. Do not replace the simulation CUDA stack with the policy-server dependencies.

## Data Processing

Eval-only draft; no LeRobot conversion or training-data processing entry is shipped. The referenced ICL consists of full decision demonstrations or ordinary official-video frames, not runtime hidden-state inputs. See `icl/manifest.json` for individual sources and hashes.

## Training

Training is pending. `jev/network.py` defines the local two-class PyTorch network (WAM / Astra). It is not TypeSafe Jev. Training code and the frozen checkpoint are planned as a follow-up to this initial review; no release date or performance claim is asserted. Eval-only status remains subject to maintainer agreement.

Instantiation initializes a network randomly, but the inference entry requires a checkpoint marked trained with matching feature version and class order. It never routes on random weights. Do not bypass the trained-checkpoint guard for evaluation.

## Evaluation

Supported target: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`, single environment per process. Native simulation verification is pending.

Set `DEXBOTIC_SECRETS` and `VLM_AGENT_PROFILE` on the policy server; set `WRSI_WAM_CHECKPOINT` to an absolute RoboDojo checkpoint directory on the policy server. Point `WRSI_JEV_CHECKPOINT` to the subsequently trained `jev.pt`. Both hosts must use identical adapter contents and configuration.

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>
# Configuration example, runnable only after weights / credentials / GPU dependencies are supplied:
bash eval.sh RoboDojo general_pickup wrsi_rc1 arx_x5 joint 0 0 0 wrsi_policy RoboDojo
```

Explicit development ablations: `WRSI_ROUTE_MODE=astra` or `wam`; default `learned` fails if weights are missing. WAM uses the adjacent adapter shared checkpoint resolver; WRSI_WAM_CHECKPOINT overrides ckpt_name. JEV uses its separate explicit checkpoint path. WAM camera order, checkpoint transforms and memory defaults come from the adjacent deploy_robodojo.yml, not the RoboTwin default.

### Custom deployment contract

Astra returns a command envelope from `get_action()` to this adapter's own `deploy.py`, not a generic list of joint actions. Our simulation-side harness resolves the commands through the current state and cuRobo, then calls `take_action`. WAM is loaded inside this policy server and accessed through wam_update_obs / wam_get_action / wam_commit_action / wam_discard_action. The client reports actual executed steps; interrupted execution discards pending predictions. Each episode resets before route selection. This divergence must be reviewed; the generic debug client is not evidence that this loop works. Batch methods fail explicitly.

## Model Assets

Skills and ICL manifests are included. Media is excluded from Git; restore it from the v2 ICL assets archive with `python import_icl_assets.py /path/to/Worldscape_RSI-ICL-assets-20261011-v2.zip`. The importer checks the archive SHA256 and every asset. A public asset download URL must be supplied before reproducible evaluation. WAM checkpoint setup and the trained JEV checkpoint distribution remain pending; this draft cannot yet reproduce a learned-router leaderboard result.

## Configuration

`deploy.yml` preserves the standard keys and our existing `vlm_agent` settings. `jev.mode` defaults to `learned`; `jev.checkpoint` defaults to `checkpoints/jev.pt`. `WRSI_JEV_CHECKPOINT` overrides that path. `WRSI_OUTPUT_DIR` sets per-episode output storage. See [integration notes](INTEGRATION_NOTES.md) for full configuration and data boundaries.

## Notes

No chunk-level handoff, terminal Astra takeover, online training or runtime native-score feedback is enabled. Original primitives were preserved. Offline unit tests are not a native simulator or official protocol certification. See [submission audit](SUBMISSION_AUDIT.md) for outstanding integration work.

## Balls success update (2026-10-11 v2)

Includes the own-success unified-v2 ICL (75 images), exact r23 task skill and shared main_route rotation guard from RSI-final-3 v13. 18 offline tests plus 8 guard cases pass. The source run succeeded at seed1/layout1 in 374/600 steps; this integrated package has not been run in simulation. The shared guard can affect other main_route rotation tasks; cross-task native regression remains pending. Other task skills and legacy Open implementations are unchanged. r24 remains archived outside the active release.
