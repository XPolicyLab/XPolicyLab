# RSIWAM

**Contributor:** RSIBot | **Paper:** Technical report planned for 2026-11-25 | **arXiv:** Not available | **Original code:** [RSIWAM integration and source provenance](MODEL_CARD.md)

RSIWAM is a recovery-adapted robot policy that combines a frozen world-model foundation, a gated recovery module, and success133 Action LoRA. This **eval-only** RoboDojo adapter supports `env_cfg_type=arx_x5`, `action_type=ee`, absolute world-frame poses and native gripper values. `bundle/` contains the vendored runtime, observation-history wrapper, identity-checking loader, exact learned adaptation weights and upstream notices; the [model card](MODEL_CARD.md) documents component provenance and the fixed evaluated candidate.

The [method review note](checks/method_review_20261102/METHOD_REVIEW_NOTE.txt) explains the frozen-branch gradient coupling, fixed inference settings and supervision scope. It includes a later partial paired development comparison, clearly separated from the official local summary; neither architectural novelty nor certified recovery capability is assumed from that comparison.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Place this complete directory at `XPolicyLab/policy/RSIWAM`; no external symlink or machine-specific source path is required. Use a dedicated policy conda environment. Install the matching CUDA PyTorch/torchvision pair first, then run the installation script in that environment:

```bash
conda activate <policy_env>
cd XPolicyLab/policy/RSIWAM
bash install.sh

# Example environment name:
conda activate rsiwam-policy
bash install.sh
```

`install.sh` requires both `torch` and `torchvision` to be installed first. It reads their installed versions without importing the GPU runtime and pins both versions for every pip invocation. The bundled requirements also constrain every invocation, including XPolicyLab's dependencies, so installing the adapter cannot replace a policy dependency with an incompatible version such as NumPy 2.x. It installs the bundled requirements, the local `recoverwam` source in editable mode, and XPolicyLab in the selected environment. Keep the adapter directory in place after installation. The simulator environment is separate.

Optional site constraints can be supplied with `RSIWAM_CONSTRAINTS=/path/to/constraints.txt bash install.sh`. The file must exist before installation starts; constraints that conflict with the preinstalled pair cause pip to fail. The installer does not select a CUDA build or certify driver compatibility.

The historical H800 service recorded `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130`; `bundle/constraints-reference.txt` preserves that partial snapshot and is not applied by default. The recorded A800 GPU debug instead used `torch==2.5.1+cu118`. These are environment observations, not a complete transitive lock or a clean-install guarantee.

Real installation validation at commit `776c9f3b177427c1078ee31daaa4db93f922008b` passed in a newly provisioned isolated Python 3.10 environment: all 27 bundled dependency constraints, `pip check` and CPU imports passed. This was a corrective reinstall in that environment, not a one-shot install of this revision from an empty environment. The [installation evidence](checks/clean_install/SUMMARY.json) includes the final package versions and unchanged logs, including the earlier NumPy constraint failure's recorded scope.

## Data Processing

Unsupported in this eval-only adapter; `process_data.sh` is intentionally absent. Evaluation consumes XPolicyLab observations with the three RGB camera arrays and the registered robot state. The model does not decode image bytes or swap color channels. It preserves the native views for the frozen runtime's camera composition.

Training used the `success133-ee16-rgb-lossless-train-v2` prepared export, with native EE16 action/provenance fields rather than the official XPolicyLab converter output. This adapter has no dataset conversion entrypoint; the separate training source/configuration artifact records the export and its masks. The bundled training/data utilities are preserved provenance and are not invoked by evaluation. The eval-only exception and the publication timeline are disclosed in the submission PR; acceptance remains subject to maintainer review.

The [training supplement](https://github.com/modaxiansheng/XPolicyLab/releases/tag/rsiwam-training-supplement-20261102) supplies original-byte dataset archive verification/extraction, exact split/mask/sampling contracts and a separate local training entry. Its data preparation does not alter camera RGB or replace pretrained normalization with dataset statistics.

## Training

Unsupported in this adapter; `train.sh` is intentionally absent. No training or weight update occurs during evaluation. The frozen candidate has 119 original optimizer steps plus 281 exact resumed steps, for step 400. The [public training-source companion](https://github.com/modaxiansheng/XPolicyLab/releases/tag/rsiwam-training-companion-20261101) documents that lineage, masks and contracts, with explicit omissions and external dependencies; it is not a verified end-to-end retraining package. The team has approved releasing the evaluated training/deployment code, exact weights, configuration and reproduction instructions within three days of verified leaderboard listing. The planned technical-report date is 2026-11-25 and is a separate deadline. This repository contains the RSIWAM evaluation code and small adaptation weights; complete evaluation results remain pending.

The [2026-11-02 training supplement](https://github.com/modaxiansheng/XPolicyLab/releases/tag/rsiwam-training-supplement-20261102) adds `train_portable.py`, fixed configuration, complete-state resume checks and 257 byte-preserved historical source files. Ten focused CPU check groups passed. The separate entry starts a new 400-update execution from the published frozen parent; it does not reconstruct the historical 119-online/281-cached handoff or claim byte-identical retraining. No GPU training was executed for this new port. Historical optimizer/RNG/cache inputs and parent-data dependencies remain explicitly documented rather than synthesized. The evaluated weights are unchanged.

## Evaluation

Prepare the base assets described below, then use the standard XPolicyLab argument order:

```bash
cd XPolicyLab/policy/RSIWAM
export RSIWAM_BASE_WEIGHTS="$PWD/checkpoints/rsiwam-original32-lora33-step400"
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Debug command to run after installation and base-weight provisioning:
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls rsiwam-original32-lora33-step400 arx_x5 ee 7 0 0 rsiwam-policy robodojo-eval
# Repeat with encoded camera transport to exercise the shared server decoder:
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls rsiwam-original32-lora33-step400 arx_x5 ee 7 0 0 rsiwam-policy robodojo-eval
```

Real CUDA debug validation passed for commit `7e5def6e61435e823f5b346f2b9d2dd5a7a78256`: the official plain and encoded debug clients each executed one 20-action episode, and their returned action chunks were byte-identical. The [selected evidence](checks/gpu_debug/SUMMARY.json) records the exact candidate and limits. These runs reused an existing environment through an observational harness; the literal `eval.sh` launcher and clean installation were not validated by that harness. The commands above remain the documented standard launch path. A later literal plain run at `776c9f3b177427c1078ee31daaa4db93f922008b` completed 10 debug episodes / 200 actions using the isolated r5 environment; its original overall attempt failed during resource-lease cleanup, and encoded did not start in that attempt. The [unchanged plain logs and failure record](checks/standard_launcher_plain/SUMMARY.json) preserve both outcomes. The later [independent cleanup closure](checks/standard_launcher_cleanup/SUMMARY.json) reconciled the exact orphan reservation while preserving the original failure. The subsequent [literal encoded continuation](checks/standard_launcher_encoded/SUMMARY.json) completed 10 debug episodes / 200 actions at the same executable revision, reached `[MAIN] eval finished`, exited 0 without a traceback, and released its resources. Both standard debug modes are therefore accepted across the preserved runs and the independent plain cleanup closure. For simulator evaluation use `EVAL_ENV_TYPE=sim`; for split-machine deployment, use the shared deployment guide.

The standard positional `seed` controls environment evaluation and checkpoint-directory resolution. The model's sampling seed is separately fixed at `model_seed: 1`; changing the evaluation seed never changes the candidate or its inference seed. The server launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts. Direct server launches must set the same variable.

The [partial native evaluation attachment](checks/native_partial_325/README.txt) preserves 325 accepted episodes from seven completed task blocks at evaluation seed 0, with original results, candidate identity records and existing video SHA references. Their episode-weighted SR is 137/325 (42.15%) and mean Score is 45.38/100. This is the set of completed blocks at a fixed progress snapshot, not a representative sample or the complete 6,300-episode benchmark. Pending blocks are excluded; no paired baseline or recovery-specific success rate is claimed. These native runs used the recorded queue and model constructor, separately from the literal standard-launcher checks above.

The unchanged official `bash scripts/robodojo.sh summarize` output and raw-result snapshot are available in [official local summary](checks/official_summary_20261102/README.txt). At its 2026-11-02 13:55:54 SGT cut, 6/126 task-seed cells are complete (300 episodes, evaluation seed 0 only); the additional accepted 25-episode standard half does not complete its paired Generalization task. The partial dimension-balanced Overview is SR 52.25% / Score 55.55. These figures use a different aggregation from the 325-episode pooled result above. Neither is a complete three-seed benchmark or an official leaderboard score. Full local evaluation is still in progress.

[Original video archives](https://github.com/modaxiansheng/XPolicyLab/releases/tag/rsiwam-eval-videos-20261102) are being delivered by complete task/seed block. The first published block contains all 150 original three-camera videos for 50 `align_blocks` episodes, including failures. Per-video manifests preserve the recorded SHA-256 and the frozen result reference; copied video bytes and the finished ZIP were verified without re-encoding. The release inventory states the actually available blocks. Other videos from the 325-episode snapshot remain in transfer; this is not complete video delivery for the 6,300-episode campaign.

## Model Assets

The recovery parent and step-400 LoRA are already included and verified in place:

| Asset | Location under this directory | SHA-256 |
| --- | --- | --- |
| Frozen recovery parent | `bundle/weights/parent/adapter_final.pt` | `861529eb89e4279e4490d6156bf04a335eee410f6c02842d0ac3db2dc90431d3` |
| Action LoRA step 400 | `bundle/weights/step000400/adapter.pt` | `eec21ec2834ef95bffdac7b3df0b48092a215aa67e7bf6952ec1643ab511b361` |

The large base weights are external. `bundle/base_weights.json` pins repository `Haodong082399/VPP2`, revision `04824ddb3aae11f13db50984ed7d540c3b77dd9f`, every required file hash, and the base manifest identity `ae66dbfc16eb3c75fa89111501412ecb14cd3ab62ce02ff9fc8a457bb3b19aba`. Preserve the listed layout under the selected base root:

```bash
# Inspect the plan: no writes, model loading or network access.
python3 download_checkpoints.py --destination "$PWD/checkpoints/rsiwam-original32-lora33-step400"

# Fetch missing base assets and verify every file before installing it.
python3 download_checkpoints.py --destination "$PWD/checkpoints/rsiwam-original32-lora33-step400" --download

# Alternatively, stage from an existing base cache without network access.
python3 download_checkpoints.py --destination "$PWD/checkpoints/rsiwam-original32-lora33-step400" \
  --cache-dir /path/to/existing/checkpoints --download --offline
```

The script verifies the fixed manifest and both included adaptation weights. Existing correct base files are reused; mismatched files cause an error and are left untouched. `BASE_ASSET_URL_AUDIT.json` records HTTP 200 for all 11 pinned URLs and matching downloaded hashes for six small files. Five large files were checked by HEAD only, so this audit does not certify their remote bytes; the download script performs that check during staging.

```text
<base-root>/
├── joint2b_s100000/{action.pt,video.pt,dataset_stats.json,manifest.json}
└── Wan2.1-I2V-14B-480P/
    ├── Wan2.1_VAE.pth
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── google/umt5-xxl/{special_tokens_map.json,spiece.model,tokenizer.json,tokenizer_config.json}
```

`rsiwam_weights_root` in `deploy.yml` takes precedence over `RSIWAM_BASE_WEIGHTS`. The adapter uses XPolicyLab's shared checkpoint resolver for these roots, the other standard explicit path keys, `ckpt_name` as a path, and conventional run directories. Relative paths resolve against `policy/RSIWAM`. With no explicit path, the normal run directory is tried before `checkpoints/<ckpt_name>`; the latter is convenient for one fixed checkpoint across evaluation seeds. Every selected base file is rehashed before model construction. Every bundled parent/LoRA and source identity is also checked; a different checkpoint is rejected regardless of its directory label.

The base weights are not redistributed in this directory. Keep all upstream notices and observe the upstream asset terms. The public model identifier is **RSIWAM**, and the fixed checkpoint display name is **rsiwam-original32-lora33-step400**; names do not override the component identities or hashes.

## Configuration

The full standard `deploy.yml` key set is present. Additional frozen settings are `model_seed=1`, `deployment_adapter=robodojo_ee16`, `num_inference_steps=10`, `sigma_shift=1`, `replan_steps=24`, `action_horizon=32`, `mixed_precision=bf16`, and `device=cuda`. `rsiwam_weights_root` selects the external base root; `default_instruction` is used only if no instruction is supplied. `request_timeout_s=1800` allows a slow policy request. Evaluation batch mode is explicitly unsupported.

The runtime consumes dual-arm `[x,y,z,qw,qx,qy,qz,gripper]` values. The top-level adapter obtains arm and gripper dimensions from `get_robot_action_dim_info`, validates the supported two-arm contract, and derives packing/slicing from it. Camera arrays remain RGB. The deployment loop records one observation per executed action, carries a client identifier, and resets the selected client's history per episode. It stops as soon as the episode ends, including inside an action chunk.

## Notes

- The nested `bundle/policy/RSIWAM` entry and its symlink instructions are historical provenance. Use this directory's top-level scripts and model instead.
- `checks/check_adapter_contract.py` checks the copied adapter import path, robot metadata and action packing, environment/model seed separation, per-action history, client isolation/reset, terminal handling, and shell arguments using CPU fixtures. These are boundary tests, not neural-network output parity tests.
- The original package's `bundle/checks/verification.json` records real parent/LoRA loading into a CPU structural skeleton. This round preserves and rehashes those files; it does not repeat that tensor test.
- Real full-model CUDA loading and plain/encoded synthetic debug passed at the recorded commit. The separate isolated-environment installation record above establishes dependency compliance after a corrective reinstall. A native RoboDojo evaluation using the same model constructor and checkpoint has started. Both literal standard `eval.sh` debug modes completed 10 episodes / 200 actions each; the original plain cleanup failure and its later independent closure are retained alongside the successful encoded continuation. Full-episode parity with the historical native implementation and the complete 6,300-episode suite remain unverified or unfinished. Synthetic debug is interface evidence, not a recovery-success measurement or a leaderboard result.
