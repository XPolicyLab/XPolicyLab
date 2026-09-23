## Policy
- Name: EchoPolicy
- Supported: bench_name=RoboDojo, env_cfg_type=arx_x5, action_type=joint
- Training support: eval-only; data conversion and training scripts are not included.
- Base policy: the existing XPolicyLab Pi05 adapter, with optional VLM subgoal planning.

## Components
- [x] install.sh reusing the Pi05 OpenPI environment
- [x] model.py + __init__.py
- [x] deploy.yml with protocol: ws
- [x] deploy.py aligned with Pi05
- [x] eval.sh + setup_eval_policy_server.sh + setup_eval_env_client.sh
- [x] policy README
- [x] Checkpoint resolution through the standard XPolicyLab Pi05 adapter
- [x] Download helper supports a mounted checkpoint directory or an approved Hugging Face repository

## Testing
- [x] bash -n + py_compile
- [x] Canonical observation mapping smoke test
- [x] Passthrough strategy smoke test
- [ ] EVAL_ENV_TYPE=debug full closed loop with Pi05 checkpoint
- [ ] Simulator evaluation using this adapter revision

## Checkpoint
The XPolicyLab Git repository contains the Pi05 loader and model code, but does not
contain the RoboDojo fine-tuned `59999` checkpoint. `download_checkpoint.sh`
supports `ECHO_POLICY_CHECKPOINT_PATH` for an externally mounted checkpoint or
`ECHO_POLICY_CHECKPOINT_REPO` for a maintainer-approved public Hugging Face repo.
No public checkpoint URL is claimed by this PR. Official leaderboard evaluation
requires maintainers to approve a reproducible checkpoint source.

## Limitations / notes
- This is an eval-only adapter. Training code/data release plans are not specified in this PR.
- VLM credentials are injected at runtime and are not part of this repository.
- Maintainer confirmation is requested for external VLM access during official evaluation.
- No official score is claimed by this PR yet.
