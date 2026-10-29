# Submit to RoboDojo's leaderboard

RoboDojo's leaderboard runs submitted policies through XPolicyLab. This tutorial takes
the agent from development to a pull request. [Evaluation through XPolicyLab](../concepts/xpolicylab.md)
explains the parts involved.

The leaderboard's rules that shape a submission (protocol of 2026-10-14):

- one fixed system and inference procedure for every task; an agent inside it is
  allowed, a router that picks policies or checkpoints by task is not;
- RoboDojo scores the submission itself, with its official client: three evaluation
  seeds (mean ± std) on the public layouts, plus hidden verification layouts (a large
  gap invalidates the entry);
- what to submit: the inference code in XPolicyLab's standard (a pull request), the
  complete local results of `bash robodojo.sh summarize`, a short description of the
  approach and the model types it uses (here: agent, VLA, LLM/VLM), and the commitment
  to make code, configuration and checkpoints public within 3 days of listing;
- to be listed, the approach needs something new (an architecture or method, which may
  be a principled integration of existing models) and a paper or technical report, or
  a concrete plan for one.

## 1. Develop under the contract

The official client hands a policy less than our worker reports: no camera calibration,
grippers as their last commands, no step limit, no task name. The agent sees exactly
that by default (`configs/default.toml`):

```toml
--8<-- "configs/default.toml"
```

```bash
harness run configs/experiments/dev.toml --deploy mine --output runs/dev
```

Results on development layouts are what the exported policy should reach.

## 2. Choose the models and export

The export freezes the agent with its model. The default is DeepSeek V4.1 Flash
(`deepseek-flash` at `https://api.deepseek.com`). Commit and push, then:

```bash
harness export xpolicylab --name HarnessPolicy \
  --out <RoboDojo>/XPolicyLab/policy/HarnessPolicy --deploy mine
```

The agent is the submission as it is: no pointer role (the VLM, the main model itself,
points) and no world model, so the package needs the kinematics service, SAM 3 and
Simate, 23.2 GB, which one free 24 GB GPU holds resident. The evaluator sets
`DEEPSEEK_API_KEY`. `harness lock` prints the identity the package will
record before you export. Development runs of the agent have the same identity, so
their results are the package's. A `--set` or `--model` on the export changes it, as
on any run. (Agent 1.4.2 was exported with `--set 'roles=["main", "vlm", "verifier"]'
--set 'services.world_models=[]'`, which 1.5.0 made the default.)

The export refuses an agent that does not run under the contract or reads task
descriptions, one that does not resolve strictly, and uncommitted code (`--allow-dirty` overrides the last, for
trials). `--vendor` copies the files the agent was resolved from into the folder
instead of letting `install.sh` clone the commit: use it while the repository is
private, since the evaluators' `install.sh` cannot clone it.

## 3. Check

```bash
harness export xpolicylab --name HarnessPolicy \
  --out <RoboDojo>/XPolicyLab/policy/HarnessPolicy --deploy mine --check --gpu 5
```

XPolicyLab's PR checks: `bash -n` on every script, `py_compile`, no image decoding in
`model.py` or `deploy.py`, and ten episodes of XPolicyLab's debug environment with
scripted model replies (`configs/scripts/xpolicylab_check.json`). The package places its
services on the GPUs it is given by itself; one 24 GB GPU is enough (the models load on
demand).

## 4. Run the official client

With `<RoboDojo>/XPolicyLab/policy/HarnessPolicy` in place, RoboDojo's own client runs
it exactly as the evaluators will:

```bash
cd <RoboDojo>/XPolicyLab/policy/HarnessPolicy
export TMPDIR=/data/me/tmp            # Isaac Lab writes under $TMPDIR/isaaclab
EVAL_NUM=1 bash eval.sh RoboDojo general_pickup HarnessPolicy arx_x5 joint 0 5 5 harness-policy harness-policy
```

Arguments: bench, task, checkpoint name (not read), robot, action type, seed, policy
GPUs, simulator GPU, policy environment, simulator environment. `EVAL_NUM` caps the
episodes. The client reads layouts from
`Assets/Eval_Layout/RoboDojo/arx_x5/<seed>/<task>_<n>.json`; to check the wiring
without touching evaluation layouts, serve a development layout as an unused seed:

```bash
mkdir -p <RoboDojo>/Assets/Eval_Layout/RoboDojo/arx_x5/9000
ln -s <dev_layout_dir>/general_pickup_3.json <RoboDojo>/Assets/Eval_Layout/RoboDojo/arx_x5/9000/general_pickup_0.json
EVAL_NUM=1 bash eval.sh RoboDojo general_pickup HarnessPolicy arx_x5 joint 9000 5 5 harness-policy harness-policy
rm -r <RoboDojo>/Assets/Eval_Layout/RoboDojo/arx_x5/9000
```

Each policy-server instance writes its traces to `results/<task>-<time>-<pid>/` in the
policy folder (one directory per episode, as `harness run` writes them, but without the
outcome: the host scores); RoboDojo writes its videos and scores under `eval_result/`.

A check of agent 1.4.1's package this way (one development layout per task served as
seed 9000): `stack_bowls`, `general_pickup` and `make_toast` succeeded, `solve_equation`
did not; no request ran into the client's 120 s limit.

For the application, run one complete evaluation locally: every task and `_random`
variant of a pack at the official episode counts (`EVAL_NUM` unset), then summarize it
with RoboDojo's `bash robodojo.sh summarize`. RoboDojo runs the three seeds itself. A
policy server needs its services' GPU memory back before the next `eval.sh` starts on
the same GPU: wait until the previous instance's services have exited.

## 5. Submit

1. Open a pull request with `policy/HarnessPolicy/` against XPolicyLab. The folder pins
   the commit (`agent.json`, `install.sh`); its README follows XPolicyLab's adapter
   template.
2. Apply at robodojo-benchmark.com/eval: the pull request's URL and commit, the
   policy and checkpoint, the benchmark, tasks, robot and action space (`arx_x5`,
   joint), the environment variables and downloads the package needs
   (`DEEPSEEK_API_KEY`), the local results and the description of the approach.

The evaluators run `install.sh`, `download_checkpoints.sh`, set `DEEPSEEK_API_KEY`,
then `eval.sh`. At start the policy resolves `agent.json` again and refuses to run
unless it gets the identity that was exported.
