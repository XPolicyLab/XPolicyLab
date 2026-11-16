# EmbodiedEvo policy for RoboDojo

This adapter connects a persistent Python coding agent to the native RoboDojo
evaluator through XPolicyLab WebSockets. `policy/EmbodiedEvo/model.py` implements
the policy interface, and `deploy.yml` registers the `EmbodiedEvo` policy name.
The native client owns physics, action limits, success checks and official
result files. This is local evaluation support, not an organizer-verified submission.

## Evaluate frozen knowledge locally

Provision the simulator and policy environments as described in
[installation](../../docs/INSTALL.md). Select a completed Playground run with
an intact `protocol.json` and `frozen_harness/`:

```bash
.venv/bin/python -m integrations.robodojo_policy.run \
  --source-run runs/YYYYMMDDHHMM_stack_bowls \
  --env-task stack_bowls --seed 1 --episodes 1 --gpu 0 \
  --run-dir "runs/$(date +%Y%m%d%H%M)_stack_bowls_native"
```

Replace the source-run placeholder with your actual frozen run. The adapter pins
its source and frozen knowledge before native scene selection. Each trial gets
a new agent and Python worker; only Skills/Lessons transfer. Test reset remains
forbidden. Preserve native reports and videos under `official/eval_result/`.
A one-episode trial is a subset, not a full leaderboard result.

## Optional image runtime in an existing Slurm allocation

The shared single-scene executor in `execution.py` supports native Playground,
Test and baseline sessions through an independently supplied Apptainer SIF.
You must provide a compatible image, its repository-local source/assets, and
the XPolicyLab policy environment. No image, site profile or node access is
included in this export.

Copy `configs/runtime.example.json` to a local ignored profile, replace all
placeholder paths and the full image ID, and set:

```bash
export EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE="$PWD/configs/runtime.local.json"
```

Then the normal `embodiedevo explore`, `test` and `run` commands use the native
XPolicyLab executor when the profile contains `sif`. Launch inside a Slurm GPU
allocation; pass a physical GPU index present in `SLURM_JOB_GPUS`. The profile's
`python`, `source_root` and `kit_ext_cache` must match the paths inside your image.
The source and assets live in this checkout's real `vendor/robodojo/` tree, never
a link to another project. `material_mirror` is optional; offline image material
lookup must already work if it is omitted.

The four-task diagnostic entry is also available:

```bash
.venv/bin/python -m integrations.robodojo_policy.debug \
  --run-dir "runs/$(date +%Y%m%d%H%M)_native_debug" \
  --sif /path/to/robodojo.sif --per-gpu 1
```

Add `--probe` for interface diagnostics without a model. Repeated camera/reset
checks use `python -m scripts.smoke_robodojo_capture --help`. These commands
require the same prepared runtime. They do not submit Slurm jobs or choose nodes.

## Transport and compatibility

Each `step(action)` yields through a host pipe to XPolicyLab, then resumes after
receiving the native observation. Agent thinking does not advance physics.
The deployment loop forwards native termination and binary success through
trial lifecycle hooks; private shaped rewards and scene layouts stay host-side.

The bootstrap uses the repository policy namespace without editing the vendor
tree. It supplies run-local logging paths and retains the adapter's reset and
camera cleanup hooks. The image route isolates simulator networking and bridges
unchanged WebSocket traffic through a Unix socket. The agent container has no
simulator or policy-server socket mounts.

Source, task and native report provenance must accompany any reported score.
Adapter failures are recorded, never silently replaced with another model attempt.
Classify them from evidence before determining an experimental denominator.
