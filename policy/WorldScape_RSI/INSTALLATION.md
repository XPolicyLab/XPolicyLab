# WorldScape_RSI runtime dependencies

The supported inference deployment is Linux with NVIDIA CUDA. CPU fixtures can
run independently on macOS/Linux without checkpoints or API credentials.

## Policy environment

1. Retain `policy/WorldScape_Policy_2` beside this adapter. The frozen manifests
   check 217 of its source files; the migration does not modify that policy.
2. Activate a dedicated conda environment and run `bash install.sh`. This installs
   XPolicyLab, the adjacent WSP dependencies and the Harness dependencies.
3. Follow the matching RoboDojo/CuRobo installation for CUDA runtime dependencies
   (including its Warp/native dependencies). The supplied CuRobo Python source
   overlay is not a replacement for that installation. GPU compatibility is not
   established by the CPU tests.
4. Set `ROBODOJO_ROOT` to a RoboDojo checkout with `Assets/Robots/x5/X5A.urdf`,
   `Assets/Robots/x5/curobo.yml` and their referenced robot assets. The official
   robot configuration also belongs in the parent workspace's `env_cfg/`, as
   required by XPolicyLab's shared dimension resolver.
5. Configure Astra and the two checkpoint paths as in README.md, then run
   `python preflight.py --check-checkpoints` before the real-resource debug loop.
   This verifies files only; it does not measure inference or task success.

The standard server launcher binds `CUDA_VISIBLE_DEVICES` from `policy_gpu_id`.
For example, exposing `0,1` allows the WAM bindings to use `cuda:0` and `cuda:1`
inside that process. Astra is an external API client; WAM inference is local to
that process. Two selected variants may occupy GPU memory concurrently.

## Bundled runtime source

`harness/vendor/runtime_support/` contains the 677 support source/assets files
from the supplied 20261015 delivery. The CuRobo Python tree is selected by
`register_vendor()` and rejects a previously loaded incompatible CuRobo module.
The native planner is imported under the policy-private package, and its two
local imports are relative, so it cannot accidentally import XPolicyLab's
unrelated top-level `utils` package. No simulator package is added globally to
`PYTHONPATH`. Robot planning uses the existing robot-only subclass; these support
sources do not grant access to scene state through the Agent tool interface.

No file under the original Desktop delivery folder is required after migration.
Model weights, provider credentials, full simulator assets and non-Python GPU
dependencies are still external. Their real-resource validation and public
checkpoint distribution remain release prerequisites.

## CPU verification

Use a fresh virtual environment and install `requirements-test.txt`; do not run
`install.sh` merely to execute the CPU tests. Run the two fixture commands in
README.md. `validation/requirements-lock.txt` records the exact dependency
versions used for this migration; it is not a Linux CUDA inference lockfile.
