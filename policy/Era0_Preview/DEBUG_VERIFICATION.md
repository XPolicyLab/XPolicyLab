# Era0-Preview debug verification

Verified on 2026-11-04 using the released model assets and the standard XPolicyLab websocket debug client. No simulator is required. Both runs used the unchanged debug client and the packaged model adapter, with its BF16 loading-order fix.

| Check | Result |
| --- | --- |
| Plain RGB observations | PASS: 10 rounds, 10 environments, 20 action steps per round |
| Encoded RGB observations (`DEBUG_OBS_ENCODED=1`) | PASS: 10 rounds, 10 environments, 20 action steps per round |
| Model-server startup, batch RPC, action dimensions and reset | PASS in both runs |
| Final launcher message | `[MAIN] eval finished` in both runs |
| Tracebacks | None in either successful run |

## Current debug command

From `policy/Era0_Preview`, after installing the policy environment and setting `ERA0_CHECKPOINT_PATH`:

```bash
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=0 \
ERA0_EXECUTION_PROFILE=official \
  bash eval.sh RoboDojo stack_bowls era0-preview arx_x5 ee 0 0 0 .venv .venv
```

Repeat with `DEBUG_OBS_ENCODED=1`. Use a real XPolicyLab checkout inside the RoboDojo workspace so the standard robot configuration is available. The policy environment can serve as the debug-client environment.

Both successful logs end with:

```text
[TestEnv] Action Step: 19 / 20 (step_limit)
[TestEnv] Check Episode End: True
[TestEnv] Check Episode End: True
[TestEnv] Episode finished
[MAIN] eval finished
```

## Clean installation verification

On 2026-11-04 the corrected `install.sh` completed in a fresh Python 3.11 environment with system site packages disabled. Both CUDA extensions were built from source using CUDA 12.8 and GCC 11.4. The installer dependency/import checks passed, and FlashAttention and causal convolution matched PyTorch reference computations on an NVIDIA L20.

Using that new environment and the released weights, the unchanged official debug client passed one round of 20 action steps with 10 environments for each of plain and encoded RGB observations. This verifies the policy environment; the separate RoboDojo/Isaac Sim environment was not reinstalled.
