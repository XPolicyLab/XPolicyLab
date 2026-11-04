# Era0-Preview debug verification

Verified on 2026-11-04 using the released model assets and the standard XPolicyLab websocket debug client. No simulator is required. Both runs used the unchanged debug client and the packaged model adapter, with its BF16 loading-order fix.

| Check | Result |
| --- | --- |
| Plain RGB observations | PASS: 10 rounds, 10 environments, 20 action steps per round |
| Encoded RGB observations (`DEBUG_OBS_ENCODED=1`) | PASS: 10 rounds, 10 environments, 20 action steps per round |
| Model-server startup, batch RPC, action dimensions and reset | PASS in both runs |
| Final launcher message | `[MAIN] eval finished` in both runs |
| Tracebacks | None in either successful run |

## Reproduce

From `policy/Era0_Preview`, after installing the policy environment and downloading the model:

```bash
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=0 HY_OBS_LAG=1 \
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

## Observation-delay regression

A separate CPU check exercised the real `Model.update_obs_batch` and `Model.reset` methods, the shared RGB decoder and the image-delay queue. Model construction and inference were omitted in this focused check; the GPU closed loops above loaded the actual weights. Distinct image values were used because the standard debug client's all-black images cannot reveal a temporal offset.

Both plain and encoded input cases passed:

- All three cameras use the preceding observation update; initial updates repeat the first image.
- State and language remain current.
- Environment queues stay independent when indices are reordered or temporarily absent.
- Mutating caller image buffers does not change stored snapshots.
- Reset clears delayed images and stored observations.
- Lag 0 uses current images; combining image delay with legacy render delay is rejected.

These checks verify transport and adapter behavior, not simulation success rates.
