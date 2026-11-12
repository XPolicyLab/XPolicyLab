# Official CLI simulator smoke test

Verified on 2026-11-04 with the public Era0-Preview model assets, the packaged adapter and RoboDojo's `scripts/robodojo.sh eval`. This limited seed-0 test checks the evaluation chain, not benchmark performance.

| Task | Episodes | Successes | Score | Camera videos |
| --- | ---: | ---: | ---: | ---: |
| press_by_number | 2 | 2 | 100.0 | 6 |
| stack_bowls | 2 | 2 | 100.0 | 6 |

Both commands exited with status 0 and reached `[MAIN] eval finished`. Each result contains two episode records; all twelve saved camera videos decode and contain multiple nonblank frames. The SDK summary command completed but skipped both short runs, including its media audit, because they do not meet native-count thresholds. A separate check using that same SDK script's record/video inventory helpers verified all four records and twelve camera videos with zero missing or extra episodes.

The run used the official execution profile, SDK rendering and the policy's default image-observation delay. Installed SDK source files were unchanged. A nonfatal Isaac Sim headless property-window callback exception appeared during initialization; both tasks continued to completion.

## Reproduce

From the RoboDojo root, after installing the adapter and downloading its model:

```bash
ERA0_EXECUTION_PROFILE=official ROBODOJO_NUM_ENVS=2 \
  bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Era0_Preview \
  --task press_by_number --ckpt era0-preview \
  --env-cfg arx_x5 --action-type ee --seed 0 --eval-num 2 \
  --policy-gpu 1 --env-gpu 2 \
  --policy-env XPolicyLab/policy/Era0_Preview/.venv --eval-env RoboDojo
```

Repeat with `--task stack_bowls`. Choose available GPU IDs and the installed simulator environment. Results and videos use the SDK's native directories.
