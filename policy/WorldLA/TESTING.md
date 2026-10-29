# WorldLA adapter verification — 2026-10-22

This record describes pipeline validation, **not** a full RoboTwin leaderboard run.

## Revisions and scope

- XPolicyLab base: `aa46382aa12dc46b360e2fdc63d0d3d39215b3e9`.
- Native RoboTwin simulator/evaluator: `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`.
- WorldLA core: matches public `d84647cb4f412af1a9b852146073e42b4e3b76ea`;
  model/policy/training/fusion code was not changed for this adapter.
- Weights: `original/` full export from Hugging Face revision
  `7fd1111c6fa8419a64bb6f26383b5025ffbf436d`.
- Hardware: one idle A100 80 GB, physical GPU 7; SAPIEN PCI `0000:e0:00.0`.
- Existing Python 3.11 robotics runtime was reused. Missing transport support
  previously installed in an isolated temporary target was supplied through
  `PYTHONPATH`; no running shared environment was upgraded.

## Completed checks

| Check | Result |
| --- | --- |
| Required files, YAML keys and ModelTemplate lifecycle | Pass |
| Shell syntax and Python compilation | Pass |
| Ruff lint | Pass |
| Image-operation audit | No adapter decoding, encoded-image writing or RGB/BGR swaps |
| New CPU contract tests | 13 passed |
| Existing WorldLA source regression suite | 51 passed |
| Actual full-checkpoint debug, decoded RGB | Exit 0; `[MAIN] eval finished` |
| Actual full-checkpoint debug, `DEBUG_OBS_ENCODED=1` | Exit 0; `[MAIN] eval finished` |
| Native simulator through new standard `eval.sh` | `press_stapler/demo_clean`, seed 100000: 1/1 success, action 111; exit 0 |
| Cleanup | All temporary policy/client processes exited; GPU 7 returned to 1 MiB |

The two debug loops each ran the official debug client's ten 20-action
episodes with the real model, not a zero/dummy policy. These episodes test
IO/lifecycle only and are not task-success measurements. CPU tests exercise
batch reorder/shrink, isolated plan/RNG state, reset, invalid inputs and
keyword-compatible deploy callbacks; they use a deterministic fake policy.

## Commands

In `policy/WorldLA`, with `WORLDLA_MODEL_DIR` selecting the existing full
release and the two environment arguments selecting their actual Python executables:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin press_stapler worldla-100k-decay \
  arx_x5 joint 0 7 7 /absolute/policy-env/bin/python /absolute/client-env/bin/python

DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh \
  RoboTwin press_stapler worldla-100k-decay arx_x5 joint 0 7 7 \
  /absolute/policy-env/bin/python /absolute/client-env/bin/python

WORLDLA_TASK_CONFIG=demo_clean WORLDLA_TEST_NUM=1 EVAL_ENV_TYPE=sim \
  bash eval.sh RoboTwin press_stapler worldla-100k-decay arx_x5 joint 0 7 7 \
  /absolute/policy-env/bin/python /absolute/simulator-env/bin/python
```

The simulator test selected installed CUDA 12.4 for the process, with a
separate Curobo extension cache and four build jobs. Five native extensions
were compiled before rollout. No task assets, expert-validity checks, seeds,
reward/success thresholds or physics were patched. A separate validation
workspace linked existing official resources read-only; results were written
to that workspace, not over historical evaluation output.

Native result tail:

```text
Success rate: 1/1 => 100.0%, current seed: 100000
Final success rate: 1/1 = 100.0%
[MAIN] eval finished
```

Relative result file:
`eval_result/press_stapler/WorldLA/demo_clean/worldla_100k_decay/2026-10-22_23-30-38_577593/_result.txt`.
The timestamp is a raw environment record, not a promise of a reproducible
directory name. Ordinary deprecation/optional-dependency warnings did not
prevent this completed rollout.

## Not established by these checks

- Fresh dependency installation on a clean machine.
- Full data conversion or 60k/100k training for this adapter PR.
- 50-task clean/randomized success rates or official leaderboard acceptance.
- Simulator SR equivalence of original and LoRA-folded arithmetic.
- RoboDojo or real-robot performance, or vectorized batch throughput.

The public route cache is sufficient for the documented cached-target
student training path with matching demonstrations and initialization; it
does not reproduce the omitted raw Fast-WAM extraction instrumentation.
