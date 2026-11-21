# Evaluation

## Native RoboDojo entry

After installation and configuration, the policy retains XPolicyLab's positional interface:

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/JoyAI_VL
bash eval.sh \
  RoboDojo stack_bowls JoyAI-VL arx_x5 joint 2 \
  0 0 unused /path/to/isaac-env
```

| Position | Meaning |
| --- | --- |
| 1 | Benchmark name, `RoboDojo` |
| 2 | Task name from the evaluator |
| 3 | Result/checkpoint label; does not select the API model |
| 4 | Stock RGB environment configuration, `arx_x5` |
| 5 | Action type, `joint` |
| 6 | Evaluator-provided seed |
| 7, 8 | Policy and simulator GPU IDs |
| 9 | Policy environment fallback; `unused` when `ISAAC_PYTHON` is configured |
| 10 | RoboDojo Isaac environment directory or Python executable |

The `seed` is forwarded to both the native policy server and environment client.
Layouts and episode counts follow the caller's RoboDojo configuration, including
`ROBODOJO_LAYOUT_IDS` and `EVAL_NUM`. RoboDojo computes the evaluation scores.

Generalization's cluttered tasks use their native `_random` names, such as
`stack_bowls_random`. The native evaluator's own task list is authoritative.

## Convenience entry

From the policy directory:

```bash
python3 tools/evaluate.py \
  --robodojo-repo /path/to/RoboDojo \
  --task stack_bowls_random --seed 12345 \
  --policy-gpu 0 --env-gpu 0 \
  --eval-env /path/to/isaac-env
```

Use `--dry-run` to print the command without starting the simulator or calling the model.
This entry runs one task with `eval_batch: false`. Use RoboDojo's scheduler for a suite.

## Effective settings

| Setting | Value |
| --- | --- |
| Reasoning effort | `max` |
| Chat template | `thinking_mode=thinking`, `reasoning_effort=max`, `drop_thinking=false` |
| Generation | temperature 0; max tokens 65,536; non-streaming |
| Reasoning history | Enabled |
| Agent tools | `python`, `finish` |
| Task skill notes / recipe text | Disabled |
| Agent budget | 100 replies, 5,400 seconds |
| Model HTTP timeout / attempts | 900 seconds / 3 |
| Command budget | 0; native episode step limit applies |
| Motion profile | `v2` |
| Perception | RGB and SAM3 masks with camera geometry |
| Images | JPEG quality 90; at most 8 per turn; history trimmed from 90 to 60 images |
| Image history byte threshold | 20,000,000 bytes |

`deploy.yml`'s bridge `request_timeout_s=120` is separate from the model HTTP timeout.
The model service supplies defaults for `top_p`, `top_k`, `min_p`, and penalties.
Record these defaults together with the served model and software versions for reproduction.

## Results and traces

Native RoboDojo writes the evaluation results. Generate its normal summary from that checkout:

```bash
cd /path/to/RoboDojo
bash scripts/robodojo.sh summarize
```

The summary uses RoboDojo's native completeness and aggregation rules.

Agent traces are under the installed policy's `runs/bridge/<task>/` unless the evaluator sets
`ROBOSHELL_RUN_ROOT`. These include model events, tool results, observations, and request
payloads. They are ignored by Git. When parallelizing multiple runs of the same task, use a
separate run root and avoid sharing output locations.
