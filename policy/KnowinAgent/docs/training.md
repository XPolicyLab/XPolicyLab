# Training and frozen knowledge

Training means simulator practice followed by a review-agent session. Practice edits task skills and lessons; review checks its evidence, merges the useful experience and freezes the result. It does not train an action-network checkpoint. `process_data.sh` is explicitly not applicable because this workflow does not consume a converted demonstration dataset.

## Run

Commands below run from `XPolicyLab/policy/KnowinAgent/`, using the bundled `train.sh` and `eval.sh` entry points. Runtime resources are resolved independently of the working directory.

Prepare RoboDojo, Codex access and the perception services described in the [adapter README](../README.md). Use the simulator-compatible interpreter through `KNOWIN_AGENT_PYTHON`, and choose an available GPU according to the execution environment's admission rules. The entry point does not start perception services or manage a machine's GPU queue.

```bash
export KNOWIN_AGENT_PYTHON=/path/to/simulator/python
export KNOWIN_AGENT_ROBODOJO_ROOT=/path/to/RoboDojo
export KNOWIN_AGENT_START_SERVICES=0
export KNOWIN_AGENT_SAM3_URL=http://127.0.0.1:8035
export KNOWIN_AGENT_DEPTH_URL=http://127.0.0.1:8034

bash train.sh RoboDojo no-checkpoint arx_x5 joint 0 0 \
  --task stack_bowls --out /path/to/new-training-output
```

The six positional arguments are `<bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>`. This implementation supports RoboDojo, `arx_x5` and joint actions. `seed` is the practice backend's encoded layout seed: `bank * 1_000_000 + ordinal`. The GPU argument selects a physical GPU in the subprocess CUDA and Kit renderer settings. Verify physical placement in the execution environment. The official evaluation launcher instead receives a bank seed and owns its episode sequence.

`--out` must name a new directory outside the policy source. Training does not write assets into the source tree. With no `--knowledge-from`, practice starts without task-specific learned experience or a shared learned library. To continue from a previous frozen result, pass `--knowledge-from /path/to/frozen` explicitly.

| Option | Default | Meaning |
| --- | --- | --- |
| `--practice-max-executions` | `150` | Accepted program execution cap for practice |
| `--practice-wall-seconds` | `10800` | Agent wall-time cap; simulator startup and cleanup are separate |
| `--review-wall-seconds` | `2400` | Review-agent session timeout |
| `--test-budget` | `100` | Budget reference supplied to review; training does not launch Test |
| `--video-stride` | `0` | Optional practice frame recording |
| `--robodojo-root` | Environment/default lookup | Explicit RoboDojo checkout |
| `--review-agent-command` | Environment or Codex default | Explicit review launcher command |

The default review launcher uses the same configured model and reasoning effort as practice. `KNOWIN_AGENT_REVIEW_AGENT_CMD` can supply a review command. When a Codex home is configured, review receives it through `CODEX_HOME`, so both phases use the intended authentication source. CLI options are listed by `bash train.sh --help`.

## Execution and output

The orchestrator writes the practice configuration and launches practice in a separate process. This is necessary because closing Isaac Sim may exit its process. Once practice has produced its result and frozen knowledge, the parent starts the real review agent. Review inputs are protected, its output is validated, and the resulting knowledge is frozen.

```text
output/
  training.json
  practice_config.json
  practice.log
  practice/
    result.json
    frozen/
    workspace/ ...
  review/
    agent.log
    review.json
    REPORT.md
    workspace/
    frozen/        final reviewed task knowledge
```

Review writes its report and task knowledge under its output workspace. Inputs are staged read-only and checked again for modifications before freezing. The freeze manifest records content hashes, source runs and the shared copy actually used. A failed phase remains a failure with its logs; a partial directory is not a successful training result.

Evaluate a newly frozen result through the official adapter. With this integration installed under `XPolicyLab/policy/KnowinAgent/`, run from that directory:

```bash
KNOWIN_AGENT_KNOWLEDGE_DIR=/path/to/training-output/review/frozen \
  bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 \
  <policy_gpu_id> <env_gpu_id> <policy_python_or_env> <client_venv_or_conda_env>
```

That override points directly to one task's frozen directory. Use it only for the matching task. Newly trained outputs from this framework are supported; compatibility with assets produced by other releases is not promised. A frozen directory that references an absent packaged library without carrying its copy is rejected. Evaluation stages skills and lessons read-only. The default shared-library switch is `off`; a self-contained frozen directory restores its own shared copy, if present. The adapter does not automatically promote training output into the repository.

## Shared-library curation

No learned shared helpers ship in this integration. The `packaged` selector fails clearly because its assets are withheld. Curation supports a separate `library_root` or `KNOWIN_AGENT_SHARED_LIBRARY_ROOT`; candidate/version output belongs there. Explicitly choose `latest` or a registered version in a custom run configuration to stage your own library; merely setting the registry environment variable does not enable it. Publishing into the installed package is rejected. Any proposed library change needs its own offline and simulator validation before inclusion in a future system release.

## Interpreting validation

The CPU tests exercise orchestration with controlled substitutes, failure handling, workspace permissions, freeze hashes and shared-library contracts. The debug evaluation exercises the official transport and action schema without a model, perception service or simulator. Neither demonstrates learned knowledge quality or task success.

For real runs, report accepted program executions, native action steps, resets, model calls, wall time and native scores separately. Practice feedback is not independent Test performance. Keep unsuccessful/incomplete runs, and report valid and planned episode denominators with the source revision and knowledge digest. Results and videos belong in separate run artifacts or release attachments, not in the policy source.
