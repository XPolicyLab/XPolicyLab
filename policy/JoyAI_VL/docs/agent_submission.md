# JoyAI-VL Agent

Joy Future Academy's JoyAI-VL uses a vision-language model API to control RoboDojo
through a shared Python robot interface. The runtime uses a shared system prompt and tool registry.

## Context and execution

- `agent/api/prompts/robodojo/system.md` documents the robot API and general execution rules.
- `roboshell/task_card.py` assembles the evaluator-provided instruction, step budget,
  initial robot state and camera names. Initial RGB images enter the model conversation.
- `roboshell/server/tools.py` uses `profiles/generic/enabled_tools.txt` for every task.
  Tools load from `tools/` and `library/`.
- `robot` dispatches perception and motion through the robot host. SAM3 provides masks;
  camera geometry supports 3-D measurement and the controller emits joint targets.
- Every episode starts a fresh model conversation, Python session and perception state.
  Actions execute through the official evaluation interface.

## Shared tools

The model calls `python` and `finish`. The robot API includes motion, gripper control,
images, segmentation, measurement, camera projection, state and budget queries.
The shared command registry additionally contains table-plane and scene perception,
RGB point/quad geometry, boundary/rim fitting, grasp/place, contact strokes,
bounded rotation and timed interception. Its inventory is fixed across tasks.

## Configuration and records

`configs/model_api.example.json` records inference settings with `skills=false`.
The task card uses the evaluator's instruction and current observations.
Task identity is used by the evaluator and in run names; tool selection is shared.
Configure endpoint, model ID and an external key file with `tools/configure.py`.
Episode logs are written under `runs/bridge/<task>/` or `ROBOSHELL_RUN_ROOT`.

See [development history](development.md) for the relationship to the earlier release,
and [reproducibility](reproducibility.md) for manifests and checks.
