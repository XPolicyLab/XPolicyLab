# PyRUA_Lean

**Contributor:** PKU DAGroup | **Paper:** PyRUA-Lean | **arXiv:** [2610.01939](https://arxiv.org/abs/2610.01939) | **Original code:** https://github.com/DAGroup-PKU/PyRUA-Lean

`PyRUA_Lean` runs a code-writing agent on RoboDojo: a language model writes Python against a small robot API (`robo`), one cell at a time, while the episode runs; each motion call becomes native joint actions. The same program-writing loop, the same six primitives and the same prompt serve every task: nothing is tuned, routed or stored per task. There is no VLA and no training entry; SAM3 (segmentation) is the only model the policy side loads. The adapter is evaluation-only and supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Inside the policy environment (any environment with a CUDA build of `torch` and the `torchvision` built for it; `install.sh` installs neither and keeps both as they are):

```bash
cd XPolicyLab/policy/PyRUA_Lean
bash install.sh              # XPolicyLab's server, PyRUA-Lean, the sam3 package
bash download_assets.sh      # SAM3's checkpoint (3.4 GB) into ./checkpoints/sam3.pt
```

`install.sh` installs PyRUA-Lean from the checkout this directory is linked from, or from the public repository when the directory was copied; it takes a PyRUA-Lean checkout or pip requirement as its argument when that is not the one to install. `download_assets.sh modelscope` fetches the checkpoint from the ModelScope mirror; `SAM3_CHECKPOINT_PATH` names a checkpoint kept elsewhere. The simulator's environment needs nothing beyond RoboDojo's own installation.

## Data Processing

Unsupported: there is no dataset. `process_data.sh` is omitted (evaluation-only adapter).

## Training

Unsupported: nothing is trained and there is no checkpoint directory. `train.sh` is omitted (evaluation-only adapter).

## Evaluation

The programs are written by a model behind an OpenAI-compatible chat endpoint:

```bash
export OPENAI_BASE_URL=https://YOUR_API_HOST/v1
export OPENAI_API_KEY=...
export PYRUALEAN_MODEL=gpt-6-astra          # the model's name at that endpoint

cd XPolicyLab/policy/PyRUA_Lean
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo stack_blocks no-checkpoint arx_x5 joint 0 0 0 pyrualean RoboDojo

# Offline wiring check (no simulator, no model call)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 pyrualean RoboDojo
```

`action_type` must be `joint`: the program's poses are solved to joint targets by PyRUA-Lean's own kinematics. `env_cfg_type` is `arx_x5`. `ckpt_name` is ignored; `no-checkpoint` is the conventional placeholder. Batch evaluation is not supported: one conversation drives one environment (`eval_batch: false`). The policy GPU holds SAM3 (about 8 GB). Each episode's programs, tool calls and transcript are kept under `result_dir` (`./results/<task>_l<layout>_s<seed>-<time>/`).

`EVAL_ENV_TYPE=debug` holds the robot where the debug client's observation has it: the loop, the action keys and their dimensions are exercised, and no episode is started.

## Configuration

Extra `deploy.yml` keys beyond the shared set:

| Key | Role |
| --- | --- |
| `robot` | `robodojo_xpl_rgb`: the robot the programs are written against, RoboDojo's default observation (three colour images and the robot's state) |
| `primitives` | `no-vla` (this adapter serves no VLA) |
| `runtime`, `model` | the agent that writes the programs: `openai` (a chat-completions endpoint) and the model's name there (or `PYRUALEAN_MODEL`) |
| `max_turns`, `budget_s` | an episode's decision budget (model replies that call a tool) and its wall-clock limit in seconds |
| `images`, `feedback` | `on-demand`: the model sees a camera when its program asks for it; `pure`: a tool call returns what the program printed and nothing else |
| `openai_options` | passed to the endpoint with every request (`max_tokens`, `extra_body.reasoning_effort`) and `nudges`, the text-only replies tolerated before the episode is ended |
| `poll_s` | the longest the policy server holds one `get_action` before answering "wait" |
| `wrap_up_s` | the longest the verdict's acknowledgement waits for the episode's record to be written |
| `result_dir` | where each episode's record is written, relative to this directory |

Environment variables: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PYRUALEAN_MODEL` (above); `SAM3_CHECKPOINT_PATH` (default `./checkpoints/sam3.pt`); `PYRUALEAN_RESULT_DIR` overrides `result_dir`.

### Observation

The policy reads what RoboDojo's client sends as it is configured (`env_cfg/arx_x5.yml`, unchanged): the three colour images, the joint and end-effector state and the instruction. It reads no depth image and no camera matrix; if a client sends them they are ignored. `setup_eval_env_client.sh` is `demo_policy`'s, and nothing in this directory touches the client's configuration.

Where an object is, is measured from its masks in the cameras that see it (SAM3): the head camera's mask bounds the object's footprint on the table, a wrist camera that sees the same object settles its height. That is good to a few millimetres for things up to about 6 cm tall, worse for tall things only one camera sees, and it cannot measure what does not stand on the table (an object in the gripper, on another object, in a rack).

## Notes

- **What the policy reads.** The observation RoboDojo's client builds and nothing else: the colour images, the joint and end-effector state and the instruction. No depth, no camera matrices, no object poses, no layout file, no success signal, no simulator handle: the policy server is a separate process that sees only what the protocol carries.
- **What is built in.** Constants of the embodiment and the default scene, the same for every task and episode, taken from RoboDojo's public configuration or measured once in the simulator: the head camera's pose and focal length and the wrist cameras' mounting (`env_cfg/camera`), the arms' base poses, link geometry and joint limits (`env_cfg/robot`, the URDF), the table top's height and extent (`env_cfg/scene/default.yml`), and the gripper's opening-to-width relation. With them a pixel is a known ray although the client sends no camera matrix. The table height and the arms' base poses are the values `policy/GPT_6_Astra_Direct_EEF` also builds in (`pose.py`).
- **What the model is told.** One text for every task (`knowledge/robodojo_xpl_rgb-novla.md` in PyRUA-Lean): how the tools behave and how to judge progress from images. It holds no task recipes. It does state general facts about how RoboDojo judges an episode that the instruction does not spell out (both arms back at their start pose with the grippers open, positions checked to about a centimetre), which were read from RoboDojo's task code.
- **`deploy.py` differs from `demo_policy`'s in three places**, each because the policy is a program a model writes while the episode runs: `get_action` is asked again while it answers "wait" (a model thinks for minutes; no request is held longer than `poll_s`); an empty answer ends the episode, and RoboDojo judges the scene as it stands; every observation carries the episode's step count and limit (`additional_info["pyrualean"]`), which the program budgets its motions by, and the verdict is reported back through the protocol's `trial_end` once the episode is over (it is recorded, the model never sees it). To have RoboDojo judge a scene the program has finished early, `deploy.py` clears the environment's success flag and calls `is_episode_end()` once more: the flag is only ever cleared, the verdict is RoboDojo's check. `deploy.py` reads the step count, the step limit, the task's name, the eval seed and the layout index from the environment object, and nothing else. (`step_lim` is read the same way by `policy/GPT_6_Astra_Direct_EEF/inspect/deploy.py`.)
- **Cost.** One episode is up to `max_turns` model calls with images; a rate-limited endpoint (HTTP 429) is waited out, not failed.
