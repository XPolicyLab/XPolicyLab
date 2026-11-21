# PyRUA_RSI

**Contributor:** PKU DAGroup | **Paper:** PyRUA-Lean | **arXiv:** [2610.01939](https://arxiv.org/abs/2610.01939) | **Original code:** https://github.com/DAGroup-PKU/PyRUA-Lean

`PyRUA_RSI` (PyRUA-RSI) plays RoboDojo with **one fixed Python program per task name** and **no language model at evaluation time**. It is the `PyRUA_Lean` adapter with one change: the code of an episode is not written by a model while the episode runs, it is read from `programs/<task>.py`. The programs use the same robot API (`robo`: six motion and perception primitives plus camera reads), the same PyRUA-Lean runtime and sandbox, and the same observation as `PyRUA_Lean`. No VLA, no training, no checkpoint; SAM3 (segmentation) is the only model the policy side loads. The adapter is evaluation-only and supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`.

Shared conventions (argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE`) are documented in the [XPolicyLab README](../../README.md).

## Installation

Inside the policy environment (any environment with a CUDA build of `torch` and the `torchvision` built for it; `install.sh` installs neither and keeps both as they are):

```bash
cd XPolicyLab/policy/PyRUA_RSI
bash install.sh              # XPolicyLab's server, PyRUA-Lean (pinned revision), scipy, the sam3 package
bash download_assets.sh      # SAM3's checkpoint (3.4 GB) into ./checkpoints/sam3.pt
```

`install.sh` and `download_assets.sh` are `PyRUA_Lean`'s (same arguments, same mirrors). The simulator's environment needs nothing beyond RoboDojo's own installation.

## Data Processing

Unsupported: there is no dataset. `process_data.sh` is omitted (evaluation-only adapter).

## Training

Unsupported: nothing is trained and there is no checkpoint directory. `train.sh` is omitted (evaluation-only adapter).

## Evaluation

No API key and no model endpoint are needed:

```bash
cd XPolicyLab/policy/PyRUA_RSI
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo push_T offline arx_x5 joint 0 0 0 pyrualean RoboDojo

# or through the benchmark's entry, from the RoboDojo root
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/PyRUA_RSI --task push_T --ckpt offline \
  --policy-env pyrualean --eval-env RoboDojo --env-cfg arx_x5 --action-type joint --seed 0 --policy-gpu 0 --env-gpu 0

# Offline wiring check (no simulator)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls offline arx_x5 joint 0 0 0 pyrualean RoboDojo
```

`action_type` must be `joint` and `env_cfg_type` `arx_x5`. `ckpt_name` is ignored (`offline` is a placeholder). Batch evaluation is not supported: one policy server drives one environment (`eval_batch: false`). The policy GPU holds SAM3 (about 8 GB). Each episode's program output and tool calls are kept under `result_dir` (`./results/<task>_l<layout>_s<seed>-<time>/`).

## How an episode runs

`model.py` starts `offline_programs.py`, a small chat-completions endpoint on `127.0.0.1` inside the policy server, and points PyRUA-Lean's `openai` runtime at it. At the first request of an episode the endpoint answers with the task's whole program as one `python` tool call; every later request is answered with `finish`. The runtime runs the program in its sandbox (a fresh namespace per episode; only the public `robo` methods and the modules the sandbox allows, such as `numpy`, `scipy`, `cv2`, `PIL`), each motion call becomes native joint actions, and RoboDojo judges the final scene as usual. `deploy.py` is `PyRUA_Lean`'s, unchanged.

## The programs

`programs/` holds 54 files, one per RoboDojo task name (the twelve Generalization tasks have a `<task>_random.py` next to `<task>.py`; for eleven of them the two files are identical, `make_toast_random.py` is its own program). They were written ahead of time by a code-writing agent loop that ran candidate programs in RoboDojo's simulator through this same entry and kept the versions that solved more episodes; the process is described in the PyRUA-Lean repository.

What a program uses at run time:

- **Read**: the three colour images, the joint and end-effector state, the instruction, the episode's step count and step limit (`set_gripper(...).step_limit`), and SAM3 masks of what the cameras see.
- **Not read**: depth images, camera matrices sent by the client, object poses, the layout file, the success signal, the simulator. A program cannot open files, import other modules or reach private attributes (each program passes a static check for this before it is used), and it never reads the layout index or the evaluation seed.
- **Not XPolicyLab bytes**: a program only ever sees image arrays handed over by `robo.image()` (the policy server decodes the observation). The mechanical greps of XPolicyLab's adapter check hit three things inside `programs/`, none on observation data: `np.frombuffer(bytes.fromhex(...))` unpacking constant bitmaps (digit templates) in `arrange_largest_number*.py` and `solve_equation.py`; `cv2.imencode` writing a small diagnostic JPEG to the program's own stdout in `store_laptop_and_headphones*.py`; an RGB→BGR conversion for OpenCV colour operations in `hang_mugs*.py` and `imitate_sorting_sequence.py`.
- **Built in**: constants of the embodiment and the default scene (as in `PyRUA_Lean`: camera mounting, the arms' base poses and joint limits, the table height), constants of the task taken from its public definition and assets (object dimensions from the asset metadata and meshes; fixed values of the task's configuration and success check, e.g. `make_kong`'s fixed group-to-slot table), and a few constants measured in simulation (offsets between a perception reading and an object's origin). Each is the same for every episode; no value depends on the layout.

## Configuration

Extra `deploy.yml` keys beyond the shared set:

| Key | Role |
| --- | --- |
| `robot` | `robodojo_xpl_rgb`: RoboDojo's default observation (three colour images and the robot's state) |
| `primitives` | `no-vla` |
| `runtime`, `model` | `openai` and `offline-program`: the runtime asks the local endpoint of `offline_programs.py` |
| `max_turns`, `budget_s` | an episode's decision budget and its wall-clock limit in seconds |
| `images`, `feedback` | as in `PyRUA_Lean`; a program reads cameras through `robo` |
| `openai_options` | sent with every request and ignored by the local endpoint |
| `poll_s`, `wrap_up_s`, `result_dir` | as in `PyRUA_Lean` |

Environment variables: `SAM3_CHECKPOINT_PATH` (default `./checkpoints/sam3.pt`); `PYRUALEAN_RESULT_DIR` overrides `result_dir`. `OPENAI_*` variables are not needed and are overridden inside the policy server.

## Results

RoboDojo's own summary (`bash scripts/robodojo.sh summarize`) of our run on eval seed 0, all 54 task configurations at their native episode counts (2,100 episodes), one `robodojo.sh eval` per task name: [`eval/robodojo_summary.md`](eval/robodojo_summary.md) (Avg SR 82.82 / Avg Score 87.16; the same programs on the held-out eval seed 2: 80.73 / 85.36).
