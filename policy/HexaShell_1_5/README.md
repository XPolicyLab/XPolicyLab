# HexaShell_1_5

**Contributor:** HexaFuture | **Paper:** TBD | **arXiv:** TBD | **Original code:** See vendored `hexashell/`.

Evaluation-only agent policy. A language model (`gpt-6-astra`, through an OpenAI-compatible chat API) works the two ARX X5 arms by writing Python cells against a `robot` object: it looks (the three RGB cameras, SAM3 segmentation, 3D measurement from several views) and moves (joint paths planned in this directory and streamed through the standard action contract), until the task is done or the step budget is spent. The harness (`hexashell/`) and the task descriptions (`prompts/ladders/`, from RoboProbe) are vendored here; SAM3 runs as a local service (`services/sam3_server.py`). The system prompt (`prompts/robodojo/`) and the per-task notes (`skills/robodojo/`) are not public yet: the contributors send them to the evaluators (see Installation).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Requirements on the policy machine: conda (or `python3.11` for a venv); a CUDA GPU for SAM3 (about 3 GB; the policy server itself uses no GPU); access to [facebook/sam3](https://huggingface.co/facebook/sam3) on Hugging Face (gated: accept its license, then `hf auth login` or set `HF_TOKEN`); an OpenAI-compatible chat completions endpoint that serves `gpt-6-astra`, and its key (for the official evaluation, the contributors send both to the evaluators, with the archive of the system prompt and the task notes). The simulator side is the stock RoboDojo setup (Isaac Sim 5.1).

```bash
cd XPolicyLab/policy/HexaShell_1_5
bash install.sh            # first run only creates config.env from config.env.example
vi config.env              # endpoint, key file, SAM3 checkpoint path
bash install.sh            # conda env "hexashell": requirements.txt, torch 2.7 (cu128), SAM3, and sam3.pt
# or: bash install.sh /path/to/venv
tar -xzf /path/to/hexashell_1_5_private.tar.gz   # prompts/robodojo/ and skills/robodojo/, from the contributors
sha256sum --check --quiet private.sha256         # every file there and unchanged
```

The system prompt and the task notes are not in the repository yet. `private.sha256` lists the SHA-256 of each of their files, and `setup_eval_policy_server.sh` does not start the policy server unless every one is there and matches (the debug check below needs none of them).

Nothing upstream is modified, and the stock `arx_x5` observation config is used.

## Data Processing

Unsupported. This is an eval-only adapter: `process_data.sh` is omitted, and it does not convert datasets.

## Training

Unsupported. `train.sh` is omitted. The policy has no weights: the model is called through the provider API at evaluation time, and the prompts and task notes (the archive, pinned by `private.sha256`) are frozen.

## Evaluation

```bash
cd XPolicyLab/policy/HexaShell_1_5
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# Example: stack_bowls, seed 0, simulator, policy server and SAM3 on GPU 0
bash eval.sh RoboDojo stack_bowls gpt-6-astra arx_x5 joint 0 0 0 hexashell <eval_env>
```

- `env_cfg_type` is the stock `arx_x5` (three 640x480 RGB cameras; depth is not used). `action_type` must stay `joint`. `ckpt_name` is a label only; `gpt-6-astra` is the convention.
- `<policy_env>` is the conda environment from `install.sh`, or the path of a venv.
- `setup_eval_policy_server.sh` starts the SAM3 service on the policy GPU (`SAM3_GPU` in `config.env` chooses another) when nothing answers on `SAM3_PORT`. The service stays up for later runs and is shared by them; `bash scripts/sam3_service.sh stop 18300` stops it.
- Cluttered Generalization variants (`_random`) use the note and the task description of the base task.
- One environment at a time (`eval_batch: false`).
- Each episode is recorded under `results/<task>_s<seed>-<time>/` (`<seed>` is the policy server's `seed` argument): `result.json`, the full prompt, every cell with its output, every action, and the pictures the model saw.
- Almost all of an episode's wall-clock time is model latency. `budget_s` caps the agent's time per episode at 90 minutes (time lost to endpoint failures is given back, up to one more budget), `max_turns` at 100 cells.

Offline wiring check (no simulator, no endpoint, no SAM3, no archive needed): with `EVAL_ENV_TYPE=debug` the policy server starts no agent and answers every step with an action that holds the joints where they are:

```bash
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo stack_bowls gpt-6-astra arx_x5 joint 0 0 0 hexashell <eval_env>
```

## Configuration

`config.env` (not tracked; see `config.env.example`):

| Variable | Role |
| --- | --- |
| `HEXASHELL_API_BASE` | OpenAI-compatible endpoint (`<base>/chat/completions`, streamed) |
| `HEXASHELL_MODEL`, `HEXASHELL_REASONING_EFFORT` | `gpt-6-astra`, `high` (the setting for the leaderboard evaluation) |
| `HEXASHELL_API_KEY_FILE` | file holding the API key (else `OPENAI_API_KEY`); read by the policy server only |
| `SAM3_PYTHON`, `SAM3_CHECKPOINT_PATH` | Python of the SAM3 service (default: the policy environment's) and `sam3.pt` |
| `SAM3_PORT`, `SAM3_GPU` | port of the service (18300) and its GPU (default: the policy GPU) |
| `POLICY_CONDA_ENV` | conda environment `install.sh` creates (`hexashell`) |

Any `deploy.yml` key below the standard ones can be overridden by a `HEXASHELL_<KEY>` variable. The keys: `api`, `api_base`, `api_key_file`, `ca_file`, `model`, `reasoning_effort`, `thinking_display`, `max_tokens`, `stream`, `history_reasoning` (the agent); `skills` (append the task note), `ladder` (the task description and scoring table in the task card); `max_turns`, `budget_s`, `nudges`, `images_max`, `images_low`, `images_bytes_max`, `jpeg_quality`, `cell_cpu_s`, `cell_idle_s`, `cell_total_s`, `motion` (limits of one episode); `sam3_url`; `poll_s`, `wrap_up_s`, `result_dir` (plumbing). Each is commented in `deploy.yml`.

## Notes

- **Prompt.** The system prompt (in the archive) is `prompts/robodojo/01_rules.md` to `06_checking.md` (rules, world and robot, the `robot` API, seeing, moving, checking), plus `07_env_icl.md` where `Env_ICL` is on. The task card is the official instruction, then the capability dimension, description and scoring table of the task from RoboProbe (commit `5230075`, Apache-2.0, copied unchanged into `prompts/ladders/` with its license; `hexashell/runtime/card.py` refuses a file whose SHA-256 differs from `prompts/ladders/manifest.json`), then the task note `skills/robodojo/<task>.md`.
- **Task notes.** One note for each of the 34 tasks outside Open (in the archive), written from our own runs of the task: how the task is judged, a flow, parameters that worked, common failures and fixes, final checks, typical step costs. They hold no coordinates or object sizes of a layout, and nothing taken from the task code or the simulator is written plainly; `python3 -I scripts/check_notes.py skills/robodojo/*.md` rejects a note with the wrong sections, more than 75 lines, wording that shows evaluator or simulator internals, seed numbers or model names, and warns about coordinates. The 8 Open tasks have no note and no task-specific code.
- **`Env_ICL`.** An object every cell sees next to `robot`, kept across cells: the objects measured so far (marked "may have moved" after a contact or a drop near them), the places where an arm ended blocked, and each grasp and how it went, filled in from the robot's results, plus the agent's own plan, facts and notes. It is off in the six Memory tasks (cover_blocks, imitate_sorting_sequence, match_and_pick_from_conveyor, press_by_number, swap_blocks, swap_T); `hexashell/robot/env_icl.py` lists them.
- **What the agent sees.** The three RGB images, joint angles, gripper openings and end-effector poses of the official observation, and the episode's step count and step limit (below). Positions are measured from the colour images: a pixel defines a ray through the camera model (`hexashell/vision/cameras.py`: fixed intrinsics and head-camera pose, wrist cameras from the reported end-effector pose), intersected with a measured plane or with the same point seen by another camera; SAM3 finds the objects' outlines. The robot's own pixels are masked with a model of its links (`hexashell/assets/`).
- **Sandbox.** The model's cells run in a separate process confined by Landlock and seccomp (`hexashell/kernel/`): it can read only the Python installation, `hexashell/` and system libraries, write only its episode workspace, and cannot open network connections, start programs or signal other processes. The API key is held by the policy server and never reaches it.
- **Episode loop (`deploy.py`).** It differs from `demo_policy/deploy.py` in four ways:
  1. `get_action` may answer "wait" while the model is still working (after `poll_s` seconds without an action); the loop asks again and the simulator does not step meanwhile.
  2. An empty answer means the agent has finished. The loop then ends the episode: it marks the running environment as not yet successful and calls `is_episode_end()`, so RoboDojo's final check judges the scene as it stands (the same check as at the step limit).
  3. Each observation carries `additional_info["hexashell"]`: the task name (for the task card), the step count and step limit (shown to the agent, which budgets its motions with them), and whether the client renders the pictures of the current state (older RoboDojo clients hand out the previous frame). Nothing else of the simulator is sent: no seed, layout or scene state.
  4. When the episode is over, the verdict (success and RoboDojo's progress score) is sent to the policy server through the standard `trial_end` call; it only goes into the record.
- **Images.** `model.py` decodes nothing: it receives the RGB arrays the policy server has decoded, and no XPolicyLab trajectory bits or datasets are read or written. Inside the policy, the pictures shown to the model are encoded as JPEG with PIL from those RGB arrays (`hexashell/vision/overlay.py`), the record keeps JPEG copies, and the SAM3 service and its client exchange the policy's own pictures and masks as PNG or raw RGB (`services/sam3_server.py`, `hexashell/vision/sam3_client.py`). These are the grep hits for `np.frombuffer` and `Image.open`; there is no `cv2.imencode`, `cv2.imdecode` or channel swap.
- **Provenance.** `hexashell/`, `prompts/ladders/`, `services/`, `scripts/sam3_service.sh`, `scripts/check_notes.py`, and the system prompt and the task notes of the archive, are HexaShell-1.5 at commit `9b39f6e`, without the ports to other benchmarks (`hexashell/benches/`, their prompts and robot model), plus the RoboDojo part of two later commits: `2dfa05e` (after a carry, a re-measure looks where the hand set the object down: `hexashell/robot/host.py`, `hexashell/vision/scene.py`) and `23a7fdc` (the picture window: at most 90 pictures and 10 MB of them as sent, was 20: `hexashell/runtime/conversation.py`, `hexashell/runtime/episode.py`). Changed here: the defaults of `api_base` (`hexashell/runtime/episode.py`) and of the SAM3 checkpoint path (`services/sam3_server.py`), the machine paths in `scripts/sam3_service.sh` (now read from `config.env`), the record names (`hexashell/runtime/records.py`, `hexashell/xpl/adapter.py`: the seed is now the policy server's argument, and the layout index is no longer sent), the usage line of `scripts/check_notes.py` (moved from `notes/robodojo_skills/`), and the line endings of `hexashell/assets/X5A.urdf`. `hexashell/assets/X5A.urdf` is a copy of the ARX X5 robot description in RoboDojo's assets (MIT), and `hexashell/assets/robot_model.npz` holds triangles derived from its meshes.
- Data processing and training are unsupported.
