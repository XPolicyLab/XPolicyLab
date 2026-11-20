# Pudu_Agent — Development and Evaluation Log

Public development log for the RoboDojo Agent Track (rules of 2026-11-14, Section 5 "Open Development and Evaluation Logs"). This records what the submitted system is, how it was built, and why it is compliant. It contains no hidden simulator ground truth, scene state, privileged metadata, evaluator state, or precomputed task answers.

## 1. Versions — harness, model, API, code, configuration
- **Harness / scaffold:** XPolicyLab L4 Inspect EEF tool-use runtime, forked from `policy/GPT_6_Astra_Direct_EEF`. Client-side agent loop in `policy/Pudu_Agent/inspect/`; policy server is a no-op (`model.py`) that only satisfies the WebSocket boundary. Action space `joint` (model-facing action is EEF; RoboDojo's CuRobo planner produces the joint path).
- **Planner model (submitted):** `claude-fable-5-2` (Anthropic Claude Fable 5.2), reasoning effort `high`.
- **API surface:** OpenAI-compatible `chat/completions` (image parts as `image_url` data URIs, function `tools`). Selected by `L4_INSPECT_PLANNER=fable`; endpoint via `L4_INSPECT_BASE_URL`, token via `L4_INSPECT_API_KEY_ENV`. `astra`/`gpt55` GPT planners remain available for reproduction.
- **VLA/VLM:** none. No checkpoint, no training. Actions are produced at eval time by the planner through tool calls.
- **Config:** `deploy.yml` (`eval_batch: false`, single environment). Per-episode LLM-call budget `L4_INSPECT_MAX_LLM_CALLS=100`.

## 2. General skill library and task-agnostic components
All skills are task-agnostic; none is keyed on the task name.
- `move_eef` — command world-frame grasp-point dimensions; the local planner plans the joint path.
- `locate` — turn a pixel in a camera image into a world point using the cameras' published intrinsics and extrinsics: single-pixel ray–plane intersection (default plane = table surface, or a passed height), or two-pixel triangulation across cameras with a reported ray gap. Geometry in `cameras.py`.
- `update_plan` — an episode-local plan the model writes and keeps current; shown back with later observations.
- `return_home` — drive both arms to the poses recorded at the start of the current episode.

Change vs. the scaffold (`GPT_6_Astra_Direct_EEF`): added `locate` and its calibrated camera geometry; added in-episode `update_plan` and `return_home`; the task text is the official Detail Description (`detailed_instruction`, falling back to the short instruction); removed the hard-coded return-tolerance numbers from the embodiment docs so no numeric checker detail is pre-written into the prompt; reasoning effort raised to `high`.

## 3. Development data sources and general evaluation procedure
- **Camera geometry** was taken from the public RoboDojo camera configuration (`env_cfg/camera/camera_config.yml`: head camera `pos [0,-0.41,1.308]`, `ori [30,0,0]`, Gemini_345Lg 640×480) and cross-checked against one official demonstration episode (`RoboDojo-Benchmark/RoboDojo`, `data/demo/arx_x5`) — a permitted official demonstration source. The head camera model reproduces that episode's recorded intrinsics and extrinsics to < 0.001 px over the tabletop volume; the wrist-camera mount reproduces the recorded wrist-camera pose to ~0.1 mm / 0.0°.
- **Development runs** used the offline/debug wiring path (no simulator): it builds the action spec from the X5 URDF and exercises `locate` / `update_plan` / `move_eef` end to end through the submitted planner. No RoboDojo rollout outcomes, failures, or trajectories were used to tune prompts, select models, or switch behavior (Section 3, rollout-feedback prohibition).

## 4. Relationship between the final version and development history
The submitted system is the fixed, task-agnostic harness + skills described above with the `fable` planner. It is frozen for evaluation and is not switched by task. No per-task prompt, program, retrieval store, or replay trajectory exists.

## 5. Observations, scores, failure analysis
Offline wiring validated (see §3). Official per-task scores are produced by the RoboDojo evaluation service; this log will be updated with the aggregate official records and failure analysis once evaluation completes.

## 6. External models and services
- Anthropic Claude Fable 5.2 (`claude-fable-5-2`) via a third-party-hosted OpenAI-compatible endpoint. Provider, model id, and inference parameters (effort `high`) are disclosed; the endpoint URL and access token are provided to evaluators in the evaluation application (Section 7, closed-source foundation-model API exception).
