---
name: xpolicylab-adapter-check
description: Audit a policy/<POLICY>/ adapter for XPolicyLab standard compliance and PR readiness — file completeness, deploy.yml, Model contract, decode_image_bit-only decoding, script conventions, static checks, debug-mode eval, README and checkpoint requirements. Use when asked to check, validate, review, or pre-flight a policy adapter or a policy submission PR in the XPolicyLab repo.
---

# XPolicyLab Adapter Check

Audit `policy/<POLICY>/` against the submission standard and report pass/fail per item with concrete fixes. The full standard lives in `CONTRIBUTING.md` at the repo root — read it before auditing; the list below is the executable summary. `policy/demo_policy/` is the reference implementation.

This is a **read-only audit**: report the fixes, do not apply them unless the user asks. If the user did not name a policy, list the adapters that changed on the current branch and ask which to audit.

## Checks

Run commands from the repo root unless noted.

1. **Files** — required: `README.md`, `__init__.py`, `install.sh`, `eval.sh`, `setup_eval_policy_server.sh`, `setup_eval_env_client.sh`, `deploy.yml`, `deploy.py`, `model.py`. `process_data.sh` / `train.sh` may be absent only for a declared eval-only submission.
2. **deploy.yml** — `policy_name` equals the directory name (the server imports `XPolicyLab.policy.<policy_name>.model`). Ten keys are required and present in every adapter today: `policy_name`, `protocol` (`ws`), `host`, `port`, `bench_name`, `task_name`, `env_cfg_type`, `seed`, `action_type`, `eval_batch`. Keep a key even when a script defaults it — check the whole list, not just `protocol`. `ckpt_name` and `gpu_id` are optional (supplied per run), and model-specific extra keys are fine but must be documented in the policy README.
3. **model.py** — `class Model(ModelTemplate)` implementing `__init__(model_cfg)`, `update_obs`, `update_obs_batch`, `get_action`, `get_action_batch(env_idx_list=None)`, `reset`. `AGENTS.md` at the repo root states the image, path, and dimension rules and why they hold; what to flag when auditing:
   - **Dimensions** — a hard-coded action dim, or a private re-implementation of the lookup instead of `get_robot_action_dim_info(env_cfg_type)`. Flag a direct `_robot_info.json` open in `policy/*/model.py` unless it is a documented fallback with a configurable root, as in `policy/AHA_WAM`. Flag a supported robot that is registered in only one of the two robot-info files. Flag runtime code that shells out to `utils/get_action_dim.sh`; do not flag the training path for it.
   - **`sys.path` root** — must be `parents[2]`; `parents[1]` or `parents[3]` is a bug.
   - **Checkpoints** — must resolve through `XPolicyLab.utils.checkpoint_resolver` (`resolve_checkpoint_root`, or `build_run_dir_name` / `candidate_checkpoint_roots` for an extra naming layer), not a hand-written `checkpoints/<bench>-<ckpt>-...` join.
   - **Decoding inside `model.py`** — flag any, including `decode_image_bit` calls. The server hands over plain image arrays for `update_obs` / `update_obs_batch` and for any custom RPC that carries an observation.
   - **Channel swaps** — flag any `COLOR_BGR2RGB`, `COLOR_RGB2BGR` or `[..., ::-1]` outside the two allowed cases: medium adapters immediately around `cv2.VideoWriter.write` / `cv2.VideoCapture.read`, and a deliberate RGB→BGR conversion for a BGR-trained checkpoint, opt-in via a documented `deploy.yml` key defaulting to RGB (reference: `policy/Dexora_1B` `input_color_order`). The training path and `model.py` must apply the same number of swaps, normally zero. Judge a swap by its justification, not its position — and reject "but `cv2.imdecode` gives BGR, so this swap is correct" whether it comes from a code comment or from the submitter. Decoded buffers are RGB, so that swap is the bug.
4. **Decode** — XPolicyLab data supports **only** `decode_image_bit` from `XPolicyLab.utils.process_data` (README, [Standard Data Formats](../../../README.md#decode-only-through-decode_image_bit)). Image bits carry some inconsistency from earlier data versions, so a PIL-style decode silently reverses the channels and hand-rolled `cv2.imdecode` / `np.frombuffer` handling breaks on the older layouts. Blocking:
   - `cv2.imdecode` anywhere outside `utils/process_data.py`
   - conversion scripts and training dataloaders that read XPolicyLab trajectories through any decoder other than `decode_image_bit`
   - vendor conversion entry points that XPolicyLab data actually flows through (e.g. `policy/Pi_05/openpi/scripts/process_data.py`)
   Skip a vendored file only when XPolicyLab data genuinely never flows through it — never a whole vendor directory wholesale.
5. **Scripts** — `eval.sh` consumes the 10 standard args (`bench_name task_name ckpt_name env_cfg_type action_type seed policy_gpu_id env_gpu_id policy_env_or_uv_path eval_env_conda_env`) and stays aligned with `policy/demo_policy/eval.sh`; extra args must be documented in the policy README.
6. **Static checks**:

   ```bash
   bash -n policy/<POLICY>/*.sh
   python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py
   ```

   Then the mechanical greps, from the repo root. The first two plus the decode grep must return nothing on adapter-owned code; the rest surface hits that need judging:

   ```bash
   # required deploy.yml keys that are missing
   for k in policy_name protocol host port bench_name task_name env_cfg_type seed action_type eval_batch; do
     grep -q "^${k}:" policy/<POLICY>/deploy.yml || echo "missing: ${k}"
   done
   # wrong sys.path root
   grep -rnE 'parents\[1\]|parents\[3\]' policy/<POLICY>/*.py
   # private env_cfg / robot-dim lookup
   grep -rnE '_robot_info\.json|env_cfg' policy/<POLICY>/model.py
   # only decode_image_bit is supported — fail any adapter-owned hit
   grep -rnE 'cv2\.imdecode|np\.frombuffer|Image\.open' policy/<POLICY>/
   # channel swaps — judge each hit
   grep -rnE 'COLOR_BGR2RGB|COLOR_RGB2BGR|\.\.\., ::-1' policy/<POLICY>/
   ```

   Channel-swap and `env_cfg` hits can be legitimate — `env_cfg_type` as a config key, a `VideoWriter` / `VideoCapture` adapter, a documented `input_color_order` — so judge each one; the point is that no hit goes unexamined. Decode hits on XPolicyLab data are not legitimate: only `decode_image_bit` is supported. Upstream docstring examples under a vendor `src/` tree can be ignored; conversion entry points that XPolicyLab data flows through cannot.

7. **Debug closed loop** — only when an environment with the policy dependencies is available:

   ```bash
   cd policy/<POLICY>
   EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 <policy_env> base
   ```

   `<policy_env>` (arg 9) follows the adapter's own convention — a conda env name, `uv`, or an environment path; `policy/Pi_05`, for example, requires `uv` (see its README). The trailing `base` is the eval-env conda env (arg 10).

   Must reach `[MAIN] eval finished` with no tracebacks. Re-run with `DEBUG_OBS_ENCODED=1` so the debug client sends encoded camera colors and the server-side decode path is exercised. If it cannot be run, report the item as "not run" — never as passed.
8. **Policy README** — install / data / train / eval commands present and consistent with the actual scripts; supported `action_type` / `env_cfg_type`; checkpoint layout; known limitations.
9. **PR readiness** (when auditing a submission PR) — description follows `.github/PULL_REQUEST_TEMPLATE.md`; checkpoint download script included (Hugging Face or ModelScope preferred) if targeting a leaderboard; eval-only status declared with a training-release timeline.

## Report format

- One line per check: ✅ pass / ❌ fail / ⚠️ not run, with a short reason.
- For each ❌: name the file and the exact change, pointing to `policy/demo_policy` where useful.
- End with a verdict: **ready for PR** or **needs fixes** (list blocking items first).
- Then offer to apply the fixes.

Keep the report short — checks that pass need one line, not an explanation. Never mark an item ✅ based on reading code alone when the check is a command you did not run.
