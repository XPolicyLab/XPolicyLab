# VAPv2 RoboDojo inference

Independent entry: `python -m scripts.vapv2.infer_robodojo`. This does **not**
start a robot/controller. It samples a recorded window and writes normalized and
physical-relative actions. For online evaluation, use the adapter's standard
`eval.sh` entrypoint.

## Contract

- Only `model_type=vapv2`, `vapv2.qwen_joint_ivava.v2` checkpoints; strict state
  loading, including the learned text projector and joint text Q/K/V/O. No optimizer.
- Use the checkpoint's own `config.yml`. The training initializer checkpoint need
  not exist. The complete checkpoint restores DiT/VAE parameters after structure
  construction; no Wan weight files are read.
- Frozen Qwen online or the same versioned raw Qwen cache from training. Cache
  overrides relocate a file and verify its identity; no T5 substitution, implicit
  backend conversion, silent truncation or cache-miss fallback. The explicit
  `--qwen-online-from-cache --qwen-model PATH` mode hashes the Qwen files against
  the training cache provenance before computing raw hidden states online.
- Explicit request-local `TextCondition(hidden_states, valid_mask)` with EOS and
  right padding. The training-only mutable mask is never used by this path.
- The uncached reference uses full joint forward `[Text; Memory; I; A0; V0; A1; V1]`.
  A0 sees only the clean condition prefix; A1 sees its executed clean A0/V0.
  Text/Memory/I retain bidirectional prefix attention, original 4D RoPE and
  negative relative history ages. RoboDojo evaluation enables the joint-v2 cache
  by default (`VAP_USE_KV_CACHE=0` restores full-forward); the offline CLI uses
  `--kv-cache`. Every rolling request rebuilds all layer K/V. State-FiLM and
  cross-attention memory banks are not supported by this cache.
- RGB view slots are `[ego, left wrist, right wrist]`, physical camera IDs
  `(1,0,2)`. Images must already match checkpoint height/width. Use uint8
  `[0,255]` or floating `[0,1]`; no evaluation-time random augmentation.

## Input archive (`np.savez`, no pickle)

- `video`: `[3,T,H,W,3]`. A0 uses `T=1`. A1 uses exactly `T=1+V` (normally 17),
  the same action/video cadence as training. I and executed V0 are encoded
  together in one causal VAE call. Do not feed consecutive control frames when
  the checkpoint expects stride-2 video sampling across 32 actions.
- For A1: `previous_normalized_action`: `[A,20]` (normally `[32,20]`), already
  in the checkpoint's normalized **local A0 frame**. It is not automatically
  rebased. Input the actual issued action prefix, not an unrelated window's chunk.
- With Memory: `history_steps`: unique integer `[N]` control-step indices,
  episode origin 0; `history_rgb`: `[N,3,H,W,3]`. Frame 0 is mandatory. No history
  may be newer than local I. History strides are control steps, not VAE latent
  frames or inference calls. Missing history is allowed; missing frame 0 is an error.
- `--window-start-step` is the step of local I, **also for A1**. No-memory
  checkpoints must omit history fields. State-FiLM checkpoints require the original
  runtime's `physical_state` contract; no-State checkpoints omit it.

Output: `raw_normalized_action`, controller-bounded `normalized_action` (clamped
to `[-1,1]`), `relative_action` (physical units after checkpoint normalization),
and JSON `metadata`. This is **not** an absolute robot command. Existing
RoboDojo coordinate conversion/rebasing and controller safety remain the caller's
responsibility. Existing outputs are never overwritten.

## Example

Supply a VAPv2 checkpoint, recorded RGB archive and compatible instruction.
Online checkpoints use `--qwen-model` to select the local official Qwen directory;
see the adapter README for its separate, hash-verified download. Cache-backed
checkpoints use `--text-cache`, or explicitly select online recomputation with
`--qwen-online-from-cache --qwen-model /absolute/path/to/Qwen3.5-9B`.

```bash
cd XPolicyLab/policy/vap/upstream
python -I -m scripts.vapv2.infer_robodojo \
  --checkpoint /absolute/path/to/VAPV2_CHECKPOINT \
  --input /absolute/path/to/recorded_window.npz \
  --output /absolute/path/to/new_prediction.npz \
  --instruction 'EXACT_INSTRUCTION' --stage a0 --window-start-step 0 \
  --qwen-model /absolute/path/to/Qwen3.5-9B \
  --normalizer-stats ../configs/normalizer_keys_robodojo_ee_v3_full_master_relative_global.json \
  --device cuda:0
```

Sampling steps/shift default to the checkpoint; explicit overrides are logged.
The installed policy environment supplies the required Qwen3.5 Transformers.

## Verification boundary

CPU tests exercise tiny joint-v2 A0/A1 sampling, padding and history invariance,
cache/full-forward trajectory parity and strict loading. These are not a
full-size checkpoint/real-video GPU smoke test or a closed-loop robot benchmark.
Before deployment, compare consecutive windows and verify action/state coordinate
anchors with the recorded episode. FP32 equivalence tests use tight tolerances;
BF16 kernels can differ when sequence shapes change. Do not assume bitwise
identity or identical closed-loop outcomes without testing the actual checkpoint.

## Action CUDA Graph + exact-rounding RoPE

With prefix KV enabled, `VAP_V2_ACTION_ACCELERATION=graph_rope` enables the
VAPv2-only decoder accelerator. RoboDojo V2 evaluation defaults to `graph_rope`
when KV is enabled. Set `VAP_V2_ACTION_ACCELERATION=off` to restore the KV baseline;
`rope` isolates precomputed phases and inference fused RoPE without CUDA Graph.
Environment overrides deployment `action_acceleration`, which overrides the default.
Explicitly disabling KV defaults acceleration to `off`; combining graph and
disabled KV fails. Unsupported modes fail explicitly. CUDA Graph does not
accelerate Qwen encoding or simulation.

Prefill is rebuilt per window. Graph-owned KV, text/history validity and action
phases refresh for every new prefix. No cross-window text KV reuse, noise capture,
action packing, fused normalization or compact attention is used. Graph pools are
bounded to two A0/A1 buckets. Weight reload invalidates them; direct `.data`
writes are unsupported (call `accelerator.clear()` explicitly). Input prefix
caches are immutable for the duration of an action sampling call. The returned
velocity owns its storage and survives subsequent replays.

`VAP_POLICY_TIMING=1` optionally logs each RoboDojo `policy.infer` wall time,
including text/memory/VAE work and action conversion, excluding transport and
simulation. `captures>0` marks a cold graph request, not steady-state latency.
This timing option does not change seeds or action processing.
