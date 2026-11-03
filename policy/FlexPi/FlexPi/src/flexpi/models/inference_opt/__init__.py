"""Deploy-time speed adapters. Nothing here is imported at training time.

Each module exposes one ``install_*(model)`` entry point that monkey-patches a
method on an already-built ``FlexPiBackbone``; all are opt-in, off by default,
and reached only from the ctor's optimization block. Measurements below are the
d7pu2 joint path on an RTX 5090 (sm_120) — see
``docs/INFERENCE_OPTIMIZATION.md`` for the full ladders, and note
that verdicts are hardware- and TRT-version-specific.

  trt_joint         ``trt_joint_engine_path``. Raw TRT engine for the joint
                    core. 1096 → 600.9 ms/call. DEPLOYED.
  trt_joint_split   ``trt_joint_{prefill,decode}_split_engine_path``. KV-split
                    pair; fastest measured stack at 413.3 ms. Mutually
                    exclusive with trt_joint.
  encoder_graphs    ``encoder_cuda_graph``. Captures the three first-frame
                    anchor encoders. 26.3 → 15.2 ms, bit-exact. Rejects
                    ``compile_encoders`` (same methods).

  trt_prefill       ``trt_prefill_engine_path``. Action-only path, 234.7 →
                    192.8 ms — but ``torch_compile`` reaches 95.6 ms on the
                    same path, so this is superseded on current hardware.
  joint_loop_graph  ``joint_loop_cuda_graph``. Captures the whole denoise loop.
                    Capture FAILS against a TRT engine (``execute_async_v3``
                    is not capturable) and silently falls back — a no-op on the
                    deployed stack. Raises under ``torch_compile`` +
                    ``scope=loop``.
  step_skip         ``dynamic_step_skip``. DreamZero-style adaptive step
                    skipping. Wired and correct, but no measurable speedup
                    on 32D unified-joint LIBERO-Plus; off in every config.

The last three are kept for re-measurement on newer TRT / other GPUs, not
because anything currently runs them.
"""
