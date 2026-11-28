# `wall_x.model.core.ops` — operator boundary contract

This package is the **single boundary** between wall-x and the internal
`xcompute` kernel library. It is also a **public (open-source) module**: the
open-source export ships `wall_x/model/core/ops/` but **not** `xcompute`, so the
PyTorch fallback in each operator is the code path open-source users actually
run. That makes the contract below load-bearing, not stylistic.

## The contract

Every operator that can be accelerated by an xcompute kernel MUST:

1. **Touch xcompute only here.** No module outside `core/ops/` may
   `import xcompute` (directly or via `from xcompute import ...`). Higher layers
   import the operator from `wall_x.model.core.ops`. Two legacy exceptions
   (`model/pi0/*`) predate this rule and are tracked for cleanup.

2. **Import cleanly without xcompute.** Kernel access is wrapped so the module
   imports on a machine with no xcompute:
   - `OpsProxy` subclasses resolve lazily via `importlib.import_module`.
   - Plain-function ops use `try: from xcompute import <sym> except Exception: <sym> = None`.
   The `from xcompute import` line must be indented under `try` (never column-0),
   so the open-source export grep guard passes.

3. **Own a self-contained PyTorch fallback.** The fallback lives **in this
   package**, computes the same result as the kernel, and is a **first-class,
   independently callable** entry — never buried in a caller module. Rationale:
   - open-source runs the fallback (no xcompute), so it must be complete here;
   - the fallback must be selectable in isolation for the bitexact test below.

4. **Select backend by capability — no env switches.** An op uses its kernel
   iff the loaded xcompute build provides it (and inputs are on CUDA); otherwise
   it runs the fallback. There are **no per-op switches and no global disable
   flag** — capability detection ("kernel present ⇒ use it") is the only
   mechanism. To exercise the fallback, run without xcompute (the open-source
   env) or, in a test, call the fallback directly / null the module's kernel
   handle.

5. **Be covered by `tests/test_ops_fallback.py`** with two assertions:
   - **fallback vs reference** — the PyTorch fallback matches an independent
     reference (this runs in the open-source overlay, guaranteeing OSS
     correctness with no xcompute present);
   - **kernel vs fallback** (`test_vs_xcompute`) — when xcompute is present, the
     kernel and the fallback agree. Use an explicit tolerance for
     native-dtype/fused rounding (fp32 ~2e-6, fp16 ~2e-2, bf16 ~3e-2); a
     divergence far outside tolerance is a **convention/layout bug**, not a
     rounding artifact — fix the operator, do not widen the tolerance.

## Two conforming shapes

- **`OpsProxy` subclass** (rmsnorm, rope, m_rope, silu_and_mul, …): define
  `_xcompute_name` and `_pytorch_fallback`; get `__call__` dispatch,
  `call_with_backend("pytorch"|"xcompute")`, and `available_backends()` for free.
  Preferred for single tensor→tensor ops.

- **Plain-function module** (gated_delta, qkv_norm_replace, attention_mask):
  when the op is not a single callable (multiple entry points, or a fused op
  with a structured signature), expose `<op>(...)` (capability-dispatched) and a
  directly-callable `<op>_pytorch(...)` fallback in the same module, plus a
  `<op>_available(...)` probe. The fallback must be a pure tensor function (it
  may take a `module` for parameters but must not embed caller-only control
  flow), so the bitexact test can call kernel and fallback side by side.

## Anti-patterns (do not ship)

- Fallback logic living in a caller module (e.g. an attention forward) while the
  ops module only wraps the kernel — the op then has no OSS-runnable, testable
  fallback.
- Any env switch (per-op or global) to force the fallback — capability
  detection covers every real need; nulling the kernel handle covers tests.
- Widening a `test_vs_xcompute` tolerance to hide a large divergence.
