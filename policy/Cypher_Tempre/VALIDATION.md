# Integration validation

Validation performed on Windows, 2026-11-13, against XPolicyLab base `6baccd57ff958570d0db1588370289ee36c8bb70`.

- Five standard-library transport tests passed: two episode boundaries retain the same ledger and generated faculty; absent memory fails without initializing an empty chain; invalid targets are rejected; the remote broker enforces authentication and one active session; non-loopback plain HTTP is refused. These tests use synthetic state, not a cryptographic or robot benchmark.
- Python compilation and shell syntax checks passed. Adapter-owned code contains no custom image decoder, image encoder or channel swap.
- A local native `stack_bowls`, seed 0 diagnostic reached the new adapter through the normal XPolicyLab imports and existing RoboDojo Cartesian executor. It used the populated live skill, one simulator environment, a local Windows rendering/pipe compatibility launcher, and `L4_INSPECT_MAX_LLM_CALLS=0`. No model process was launched. The native client exited 0; its log reported `0 llm calls` and normal evaluation completion. The agent stopped immediately, producing one failed zero-score diagnostic episode. This is expected for the zero-call check, is excluded from performance claims, and does not validate task-solving ability or Cartesian motion quality. The existing Timechain ledger was unchanged.
- Generic debug plain/encoded loops are **not passed**: the shared debug `TestEnv` does not provide the `robot_manager`/Cartesian planner the inherited EEF executor requires. Native simulator testing is the appropriate next gate for model-selected motion.
- No live remote TLS endpoint, actual continuing Codex-controller session, scored native model trial, official cloud run or hidden verification has been validated for this adapter.

Run the transport checks from the parent of the checkout:

```bash
python -m unittest XPolicyLab.policy.Cypher_Tempre.tests.test_transport -v
```

This draft still needs a reproducible evaluated-state/access arrangement and benchmark approval of the disclosed continual-learning protocol before official evaluation.
