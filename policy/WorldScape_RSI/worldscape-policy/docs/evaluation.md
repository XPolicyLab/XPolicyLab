# RoboDojo evaluation

Use the XPolicyLab adapter's `serve_robodojo.sh` or `eval.sh` from its parent
policy directory. Both `deploy.yml` and `deploy_robodojo.yml` select the RoboDojo
backend; unsupported runtime profiles and benchmarks fail before model loading.

```bash
bash serve_robodojo.sh flash --checkpoint /path/to/flash-checkpoint --gpus 0
bash serve_robodojo.sh memory --checkpoint /path/to/memory-checkpoint --gpus 1
```

Run independently or allocate distinct GPUs and ports. See the
[server guide](../../robodojo_server/README.md) for the RoboDojo client loop,
sessions and execution acknowledgements, and the [adapter guide](../../README.md)
for standard evaluation commands.

A checkpoint must include its native configuration, tokenizer, transforms and
checksummed weights. Runtime normalization and preprocessing come from checkpoint
metadata. The environment sends RGB observations and receives absolute 14-D joint
actions. Execution must be acknowledged with commit/discard; only executed history
is retained for Memory. Flash and Memory share the same explicit-commit protocol.

Full evaluation requires external RoboDojo simulator/robot assets. A successful
weight load, replay or training smoke test does not establish task success.
