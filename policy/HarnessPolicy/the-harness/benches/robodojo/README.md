# RoboDojo

The RoboDojo bench: a simulator worker serving one task per process over the worker
protocol (`runtime.py`, on top of `general.py` and `adapter.py`), cuRobo kinematics
(`kinematics.py`, also behind the kinematics service), and development layouts disjoint
from the evaluation packs (`devlayouts.py`).

```bash
harness serve --factory benches.robodojo.runtime:create_environment \
  --options configs/workers/robodojo_runtime_dev.toml --port 8771
```

- The worker, actions, observations, layouts and measured facts:
  [docs/reference/benches.md](../../docs/reference/benches.md)
- The robot API it serves: [docs/reference/worker-protocol.md](../../docs/reference/worker-protocol.md)
