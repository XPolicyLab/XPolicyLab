# Contributing to Focus-VLWA

Focus-VLWA intentionally keeps a narrow scope: dataset processing, post-training, checkpoint compatibility, and
inference. Changes should preserve released-checkpoint compatibility and avoid adding unrelated robot stacks,
experiment dashboards, or cluster-specific infrastructure.

Before opening a pull request, run:

```bash
pytest
ruff check .
ruff format --check .
```

Bug reports should include the Python, PyTorch, CUDA, and GPU versions, a minimal reproduction, and the complete error
traceback.
