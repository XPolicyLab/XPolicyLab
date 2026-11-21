"""Numerical controls recorded by the checkpoint-1000 node-2 evaluation."""
import os
import random


def configure_backend(seed):
    if os.environ.get("PYTHONHASHSEED") != "1":
        raise RuntimeError("Launch Python with PYTHONHASHSEED=1 (see setup_eval_policy_server.sh)")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Launch Python with CUBLAS_WORKSPACE_CONFIG=:4096:8")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}
