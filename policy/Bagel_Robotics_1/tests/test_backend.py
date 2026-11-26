"""Check required launch controls without touching a GPU."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


def test_backend_refuses_missing_startup_controls():
    policy = Path(__file__).resolve().parents[1]
    source = "from backend import configure_backend; configure_backend(1)"
    env = dict(os.environ, PYTHONPATH=str(policy), CUDA_VISIBLE_DEVICES="")
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "PYTHONHASHSEED=1" in result.stderr
    env["PYTHONHASHSEED"] = "1"
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "CUBLAS_WORKSPACE_CONFIG" in result.stderr


def test_backend_cpu_configuration_and_seed_reproducibility():
    policy = Path(__file__).resolve().parents[1]
    source = """
import json, random
import numpy as np
import torch
from backend import configure_backend
report = configure_backend(42)
first = (random.random(), float(np.random.random()), float(torch.rand(1)))
configure_backend(42)
second = (random.random(), float(np.random.random()), float(torch.rand(1)))
assert first == second
assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
assert torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
assert not torch.is_deterministic_algorithms_warn_only_enabled()
print(json.dumps(report))
"""
    env = dict(os.environ, PYTHONPATH=str(policy), CUDA_VISIBLE_DEVICES="",
               PYTHONHASHSEED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["deterministic_algorithms"] is True
    assert report["float32_matmul_precision"] == "highest"
