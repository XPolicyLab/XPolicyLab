from __future__ import annotations

import builtins
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldscape_policy.conditioning.auto_conditioner import AutoConditioner
from worldscape_policy.conditioning.text.t5 import WanTextEncoder
from worldscape_policy.conditioning.vlm import qwen3vl
from worldscape_policy.conditioning.vlm.qwen3vl import (
    QwenPlanningEncoder,
    _transformers,
)


def test_owned_t5_state_dict_keeps_legacy_parameter_names():
    encoder = WanTextEncoder(
        vocab=16,
        dim=8,
        dim_attn=8,
        dim_ffn=16,
        num_heads=2,
        num_layers=1,
        num_buckets=4,
    )
    assert tuple(encoder.state_dict()) == (
        "token_embedding.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn.q.weight",
        "blocks.0.attn.k.weight",
        "blocks.0.attn.v.weight",
        "blocks.0.attn.o.weight",
        "blocks.0.norm2.weight",
        "blocks.0.ffn.gate.0.weight",
        "blocks.0.ffn.fc1.weight",
        "blocks.0.ffn.fc2.weight",
        "blocks.0.pos_embedding.embedding.weight",
        "norm.weight",
    )


def test_missing_transformers_fails_clearly_for_auto(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="Auto conditioning.*transformers"):
        _transformers()


def test_last_token_mode_does_not_construct_qformer(monkeypatch):
    class FakeInterface(nn.Module):
        def __init__(self, **_):
            super().__init__()
            self.model = nn.Module()
            self.model.config = SimpleNamespace(hidden_size=8)

    monkeypatch.setattr(qwen3vl, "Qwen3VLInterface", FakeInterface)

    encoder = QwenPlanningEncoder(vlm_token_mode="last")

    assert encoder.qformer is None
    assert not any(key.startswith("qformer.") for key in encoder.state_dict())


def test_qformer_mode_constructs_qformer(monkeypatch):
    class FakeInterface(nn.Module):
        def __init__(self, **_):
            super().__init__()
            self.model = nn.Module()
            self.model.config = SimpleNamespace(hidden_size=8)

    monkeypatch.setattr(qwen3vl, "Qwen3VLInterface", FakeInterface)

    encoder = QwenPlanningEncoder(
        vlm_token_mode="qformer",
        qformer_start_layer=0,
        qformer_end_layer=1,
        qformer_num_heads=2,
        qformer_output_dim=8,
    )

    assert encoder.qformer is not None
    assert any(key.startswith("qformer.") for key in encoder.state_dict())


def test_auto_conditioner_rejects_mixed_perception_and_planning_widths():
    with pytest.raises(ValueError, match="token widths differ"):
        AutoConditioner._require_matching_token_widths(
            torch.zeros(2, 8, 4096),
            torch.zeros(2, 4, 2560),
        )


def test_interactive_package_imports_without_groot_or_transformers():
    script = """
import builtins
import importlib.util
import sys
sys.path[:0] = ["src", "."]
real_find_spec = importlib.util.find_spec
real_import = builtins.__import__
def find_spec(name, *args, **kwargs):
    if name == "transformers" or name.startswith("transformers."):
        return None
    return real_find_spec(name, *args, **kwargs)
def blocked_import(name, *args, **kwargs):
    if name == "groot" or name.startswith("groot."):
        raise ModuleNotFoundError("blocked groot")
    if name == "transformers" or name.startswith("transformers."):
        raise ModuleNotFoundError("blocked transformers")
    return real_import(name, *args, **kwargs)
importlib.util.find_spec = find_spec
builtins.__import__ = blocked_import
from worldscape_policy.conditioning.interactive_conditioner import InteractiveConditioner
assert InteractiveConditioner.__module__.startswith("worldscape_policy.")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
