from __future__ import annotations

import math
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from worldscape_policy.training.runtime import (
    NativeDistributedConfig,
    NativeTrainingRuntime,
    TrainingForwardAdapter,
)
from worldscape_policy.training.scheduler import (
    NativeLRSchedulerConfig,
    build_lr_scheduler,
    lr_factor,
)


class _LinearTrainingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def training_forward(self, *, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value


def _ddp_worker(rank: int, world_size: int, port: int, queue) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    runtime = NativeTrainingRuntime(
        NativeDistributedConfig(
            backend="ddp",
            process_group_backend="gloo",
            seed=17,
            device="cpu",
        )
    )
    policy = _LinearTrainingPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    runtime.setup(policy, optimizer)
    output = runtime.forward(value=torch.tensor(float(rank + 1)))
    runtime.backward(output.square())
    runtime.step(optimizer)
    reduced = runtime.all_reduce_metrics({"rank_value": float(rank + 1)})
    queue.put((rank, policy.weight.item(), reduced["rank_value"], runtime.seed))
    runtime.close()


def test_hf_linear_warmup_factors_are_exact() -> None:
    factors = [
        lr_factor(step, total_steps=10, warmup_steps=2, schedule="linear")
        for step in range(11)
    ]
    assert factors == pytest.approx(
        [0.0, 0.5, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
    )
    assert [
        lr_factor(step, total_steps=4, warmup_steps=1, schedule="constant")
        for step in range(4)
    ] == [0.0, 1.0, 1.0, 1.0]


def test_hf_cosine_warmup_factors_are_exact() -> None:
    factors = [
        lr_factor(step, total_steps=10, warmup_steps=2, schedule="cosine")
        for step in (0, 1, 2, 6, 10)
    ]
    assert factors == pytest.approx(
        [
            0.0,
            0.5,
            1.0,
            0.5 * (1.0 + math.cos(math.pi * 0.5)),
            0.0,
        ]
    )


def test_scheduler_state_resume_preserves_factor() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=2.0)
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=10,
        config=NativeLRSchedulerConfig(warmup_ratio=0.2),
    )
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()

    other_parameter = nn.Parameter(torch.tensor(1.0))
    other_optimizer = torch.optim.SGD([other_parameter], lr=2.0)
    resumed = build_lr_scheduler(
        other_optimizer,
        total_steps=10,
        config=NativeLRSchedulerConfig(warmup_ratio=0.2),
    )
    resumed.load_state_dict(state)
    other_optimizer.load_state_dict(optimizer.state_dict())
    optimizer.step()
    scheduler.step()
    other_optimizer.step()
    resumed.step()
    assert other_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed unavailable")
def test_cpu_gloo_ddp_synchronizes_forward_gradients(tmp_path: Path) -> None:
    del tmp_path
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(_ddp_worker, args=(2, port, queue), nprocs=2, join=True)
    results = sorted(queue.get() for _ in range(2))
    assert results[0][1] == pytest.approx(results[1][1])
    assert results[0][2] == results[1][2] == pytest.approx(1.5)
    assert [result[3] for result in results] == [17, 18]


def test_deepspeed_missing_dependency_is_actionable(monkeypatch) -> None:
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "deepspeed":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    runtime = NativeTrainingRuntime(NativeDistributedConfig(backend="deepspeed"))
    policy = _LinearTrainingPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    with pytest.raises(ImportError, match=r"worldscape-policy\[train\]"):
        runtime.setup(policy, optimizer)


@pytest.mark.parametrize("stage", [2, 3])
def test_deepspeed_accepts_supported_zero_stages(monkeypatch, stage) -> None:
    captured = {}

    def initialize(**kwargs):
        captured["config"] = kwargs["config"]
        return kwargs["model"], kwargs["optimizer"], None, None

    monkeypatch.setitem(
        sys.modules, "deepspeed", SimpleNamespace(initialize=initialize)
    )
    runtime = NativeTrainingRuntime(
        NativeDistributedConfig(
            backend="deepspeed",
            device="cpu",
            deepspeed_config={"zero_optimization": {"stage": stage}},
        )
    )
    policy = _LinearTrainingPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    runtime.setup(policy, optimizer)

    assert captured["config"]["zero_optimization"]["stage"] == stage


def test_deepspeed_checkpoint_operations_delegate_to_engine(tmp_path: Path) -> None:
    calls = []

    class Engine:
        def save_checkpoint(self, directory, *, tag):
            calls.append(("save", directory, tag))
            return True

        def load_checkpoint(self, directory, *, tag, **kwargs):
            calls.append(("load", directory, tag, kwargs))
            return str(Path(directory) / tag), {}

        def save_16bit_model(self, directory, *, save_filename):
            calls.append(("export", directory, save_filename))
            return True

    runtime = NativeTrainingRuntime(
        NativeDistributedConfig(backend="single", device="cpu")
    )
    runtime.engine = Engine()
    engine_dir = str(tmp_path / "engine")
    export_dir = str(tmp_path / "model")

    runtime.save_deepspeed_checkpoint(engine_dir, tag="step")
    runtime.load_deepspeed_checkpoint(engine_dir, tag="step")
    runtime.save_deepspeed_16bit_model(
        export_dir,
        save_filename="policy.pt",
    )

    assert calls == [
        ("save", engine_dir, "step"),
        (
            "load",
            engine_dir,
            "step",
            {
                "load_module_strict": True,
                "load_optimizer_states": True,
                "load_lr_scheduler_states": False,
            },
        ),
        ("export", export_dir, "policy.pt"),
    ]


def test_deepspeed_adapter_state_has_policy_prefix_for_export_normalization() -> None:
    policy = _LinearTrainingPolicy()
    adapter = TrainingForwardAdapter(policy)

    assert set(adapter.state_dict()) == {"policy.weight"}
    assert set(policy.state_dict()) == {"weight"}
