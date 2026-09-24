from __future__ import annotations

import numpy as np
import pytest
import torch

import worldscape_policy.cli.serve as server_module
from worldscape_policy.cli.serve import (
    NativePolicyHandler,
    _decode_message,
    _encode_message,
    ensure_single_rank,
    parse_request,
)
from worldscape_policy.types import WorldActionOutput


class FakeRuntime:
    def __init__(self) -> None:
        self.mode = None
        self.pending = False
        self.observation = None
        self.prompts = None

    def reset(self, mode) -> None:
        self.mode = mode
        self.pending = False

    def predict(self, *, observation, prompts, generator):
        if self.mode is None:
            raise RuntimeError("reset required")
        if self.pending:
            raise RuntimeError("prediction pending")
        self.pending = True
        self.observation = observation
        self.prompts = prompts
        return WorldActionOutput(
            action=torch.tensor([[[1.0, 2.0]]]),
            metrics={"sample": torch.tensor(3.0)},
        )

    def commit(self) -> None:
        if not self.pending:
            raise RuntimeError("nothing pending")
        self.pending = False

    def discard(self) -> None:
        self.pending = False


def _predict_request() -> dict:
    return {
        "operation": "predict",
        "observation": {
            "images": np.zeros((1, 2, 1, 3, 4, 4), dtype=np.uint8),
            "head_view": np.zeros((1, 1, 3, 4, 4), dtype=np.uint8),
            "proprioception": np.zeros((1, 1, 5), dtype=np.float32),
            "embodiment_id": np.array([2], dtype=np.int64),
        },
        "prompts": {"language_instruction": ["pick up the cube"]},
    }


def test_handler_enforces_predict_commit_lifecycle():
    runtime = FakeRuntime()
    handler = NativePolicyHandler(
        runtime,
        device="cpu",
        visual_input_range="uint8",
        clock=iter((1.0, 1.01, 2.0, 2.02, 3.0, 3.03)).__next__,
    )

    reset = handler.handle(
        {"operation": "reset", "mode": "interactive", "seed": 17}
    )
    predicted = handler.handle(_predict_request())
    prediction_id = predicted["result"]["prediction_id"]
    committed = handler.handle(
        {"operation": "commit", "prediction_id": prediction_id}
    )

    assert reset["ok"] is True
    assert predicted["ok"] is True
    np.testing.assert_array_equal(
        predicted["result"]["action"], np.array([[[1.0, 2.0]]], dtype=np.float32)
    )
    assert predicted["record"]["latency_ms"] == pytest.approx(20.0)
    assert committed["ok"] is True
    assert runtime.pending is False
    assert runtime.observation.images.dtype is torch.uint8
    assert runtime.prompts.language_instruction == ["pick up the cube"]


def test_handler_returns_typed_error_record_and_can_discard():
    runtime = FakeRuntime()
    handler = NativePolicyHandler(
        runtime,
        device="cpu",
        visual_input_range="uint8",
    )

    failed = handler.handle(_predict_request())
    assert failed["ok"] is False
    assert failed["record"]["status"] == "failed"
    assert failed["error"]["type"] == "RuntimeError"

    handler.handle({"operation": "reset", "mode": "interactive"})
    predicted = handler.handle(_predict_request())
    discarded = handler.handle(
        {
            "operation": "discard",
            "prediction_id": predicted["result"]["prediction_id"],
        }
    )
    assert discarded["ok"] is True
    assert runtime.pending is False


@pytest.mark.parametrize("operation", ["commit", "discard"])
def test_commit_and_discard_require_exact_pending_prediction_id(operation):
    runtime = FakeRuntime()
    handler = NativePolicyHandler(
        runtime,
        device="cpu",
        visual_input_range="uint8",
    )
    handler.handle({"operation": "reset", "mode": "interactive"})
    predicted = handler.handle(_predict_request())
    prediction_id = predicted["result"]["prediction_id"]

    missing = handler.handle({"operation": operation})
    stale = handler.handle(
        {"operation": operation, "prediction_id": f"stale-{prediction_id}"}
    )

    assert missing["ok"] is False
    assert missing["error"]["message"] == "Missing required field 'prediction_id'"
    assert stale["ok"] is False
    assert "does not match" in stale["error"]["message"]
    assert runtime.pending is True

    accepted = handler.handle(
        {"operation": operation, "prediction_id": prediction_id}
    )
    assert accepted["ok"] is True
    assert accepted["result"]["prediction_id"] == prediction_id
    assert runtime.pending is False


def test_visual_range_is_explicit_and_validated():
    handler = NativePolicyHandler(
        FakeRuntime(),
        device="cpu",
        visual_input_range="zero_one",
    )
    handler.handle({"operation": "reset", "mode": "interactive"})
    request = _predict_request()
    request["observation"]["images"] = np.full(
        (1, 2, 1, 3, 4, 4), 255, dtype=np.uint8
    )

    response = handler.handle(request)

    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"
    assert "visual_input_range" in response["error"]["message"]


@pytest.mark.parametrize(
    ("visual_fields", "goal_shape", "demo_shape"),
    [
        ({}, None, None),
        (
            {"goal_images": np.zeros((1, 1, 3, 4, 4), dtype=np.uint8)},
            (1, 1, 3, 4, 4),
            None,
        ),
        (
            {"demo_videos": np.zeros((1, 50, 1, 3, 4, 4), dtype=np.uint8)},
            None,
            (1, 50, 1, 3, 4, 4),
        ),
        (
            {"demo_videos": np.zeros((1, 50, 3, 3, 4, 4), dtype=np.uint8)},
            None,
            (1, 50, 3, 3, 4, 4),
        ),
    ],
)
def test_server_accepts_text_goal_and_uniform_video_payloads(
    visual_fields, goal_shape, demo_shape
):
    runtime = FakeRuntime()
    handler = NativePolicyHandler(
        runtime,
        device="cpu",
        visual_input_range="uint8",
    )
    handler.handle({"operation": "reset", "mode": "interactive"})
    request = _predict_request()
    request["prompts"].update(visual_fields)

    response = handler.handle(request)

    assert response["ok"] is True
    assert (
        None
        if runtime.prompts.goal_images is None
        else tuple(runtime.prompts.goal_images.shape)
    ) == goal_shape
    assert (
        None
        if runtime.prompts.demo_videos is None
        else tuple(runtime.prompts.demo_videos.shape)
    ) == demo_shape


def test_request_schema_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        parse_request(
            {
                "operation": "reset",
                "mode": "interactive",
                "untyped_escape_hatch": True,
            }
        )


@pytest.mark.parametrize("encoding", ["json", "msgpack"])
def test_json_and_msgpack_roundtrip(encoding):
    value = {"ok": True, "array": np.arange(4, dtype=np.float32)}

    encoded = _encode_message(value, encoding)
    actual_encoding, decoded = _decode_message(encoded)

    assert actual_encoding == encoding
    np.testing.assert_array_equal(decoded["array"], value["array"])


def test_native_server_rejects_distributed_launch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(RuntimeError, match="exactly one rank"):
        ensure_single_rank()


def test_server_command_is_native_and_rejects_removed_legacy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        server_module, "native_main", lambda argv: calls.append(("native", argv))
    )

    server_module.main(["--checkpoint-dir", "/native"])
    with pytest.raises(SystemExit, match="Legacy serving was removed"):
        server_module.main(["--legacy", "--model-path", "/legacy"])

    assert calls == [("native", ["--checkpoint-dir", "/native"])]
