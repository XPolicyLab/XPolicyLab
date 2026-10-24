"""WorldScape-owned websocket policy server and legacy command dispatch."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from worldscape_policy.memory.visual.normalization import VisualInputRange
from worldscape_policy.types import (
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)
from worldscape_policy.wam.wan22.distributed import Wan22DistributedContext

LOGGER = logging.getLogger(__name__)
Operation = Literal["reset", "predict", "commit", "discard"]


@dataclass(frozen=True)
class ObservationRequest:
    """Wire representation of the public observation batch."""

    images: Any
    head_view: Any
    proprioception: Any
    embodiment_id: Any

    @classmethod
    def parse(cls, value: Any) -> ObservationRequest:
        data = _mapping(value, "observation")
        return cls(**{name: _required(data, name) for name in cls.__annotations__})


@dataclass(frozen=True)
class PromptsRequest:
    """Wire representation of mode-specific text and visual prompts."""

    vlm_planning_text: list[str] | None = None
    language_instruction: list[str] | None = None
    negative_vlm_text: list[str] | None = None
    negative_language_instruction: list[str] | None = None
    planning_labels_text: list[str | None] | None = None
    goal_images: Any | None = None
    demo_videos: Any | None = None

    @classmethod
    def parse(cls, value: Any) -> PromptsRequest:
        data = _mapping(value, "prompts")
        unknown = set(data) - set(cls.__annotations__)
        if unknown:
            raise ValueError(f"prompts contains unknown fields: {sorted(unknown)}")
        values: dict[str, Any] = {}
        text_fields = {
            "vlm_planning_text",
            "language_instruction",
            "negative_vlm_text",
            "negative_language_instruction",
            "planning_labels_text",
        }
        for name in cls.__annotations__:
            item = data.get(name)
            if name in text_fields:
                item = _optional_string_list(item, f"prompts.{name}")
            values[name] = item
        return cls(**values)


@dataclass(frozen=True)
class ResetRequest:
    operation: Literal["reset"]
    mode: InteractionMode
    seed: int


@dataclass(frozen=True)
class PredictRequest:
    operation: Literal["predict"]
    observation: ObservationRequest
    prompts: PromptsRequest


@dataclass(frozen=True)
class CommitRequest:
    operation: Literal["commit"]
    prediction_id: str


@dataclass(frozen=True)
class DiscardRequest:
    operation: Literal["discard"]
    prediction_id: str


ServerRequest = ResetRequest | PredictRequest | CommitRequest | DiscardRequest


@dataclass(frozen=True)
class RequestRecord:
    operation: str
    status: Literal["completed", "failed"]
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class NativeServerConfig:
    checkpoint_dir: str
    visual_input_range: VisualInputRange
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"
    expected_mode: str | None = None
    default_seed: int = 0
    include_video: bool = False
    distributed_context: Wan22DistributedContext | None = None


def parse_request(value: Any, *, default_seed: int = 0) -> ServerRequest:
    """Validate and type one JSON/msgpack request."""

    data = _mapping(value, "request")
    operation = data.get("operation", data.get("endpoint"))
    if operation == "infer":
        operation = "predict"
    if operation == "reset":
        _only(data, {"operation", "endpoint", "mode", "seed"}, "reset request")
        return ResetRequest(
            operation="reset",
            mode=InteractionMode.parse(_required(data, "mode")),
            seed=_integer(data.get("seed", default_seed), "seed"),
        )
    if operation == "predict":
        _only(
            data,
            {"operation", "endpoint", "observation", "prompts"},
            "predict request",
        )
        return PredictRequest(
            operation="predict",
            observation=ObservationRequest.parse(_required(data, "observation")),
            prompts=PromptsRequest.parse(data.get("prompts", {})),
        )
    if operation in {"commit", "discard"}:
        _only(data, {"operation", "endpoint", "prediction_id"}, f"{operation} request")
        prediction_id = _required(data, "prediction_id")
        if not isinstance(prediction_id, str):
            raise TypeError("prediction_id must be a string")
        if not prediction_id:
            raise ValueError("prediction_id must be a non-empty string")
        if operation == "commit":
            return CommitRequest(operation="commit", prediction_id=prediction_id)
        return DiscardRequest(operation="discard", prediction_id=prediction_id)
    raise ValueError("operation must be one of: reset, predict, commit, discard")


class NativePolicyHandler:
    """Dependency-light request handler around a transactional PolicyRuntime."""

    def __init__(
        self,
        runtime: Any,
        *,
        device: str | torch.device,
        visual_input_range: VisualInputRange,
        default_seed: int = 0,
        include_video: bool = False,
        clock: Any = time.perf_counter,
    ) -> None:
        if visual_input_range not in {"uint8", "zero_one", "minus_one_one"}:
            raise ValueError(f"Unknown visual input range: {visual_input_range!r}")
        self.runtime = runtime
        self.device = torch.device(device)
        self.visual_input_range = visual_input_range
        self.default_seed = default_seed
        self.include_video = include_video
        self.clock = clock
        self._generator: torch.Generator | None = None
        self._pending_id: str | None = None

    def handle(self, payload: Any) -> dict[str, Any]:
        """Handle one decoded request and always return a structured response."""

        started = self.clock()
        operation = _operation_hint(payload)
        try:
            request = parse_request(payload, default_seed=self.default_seed)
            operation = request.operation
            result = self._dispatch(request)
            record = RequestRecord(
                operation=operation,
                status="completed",
                latency_ms=max(0.0, (self.clock() - started) * 1000.0),
            )
            response = {
                "ok": True,
                "operation": operation,
                "result": result,
                "record": asdict(record),
            }
            LOGGER.info("policy_request %s", json.dumps(asdict(record), sort_keys=True))
            return response
        except Exception as exc:
            record = RequestRecord(
                operation=operation,
                status="failed",
                latency_ms=max(0.0, (self.clock() - started) * 1000.0),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            LOGGER.warning("policy_request %s", json.dumps(asdict(record), sort_keys=True))
            return {
                "ok": False,
                "operation": operation,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "record": asdict(record),
            }

    def _dispatch(self, request: ServerRequest) -> dict[str, Any]:
        if isinstance(request, ResetRequest):
            self._generator = None
            self._pending_id = None
            self.runtime.reset(request.mode)
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(request.seed)
            return {"mode": request.mode.value, "seed": request.seed}
        if isinstance(request, PredictRequest):
            if self._generator is None:
                raise RuntimeError("Call reset before predict")
            observation = self._observation(request.observation)
            prompts = self._prompts(request.prompts, observation.images.shape[0])
            output = self.runtime.predict(
                observation=observation,
                prompts=prompts,
                generator=self._generator,
            )
            prediction_id = uuid.uuid4().hex
            self._pending_id = prediction_id
            return self._output(output, prediction_id)
        self._check_prediction_id(request.prediction_id)
        if isinstance(request, CommitRequest):
            self.runtime.commit()
        else:
            self.runtime.discard()
        prediction_id = self._pending_id
        self._pending_id = None
        return {"prediction_id": prediction_id}

    def _observation(self, request: ObservationRequest) -> ObservationBatch:
        observation = ObservationBatch(
            images=self._visual_tensor(request.images, "observation.images"),
            head_view=self._visual_tensor(
                request.head_view, "observation.head_view"
            ),
            proprioception=_tensor(
                request.proprioception,
                name="observation.proprioception",
                dtype=torch.float32,
                device=self.device,
            ),
            embodiment_id=_tensor(
                request.embodiment_id,
                name="observation.embodiment_id",
                dtype=torch.long,
                device=self.device,
            ),
        )
        observation.validate()
        return observation

    def _prompts(self, request: PromptsRequest, batch_size: int) -> PromptBatch:
        prompts = PromptBatch(
            vlm_planning_text=request.vlm_planning_text,
            language_instruction=request.language_instruction,
            negative_vlm_text=request.negative_vlm_text,
            negative_language_instruction=request.negative_language_instruction,
            planning_labels_text=request.planning_labels_text,
            goal_images=(
                self._visual_tensor(request.goal_images, "prompts.goal_images")
                if request.goal_images is not None
                else None
            ),
            demo_videos=(
                self._visual_tensor(request.demo_videos, "prompts.demo_videos")
                if request.demo_videos is not None
                else None
            ),
        )
        prompts.validate(batch_size)
        return prompts

    def _visual_tensor(self, value: Any, name: str) -> Tensor:
        dtype = torch.uint8 if self.visual_input_range == "uint8" else torch.float32
        source = value if isinstance(value, Tensor) else np.asarray(value)
        tensor = _tensor(value, name=name, dtype=dtype, device=self.device)
        if self.visual_input_range == "uint8":
            source_dtype = getattr(source, "dtype", None)
            if source_dtype not in {np.dtype("uint8"), torch.uint8} and isinstance(
                value, (np.ndarray, Tensor)
            ):
                raise TypeError(f"{name} must be uint8 for visual_input_range='uint8'")
            if not isinstance(value, (np.ndarray, Tensor)) and source.dtype.kind not in {
                "i",
                "u",
            }:
                raise TypeError(f"{name} JSON values must be integers for uint8 images")
            source_size = source.numel() if isinstance(source, Tensor) else source.size
            if source_size and (float(source.min()) < 0 or float(source.max()) > 255):
                raise ValueError(f"{name} values must be in [0, 255]")
        elif tensor.numel():
            minimum, maximum = float(tensor.min()), float(tensor.max())
            low = -1.0 if self.visual_input_range == "minus_one_one" else 0.0
            if minimum < low or maximum > 1.0:
                raise ValueError(
                    f"{name} values [{minimum}, {maximum}] are outside "
                    f"visual_input_range={self.visual_input_range!r}"
                )
        return tensor

    def _output(
        self, output: WorldActionOutput, prediction_id: str
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "prediction_id": prediction_id,
            "action": _wire_tensor(output.require_action()),
            "metrics": {
                name: _wire_tensor(value) for name, value in output.metrics.items()
            },
        }
        if self.include_video and output.video is not None:
            result["video"] = _wire_tensor(output.video)
        return result

    def _check_prediction_id(self, prediction_id: str) -> None:
        if self._pending_id is None:
            raise RuntimeError("There is no pending prediction")
        if prediction_id != self._pending_id:
            raise ValueError("prediction_id does not match the pending prediction")


class NativeWebsocketPolicyServer:
    """Transport adapter; protocol logic remains testable without websockets."""

    def __init__(
        self,
        handler: NativePolicyHandler,
        *,
        host: str,
        port: int,
    ) -> None:
        self.handler = handler
        self.host = host
        self.port = port
        self._connection_lock = asyncio.Lock()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise RuntimeError(
                "Native serving requires the server extra: pip install '.[server]'"
            ) from exc
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
        ) as server:
            LOGGER.info("Native policy server listening on %s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket: Any) -> None:
        if self._connection_lock.locked():
            await websocket.close(
                code=1013,
                reason="Native PolicyRuntime already has an active client",
            )
            return
        async with self._connection_lock:
            metadata = {
                "protocol": "worldscape-policy-v1",
                "encoding": ["msgpack", "json"],
                "visual_input_range": self.handler.visual_input_range,
                "operations": ["reset", "predict", "commit", "discard"],
            }
            await websocket.send(_encode_message(metadata, "msgpack"))
            async for message in websocket:
                encoding, payload = _decode_message(message)
                response = await asyncio.to_thread(self.handler.handle, payload)
                await websocket.send(_encode_message(response, encoding))


def ensure_single_rank(
    distributed_context: Wan22DistributedContext | None = None,
) -> None:
    """Reject distributed serving until a coordinator/worker server exists.

    ``distributed_context`` remains accepted so callers can share construction
    code with the multi-rank WAM library, but it does not make websocket
    serving safe: the native server has neither a worker loop nor request
    coordination and every process would otherwise try to bind the same port.
    """

    del distributed_context
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    if world_size != 1 or rank != 0:
        raise RuntimeError(
            "The native websocket server supports exactly one rank; "
            "distributed WAM serving requires a coordinator/worker request "
            "protocol and worker loop, which are not implemented. "
            f"got rank={rank}, world_size={world_size}. "
            "Run the server with plain Python. Multi-rank WAM remains available "
            "through the library API, not this websocket transport."
        )


def build_native_server(config: NativeServerConfig) -> NativeWebsocketPolicyServer:
    """Build the native checkpoint policy, runtime, handler, and transport."""

    ensure_single_rank(config.distributed_context)
    from worldscape_policy.native_builder import build_wan22_policy_from_checkpoint
    from worldscape_policy.rollout.session import PolicyRuntime

    policy = build_wan22_policy_from_checkpoint(
        config.checkpoint_dir,
        visual_input_range=config.visual_input_range,
        device=config.device,
        expected_mode=config.expected_mode,
        training=False,
        distributed_context=config.distributed_context,
    )
    return NativeWebsocketPolicyServer(
        NativePolicyHandler(
            PolicyRuntime(policy),
            device=config.device,
            visual_input_range=config.visual_input_range,
            default_seed=config.default_seed,
            include_video=config.include_video,
        ),
        host=config.host,
        port=config.port,
    )


def native_main(argv: Sequence[str] | None = None) -> None:
    """Launch the WorldScape-owned single-rank server."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", "--model-path", required=True)
    parser.add_argument(
        "--visual-input-range",
        required=True,
        choices=("uint8", "zero_one", "minus_one_one"),
        help="Exact numeric range used by all observation and prompt images.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-mode", choices=("auto", "interactive"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-video", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    build_native_server(
        NativeServerConfig(
            checkpoint_dir=args.checkpoint_dir,
            visual_input_range=args.visual_input_range,
            host=args.host,
            port=args.port,
            device=args.device,
            expected_mode=args.expected_mode,
            default_seed=args.seed,
            include_video=args.include_video,
        )
    ).serve_forever()


def main(argv: Sequence[str] | None = None) -> None:
    """Launch native WorldScape serving."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--native" in arguments:
        arguments.remove("--native")
    forbidden = {"--legacy", "--source-only-legacy"}.intersection(arguments)
    if forbidden:
        raise SystemExit(
            "Legacy serving was removed; convert the checkpoint and use wsp-serve"
        )
    native_main(arguments)


def _decode_message(message: str | bytes) -> tuple[Literal["json", "msgpack"], Any]:
    if isinstance(message, str):
        return "json", json.loads(message)
    import msgpack
    import msgpack_numpy

    return "msgpack", msgpack.unpackb(
        message, raw=False, object_hook=msgpack_numpy.decode
    )


def _encode_message(value: Any, encoding: Literal["json", "msgpack"]) -> str | bytes:
    if encoding == "json":
        return json.dumps(_json_value(value), separators=(",", ":"))
    import msgpack
    import msgpack_numpy

    return msgpack.packb(value, use_bin_type=True, default=msgpack_numpy.encode)


def _wire_tensor(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    try:
        return torch.as_tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} is not a rectangular numeric tensor") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _required(value: Mapping[str, Any], name: str) -> Any:
    if name not in value:
        raise ValueError(f"Missing required field {name!r}")
    return value[name]


def _only(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _optional_string_list(value: Any, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, list):
        raise TypeError(f"{name} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain only strings")
    return value


def _operation_hint(payload: Any) -> str:
    if isinstance(payload, Mapping):
        value = payload.get("operation", payload.get("endpoint", "unknown"))
        return str(value)
    return "unknown"
