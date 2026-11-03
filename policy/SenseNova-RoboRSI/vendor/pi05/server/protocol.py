"""Binary ndarray protocol and a small websocket inference server."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
import http
import logging
import math
from typing import Any, Hashable, Protocol, Sequence

import msgpack
import numpy as np


logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
_NDARRAY_MARKER = b"__ndarray__"
_NUMPY_SCALAR_MARKER = b"__npgeneric__"
_GREEN = "\033[1;32m"
_RESET = "\033[0m"


class FatalRequestError(RuntimeError):
    """Request validation failure that requires the model server to stop."""


class InferenceService(Protocol):
    """Synchronous inference contract hosted by the websocket server."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def infer(self, request: dict[str, Any]) -> dict[str, Any]: ...


class BatchInferenceService(InferenceService, Protocol):
    """Inference contract for services that can process compatible requests together."""

    def batch_key(self, request: dict[str, Any]) -> Hashable: ...

    def infer_batch(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]: ...


@dataclass
class _PendingInference:
    request: dict[str, Any]
    batch_key: Hashable
    future: asyncio.Future[dict[str, Any]]


class DynamicRequestBatcher:
    """Collect compatible requests for one synchronous batched inference call."""

    def __init__(
        self,
        service: BatchInferenceService,
        *,
        max_batch_size: int,
        batch_wait_ms: float,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be finite and non-negative")

        self.service = service
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self._queue: asyncio.Queue[_PendingInference] = asyncio.Queue()
        self._deferred: deque[_PendingInference] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._inflight: list[_PendingInference] = []
        self._closed = False

    async def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Queue one request and resolve its response in original request order."""
        if self._closed:
            raise RuntimeError("Dynamic request batcher is closed")

        batch_key = self.service.batch_key(request)
        hash(batch_key)
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _PendingInference(
                request=request,
                batch_key=batch_key,
                future=future,
            )
        )
        self._ensure_worker()
        return await future

    async def close(self) -> None:
        """Stop pending work and fail requests that were not executed."""
        self._closed = True
        worker = self._worker
        if worker is not None:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._worker = None
        for item in self._inflight:
            self._fail_pending(item, RuntimeError("Server stopped"))
        self._inflight = []
        while self._deferred:
            self._fail_pending(self._deferred.popleft(), RuntimeError("Server stopped"))
        while not self._queue.empty():
            self._fail_pending(self._queue.get_nowait(), RuntimeError("Server stopped"))

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="dynamic-inference-batcher")

    async def _run(self) -> None:
        try:
            while not self._closed:
                first = self._next_pending()
                if first is None:
                    return
                pending = await self._collect_batch(first)
                await self._execute_batch(pending)
        finally:
            self._worker = None
            if not self._closed and (self._deferred or not self._queue.empty()):
                self._ensure_worker()

    def _next_pending(self) -> _PendingInference | None:
        if self._deferred:
            return self._deferred.popleft()
        if self._queue.empty():
            return None
        return self._queue.get_nowait()

    async def _collect_batch(
        self,
        first: _PendingInference,
    ) -> list[_PendingInference]:
        pending = [first]
        deadline = asyncio.get_running_loop().time() + self.batch_wait_s
        while len(pending) < self.max_batch_size:
            # Do not take from ``_deferred`` while assembling this batch.  An
            # incompatible deferred request would be put straight back into
            # the same deque and be selected again forever.
            candidate = self._next_queued_pending()
            if candidate is None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(self._queue.get(), remaining)
                except asyncio.TimeoutError:
                    break

            if candidate.batch_key == first.batch_key:
                pending.append(candidate)
            else:
                self._deferred.append(candidate)
        return pending

    def _next_queued_pending(self) -> _PendingInference | None:
        if self._queue.empty():
            return None
        return self._queue.get_nowait()

    async def _execute_batch(self, pending: list[_PendingInference]) -> None:
        active = [item for item in pending if not item.future.cancelled()]
        if not active:
            return
        self._inflight = active
        try:
            results = await asyncio.to_thread(
                self.service.infer_batch,
                [item.request for item in active],
            )
            if len(results) != len(active):
                raise RuntimeError(
                    "Batched inference returned a different number of results than requests"
                )
        except asyncio.CancelledError:
            if self._closed:
                for item in active:
                    self._fail_pending(item, RuntimeError("Server stopped"))
            raise
        except Exception as exc:
            for item in active:
                self._fail_pending(item, exc)
        else:
            for item, result in zip(active, results, strict=True):
                if not item.future.cancelled() and not item.future.done():
                    item.future.set_result(dict(result))
        finally:
            self._inflight = []

    @staticmethod
    def _fail_pending(item: _PendingInference, error: Exception) -> None:
        if not item.future.cancelled() and not item.future.done():
            item.future.set_exception(error)


def log_server_ready(
    server_logger: logging.Logger,
    *,
    service: str,
    host: str,
    port: int,
) -> None:
    """Emit one prominent log after model loading and socket binding finish."""
    server_logger.info(
        "%s✓ %s READY — ws://%s:%d — health http://localhost:%d/healthz%s",
        _GREEN,
        service.upper(),
        host,
        port,
        port,
        _RESET,
    )


def pack_message(value: Any) -> bytes:
    """Pack values using the same ndarray wire format as OpenPI clients."""
    return msgpack.packb(value, default=_encode_value, use_bin_type=True)


def unpack_message(payload: bytes | bytearray | memoryview) -> Any:
    """Unpack one binary message produced by :func:`pack_message`."""
    return msgpack.unpackb(payload, object_hook=_decode_value, raw=False)


def _encode_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in {
        "V",
        "O",
        "c",
    }:
        raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _NDARRAY_MARKER: True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            _NUMPY_SCALAR_MARKER: True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Unsupported protocol value: {type(value).__name__}")


def _decode_value(value: dict[Any, Any]) -> Any:
    if _NDARRAY_MARKER in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if _NUMPY_SCALAR_MARKER in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class WebsocketInferenceServer:
    """Host one preloaded inference service over websocket + msgpack."""

    def __init__(
        self,
        service: InferenceService,
        *,
        host: str,
        port: int,
    ) -> None:
        self.service = service
        self.host = host
        self.port = port
        self._inference_lock = asyncio.Lock()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets.asyncio.server as websocket_server

        async with websocket_server.serve(
            self._handle_connection,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=health_check,
        ) as server:
            log_server_ready(
                logger,
                service=str(self.service.metadata.get("service", "model")),
                host=self.host,
                port=self.port,
            )
            await server.serve_forever()

    async def _handle_connection(self, websocket: Any) -> None:
        import websockets

        await websocket.send(
            pack_message(
                {
                    "type": "metadata",
                    "protocol_version": PROTOCOL_VERSION,
                    **self.service.metadata,
                }
            )
        )
        while True:
            try:
                payload = await websocket.recv()
            except websockets.ConnectionClosed:
                return

            if not isinstance(payload, bytes):
                await websocket.send(
                    pack_message(
                        {
                            "ok": False,
                            "error": {
                                "type": "TypeError",
                                "message": "Inference requests must be binary msgpack messages",
                            },
                        }
                    )
                )
                continue

            try:
                request = unpack_message(payload)
                if not isinstance(request, dict):
                    raise TypeError("Inference request must be a mapping")
                async with self._inference_lock:
                    result = await asyncio.to_thread(self.service.infer, request)
                response = {"ok": True, "result": result}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "%s inference failed",
                    self.service.metadata.get("service"),
                )
                response = {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }

            with contextlib.suppress(websockets.ConnectionClosed):
                await websocket.send(pack_message(response))


class DynamicBatchInferenceServer:
    """Websocket server that batches compatible requests without changing the wire API."""

    def __init__(
        self,
        service: BatchInferenceService,
        *,
        host: str,
        port: int,
        max_batch_size: int = 1,
        batch_wait_ms: float = 0,
    ) -> None:
        self.service = service
        self.host = host
        self.port = port
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        self._batcher = DynamicRequestBatcher(
            service,
            max_batch_size=max_batch_size,
            batch_wait_ms=batch_wait_ms,
        )
        self._shutdown_event = asyncio.Event()
        self._fatal_error: FatalRequestError | None = None

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets.asyncio.server as websocket_server

        try:
            async with websocket_server.serve(
                self._handle_connection,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                process_request=health_check,
            ):
                log_server_ready(
                    logger,
                    service=str(self.service.metadata.get("service", "model")),
                    host=self.host,
                    port=self.port,
                )
                await self._shutdown_event.wait()
                if self._fatal_error is not None:
                    raise RuntimeError(
                        f"{self.service.metadata.get('service', 'model')} server stopped "
                        f"after a fatal request error: {self._fatal_error}"
                    ) from self._fatal_error
        finally:
            await self._batcher.close()

    async def _handle_connection(self, websocket: Any) -> None:
        import websockets

        await websocket.send(
            pack_message(
                {
                    "type": "metadata",
                    "protocol_version": PROTOCOL_VERSION,
                    "dynamic_batching": True,
                    "max_batch_size": self.max_batch_size,
                    "batch_wait_ms": self.batch_wait_ms,
                    **self.service.metadata,
                }
            )
        )
        while True:
            try:
                payload = await websocket.recv()
            except websockets.ConnectionClosed:
                return

            fatal_error: FatalRequestError | None = None
            if not isinstance(payload, bytes):
                response = {
                    "ok": False,
                    "error": {
                        "type": "TypeError",
                        "message": "Inference requests must be binary msgpack messages",
                    },
                }
            else:
                try:
                    request = unpack_message(payload)
                    if not isinstance(request, dict):
                        raise TypeError("Inference request must be a mapping")
                    result = await self._batcher.infer(request)
                    response = {"ok": True, "result": result}
                except asyncio.CancelledError:
                    raise
                except FatalRequestError as exc:
                    logger.critical(
                        "%s received a fatal request; shutting down the server: %s",
                        self.service.metadata.get("service"),
                        exc,
                        exc_info=True,
                    )
                    fatal_error = exc
                    response = {
                        "ok": False,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                except Exception as exc:
                    logger.exception(
                        "%s inference failed",
                        self.service.metadata.get("service"),
                    )
                    response = {
                        "ok": False,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }

            with contextlib.suppress(websockets.ConnectionClosed):
                await websocket.send(pack_message(response))
            if fatal_error is not None:
                if self._fatal_error is None:
                    self._fatal_error = fatal_error
                self._shutdown_event.set()
                return


def health_check(connection: Any, request: Any) -> Any | None:
    """Serve the common HTTP health endpoint during websocket handshake."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
