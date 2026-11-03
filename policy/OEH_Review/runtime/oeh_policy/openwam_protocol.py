"""OpenWAM in a session of a shared service (oeh_eval.shared_openwam), as the code that called it in process sees it.

Two flavors, one per way OpenWAM is evaluated.

The official server's methods, over its WebSocket protocol (LIBERO family, RoboTwin):

`OfficialModelBoundary` (libero_boundary) calls `reset` and `predict` on the official `PolicyServer` it was given, and
`LiberoService.ping` reads its `_ckpt_contract()`. With shared OpenWAM serving (oeh_eval.shared_openwam) the official
server lives in the GPU's service process, in a session leased for this attempt; this client gives the boundary the
same three methods over one persistent connection to that session:

- `reset()` sends `reset` and expects `reset_ack`;
- `predict(obs)` sends one `obs` and returns the reply's mapping (`action`, `step`, `latency_ms`, and the service's
  `queue_ms`, `gpu_ms`, `generated`). It is sent once: a dropped connection fails the call, as an ambiguous request
  is never retried on the OEH side (the official client resends; that is the direct condition's behaviour);
- `_ckpt_contract()` pings the session and returns the pong's contract fields.

JSON does not change an action: the official server sends `action.tolist()`, and a float32 survives the round trip
through a Python float. `take_timing()` returns the service time of the calls since the previous take, for the
runtime's model_timing events.

XPolicyLab's OpenWAM Model with its engine in the service (RoboDojo): `remote_xpolicylab_model(Model)` subclasses the
official XPolicyLab adapter, so encoding the observation (world -> robot base -> EEF20), the action decoding back to
ee dicts and the replan cut stay the official code, in this process. Only what touched the weights moves: the
service runs the adapter's `_conditions` (the official `ObsPreprocessor`), `engine.generate_batch` and
`_project_binary_dims` with the Model it loaded once, and returns the actions. RGB images use lossless PNG when smaller than raw bytes;
legacy dtype/shape arrays remain accepted. No input pixels or action values change.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import uuid
from typing import Any, Mapping

MAX_MESSAGE_BYTES = 32 * 1024 * 1024  # the official server's cap
XPOLICYLAB_MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # three raw camera frames per request, base64
# Where an attempt's launcher puts its leased session for the policy server it starts (RoboDojo).
SESSION_ENV = "OEH_OPENWAM_SESSION"


class ProtocolError(RuntimeError):
    pass


class SharedTransportError(ProtocolError):
    """A shared inference connection failed; the request is never replayed."""


def websockets_connect(**fixed):
    """websockets' sync `connect`, given only the options this version names (the pods carry 12.0 to 16.x).

    Before 15 there is no `ping_interval` or `proxy`, and an unknown option reaches socket.create_connection (13.x)
    or raises at once (12.0). From 15 on, `proxy` defaults to HTTP(S)_PROXY / ALL_PROXY, 127.0.0.1 included unless
    no_proxy names it, so it is turned off. `fixed` overrides what the caller passes."""
    import inspect

    from websockets.sync.client import connect

    named = {name for name, p in inspect.signature(connect).parameters.items() if p.kind is p.KEYWORD_ONLY}

    def filtered(url: str, **options):
        options = {"proxy": None, **options, **fixed}
        return connect(url, **{k: v for k, v in options.items() if k in named})

    return filtered


class ProtocolPolicyClient:
    def __init__(self, address: str, *, timeout: float = 600.0, connect=None):
        connect = connect or websockets_connect()
        self.address, self.timeout = address, timeout
        self._connection = connect(address, max_size=MAX_MESSAGE_BYTES, open_timeout=30, ping_interval=None,
                                   close_timeout=10).__enter__()
        self._lock = threading.Lock()
        self._transport_records = []
        self._timing = {"calls": 0, "generates": 0, "queue_ms": 0.0, "gpu_ms": 0.0}

    def _ask(self, message: Mapping[str, Any]) -> dict:
        from websockets.exceptions import ConnectionClosed
        message = {"request_id": uuid.uuid4().hex, **message}
        record = {"request_id": message["request_id"], "type": message.get("type"),
                  "stage": "serialize", "started_utc": time.time()}
        if "client_encode_ms" in message:
            record["encode_ms"] = message["client_encode_ms"]
        start = time.monotonic()
        try:
            with self._lock:
                try:
                    raw = json.dumps(message, separators=(",", ":"))
                    record.update(serialize_ms=(time.monotonic()-start)*1000,
                                  request_bytes=len(raw.encode("utf-8")), stage="send")
                    sent = time.monotonic()
                    self._connection.send(raw)
                    record.update(send_ms=(time.monotonic()-sent)*1000, stage="receive")
                    received = time.monotonic()
                    answer = self._connection.recv(timeout=self.timeout)
                    record.update(receive_ms=(time.monotonic()-received)*1000,
                                  reply_bytes=len(answer.encode("utf-8") if isinstance(answer, str) else answer),
                                  stage="decode")
                    reply = json.loads(answer)
                    record["stage"] = "completed"
                except BaseException as exc:
                    record["error"] = type(exc).__name__
                    raise
                finally:
                    record["elapsed_ms"] = (time.monotonic()-start)*1000
                    # Keep failed send/recv evidence under the same lock as timing;
                    # abort_request releases a stuck socket before timing is drained.
                    if not hasattr(self, "_transport_records"):
                        self._transport_records = []
                    self._transport_records.append(record)
                    self._transport_records = self._transport_records[-64:]
        except (OSError, TimeoutError, ConnectionClosed) as exc:
            raise SharedTransportError(f"shared OpenWAM {message.get('type')} transport failed: "
                                       f"{type(exc).__name__}") from exc
        if not isinstance(reply, dict):
            raise ProtocolError("shared OpenWAM session replied with no object")
        if reply.get("request_id", message["request_id"]) != message["request_id"]:
            raise ProtocolError("shared OpenWAM response request id differs")
        reply.pop("request_id", None)  # transport metadata isn't a model contract field
        if reply.get("type") == "error":
            raise ProtocolError(f"shared OpenWAM session: {reply.get('code')}: {reply.get('message')}")
        return reply

    def set_request_timeout(self, seconds: float) -> None:
        import math
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("shared inference timeout must be finite and positive")
        self.timeout = float(seconds)

    def abort_request(self) -> None:
        """Unblock a send/receive without acquiring the request's held lock."""
        import socket
        try:
            self._connection.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already disconnected

    def reset(self) -> None:
        reply = self._ask({"type": "reset"})
        if reply.get("type") != "reset_ack":
            raise ProtocolError(f"reset answered with {reply.get('type')!r}")

    def predict(self, obs: Mapping[str, Any]) -> dict:
        reply = self._ask({**obs, "type": "obs"})
        if reply.get("type") != "action":
            raise ProtocolError(f"obs answered with {reply.get('type')!r}")
        with self._lock:
            self._timing["calls"] += 1
            self._timing["generates"] += bool(reply.get("generated"))
            self._timing["queue_ms"] += float(reply.get("queue_ms") or 0.0)
            self._timing["gpu_ms"] += float(reply.get("gpu_ms") or 0.0)
        return {key: value for key, value in reply.items() if key != "type"}

    def _ckpt_contract(self) -> dict:
        reply = self._ask({"type": "ping"})
        if reply.get("type") != "pong":
            raise ProtocolError(f"ping answered with {reply.get('type')!r}")
        return {key: value for key, value in reply.items() if key != "type"}

    def take_timing(self) -> dict:
        """Service time of the predicts since the last take: calls, generates, queue_ms, gpu_ms."""
        with self._lock:
            taken = dict(self._timing, queue_ms=round(self._timing["queue_ms"], 3),
                         gpu_ms=round(self._timing["gpu_ms"], 3),
                         transport=getattr(self, "_transport_records", []))
            self._transport_records = []
            self._timing = {"calls": 0, "generates": 0, "queue_ms": 0.0, "gpu_ms": 0.0}
        return taken

    def shutdown(self) -> None:
        try:
            self._connection.close()
        except Exception:  # noqa: BLE001 - closing a dropped connection
            pass


# -- XPolicyLab (RoboDojo) ----------------------------------------------------------------------------------------

def array_to_wire(value: Any) -> dict:
    """An array as dtype, shape and its raw bytes: exact for every dtype, unlike a JSON list of floats."""
    import numpy as np

    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode()}


def array_from_wire(value: Mapping[str, Any]) -> Any:
    import numpy as np

    array = np.frombuffer(base64.b64decode(value["data"]), dtype=np.dtype(value["dtype"]))
    return array.reshape([int(n) for n in value["shape"]]).copy()


def payload_to_wire(payload: Mapping[str, Any], *, lossless: bool = True) -> dict:
    """The adapter's RGB images, prompt and EEF20 state, with lossless image compression."""
    import numpy as np
    from PIL import Image

    images = {}
    for name, image in payload["images"].items():
        mode = getattr(image, "mode", "RGB")
        if mode != "RGB":
            raise ValueError(f"camera {name}: expected an RGB image, got mode {mode}")
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"camera {name}: expected HxWx3 RGB")
        if not lossless:
            images[name] = array_to_wire(array)
            continue
        buf = io.BytesIO()
        # PNG is lossless: no resize, color conversion, or changed model pixels.
        Image.fromarray(array).save(buf, format="PNG", compress_level=1)
        encoded = buf.getvalue()
        images[name] = ({"encoding": "png", "shape": list(array.shape),
                         "data": base64.b64encode(encoded).decode()}
                        if len(encoded) < array.nbytes else array_to_wire(array))
    return {"images": images, "prompt": payload["prompt"], "state": payload["state"]}


def payload_from_wire(wire: Mapping[str, Any]) -> dict:
    from PIL import Image

    images = {}
    for name, value in wire["images"].items():
        if value.get("encoding") == "png":
            shape = value["shape"]
            if (len(shape) != 3 or shape[2] != 3 or min(shape) <= 0
                    or shape[0]*shape[1]*3 > XPOLICYLAB_MAX_MESSAGE_BYTES):
                raise ValueError("invalid RGB image shape")
            with Image.open(io.BytesIO(base64.b64decode(value["data"], validate=True))) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (shape[1], shape[0]):
                    raise ValueError("PNG image differs from declared RGB shape")
                images[name] = image.copy()
        elif "encoding" in value:
            raise ValueError("unknown image encoding")
        else:
            images[name] = Image.fromarray(array_from_wire(value), mode="RGB")
    return {"images": images, "prompt": wire["prompt"], "state": wire["state"]}


class XPolicyLabSessionClient(ProtocolPolicyClient):
    """The shared engine as the XPolicyLab adapter calls it: `generate_batch(payloads)` -> {"actions": (B, T, D)}.

    The service has already applied `_project_binary_dims`; the returned actions are what the in-process adapter
    holds after that line."""

    def __init__(self, address: str, *, timeout: float = 600.0, connect=None):
        super().__init__(address, timeout=timeout,
                         connect=connect or websockets_connect(max_size=XPOLICYLAB_MAX_MESSAGE_BYTES))
        self._image_encodings = None

    def _ckpt_contract(self) -> dict:
        contract = super()._ckpt_contract()
        self._image_encodings = contract.get("rgb_encodings", ["raw"])
        return contract

    def generate_batch(self, payloads: list) -> dict:
        if self._image_encodings is None:
            self._ckpt_contract()
        started = time.monotonic()
        encoded = [payload_to_wire(p, lossless="png" in self._image_encodings) for p in payloads]
        reply = self._ask({"type": "chunk", "payloads": encoded,
                           "client_encode_ms": (time.monotonic()-started)*1000})
        if reply.get("type") != "actions":
            raise ProtocolError(f"chunk answered with {reply.get('type')!r}")
        with self._lock:
            self._timing["calls"] += 1
            self._timing["generates"] += bool(reply.get("generated"))
            self._timing["queue_ms"] += float(reply.get("queue_ms") or 0.0)
            self._timing["gpu_ms"] += float(reply.get("gpu_ms") or 0.0)
        return {"actions": array_from_wire(reply["actions"]), "video": None}


class _ProjectedByService:
    """Stands in for the adapter's `_wam_policy`: the service applied the projection before replying."""

    @staticmethod
    def _project_binary_dims(actions):
        return actions


def remote_xpolicylab_model(official_model: type) -> type:
    """A subclass of XPolicyLab's official OpenWAM `Model` whose engine is a shared service session.

    It is constructed with the launcher's model_cfg and the session address. The official constructor runs with
    `allow_dummy_policy` set, which validates the robot and action type, resolves the OpenWAM source root and loads
    the calibration but loads no checkpoint (its "checkpoint loading skipped" line says only that); the instance then
    serves real actions. `replan_steps` resolves as in process: the model_cfg's value, else the engine's inference
    horizon, which the session's pong reports.
    """

    class RemoteModel(official_model):
        def __init__(self, model_cfg, address: str):
            cfg = dict(model_cfg)
            if str(cfg.get("allow_dummy_policy", False)).strip().lower() in {"1", "true", "yes", "y", "on"}:
                raise ValueError("dummy policy cannot run on a shared OpenWAM session")
            super().__init__({**cfg, "allow_dummy_policy": True})
            self.model_cfg = cfg
            self.allow_dummy_policy = False
            self._remote = XPolicyLabSessionClient(address)
            contract = self._remote._ckpt_contract()
            if contract.get("flavor") != "xpolicylab":
                raise ProtocolError(f"{address} serves the {contract.get('flavor')!r} flavor, not 'xpolicylab'")
            if self.replan_steps is None:
                horizon = contract.get("inference_horizon")
                self.replan_steps = None if horizon is None else int(horizon)
            self._engine = self._remote
            self._wam_policy = _ProjectedByService()
            self._preprocessor = None
            print(f"[OpenWAM] shared session {address} | replan_steps={self.replan_steps or 'full chunk'}",
                  flush=True)

        def _conditions(self, payload: dict) -> dict:
            return payload  # the service runs the official `_conditions` on its side

        def take_serving_timing(self) -> dict:
            return self._remote.take_timing()

        def set_request_timeout(self, seconds: float) -> None:
            self._remote.set_request_timeout(seconds)

        def abort_inference(self) -> None:
            self._remote.abort_request()

        def shutdown(self) -> None:
            self._remote.shutdown()

    RemoteModel.__name__ = RemoteModel.__qualname__ = f"Remote{official_model.__name__}"
    return RemoteModel
