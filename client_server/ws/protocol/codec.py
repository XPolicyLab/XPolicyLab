"""msgpack encode/decode for WebSocket binary frames."""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
from pydantic import ValidationError

from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.schemas import Frame


def _encode_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise WsError(
                ErrorCode.INVALID_FRAME, "object dtype numpy arrays are not supported"
            )
        return msgpack_numpy.encode(obj)
    if isinstance(obj, np.generic):
        if obj.dtype.kind == "O":
            raise WsError(
                ErrorCode.INVALID_FRAME, "object dtype numpy scalars are not supported"
            )
    return msgpack_numpy.encode(obj)


def _decode_numpy(obj: dict[Any, Any]) -> Any:
    if obj.get(b"nd") is True and obj.get(b"kind") == b"O":
        raise ValueError("object dtype numpy arrays are not supported")
    decoded = msgpack_numpy.decode(obj)
    if isinstance(decoded, np.ndarray) and decoded.dtype.kind == "O":
        raise ValueError("object dtype numpy arrays are not supported")
    return decoded


def encode_frame(frame: Frame | dict[str, Any]) -> bytes:
    if isinstance(frame, Frame):
        wire = frame.to_wire_dict()
    else:
        wire = dict(frame)
    try:
        return msgpack.packb(wire, default=_encode_numpy, use_bin_type=True)
    except WsError:
        raise
    except Exception as exc:
        raise WsError(ErrorCode.INVALID_FRAME, f"msgpack encode failed: {exc}") from exc


def decode_frame(data: bytes) -> dict[str, Any]:
    try:
        obj = msgpack.unpackb(data, raw=False, object_hook=_decode_numpy)
    except Exception as exc:
        raise WsError(ErrorCode.INVALID_FRAME, f"msgpack decode failed: {exc}") from exc
    if not isinstance(obj, dict):
        raise WsError(ErrorCode.INVALID_FRAME, "frame must be a msgpack map")
    return obj


def decode_envelope(data: bytes) -> Frame:
    try:
        return Frame.from_wire_dict(decode_frame(data))
    except WsError:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        raise WsError(
            ErrorCode.INVALID_FRAME, f"invalid frame envelope: {exc}"
        ) from exc
