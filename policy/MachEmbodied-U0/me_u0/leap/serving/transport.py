"""Msgpack serialization with NumPy array support.

Provides encode/decode functions for transmitting dicts containing NumPy arrays
over the network. Adapted from ref_code/starVLA msgpack_numpy.py.

No torch dependency — only numpy and msgpack.
"""

import msgpack
import numpy as np


def _pack_default(obj):
    """Msgpack default handler: convert numpy types to serializable dicts."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported numpy dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_hook(obj):
    """Msgpack object_hook: restore numpy types from serialized dicts."""
    if b"__ndarray__" in obj:
        return np.frombuffer(
            obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
        ).reshape(obj[b"shape"]).copy()

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


def encode(obj: dict) -> bytes:
    """Serialize a dict (may contain np.ndarray values) to msgpack bytes."""
    return msgpack.packb(obj, default=_pack_default)


def decode(data: bytes) -> dict:
    """Deserialize msgpack bytes back to a dict (np.ndarrays restored)."""
    return msgpack.unpackb(data, object_hook=_unpack_hook)
