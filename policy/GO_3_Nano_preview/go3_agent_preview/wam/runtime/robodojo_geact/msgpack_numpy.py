# Adapted from https://github.com/lebedov/msgpack-numpy

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

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


def unpack_array(obj):
    if b"__ndarray__" in obj or "__ndarray__" in obj:
        data = obj.get(b"data", obj.get("data"))
        dtype = obj.get(b"dtype", obj.get("dtype"))
        shape = obj.get(b"shape", obj.get("shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape)

    if b"__npgeneric__" in obj or "__npgeneric__" in obj:
        data = obj.get(b"data", obj.get("data"))
        dtype = obj.get(b"dtype", obj.get("dtype"))
        return np.dtype(dtype).type(data)

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array, use_bin_type=True)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array, raw=False)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array, raw=False)
