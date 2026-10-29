from __future__ import annotations

import threading
from typing import Any

import msgpack
import numpy as np
from websockets.sync.client import connect


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in {"V", "O", "c"}:
            raise ValueError(f"unsupported NumPy dtype: {obj.dtype}")
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


def _unpack_array(obj: dict[Any, Any]) -> Any:
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


class WAM_policyBackendClient:
    def __init__(
        self,
        url: str,
        *,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 600.0,
    ) -> None:
        self.url = str(url)
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self._connection = None
        self._lock = threading.Lock()
        self._packer = msgpack.Packer(default=_pack_array, use_bin_type=True)

    def _open(self):
        if self._connection is None:
            self._connection = connect(
                self.url,
                compression=None,
                max_size=None,
                open_timeout=self.connect_timeout_s,
                close_timeout=5.0,
            )
        return self._connection

    def close(self) -> None:
        with self._lock:
            connection, self._connection = self._connection, None
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def infer(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        prompt: str,
        num_ae_steps: int,
        bbox: np.ndarray | None = None,
        bbox_reference_images: np.ndarray | None = None,
    ) -> dict[str, Any]:
        request = {
            "method": "infer",
            "params": {
                "obs": np.ascontiguousarray(obs, dtype=np.float32),
                "state": np.ascontiguousarray(state, dtype=np.float32),
                "prompt": str(prompt),
                "num_ae_steps": int(num_ae_steps),
            },
        }
        if bbox is not None:
            request["params"]["bbox"] = np.asarray(bbox, dtype=np.float32)
        if bbox_reference_images is not None:
            request["params"]["bbox_reference_images"] = np.asarray(bbox_reference_images, dtype=np.float32)
        return self._request(request)

    def metadata(self) -> dict[str, Any]:
        response = self._request({"method": "metadata"})
        return response["metadata"]

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            try:
                connection = self._open()
                connection.send(self._packer.pack(request))
                raw = connection.recv(timeout=self.request_timeout_s)
                if not isinstance(raw, (bytes, bytearray, memoryview)):
                    raise TypeError("WAM_policy backend returned a text websocket frame")
                response = msgpack.unpackb(
                    bytes(raw), object_hook=_unpack_array, raw=False
                )
                if not isinstance(response, dict):
                    raise TypeError(
                        "WAM_policy backend response must be a mapping, got "
                        f"{type(response).__name__}"
                    )
                if response.get("error"):
                    raise RuntimeError(f"WAM_policy backend inference failed:\n{response['error']}")
                return response
            except Exception as error:
                stale, self._connection = self._connection, None
                if stale is not None:
                    try:
                        stale.close()
                    except Exception:
                        pass
                raise RuntimeError(
                    f"WAM_policy backend request failed at {self.url}: {error}"
                ) from error
