"""Direct HTTP RPC with total wall-clock and response-size limits.

Literal-IP endpoints avoid unbounded DNS resolution. A watchdog interrupts the
socket even when a peer keeps trickling headers or body bytes. A timeout does
not cancel remote actions: use invoke/status and retain uncertain outcomes.
"""

import http.client
import ipaddress
import json
import math
import socket
import threading
import time
from urllib.parse import urlsplit

from .http_rpc import _NumpyEncoder, _from_json
from .rpc_client import RpcClient, RpcError, check_response


class DeadlineHttpRpcClient(RpcClient):
    def __init__(self, endpoint, *, response_bytes=1024 * 1024):
        super().__init__()
        url = urlsplit(endpoint)
        if (
            url.scheme != "http"
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError(
                "Expected a direct HTTP endpoint without credentials or path"
            )
        ipaddress.ip_address(url.hostname)
        if type(response_bytes) is not int or response_bytes <= 0:
            raise ValueError("Positive response size limit required")
        self.host, self.port = url.hostname, url.port or 80
        self.response_bytes = response_bytes

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        if timeout_s is None or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("A positive finite RPC deadline is required")
        deadline = time.monotonic() + timeout_s
        body = json.dumps(
            dict(method=method, args=list(args), kwargs=kwargs or {}),
            cls=_NumpyEncoder,
            allow_nan=False,
        ).encode()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout_s)
        watchdog = None
        expired = threading.Event()
        try:
            connection.connect()
            wire = connection.sock

            def expire():
                expired.set()
                try:
                    wire.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RPC deadline reached during connection")
            watchdog = threading.Timer(remaining, expire)
            watchdog.daemon = True
            watchdog.start()
            connection.request(
                "POST", "/call", body=body, headers={"Content-Type": "application/json"}
            )
            with connection.getresponse() as response:
                raw = response.read(self.response_bytes + 1)
                if len(raw) > self.response_bytes:
                    raise ValueError("RPC response exceeds size limit")
                if response.status != 200:
                    raise RpcError(method, "HTTP status " + str(response.status))
            if expired.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("RPC total deadline reached")
            return check_response(_from_json(json.loads(raw)), method)
        except (OSError, http.client.HTTPException) as error:
            if expired.is_set():
                raise TimeoutError("RPC total deadline reached") from error
            raise RpcError(method, str(error)) from error
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join()
            connection.close()
