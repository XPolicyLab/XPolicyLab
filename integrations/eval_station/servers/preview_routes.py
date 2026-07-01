"""HTTP camera preview routes for the eval-station daemon.

The daemon (not the browser) owns the Orbbec cameras; these routes expose the
latest color frames so x-policy-web renders previews from MJPEG streams
instead of opening the head camera through ``getUserMedia``.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any, Optional, Protocol, Tuple
from urllib.parse import urlparse

MJPEG_BOUNDARY = "robodojo-preview-frame"
_STREAM_MAX_FPS = 15.0
_STREAM_FRAME_WAIT_S = 2.0
_STREAM_SOCKET_TIMEOUT_S = 15.0


class PreviewManagerLike(Protocol):
    def roles(self) -> Tuple[str, ...]: ...

    def ensure_started(self) -> None: ...

    def pause(self) -> None: ...

    def resume_async(self) -> None: ...

    def frame(self, role: str) -> Optional[Tuple[bytes, int]]: ...

    def wait_frame(
        self, role: str, after_timestamp_ns: int = 0, timeout_s: float = 1.0
    ) -> Optional[Tuple[bytes, int]]: ...

    def placeholder_jpeg(self) -> bytes: ...

    def status(self) -> dict: ...


def parse_preview_route(path: str) -> tuple[str, str | None] | None:
    """Return (action, role) for /v1/preview/* paths, else None.

    Actions: "status", "pause", "resume", "stream" (role.mjpeg), "snapshot"
    (role.jpg).
    """
    parts = urlparse(path).path.strip("/").split("/")
    match parts:
        case ["v1", "preview", "status"]:
            return ("status", None)
        case ["v1", "preview", "pause"]:
            return ("pause", None)
        case ["v1", "preview", "resume"]:
            return ("resume", None)
        case ["v1", "preview", leaf] if leaf.endswith(".mjpeg"):
            role = leaf[: -len(".mjpeg")]
            return ("stream", role) if role else None
        case ["v1", "preview", leaf] if leaf.endswith(".jpg"):
            role = leaf[: -len(".jpg")]
            return ("snapshot", role) if role else None
    return None


# Raised when the browser aborts a fetch (watchdog timeout) or closes the
# tab while we are writing; never crash the request thread over these.
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, TimeoutError, OSError)


def _write_json(handler: Any, status_code: HTTPStatus, body: dict) -> None:
    payload = json.dumps(body, sort_keys=True).encode("utf-8")
    try:
        handler.send_response(status_code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(payload)
    except _CLIENT_DISCONNECT_ERRORS:
        return


def _disabled(handler: Any) -> None:
    _write_json(
        handler,
        HTTPStatus.SERVICE_UNAVAILABLE,
        {"error": "camera preview is not enabled on this daemon"},
    )


def _unknown_role(handler: Any, manager: PreviewManagerLike, role: str) -> None:
    _write_json(
        handler,
        HTTPStatus.NOT_FOUND,
        {"error": f"unknown camera role '{role}'", "roles": list(manager.roles())},
    )


def handle_preview_get(
    handler: Any,
    manager: Optional[PreviewManagerLike],
    action: str,
    role: str | None,
) -> None:
    if manager is None:
        _disabled(handler)
        return

    if action == "status":
        manager.ensure_started()
        body = manager.status()
        body["roles"] = list(manager.roles())
        _write_json(handler, HTTPStatus.OK, body)
        return

    assert role is not None
    if role not in manager.roles():
        _unknown_role(handler, manager, role)
        return

    manager.ensure_started()
    if action == "snapshot":
        _serve_snapshot(handler, manager, role)
    else:
        _serve_mjpeg(handler, manager, role)


def handle_preview_post(
    handler: Any,
    manager: Optional[PreviewManagerLike],
    action: str,
) -> None:
    if manager is None:
        _disabled(handler)
        return

    if action == "pause":
        manager.pause()
        _write_json(handler, HTTPStatus.OK, {"status": "paused"})
    else:
        manager.resume_async()
        _write_json(handler, HTTPStatus.OK, {"status": "resuming"})


def _serve_snapshot(handler: Any, manager: PreviewManagerLike, role: str) -> None:
    item = manager.frame(role)
    jpeg = item[0] if item else manager.placeholder_jpeg()
    try:
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Content-Length", str(len(jpeg)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(jpeg)
    except _CLIENT_DISCONNECT_ERRORS:
        return


def _write_mjpeg_part(handler: Any, jpeg: bytes) -> None:
    handler.wfile.write(
        b"--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
        % (MJPEG_BOUNDARY.encode("ascii"), len(jpeg))
    )
    handler.wfile.write(jpeg)
    handler.wfile.write(b"\r\n")


def _serve_mjpeg(handler: Any, manager: PreviewManagerLike, role: str) -> None:
    """Stream multipart MJPEG until the client disconnects.

    On frame timeouts the latest (possibly stale) frame or a placeholder is
    re-sent so broken connections surface as write errors instead of leaking
    blocked threads.
    """
    import time

    min_period = 1.0 / _STREAM_MAX_FPS
    last_ts = 0
    try:
        handler.connection.settimeout(_STREAM_SOCKET_TIMEOUT_S)
        handler.send_response(HTTPStatus.OK)
        handler.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}"
        )
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()

        while True:
            started = time.monotonic()
            item = manager.wait_frame(
                role,
                after_timestamp_ns=last_ts,
                timeout_s=_STREAM_FRAME_WAIT_S,
            )
            if item is not None:
                jpeg, last_ts = item
            else:
                jpeg = manager.placeholder_jpeg()
            _write_mjpeg_part(handler, jpeg)
            elapsed = time.monotonic() - started
            if elapsed < min_period:
                time.sleep(min_period - elapsed)
    except _CLIENT_DISCONNECT_ERRORS:
        return
