"""The asyncio JSON-RPC 2.0 client for the kernel's boundary.

One line of JSON per message, both directions, over a Unix domain socket (or any pair
of asyncio streams). The connection is symmetric: the client calls the kernel, and the
kernel calls *back* — `capability/execute` and `approval/request` are requests the
kernel sends to us and expects an answer to, and `brain/turn`, `event`, `plugin/state`,
`shutdown` and friends are notifications it pushes. So this is not a request/response
client with a callback bolted on; it is a peer that happens to open the socket.

Three tasks per connection:

* a **read** task — the only thing that touches the reader. It decodes one line at a
  time and routes it by shape: `id` + `method` is a kernel request, `id` alone is a
  response to one of ours, `method` alone is a notification.
* a **write** task — the only thing that touches the writer, fed by an unbounded
  queue. This is what lets :meth:`Client.notify` be an ordinary `def`: enqueueing is
  synchronous and cannot block, so a capability heartbeat can be sent from anywhere,
  including from inside a synchronous notification handler.
* one task per in-flight kernel request or coroutine notification handler.

Why handlers never run *inside* the read loop when they are coroutines: a handler is
allowed to call back into the kernel (an approver may want to read `robot/state`
before deciding). Awaiting it in the read loop would mean nothing is reading the
socket while it waits for a reply that can only arrive through that same loop — an
instant deadlock. Synchronous handlers do run inline, because they cannot await and
because in-order delivery is exactly what a `log/tail` consumer wants.

Request ids are integers counting from 1. The kernel's own request ids are strings
(`"k1"`, `"k2"`, ...), which is why the two never collide and why this client echoes
whatever id it was given back verbatim instead of parsing it.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from typing import Any, Awaitable, Callable, Optional, Union

from . import protocol as P

log = logging.getLogger("oeh_sdk.rpc")

#: Longest line (excluding the newline) the kernel will send or accept, in bytes.
#: Matches `oeh_rpc::codec::MAX_LINE`; a longer line is a transport error, not a
#: protocol error, and kills the connection on both sides.
MAX_LINE = 16 * 1024 * 1024

RequestHandler = Callable[[dict], Awaitable[Any]]
NotificationHandler = Callable[[dict], Union[Awaitable[None], None]]


class RpcError(Exception):
    """A JSON-RPC error object. `code` is one of :class:`oeh_sdk.protocol.code`."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RpcError":
        return cls(
            int(d.get("code", P.code.INTERNAL)),
            str(d.get("message", "")),
            d.get("data"),
        )


class Client:
    """A connection to one kernel.

    Construct it with :meth:`connect_uds` or :meth:`from_streams`; both start the read
    and write tasks immediately.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        default_timeout: Optional[float] = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._pending: dict[Any, asyncio.Future] = {}
        self._requests: dict[str, RequestHandler] = {}
        self._notifications: dict[str, list[NotificationHandler]] = {}
        self._outbox: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._closed = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self.default_timeout = default_timeout
        self._read_task = asyncio.create_task(self._read_loop(), name="oeh-rpc-read")
        self._write_task = asyncio.create_task(self._write_loop(), name="oeh-rpc-write")

    # -- construction ------------------------------------------------------------------

    @classmethod
    async def connect_uds(
        cls, path: Any, *, default_timeout: Optional[float] = None
    ) -> "Client":
        """Connect to a kernel's `rpc-server` UDS listener (`OEH_RPC_UDS` / profile `uds`)."""
        reader, writer = await asyncio.open_unix_connection(str(path), limit=MAX_LINE)
        return cls(reader, writer, default_timeout=default_timeout)

    @classmethod
    def from_streams(
        cls,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        default_timeout: Optional[float] = None,
    ) -> "Client":
        """Wrap an existing stream pair — a subprocess' stdio (`oeh run --rpc-stdio`),
        a TCP connection, an in-memory pipe in a test."""
        return cls(reader, writer, default_timeout=default_timeout)

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- outbound ----------------------------------------------------------------------

    async def call(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send a request and wait for its response. Raises :class:`RpcError`.

        `timeout` (or `default_timeout`) is the *client's* patience, nothing the kernel
        knows about; on expiry the call raises TIMEOUT and stops waiting, but whatever
        it asked for may still happen. Methods that wait on the world — `turn/propose`
        and its `say`/`ask_human` siblings block for the whole admission round trip,
        execution included — are the reason the default is "no timeout at all": the
        honest bound there is the connection itself, and a dropped connection fails
        every pending call with KERNEL_GONE.
        """
        if self._closed.is_set():
            raise RpcError(P.code.KERNEL_GONE, "connection closed")
        rid = next(self._ids)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        # Belt and braces, not a race: `create_future()` has no suspension point, so on
        # a single-threaded loop nothing can run between the check above and this line,
        # and `_mark_closed` only ever runs in another task. The re-check costs one
        # attribute read and keeps the invariant true for any caller who hands
        # `from_streams` a reader driven from somewhere else. If it ever does fire,
        # `_mark_closed` has already set an exception on this future, so retrieve it —
        # an un-retrieved exception is reported at garbage-collection time as a bare
        # "Future exception was never retrieved", with no clue where it came from.
        if self._closed.is_set():
            self._pending.pop(rid, None)
            if fut.done():
                fut.exception()
            raise RpcError(P.code.KERNEL_GONE, "connection closed")
        self._send(
            {
                "jsonrpc": P.JSONRPC,
                "id": rid,
                "method": method,
                "params": {} if params is None else params,
            }
        )
        t = self.default_timeout if timeout is None else timeout
        try:
            if t is None:
                return await fut
            return await asyncio.wait_for(fut, t)
        except asyncio.TimeoutError:
            raise RpcError(P.code.TIMEOUT, f"`{method}` timed out after {t}s") from None
        finally:
            self._pending.pop(rid, None)

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Fire a notification. Synchronous on purpose: no id, no answer, no waiting —
        which is what makes it callable from a capability heartbeat or a `Drop`-shaped
        cleanup path."""
        self._send(
            {
                "jsonrpc": P.JSONRPC,
                "method": method,
                "params": {} if params is None else params,
            }
        )

    # -- inbound handlers --------------------------------------------------------------

    def on_request(self, method: str, handler: RequestHandler) -> None:
        """Answer a kernel-originated request (`capability/execute`, `approval/request`).

        The handler is a coroutine taking the params dict; its return value becomes the
        result. Raising :class:`RpcError` sends that error back; any other exception
        becomes INTERNAL. Registering a second handler for the same method **replaces**
        the first — the kernel only ever expects one answer, so there is nothing
        sensible to do with two.

        That replacement is why one connection serves one :class:`~oeh_sdk.plugin.Plugin`
        and one :class:`~oeh_sdk.approval.Approver`: a second `Plugin` would take over
        `capability/execute` and answer "no body for X" to the first one's capabilities.
        `Plugin.register` refuses rather than letting that happen silently.
        """
        self._requests[method] = handler

    def on_notification(self, method: str, handler: NotificationHandler) -> None:
        """Subscribe to a kernel notification. Several handlers per method are allowed
        and run in registration order.

        A plain `def` handler runs inline in the read loop, so notifications reach it in
        wire order; an `async def` handler is spawned as a task and therefore must not
        assume ordering between two of its own invocations.
        """
        self._notifications.setdefault(method, []).append(handler)

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def closed(self) -> asyncio.Event:
        """Set once this connection is finished, for whatever reason."""
        return self._closed

    async def close(self) -> None:
        """Close the connection and fail everything still waiting on it.

        Safe to call from inside a handler — `on_notification("shutdown", lambda p:
        client.close())` is the obvious way to react to the kernel going away, and that
        handler runs in one of the tasks this method cancels. The task doing the
        closing is excluded from that sweep; cancelling it would raise `CancelledError`
        straight out of `close()` and leave the connection half shut.
        """
        self._mark_closed()
        me = asyncio.current_task()
        if self._read_task is not me:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._read_task
        if self._write_task is not me:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._write_task
        others = [t for t in self._tasks if t is not me]
        for t in others:
            t.cancel()
        if others:
            await asyncio.gather(*others, return_exceptions=True)
        with contextlib.suppress(Exception):
            self._writer.close()
            await self._writer.wait_closed()

    # -- internals ---------------------------------------------------------------------

    def _send(self, msg: dict) -> None:
        if self._closed.is_set():
            return
        self._outbox.put_nowait(json.dumps(msg, separators=(",", ":")).encode() + b"\n")

    def _spawn(self, coro: Awaitable[Any]) -> None:
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _mark_closed(self) -> None:
        """Idempotent: set the flag, fail every call still waiting, and wind the write
        side down.

        KERNEL_GONE rather than a bare `ConnectionError` because that is the code the
        kernel itself uses for "the thing you are calling is not there any more" — a
        caller should not have to tell "the kernel stopped" apart from "the socket to
        the kernel stopped" to decide that retrying is pointless.

        The sentinel goes in here and not only in `close()`: a connection that ends
        because the *kernel* hung up (EOF, `shutdown`, SIGINT) would otherwise leave the
        write task parked on an empty queue and the socket open, so an application that
        reacts to `closed` by dropping the `Client` would leak one task and one fd per
        connection. Queued messages ahead of the sentinel still go out.
        """
        if self._closed.is_set():
            return
        self._closed.set()
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(RpcError(P.code.KERNEL_GONE, "connection closed"))
        self._pending.clear()
        with contextlib.suppress(Exception):
            self._outbox.put_nowait(None)

    async def _write_loop(self) -> None:
        try:
            while True:
                item = await self._outbox.get()
                if item is None:
                    break
                self._writer.write(item)
                await self._writer.drain()
        except (ConnectionError, OSError, asyncio.CancelledError):
            pass
        finally:
            self._mark_closed()
            # The writer belongs to this task; closing it here is what makes the EOF
            # path release the fd without anyone calling `close()`.
            with contextlib.suppress(Exception):
                self._writer.close()

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    line = await self._reader.readline()
                except (ConnectionError, OSError):
                    break
                except ValueError:
                    # A line longer than `limit` — the framing is gone, so is the
                    # connection.
                    log.error("oversized line (> %d bytes); closing", MAX_LINE)
                    break
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError as e:
                    log.error("undecodable line from kernel: %s", e)
                    continue
                if not isinstance(msg, dict):
                    log.error("line from kernel is not a JSON object")
                    continue
                self._dispatch(msg)
        finally:
            self._mark_closed()

    def _dispatch(self, msg: dict) -> None:
        method = msg.get("method")
        has_id = "id" in msg
        if method is not None and has_id:
            self._spawn(self._serve(msg["id"], str(method), msg.get("params") or {}))
        elif has_id:
            self._resolve(msg)
        elif method is not None:
            self._fanout(str(method), msg.get("params") or {})
        else:
            log.error("message with neither id nor method")

    def _resolve(self, msg: dict) -> None:
        fut = self._pending.pop(msg["id"], None)
        if fut is None:
            # A response to a call that already timed out or was cancelled. Not an
            # error: the caller stopped waiting, the kernel did not know that.
            log.debug("response for unknown id %r", msg.get("id"))
            return
        if fut.done():
            return
        err = msg.get("error")
        if err is not None:
            fut.set_exception(RpcError.from_dict(err))
        else:
            fut.set_result(msg.get("result"))

    async def _serve(self, rid: Any, method: str, params: dict) -> None:
        handler = self._requests.get(method)
        if handler is None:
            self._send(
                {
                    "jsonrpc": P.JSONRPC,
                    "id": rid,
                    "error": {
                        "code": P.code.METHOD_NOT_FOUND,
                        "message": f"no handler for `{method}`",
                    },
                }
            )
            return
        try:
            result = await handler(params)
        except RpcError as e:
            self._send({"jsonrpc": P.JSONRPC, "id": rid, "error": e.to_dict()})
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a handler bug must not kill the link
            log.exception("handler for `%s` raised", method)
            self._send(
                {
                    "jsonrpc": P.JSONRPC,
                    "id": rid,
                    "error": {"code": P.code.INTERNAL, "message": repr(e)},
                }
            )
            return
        self._send(
            {"jsonrpc": P.JSONRPC, "id": rid, "result": {} if result is None else result}
        )

    def _fanout(self, method: str, params: dict) -> None:
        for handler in list(self._notifications.get(method, ())):
            try:
                r = handler(params)
            except Exception:  # noqa: BLE001
                log.exception("notification handler for `%s` raised", method)
                continue
            if asyncio.iscoroutine(r) or isinstance(r, asyncio.Future):
                self._spawn(self._guard(method, r))

    async def _guard(self, method: str, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("notification handler for `%s` raised", method)
