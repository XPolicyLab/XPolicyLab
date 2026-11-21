"""Frames: the other half of the boundary, and the half that never touches JSON-RPC.

The protocol in :mod:`oeh_sdk.protocol` carries *decisions* — proposals, approvals,
capability calls, log lines. It does not carry frames, not one. A 500 Hz `lowstate`
re-encoded as JSON and squeezed through a single Unix socket would drown the control
plane in sensor data, so the boundary protocol only ever **names** a channel and gets
out of the way: `plugin/register` answers with a `channel` per declared stream, and
`stream/subscribe` answers with the `channel` of a stream the kernel produces. The
bytes themselves go through iceoryx2 shared memory, which is what this module speaks.

    from oeh_sdk.frames import FramePublisher, FrameSubscriber

    await p.register(client)
    pub = FramePublisher(p.channels["arm_state"].channel)
    pub.publish_json({"q": [0.0] * 7})

    ch = await client.call("stream/subscribe", {"stream_id": "lowstate"})
    sub = FrameSubscriber(ch["channel"])
    for frame in sub.poll_json():
        ...

Never compute a channel name. It carries the run tag (`oeh/<run_tag>/<stream_id>`)
because iceoryx2 service names are *machine-global*: two kernels on one host — one
live, one replaying a log — must not land their `lowstate` in the same shared memory.
The kernel assigns the tag per run and hands the name back; a client that derives one
itself publishes into a channel nobody reads, and finds out by seeing no frames.

Four things about this module are load-bearing:

* **The QoS numbers below must equal the Rust side's** (`crates/oeh-rpc/src/frames.rs`).
  Both sides call `open_or_create`, so whoever arrives first *creates* the service and
  the other side's numbers become requirements checked against it. Matching numbers is
  the only arrangement where it does not matter who is first.
* **No Listener, ever.** iceoryx2's event `Listener` changes the whole **process**
  umask while it binds its unix datagram socket, and any directory another thread
  creates in that window comes out without its `x` bit. The symptom lands on an
  innocent victim (a log directory that cannot be opened) with nothing pointing back
  here. So a subscriber *polls*, and gets the notification service only to pin its QoS.
  A `Notifier` is fine — it calls `connect()`, not `bind()` — and a publisher keeps one
  so that other subscribers (a console, a recorder) can wait on an event rather than
  spin. The kernel's own ingress does not listen either; it polls with a backoff.
* **`history_size(0)`: a new subscriber gets *now*, not a replay.** Frame history lives
  in the kernel's `FrameStore`, not in the channel. Construct a :class:`FrameSubscriber`
  and *then* wait for frames; frames published before it existed are gone.
* **Nothing here blocks.** Loaning, sending and receiving are lock-free shared-memory
  operations, so these calls are safe to make from inside an asyncio loop. `poll`
  returns what is there and returns immediately; there is no `await` to add.

The `iceoryx2` binding is the `frames` extra (`pip install oeh-sdk[frames]`), never a
dependency of the RPC client. A process that only proposes and approves does not need
a shared-memory stack, and this module is the only thing in the package that imports it.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any, Optional

try:
    import iceoryx2 as iox2
except ImportError as e:  # pragma: no cover - exercised by not installing the extra
    iox2 = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[ImportError] = e
else:
    _IMPORT_ERROR = None

__all__ = [
    "DEFAULT_MAX_BYTES",
    "FrameError",
    "FramePublisher",
    "FrameSubscriber",
    "HISTORY",
    "MAX_LISTENERS",
    "MAX_NOTIFIERS",
    "MAX_PUBLISHERS",
    "MAX_SUBSCRIBERS",
    "SUB_BUFFER",
    "event_name",
]


class FrameError(RuntimeError):
    """A frame-transport failure, with the channel and size that caused it attached.

    The `iceoryx2` binding raises its own exceptions (`LoanError` and friends) with
    only a terse enum name (`ExceedsMaxLoanSize`) and no indication of which channel
    or how large a payload was involved. This wraps those with that context; the
    original exception is chained (`__cause__`), so nothing about it is lost.
    """

#: A subscriber's queue depth. **Both sides use this value** — see the module note on
#: `open_or_create`. It is also the natural bound on one :meth:`FrameSubscriber.poll`:
#: sixteen samples *is* "everything queued up right now".
SUB_BUFFER = 16
#: No replay for a late subscriber. The frame history is the kernel's `FrameStore`.
HISTORY = 0
#: Publishers per channel. Normally one; the slack is for an adapter that reconnects
#: before its previous incarnation has been cleaned up.
MAX_PUBLISHERS = 4
#: Subscribers per channel. The kernel's ingress takes one, the rest are for onlookers.
MAX_SUBSCRIBERS = 8
#: Event-service QoS, aligned one-per-port with the two above. Pinned rather than left
#: to the library defaults (16/16) so that swapping an `iceoryx2.toml` on either side
#: cannot quietly reshape the other side's queues.
MAX_LISTENERS = MAX_SUBSCRIBERS
#: See :data:`MAX_LISTENERS`.
MAX_NOTIFIERS = MAX_PUBLISHERS

#: Default publisher segment size: the largest frame that can be loaned before the
#: allocator has to grow. A `lowstate`-sized JSON frame is a few hundred bytes; a
#: megabyte covers a compressed camera frame without anyone having to think about it,
#: and `AllocationStrategy.PowerOfTwo` grows past it rather than failing.
DEFAULT_MAX_BYTES = 1 << 20


def event_name(channel: str) -> str:
    """The notification service that belongs to a frame channel: `<channel>/ev`.

    This is a convention both sides hard-code, not something the protocol reports —
    which is why it is a function here and a `fn event_name` there.
    """
    return f"{channel}/ev"


def _require_iceoryx2() -> Any:
    if iox2 is None:
        raise ImportError(
            "frame transport needs the `frames` extra: "
            "`pip install 'oeh-sdk[frames]'` (iceoryx2). The RPC client does not need "
            "it — frames do not travel over JSON-RPC."
        ) from _IMPORT_ERROR
    return iox2


class _Channel:
    """The iceoryx2 objects behind one channel, opened the way the kernel opens them.

    Everything is held as an attribute on purpose. An iceoryx2 `Node` owns the service
    registration; letting it fall out of scope while a publisher built from it is still
    in use is how a channel goes quiet with no error anywhere.
    """

    def __init__(self, channel: str, node_name: str) -> None:
        x = _require_iceoryx2()
        self.channel = channel
        #: `Disabled`, not the default: the default installs process-wide handlers for
        #: SIGINT/SIGTERM so that iceoryx2's *blocking* calls can be interrupted. This
        #: module never blocks, and a library has no business taking a host process's
        #: Ctrl-C away from it.
        self._node = (
            x.NodeBuilder.new()
            .name(x.NodeName.new(node_name))
            .signal_handling_mode(x.SignalHandlingMode.Disabled)
            .create(x.ServiceType.Ipc)
        )
        self.service = (
            self._node.service_builder(x.ServiceName.new(channel))
            .publish_subscribe(x.Slice[ctypes.c_uint8])
            .subscriber_max_buffer_size(SUB_BUFFER)
            .history_size(HISTORY)
            .max_publishers(MAX_PUBLISHERS)
            .max_subscribers(MAX_SUBSCRIBERS)
            .open_or_create()
        )
        # Opened, and on the subscribing side never used for anything: opening it is
        # what pins its QoS, so the shape of the event queues does not depend on which
        # side happened to create the service first.
        self.event = (
            self._node.service_builder(x.ServiceName.new(event_name(channel)))
            .event()
            .max_listeners(MAX_LISTENERS)
            .max_notifiers(MAX_NOTIFIERS)
            .open_or_create()
        )

    def close(self) -> None:
        """Drop the `Node` and both service handles.

        Neither `Node` nor a service factory has a `delete()` of its own — unlike
        `Publisher`/`Subscriber`/`Notifier`, they are released the ordinary Python way,
        by dropping every reference to them. Without this, `FramePublisher.close()`/
        `FrameSubscriber.close()` freed only the publisher/subscriber *port* slot and
        left this channel's `Node` (and its service registrations) alive until the
        whole `FramePublisher`/`FrameSubscriber` object was itself collected — harmless
        under the `with` idiom this module recommends, but not what "close" says on the
        tin for code that keeps the object around after calling it.
        """
        self._node = None
        self.service = None
        self.event = None


class FramePublisher:
    """Publishes frames on one channel — the channel the *kernel* named.

    `channel` comes from `plugin/register`'s `streams[].channel` (see
    :attr:`oeh_sdk.plugin.Plugin.channels`). The kernel starts an ingress thread for a
    remotely-published stream during `plugin/register` and only answers once that
    thread's ports are open, so a publisher built right after `register` returns has
    someone on the other end. What it does *not* have is delivery of frames sent before
    it existed: with :data:`HISTORY` at zero, a frame published into a channel with no
    subscriber is simply dropped.

    Not thread-safe, and deliberately so: iceoryx2's ports are not `Send`, so one
    publisher belongs to the thread that made it.
    """

    def __init__(self, channel: str, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        x = _require_iceoryx2()
        self._c = _Channel(channel, "oeh-sdk-pub")
        self.channel = channel
        self.max_bytes = max_bytes
        self._publisher = (
            self._c.service.publisher_builder()
            .initial_max_slice_len(max_bytes)
            .allocation_strategy(x.AllocationStrategy.PowerOfTwo)
            .create()
        )
        #: Kept even though the kernel's ingress polls rather than listens: a Python
        #: subscriber, a console or a recorder may well be waiting on this event, and
        #: a `Notifier` costs nothing and touches no umask (see the module note).
        self._notifier = self._c.event.notifier_builder().create()
        self._closed = False

    def publish(self, payload: bytes) -> None:
        """Copy `payload` into a loaned sample and send it, then notify.

        One copy, into shared memory, and that is the whole cost — the kernel's ingress
        reads the same bytes without a second serialization step.
        """
        if self._closed:
            raise RuntimeError(f"publisher for `{self.channel}` is closed")
        try:
            sample = self._publisher.loan_slice_uninit(len(payload))
        except Exception as e:
            raise FrameError(
                f"loan of {len(payload)} bytes failed on channel `{self.channel}`: {e}"
            ) from e
        # `as_memory_view()` hands back a ctypes view whose format is `<B`, which
        # CPython refuses to assign a bytes object to; `cast("B")` is the same memory
        # relabelled as plain unsigned bytes, still without a copy of its own.
        sample.payload().as_memory_view().cast("B")[:] = payload
        sample.assume_init().send()
        self._notifier.notify()

    def publish_json(self, obj: Any) -> None:
        """`publish(json.dumps(obj))`, compactly.

        Compact separators because this is the format the Rust `serde_json` side emits
        and reads, and because at 500 Hz the whitespace is real bandwidth.
        """
        self.publish(json.dumps(obj, separators=(",", ":")).encode())

    def close(self) -> None:
        """Drop the publisher, its notifier, and this channel's `Node`/service handles.

        Optional — Python's GC gets there eventually — but an adapter that stops
        publishing should say so now rather than hold one of the channel's four
        publisher slots (or its `Node` registration) until a collection happens to run.
        """
        if self._closed:
            return
        self._closed = True
        self._publisher.delete()
        self._notifier.delete()
        self._c.close()

    def __enter__(self) -> "FramePublisher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FrameSubscriber:
    """Reads frames from one channel — again, the channel the kernel named.

    `channel` comes from `stream/subscribe`'s result. That call is what makes a
    *locally* produced stream leave the kernel at all: the egress thread is started
    lazily by the subscribe and its ports are open before the result is sent, so the
    channel is really there by the time this constructor runs. Frames published before
    this object existed are not replayed (:data:`HISTORY` is zero), so subscribe first
    and then wait.

    There is no `wait()` here, and the module docstring says why: an iceoryx2 event
    `Listener` would rewrite the process umask while it binds. Poll instead.
    """

    def __init__(self, channel: str) -> None:
        self._c = _Channel(channel, "oeh-sdk-sub")
        self.channel = channel
        self._subscriber = self._c.service.subscriber_builder().create()
        self._closed = False

    def poll(self, limit: int = SUB_BUFFER) -> list[bytes]:
        """Drain what is queued right now, oldest first; `[]` when nothing is.

        `limit` defaults to the queue depth, which is exactly "everything currently
        receivable" — and is also what stops a reader from being pinned in this loop
        forever by a publisher that is faster than it.
        """
        if self._closed:
            raise RuntimeError(f"subscriber for `{self.channel}` is closed")
        out: list[bytes] = []
        while len(out) < limit:
            sample = self._subscriber.receive()
            if sample is None:
                break
            out.append(bytes(sample.payload().as_memory_view()))
        return out

    def poll_json(self, limit: int = SUB_BUFFER) -> list[Any]:
        """:meth:`poll`, with each frame decoded.

        Raises `json.JSONDecodeError` on a frame that is not JSON rather than skipping
        it: every schema with an `@json` suffix is, so a frame that is not means the
        channel is carrying something other than what `schema_id` promised. Use
        :meth:`poll` for a binary schema.
        """
        return [json.loads(b) for b in self.poll(limit)]

    def close(self) -> None:
        """Drop the subscriber and this channel's `Node`/service handles."""
        if self._closed:
            return
        self._closed = True
        self._subscriber.delete()
        self._c.close()

    def __enter__(self) -> "FrameSubscriber":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
