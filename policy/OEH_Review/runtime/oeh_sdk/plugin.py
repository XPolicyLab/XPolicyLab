"""Cross-process plugins: `plugin/register`, heartbeats, and `capability/execute`.

A `plugin/register` is the one method on this boundary that changes the kernel tree.
After it returns, this client *is* a fiber: it shows up in `robot/health`, its state
changes are broadcast as `plugin/state`, its declared streams appear in `stream/list`,
and the capabilities it declared are on the capability surface for every brain in the
system. Disposing that fiber undoes all of it at once — which is why losing the
connection and calling `plugin/unregister` have exactly the same effect.

Three things keep a registration alive or end it:

* **heartbeats.** The kernel's watchdog disposes the fiber after `3 x heartbeat_ms`
  without one. :meth:`Plugin.register` starts a task that beats every
  `heartbeat_ms / 2`, so one lost beat is not fatal and two are noticed quickly.
  `plugin/heartbeat` is a *request*: the kernel routes only `capability/heartbeat` and
  `capability/result` as notifications, so a notification-shaped heartbeat is dropped
  and the fiber dies on schedule.
* **disconnect.** The socket closing disposes the fiber. No cleanup call is needed,
  and none can be skipped.
* **`plugin/unregister`.** Waits for the kernel to finish disposing before it returns,
  so an adapter knows its capabilities are *off* the surface before it powers down.

And when the kernel ends it *for* you — a missed beat, an operator dispose — it says so
with `plugin/state{state: "disposed"}`. :attr:`Plugin.disposed` is set from that
notification, the heartbeat stops, and further `capability/execute` requests are
refused. Nothing else in the protocol tells a client its fiber is gone.

Executing a capability is a reverse request. The kernel sends `capability/execute` and
wants an immediate `{"accepted": true}` — within `execute_accept_ms` (2 s by default),
which is a bound on the *acknowledgement*, not on the work. So the handler here never
runs the capability body: it accepts, spawns a task, and the body's return value goes
back later as a `capability/result` notification. A body that runs long should call
:meth:`ExecCtx.heartbeat` to hold its lease open, and should watch
:attr:`ExecCtx.cancelled`, which the kernel sets via `capability/cancel` on an e-stop,
an interrupt or an expired lease.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from . import protocol as P
from .protocol import PluginRegisterResult, StreamChannel
from .rpc import Client, RpcError

log = logging.getLogger("oeh_sdk.plugin")

#: What a capability body looks like.
CapabilityFn = Callable[[dict, "ExecCtx"], Awaitable[Any]]

#: Which `Plugin` owns which connection's `capability/execute` handler. Weak keys: a
#: dropped `Client` must not be kept alive by this bookkeeping.
_REGISTERED: "weakref.WeakKeyDictionary[Client, Plugin]" = weakref.WeakKeyDictionary()

#: Consecutive non-terminal heartbeat failures before the plugin gives up and calls
#: itself disposed. Three beats is one and a half kernel deadlines: long enough that a
#: GC pause or a busy kernel is ridden out, short enough that a genuinely wedged
#: connection is reported rather than beaten at forever.
MAX_BEAT_FAILURES = 3


@dataclass
class StreamDecl:
    """A stream this plugin produces. `tier` is `"small"` or `"large"`.

    `schema` is a name, not a channel: the kernel answers `plugin/register` with the
    `channel` and `schema_id` to actually publish on (see :attr:`Plugin.channels`).
    """

    stream_id: str
    schema: str
    nominal_hz: float
    tier: str = "small"


@dataclass
class CapabilityDecl:
    """A capability this plugin offers. `class_` is on the wire as `"class"`.

    `class_` is one of `"motion"` / `"speech"` / `"query"` / `"control"` and
    `risk_level` one of `"none"` / `"low"` / `"medium"` / `"high"`; both are what the
    rule chain branches on, and a misspelling comes back as INVALID_PARAMS from
    `plugin/register` rather than a fiber that fails to load.

    `lease_ttl_s` is how long the kernel will wait between heartbeats of a *running*
    execution before it cancels it; `0` means the kernel's own default.
    """

    name: str
    class_: str
    risk_level: str
    params_schema: Any = None
    lease_ttl_s: float = 0.0
    fn: Optional[CapabilityFn] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return P.CapabilityDecl(
            name=self.name,
            class_=self.class_,
            risk_level=self.risk_level,
            params_schema=self.params_schema,
            lease_ttl_s=self.lease_ttl_s,
        ).to_dict()


class ExecCtx:
    """One in-flight execution, handed to a capability body."""

    def __init__(self, client: Client, exec_id: str, trace_id: str, params: dict) -> None:
        self.exec_id = exec_id
        #: The turn this execution belongs to — the same `trace_id` that ties the
        #: brain's `brain/turn` to the `act/*` lines in the world log.
        self.trace_id = trace_id
        #: The post-rule-chain params, identical to `proposal["params"]`.
        self.params = params
        #: Set when the kernel sends `capability/cancel` (e-stop, interrupt, or an
        #: expired lease). A body that can stop should check it and return promptly;
        #: the kernel waits `cancel_grace_ms` for a real result before inventing one.
        self.cancelled = asyncio.Event()
        self._client = client

    def heartbeat(self) -> None:
        """Hold the lease open for another `lease_ttl_s`. Safe to call from anywhere:
        it is one queued notification, no awaiting."""
        self._client.notify(P.CAPABILITY_HEARTBEAT, {"exec_id": self.exec_id})


class Plugin:
    """A cross-process plugin.

    Either instantiate it directly::

        p = Plugin(name="py-arm", provide=["adapter.arm"],
                   streams=[StreamDecl("arm_state", "oeh.ArmState", 50.0)])

    or subclass it and set the class attributes. Capability bodies are registered with
    the :meth:`capability` decorator before :meth:`register`.

    **One plugin per connection.** `capability/execute` is a single reverse-request
    handler on the `Client` (the kernel expects one answer, so `on_request` replaces
    rather than appends), which means a second `Plugin` registering on the same
    connection would take the handler over and answer "no body for X" to the first
    one's capabilities. The kernel itself permits several fibers per connection, so
    this is the SDK's limit, not the protocol's: :meth:`register` raises rather than
    letting the first plugin go quietly dead. Give each `Plugin` its own `Client`.

    The claim is released the moment the plugin leaves the capability surface — by
    :meth:`unregister`, or by the kernel disposing the fiber (:attr:`disposed`) — so
    the connection can be handed to a different `Plugin` afterwards. Re-registering the
    *same* `Plugin` is always allowed.
    """

    name: str = ""
    # Class-level defaults are shared by every instance, so these are immutable and
    # `__init__` always rebinds to a fresh list — otherwise `p.streams.append(...)` on
    # one default-constructed plugin would add a stream to every other one.
    inject: Sequence[str] = ()
    provide: Sequence[str] = ()
    streams: Sequence[StreamDecl] = ()
    #: What this client promises the kernel. The kernel disposes the fiber after three
    #: missed periods, and answers `plugin/register` with the period it actually used.
    heartbeat_ms: int = 1000
    config: Any = None

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        inject: Optional[Sequence[str]] = None,
        provide: Optional[Sequence[str]] = None,
        streams: Optional[Sequence[StreamDecl]] = None,
        heartbeat_ms: Optional[int] = None,
        config: Any = None,
    ) -> None:
        if name is not None:
            self.name = name
        self.inject = list(self.inject if inject is None else inject)
        self.provide = list(self.provide if provide is None else provide)
        self.streams = list(self.streams if streams is None else streams)
        if heartbeat_ms is not None:
            self.heartbeat_ms = heartbeat_ms
        if config is not None:
            self.config = config
        self._caps: dict[str, CapabilityDecl] = {}
        self._execs: dict[str, ExecCtx] = {}
        self._client: Optional[Client] = None
        self._beat: Optional[asyncio.Task] = None
        self._running: dict[str, asyncio.Task] = {}
        #: `stream_id` -> where to publish it. Only populated after `register`.
        self.channels: dict[str, StreamChannel] = {}
        #: The fiber this plugin became, or `None` before `register`.
        self.fiber_id: Optional[int] = None
        #: Set when the kernel disposes this fiber out from under us — a heartbeat
        #: timeout, an operator dispose, a failed re-settle. Until something watches
        #: this, the only symptom of a disposed fiber is that `capability/execute`
        #: quietly stops arriving.
        #:
        #: A disposal *this* plugin asked for (`unregister`) does not set it. The one
        #: thing a supervisor does with this event is put the plugin back, and a
        #: deliberate shutdown is exactly when it must not.
        #:
        #: The object is created once and `clear()`ed on re-register, never rebound:
        #: a supervisor parked on `await p.disposed.wait()` across a re-register would
        #: otherwise be waiting on an orphan that nothing will ever set.
        self.disposed = asyncio.Event()
        #: Why, when the kernel said (`plugin/state`'s `error`).
        self.disposed_error: Optional[str] = None
        #: True while our own `plugin/unregister` is in flight — see `disposed`.
        self._unregistering = False
        #: The client whose handlers we have already subscribed to.
        self._wired: Optional[Client] = None

    # -- declaration -------------------------------------------------------------------

    def capability(
        self,
        name: str,
        class_: str,
        risk_level: str,
        params_schema: Any = None,
        lease_ttl_s: float = 0.0,
    ) -> Callable[[CapabilityFn], CapabilityFn]:
        """Decorator: register `async def fn(params: dict, ctx: ExecCtx) -> Any` as the
        body of a capability. Must be used before :meth:`register` — the declaration
        goes out with it."""

        def deco(fn: CapabilityFn) -> CapabilityFn:
            self._caps[name] = CapabilityDecl(
                name=name,
                class_=class_,
                risk_level=risk_level,
                params_schema=params_schema,
                lease_ttl_s=lease_ttl_s,
                fn=fn,
            )
            return fn

        return deco

    def manifest(self) -> dict:
        return P.Manifest(
            name=self.name,
            inject=list(self.inject),
            provide=list(self.provide),
            streams=[
                P.StreamSpec(s.stream_id, s.schema, s.nominal_hz, s.tier)
                for s in self.streams
            ],
        ).to_dict()

    # -- lifecycle ---------------------------------------------------------------------

    async def register(
        self, client: Client, timeout: Optional[float] = None
    ) -> PluginRegisterResult:
        """Become a fiber on this kernel, then start heartbeating.

        The reverse-request handlers go on *before* the call, not after: the kernel can
        route a proposal to a freshly registered capability while the `plugin/register`
        response is still on the wire, and a `capability/execute` arriving before the
        handler exists would be answered METHOD_NOT_FOUND — a capability that fails
        exactly once, right after startup, for no reason a log would explain.

        Raises `RuntimeError` if another `Plugin` already holds this connection; see
        the class docstring.
        """
        held = _REGISTERED.get(client)
        if held is not None and held is not self:
            raise RuntimeError(
                f"this connection already serves plugin `{held.name}`: "
                "`capability/execute` is one handler per connection, so registering "
                f"`{self.name}` on it would silently disable `{held.name}`'s "
                "capabilities. Give each Plugin its own Client."
            )
        self._client = client
        # `clear()`, not a new Event: see the attribute's note in `__init__`.
        self.disposed.clear()
        self.disposed_error = None
        self._unregistering = False
        # Once per client, not once per register cycle: `on_notification` appends, so
        # a connection that cycles plugins would grow these lists without bound — the
        # same hazard `Brain.attach` and `Approver.attach` guard against. (`on_request`
        # replaces, so it is idempotent already.)
        if self._wired is not client:
            client.on_notification(P.CAPABILITY_CANCEL, self._on_cancel)
            client.on_notification(P.PLUGIN_STATE, self._on_plugin_state)
            self._wired = client
        client.on_request(P.CAPABILITY_EXECUTE, self._on_execute)
        _REGISTERED[client] = self
        try:
            raw = await client.call(
                P.PLUGIN_REGISTER,
                P.PluginRegister(
                    manifest=P.Manifest.from_dict(self.manifest()),
                    capabilities=[
                        P.CapabilityDecl.from_dict(c.to_dict())
                        for c in self._caps.values()
                    ],
                    config=self.config,
                    heartbeat_ms=self.heartbeat_ms,
                ).to_dict(),
                timeout=timeout,
            )
        except BaseException:
            # A register that never produced a fiber must not leave this connection
            # claimed: nothing is on the capability surface, so another plugin may
            # legitimately take it.
            if held is None and self.fiber_id is None:
                _REGISTERED.pop(client, None)
            raise
        result = PluginRegisterResult.from_dict(raw)
        self.fiber_id = result.fiber_id
        self.channels = {c.stream_id: c for c in result.streams}
        # Honour the period the *kernel* reports, not the one we asked for: a kernel
        # configured with a different `heartbeat_ms` overrides the request, and beating
        # at our own slower rate would be a watchdog kill a few seconds later.
        self.heartbeat_ms = result.heartbeat_ms or self.heartbeat_ms
        self._beat = asyncio.create_task(self._heartbeat_loop(), name="oeh-plugin-beat")
        return result

    async def unregister(self) -> None:
        """Dispose the fiber and stop heartbeating.

        Waits for the kernel to finish disposing, which is the difference an adapter
        needs before it powers down: not "the command went out" but "my capabilities
        are off the surface".

        **Errors propagate**, with one exception: NOT_ATTACHED means the fiber is
        already gone (a heartbeat timeout got there first, or this is a second call),
        which is the state being asked for. Anything else — REFUSED, a timeout,
        INTERNAL — is raised, because returning quietly would tell a caller its
        capabilities are off the surface when they may still be on it. The heartbeat is
        stopped only *after* the call succeeds, so a failed `unregister` leaves a live
        fiber that a retry can still address.

        This does **not** set :attr:`disposed`. The kernel broadcasts
        `plugin/state{disposed}` for *this* dispose too, to every connection including
        ours, so the notification arrives either way — `_unregistering` is what tells
        the two apart. A supervisor watching `disposed` in order to put the plugin back
        would otherwise re-register the plugin it had just deliberately removed, and
        every clean shutdown would log a WARNING about being disposed.
        """
        client, fiber_id = self._client, self.fiber_id
        if client is not None and fiber_id is not None and not client.closed.is_set():
            self._unregistering = True
            try:
                await client.call(P.PLUGIN_UNREGISTER, {"fiber_id": fiber_id})
            except RpcError as e:
                if e.code != P.code.NOT_ATTACHED:
                    # Still ours, still beating: the flag has to come back off or a
                    # later kernel-side disposal of this same fiber would be ignored.
                    self._unregistering = False
                    raise
                log.debug("fiber %s was already gone: %s", fiber_id, e.message)
            except BaseException:
                self._unregistering = False
                raise
        self._stop_beat()
        self.fiber_id = None
        self._unregistering = False
        for t in list(self._running.values()):
            t.cancel()
        self._running.clear()
        self._execs.clear()
        if client is not None and _REGISTERED.get(client) is self:
            _REGISTERED.pop(client, None)

    def _stop_beat(self) -> None:
        if self._beat is not None:
            self._beat.cancel()
            self._beat = None

    async def _heartbeat_loop(self) -> None:
        """Beat every `heartbeat_ms / 2`, against the kernel's `3 x heartbeat_ms`
        deadline: one dropped beat still leaves room for two more.

        `plugin/heartbeat` is a **request**, not a notification. The kernel routes only
        `capability/heartbeat` and `capability/result` as notifications and discards
        everything else with a debug line, so a notification-shaped heartbeat refreshes
        nothing and every plugin is disposed after `3 x heartbeat_ms` — silently, with
        the only symptom being that `capability/execute` stops arriving.

        Being a request costs one round trip per beat, so the real interval is
        `period + round_trip`, not `period`. At the default 1000 ms that is up to about
        1.5 s against a 3 s deadline — still two beats per deadline, but less headroom
        than "twice per period" sounds like.

        Two kinds of failure, and they must not be treated alike:

        * **NOT_ATTACHED / REFUSED** — the kernel is telling us this fiber is gone.
          Beating on is pointless; stop and mark the plugin disposed.
        * **anything else** (a client-side TIMEOUT because one beat ran long on a loaded
          box, an INTERNAL, a transport hiccup) — transient. Giving up on the first one
          would throw away exactly the headroom the `3 x` deadline exists to provide:
          the fiber would then die `3 x heartbeat_ms` later for a reason that had
          already passed. Retry on the next tick, and only give up after
          `MAX_BEAT_FAILURES` in a row.

        The wait is on `client.closed` rather than a plain sleep, so a closed
        connection ends this task at once instead of up to half a period later — a task
        that outlives its connection is what leaves "Task was destroyed but it is
        pending" behind at shutdown.
        """
        period = max(self.heartbeat_ms, 1) / 2000.0
        # Bound the round trip so a wedged kernel cannot park this loop past its own
        # deadline; one period is already twice what a healthy beat needs.
        beat_timeout = max(period * 2, 1.0)
        client = self._client
        assert client is not None
        failures = 0
        try:
            while self.fiber_id is not None and not self.disposed.is_set():
                try:
                    await asyncio.wait_for(client.closed.wait(), period)
                    return
                except asyncio.TimeoutError:
                    pass
                fiber_id = self.fiber_id
                if fiber_id is None or self.disposed.is_set():
                    return
                try:
                    await client.call(
                        P.PLUGIN_HEARTBEAT, {"fiber_id": fiber_id}, timeout=beat_timeout
                    )
                except RpcError as e:
                    if e.code in (P.code.NOT_ATTACHED, P.code.REFUSED):
                        self._mark_disposed(f"heartbeat refused: {e.message}")
                        return
                    failures += 1
                    if failures >= MAX_BEAT_FAILURES:
                        self._mark_disposed(
                            f"{failures} heartbeats in a row failed; last: {e.message}"
                        )
                        return
                    log.warning(
                        "heartbeat %d/%d for fiber %s failed, retrying: %s",
                        failures,
                        MAX_BEAT_FAILURES,
                        fiber_id,
                        e.message,
                    )
                    continue
                failures = 0
        except asyncio.CancelledError:
            pass

    # -- plugin/state ------------------------------------------------------------------

    def _on_plugin_state(self, params: dict) -> None:
        """Notice the kernel disposing this fiber out from under us.

        Without this, a disposal (heartbeat timeout, operator dispose, a dependency
        going away and the fiber failing to re-settle) is invisible: `fiber_id` and
        `channels` stay populated, the beat loop keeps beating a dead id, and frames go
        into a channel nobody ingests any more.
        """
        info = P.FiberStateInfo.from_dict(params)
        if self.fiber_id is None or info.fiber_id != self.fiber_id:
            return
        if info.state == "disposed":
            if self._unregistering:
                # Our own `plugin/unregister`, coming back as the same broadcast the
                # kernel sends for any dispose. `disposed` means "taken from us".
                log.debug("fiber %s disposed as requested", info.fiber_id)
                return
            self._mark_disposed(info.error or "disposed by the kernel")
        elif info.state == "failed":
            log.warning("fiber %s failed: %s", info.fiber_id, info.error)

    def _mark_disposed(self, error: Optional[str]) -> None:
        if self.disposed.is_set():
            return
        self.disposed_error = error
        self.disposed.set()
        self._stop_beat()
        # Release the connection: this plugin is not on the capability surface any
        # more, so another may take it. Leaving the claim would refuse a new `Plugin`
        # in the name of one that no longer exists.
        client = self._client
        if client is not None and _REGISTERED.get(client) is self:
            _REGISTERED.pop(client, None)
        log.warning("plugin `%s` (fiber %s) was disposed: %s", self.name, self.fiber_id, error)

    # -- capability/execute ------------------------------------------------------------

    async def _on_execute(self, params: dict) -> dict:
        """Accept (or refuse) — never execute. See the module docstring."""
        call = P.CapabilityExecute.from_dict(params)
        if self.disposed.is_set() or self.fiber_id is None:
            # The fiber is gone; anything this body did would be outside the kernel's
            # accounting. Refusing is the fail-closed answer, and it is an *answer* —
            # letting the request time out would burn the kernel's `execute_accept_ms`
            # before the proposal could fail.
            return {"accepted": False, "reason": "fiber disposed"}
        decl = self._caps.get(call.capability)
        if decl is None or decl.fn is None:
            return {"accepted": False, "reason": f"no body for `{call.capability}`"}
        client = self._client
        if client is None:
            return {"accepted": False, "reason": "plugin is not registered"}
        proposal = call.proposal if isinstance(call.proposal, dict) else {}
        ctx = ExecCtx(
            client,
            call.exec_id,
            str(proposal.get("trace_id") or ""),
            call.params if isinstance(call.params, dict) else {},
        )
        self._execs[call.exec_id] = ctx
        self._running[call.exec_id] = asyncio.create_task(
            self._run(decl, call, ctx), name=f"oeh-exec-{call.exec_id}"
        )
        return {"accepted": True}

    async def _run(
        self, decl: CapabilityDecl, call: P.CapabilityExecute, ctx: ExecCtx
    ) -> None:
        assert decl.fn is not None
        client = self._client
        assert client is not None
        # The clamped params, which are the only ones a body may act on: the kernel
        # guarantees `params == proposal["params"]` precisely so that reading either
        # cannot silently skip the rule chain.
        args = call.params if isinstance(call.params, dict) else {}
        try:
            result = await decl.fn(args, ctx)
        except asyncio.CancelledError:
            client.notify(
                P.CAPABILITY_RESULT,
                {"exec_id": call.exec_id, "ok": False, "error": "cancelled"},
            )
            raise
        except Exception as e:  # noqa: BLE001
            # `repr(e)` and not `str(e)`: this string ends up in the brain's
            # `Outcome::Failed{error}` and in the world log, where `KeyError('times')`
            # is a diagnosis and a bare `'times'` is a riddle.
            log.exception("capability `%s` raised", call.capability)
            client.notify(
                P.CAPABILITY_RESULT,
                {"exec_id": call.exec_id, "ok": False, "error": repr(e)},
            )
        else:
            client.notify(
                P.CAPABILITY_RESULT,
                {
                    "exec_id": call.exec_id,
                    "ok": True,
                    "result": {} if result is None else result,
                },
            )
        finally:
            self._execs.pop(call.exec_id, None)
            self._running.pop(call.exec_id, None)

    def _on_cancel(self, params: dict) -> None:
        """`capability/cancel` is advisory: set the flag and let the body finish its own
        way. Killing the task instead would report `cancelled` for work the robot has
        in fact already done — and the kernel waits a grace period exactly so a
        well-behaved executor can say what really happened."""
        ctx = self._execs.get(str(params.get("exec_id", "")))
        if ctx is not None:
            ctx.cancelled.set()
