"""The boundary protocol, transcribed for Python.

This module is the Python half of `crates/oeh-proto/src/rpc.rs`: the same 36 method
names, the same error codes, and one dataclass per `params` / `result` shape with the
same field names as the Rust structs. Nothing here talks to a socket — see
:mod:`oeh_sdk.rpc` for that.

Why transcribe instead of generate: the wire format is small, stable and reviewed, and
a generator would put a build step between a kernel change and the SDK noticing it.
What keeps the two copies honest is `tests/test_protocol.py`, which walks
`crates/oeh-proto/fixtures/rpc/*.json` — the *same* fixtures the Rust side asserts
against — and fails if Python knows a method Rust does not, if Rust has a method
Python does not, or if any fixture's `params`/`result` does not survive a round trip
through the dataclass for that method.

Two field names cannot be spelled the way Rust spells them, because they are Python
keywords. Both carry an explicit wire mapping (`_WIRE`) so the JSON is unchanged:

* `CapabilityDecl.class_`  -> `"class"`
* `LogTailResult.from_`    -> `"from"`

Enumerations stay strings, exactly as they are on the wire. In particular the FSM
values are SCREAMING_SNAKE_CASE (`"STANDING"`, `"ACTING"`, `"EMERGENCY_STOP"`), and
`Outcome` is internally tagged: `{"outcome": "executed", "result": ...}` /
`{"outcome": "refused", "reason": ...}` / `failed` / `cancelled` / `preempted`.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Optional

JSONRPC = "2.0"

# --------------------------------------------------------------------------------------
# error codes (`oeh_proto::rpc::code`)
# --------------------------------------------------------------------------------------


class code:
    """JSON-RPC 2.0's own codes, plus this boundary's four.

    The kernel-specific ones are the ones worth branching on:

    * ``KERNEL_GONE``  — the kernel's Op channel is closed; retrying will not help.
    * ``NOT_ATTACHED`` — this *connection* holds no such brain / turn / fiber.
    * ``REFUSED``      — a policy refusal (reserved event kind, duplicate register,
      ``brain/attach`` racing a detach on the same connection...), not a malformed call.
    * ``TIMEOUT``      — the kernel gave up waiting, not the peer saying no.
    """

    PARSE = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL = -32603
    KERNEL_GONE = -32000
    NOT_ATTACHED = -32001
    REFUSED = -32002
    TIMEOUT = -32003


# --------------------------------------------------------------------------------------
# method table (`oeh_proto::rpc::method`) — identical strings, identical order
# --------------------------------------------------------------------------------------

ROBOT_CAPABILITIES = "robot/capabilities"
ROBOT_STATE = "robot/state"
ROBOT_HEALTH = "robot/health"
BRAIN_ATTACH = "brain/attach"
BRAIN_DETACH = "brain/detach"
BRAIN_CURSOR = "brain/cursor"
BRAIN_WAKE_POLICY_UPDATE = "brain/wake_policy/update"
BRAIN_INJECT = "brain/inject"
TURN_STEER = "turn/steer"
TURN_INTERRUPT = "turn/interrupt"
TURN_PROPOSE = "turn/propose"
TURN_SAY = "turn/say"
TURN_ASK_HUMAN = "turn/ask_human"
TURN_DONE = "turn/done"
STREAM_LIST = "stream/list"
STREAM_SUBSCRIBE = "stream/subscribe"
PLUGIN_REGISTER = "plugin/register"
PLUGIN_HEARTBEAT = "plugin/heartbeat"
PLUGIN_UNREGISTER = "plugin/unregister"
EVENT_EMIT = "event/emit"
LOG_TAIL = "log/tail"
LOG_REPLAY = "log/replay"
LOG_FLUSH = "log/flush"
APPROVAL_ATTACH = "approval/attach"
APPROVAL_DETACH = "approval/detach"
#: kernel -> client notification: another connection took over the approval channel.
APPROVAL_DETACHED = "approval/detached"
# kernel -> client requests
APPROVAL_REQUEST = "approval/request"
CAPABILITY_EXECUTE = "capability/execute"
# client -> kernel notifications (while an execution is in flight)
CAPABILITY_HEARTBEAT = "capability/heartbeat"
CAPABILITY_RESULT = "capability/result"
# kernel -> client notifications
CAPABILITY_CANCEL = "capability/cancel"
BRAIN_TURN = "brain/turn"
BRAIN_STEER = "brain/steer"
BRAIN_INTERRUPT = "brain/interrupt"
EVENT = "event"
PLUGIN_STATE = "plugin/state"
SHUTDOWN = "shutdown"

#: Every method on the boundary, in `oeh_proto::rpc::method::ALL` order.
METHODS: list[str] = [
    ROBOT_CAPABILITIES,
    ROBOT_STATE,
    ROBOT_HEALTH,
    BRAIN_ATTACH,
    BRAIN_DETACH,
    BRAIN_CURSOR,
    BRAIN_WAKE_POLICY_UPDATE,
    BRAIN_INJECT,
    TURN_STEER,
    TURN_INTERRUPT,
    TURN_PROPOSE,
    TURN_SAY,
    TURN_ASK_HUMAN,
    TURN_DONE,
    STREAM_LIST,
    STREAM_SUBSCRIBE,
    PLUGIN_REGISTER,
    PLUGIN_HEARTBEAT,
    PLUGIN_UNREGISTER,
    EVENT_EMIT,
    LOG_TAIL,
    LOG_REPLAY,
    LOG_FLUSH,
    APPROVAL_ATTACH,
    APPROVAL_DETACH,
    APPROVAL_DETACHED,
    APPROVAL_REQUEST,
    CAPABILITY_EXECUTE,
    CAPABILITY_HEARTBEAT,
    CAPABILITY_RESULT,
    CAPABILITY_CANCEL,
    BRAIN_TURN,
    BRAIN_STEER,
    BRAIN_INTERRUPT,
    EVENT,
    PLUGIN_STATE,
    SHUTDOWN,
]

#: Methods the *kernel* sends as requests, and that a client therefore has to answer.
KERNEL_REQUESTS: frozenset[str] = frozenset({APPROVAL_REQUEST, CAPABILITY_EXECUTE})

#: Methods the kernel sends as notifications.
KERNEL_NOTIFICATIONS: frozenset[str] = frozenset(
    {
        APPROVAL_DETACHED,
        CAPABILITY_CANCEL,
        BRAIN_TURN,
        BRAIN_STEER,
        BRAIN_INTERRUPT,
        EVENT,
        PLUGIN_STATE,
        SHUTDOWN,
    }
)


# --------------------------------------------------------------------------------------
# dataclass plumbing
# --------------------------------------------------------------------------------------


def _from_wire(tp: Any, v: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        if v is None:
            return None
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _from_wire(args[0], v) if args else v
    if origin is list:
        args = typing.get_args(tp)
        inner = args[0] if args else Any
        return [_from_wire(inner, i) for i in (v or [])]
    if isinstance(tp, type) and issubclass(tp, Wire):
        return tp.from_dict(v or {})
    return v


def _to_wire(v: Any) -> Any:
    if isinstance(v, Wire):
        return v.to_dict()
    if isinstance(v, list):
        return [_to_wire(i) for i in v]
    return v


class Wire:
    """Base for every params/result shape: `to_dict()` / `from_dict()` and value equality.

    Unknown keys are dropped on the way in. That is deliberate forward compatibility:
    a kernel that grows a field must not break an older SDK, and the fixture round-trip
    test is what catches a field this module has gone and *forgotten*.
    """

    #: Python attribute name -> JSON key, for the names Python will not let us spell.
    _WIRE: ClassVar[Mapping[str, str]] = {}

    @classmethod
    def _hints(cls) -> dict[str, Any]:
        cached = cls.__dict__.get("__wire_hints__")
        if cached is None:
            cached = typing.get_type_hints(cls)
            setattr(cls, "__wire_hints__", cached)
        return cached

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Any:
        hints = cls._hints()
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):  # type: ignore[arg-type]
            key = cls._WIRE.get(f.name, f.name)
            if key in d:
                kwargs[f.name] = _from_wire(hints.get(f.name, Any), d[key])
        return cls(**kwargs)  # type: ignore[call-arg]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            # An optional field (one whose declared default is `None`) that is `None`
            # is left off the wire rather than sent as an explicit null: every such
            # field is `#[serde(default)]` on the Rust side, so absence and null mean
            # the same thing there, and absence is what the fixtures show.
            if v is None and f.default is None:
                continue
            out[type(self)._WIRE.get(f.name, f.name)] = _to_wire(v)
        return out


# --------------------------------------------------------------------------------------
# shared shapes (`oeh_proto`, not `oeh_proto::rpc`)
# --------------------------------------------------------------------------------------


@dataclass
class StreamSpec(Wire):
    """One stream an adapter declares. `tier` is `"small"` | `"large"`."""

    stream_id: str
    schema: str
    nominal_hz: float
    tier: str = "small"


@dataclass
class Manifest(Wire):
    """What a fiber declares about itself — the same struct in-process plugins use."""

    name: str
    inject: list[str] = field(default_factory=list)
    provide: list[str] = field(default_factory=list)
    streams: list[StreamSpec] = field(default_factory=list)
    config_schema: Optional[Any] = None


@dataclass
class FrameRef(Wire):
    stream_id: str
    frame_seq: int
    t_kernel: int


@dataclass
class WorldEvent(Wire):
    """One line of the world log; also the params of the `event` notification."""

    seq: int
    t_kernel: int
    kind: str
    source: str
    payload: Any
    frame_refs: list[FrameRef] = field(default_factory=list)
    salience: int = 0
    collapse_key: Optional[str] = None
    trace_id: Optional[str] = None
    required: bool = True


# --------------------------------------------------------------------------------------
# `oeh_proto::rpc::types`
# --------------------------------------------------------------------------------------


@dataclass
class Empty(Wire):
    """`{}` — the params of a nullary method, and the result of a method with no answer."""


@dataclass
class CapabilityDecl(Wire):
    """One capability offered to the admission pipeline.

    `class_` is on the wire as `"class"`. Both it and `risk_level` are the lowercase
    serde names (`"motion"` / `"speech"` / `"query"` / `"control"`, and `"none"` /
    `"low"` / `"medium"` / `"high"`); a misspelling is rejected at `plugin/register`
    with INVALID_PARAMS rather than turning into a failed fiber.

    `provider` and `overridable` are the kernel's side of the shape: they come back
    from `robot/capabilities` and say who executes this name and whether a plugin may
    take it over (only the built-in `say` may be). Sending them with a
    `plugin/register` changes nothing — the kernel fills in both — so they default to
    `None` and stay off that wire.
    """

    _WIRE: ClassVar[Mapping[str, str]] = {"class_": "class"}

    name: str
    class_: str
    risk_level: str
    params_schema: Any = None
    lease_ttl_s: float = 0.0
    provider: Optional[str] = None
    overridable: Optional[bool] = None


@dataclass
class RobotCapabilitiesResult(Wire):
    capabilities: list[CapabilityDecl] = field(default_factory=list)


@dataclass
class RobotStateResult(Wire):
    """`fsm` is SCREAMING_SNAKE_CASE (`"STANDING"`), not the lowercase serde style."""

    fsm: str
    state: Any
    t_kernel: int


@dataclass
class FiberStateInfo(Wire):
    """`state` is `"pending"` | `"active"` | `"failed"` | `"disposed"`.

    `"loading"` exists in the kernel enum but is never broadcast — a consumer must not
    wait for it.
    """

    fiber_id: int
    plugin: str
    state: str
    error: Optional[str] = None


@dataclass
class StreamHealth(Wire):
    stream_id: str
    schema: str
    nominal_hz: float
    frames: int
    age_ms: Optional[int] = None


@dataclass
class RobotHealthResult(Wire):
    fibers: list[FiberStateInfo] = field(default_factory=list)
    streams: list[StreamHealth] = field(default_factory=list)
    log_sink_errors: int = 0
    brains_dropped_events: int = 0
    connections: int = 0


@dataclass
class BrainAttach(Wire):
    brain_id: str
    wake_policy: Optional[Any] = None


@dataclass
class BrainAttachResult(Wire):
    cursor: int


@dataclass
class BrainRef(Wire):
    brain_id: str


@dataclass
class BrainCursorResult(Wire):
    """`status` is `"idle"` | `"running"` | `"maintenance"`."""

    cursor: int
    status: str


@dataclass
class WakePolicyUpdate(Wire):
    brain_id: str
    wake_policy: Any


@dataclass
class BrainInject(Wire):
    """`brain/inject` params.

    The injector's identity is *not* a parameter: the kernel fills the event's
    `source` from the connection's name (`rpc:<name>`), the same rule `event/emit`
    follows. `text` is 1..=4096 bytes; anything else is INVALID_PARAMS, and so is a
    `brain_id` no brain is attached under.
    """

    brain_id: str
    text: str


@dataclass
class BrainInjectResult(Wire):
    """The seq of the `brain/inject` event, to line the call up with the world log."""

    seq: int


@dataclass
class TurnSteer(Wire):
    brain_id: str
    text: str


@dataclass
class TurnPropose(Wire):
    brain_id: str
    turn: int
    capability: str
    params: Optional[Any] = None


@dataclass
class TurnSay(Wire):
    brain_id: str
    turn: int
    text: str


@dataclass
class TurnAskHuman(Wire):
    brain_id: str
    turn: int
    question: str


@dataclass
class TurnDone(Wire):
    brain_id: str
    turn: int
    reason: str = ""


@dataclass
class TurnProposeResult(Wire):
    """`outcome` is the internally tagged `Outcome`: `{"outcome": "executed", ...}`."""

    proposal_id: str
    outcome: Any


@dataclass
class StreamChannel(Wire):
    """Where a stream's frames live. Never compute `channel`/`schema_id` yourself —
    they are assigned per run (the channel carries the run tag) and handed back by
    `plugin/register` / `stream/subscribe`."""

    stream_id: str
    channel: str
    schema_id: str


@dataclass
class StreamListResult(Wire):
    streams: list[StreamSpec] = field(default_factory=list)


@dataclass
class StreamSubscribe(Wire):
    stream_id: str


@dataclass
class PluginRegister(Wire):
    """`heartbeat_ms == 0` means "use the kernel default" (1000 ms)."""

    manifest: Manifest
    capabilities: list[CapabilityDecl] = field(default_factory=list)
    config: Optional[Any] = None
    heartbeat_ms: int = 0


@dataclass
class PluginRegisterResult(Wire):
    """`state == "failed"` is still a *successful* call: the kernel took the row, tried
    to activate it, and is telling you why it did not come up. `fiber_id` is valid
    either way, and `plugin/unregister` still applies to it."""

    fiber_id: int
    state: str
    error: Optional[str] = None
    streams: list[StreamChannel] = field(default_factory=list)
    heartbeat_ms: int = 0


@dataclass
class FiberRef(Wire):
    fiber_id: int


@dataclass
class EventEmit(Wire):
    """`source` is not a field: the kernel stamps `rpc:<connection name>` itself.

    `act/*`, `safety/fsm` and `safety/tamper_attempt` are the kernel's own bookkeeping
    and come back REFUSED.
    """

    kind: str
    payload: Optional[Any] = None
    salience: Optional[int] = None
    collapse_key: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class EventEmitResult(Wire):
    seq: int


@dataclass
class LogTail(Wire):
    """`prefixes` matches whole kind *segments*: `"obs"` hits `obs/person`, not
    a future `obsolete/x`. Empty = everything."""

    after: Optional[int] = None
    prefixes: list[str] = field(default_factory=list)


@dataclass
class LogTailResult(Wire):
    """`from_` is `"from"` on the wire: the earliest seq this subscription covers.

    Backfill is capped, so `from_` can be later than the `after` you asked for —
    anything earlier is `log/replay`'s job.
    """

    _WIRE: ClassVar[Mapping[str, str]] = {"from_": "from"}

    from_: int


@dataclass
class LogReplay(Wire):
    dir: Optional[str] = None
    after: int = 0
    limit: int = 1000


@dataclass
class LogReplayResult(Wire):
    events: list[WorldEvent] = field(default_factory=list)
    done: bool = False


@dataclass
class ApprovalAttach(Wire):
    name: str = ""


@dataclass
class ApprovalDetached(Wire):
    """Today `reason` is only ever `"replaced"` — `approval/attach` is preemptive."""

    reason: str


@dataclass
class ApprovalRequest(Wire):
    ask_id: str
    proposal: Any
    spec: Any
    reason: str
    trace: Optional[Any] = None


@dataclass
class ApprovalDecision(Wire):
    """`decision` is `"allow"` | `"deny"`. An empty `by` falls back to the name the
    approver attached with."""

    decision: str
    by: str
    reason: str


@dataclass
class CapabilityExecute(Wire):
    """Invariant: `params == proposal["params"]`, and both are the post-rule-chain
    (clamped, rate-limited) copy. An executor that reads `proposal["params"]` instead
    of the top-level `params` would silently bypass the rule chain — they are equal
    precisely so that mistake cannot happen."""

    exec_id: str
    capability: str
    params: Any
    proposal: Any
    lease_ttl_s: float


@dataclass
class CapabilityAccepted(Wire):
    accepted: bool
    reason: Optional[str] = None


@dataclass
class ExecRef(Wire):
    exec_id: str


@dataclass
class CapabilityResult(Wire):
    exec_id: str
    ok: bool
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class CapabilityCancel(Wire):
    exec_id: str
    reason: str


@dataclass
class BrainTurn(Wire):
    brain_id: str
    turn: int
    trace_id: str
    observation: Any


@dataclass
class BrainSteerNotice(Wire):
    brain_id: str
    observation: Any


@dataclass
class ShutdownNotice(Wire):
    """Sent once, just before the kernel stops answering."""

    reason: str


# --------------------------------------------------------------------------------------
# method -> shape
# --------------------------------------------------------------------------------------

#: The `params` type of every method on the boundary.
PARAMS: dict[str, type[Wire]] = {
    ROBOT_CAPABILITIES: Empty,
    ROBOT_STATE: Empty,
    ROBOT_HEALTH: Empty,
    BRAIN_ATTACH: BrainAttach,
    BRAIN_DETACH: BrainRef,
    BRAIN_CURSOR: BrainRef,
    BRAIN_WAKE_POLICY_UPDATE: WakePolicyUpdate,
    BRAIN_INJECT: BrainInject,
    TURN_STEER: TurnSteer,
    TURN_INTERRUPT: BrainRef,
    TURN_PROPOSE: TurnPropose,
    TURN_SAY: TurnSay,
    TURN_ASK_HUMAN: TurnAskHuman,
    TURN_DONE: TurnDone,
    STREAM_LIST: Empty,
    STREAM_SUBSCRIBE: StreamSubscribe,
    PLUGIN_REGISTER: PluginRegister,
    PLUGIN_HEARTBEAT: FiberRef,
    PLUGIN_UNREGISTER: FiberRef,
    EVENT_EMIT: EventEmit,
    LOG_TAIL: LogTail,
    LOG_REPLAY: LogReplay,
    LOG_FLUSH: Empty,
    APPROVAL_ATTACH: ApprovalAttach,
    APPROVAL_DETACH: Empty,
    APPROVAL_DETACHED: ApprovalDetached,
    APPROVAL_REQUEST: ApprovalRequest,
    CAPABILITY_EXECUTE: CapabilityExecute,
    CAPABILITY_HEARTBEAT: ExecRef,
    CAPABILITY_RESULT: CapabilityResult,
    CAPABILITY_CANCEL: CapabilityCancel,
    BRAIN_TURN: BrainTurn,
    BRAIN_STEER: BrainSteerNotice,
    BRAIN_INTERRUPT: BrainRef,
    EVENT: WorldEvent,
    PLUGIN_STATE: FiberStateInfo,
    SHUTDOWN: ShutdownNotice,
}

#: The `result` type of every method that answers. `None` = a notification (no answer).
RESULTS: dict[str, Optional[type[Wire]]] = {
    ROBOT_CAPABILITIES: RobotCapabilitiesResult,
    ROBOT_STATE: RobotStateResult,
    ROBOT_HEALTH: RobotHealthResult,
    BRAIN_ATTACH: BrainAttachResult,
    BRAIN_DETACH: Empty,
    BRAIN_CURSOR: BrainCursorResult,
    BRAIN_WAKE_POLICY_UPDATE: Empty,
    BRAIN_INJECT: BrainInjectResult,
    TURN_STEER: Empty,
    TURN_INTERRUPT: Empty,
    TURN_PROPOSE: TurnProposeResult,
    TURN_SAY: TurnProposeResult,
    TURN_ASK_HUMAN: TurnProposeResult,
    TURN_DONE: Empty,
    STREAM_LIST: StreamListResult,
    STREAM_SUBSCRIBE: StreamChannel,
    PLUGIN_REGISTER: PluginRegisterResult,
    PLUGIN_HEARTBEAT: Empty,
    PLUGIN_UNREGISTER: Empty,
    EVENT_EMIT: EventEmitResult,
    LOG_TAIL: LogTailResult,
    LOG_REPLAY: LogReplayResult,
    LOG_FLUSH: Empty,
    APPROVAL_ATTACH: Empty,
    APPROVAL_DETACH: Empty,
    APPROVAL_DETACHED: None,
    APPROVAL_REQUEST: ApprovalDecision,
    CAPABILITY_EXECUTE: CapabilityAccepted,
    CAPABILITY_HEARTBEAT: None,
    CAPABILITY_RESULT: None,
    CAPABILITY_CANCEL: None,
    BRAIN_TURN: None,
    BRAIN_STEER: None,
    BRAIN_INTERRUPT: None,
    EVENT: None,
    PLUGIN_STATE: None,
    SHUTDOWN: None,
}
