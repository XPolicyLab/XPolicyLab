"""Model-written guards and controllers evaluated at every control tick.

Reactive code must be a pure function of the observation and its own state.
While it runs, an audit hook refuses file, network and process operations,
and a per-tick time budget bounds it. Any violation is a REACTIVE_FAULT.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import signal
import sys
import threading
import time
from typing import Any, Callable


# Guard actions a session takes itself. Every embodiment stops and holds; a robot also
# opens its grippers (EpisodeCore.guard_actions, RobotSession.guard_actions).
STOP = "stop"
HOLD = "hold"
OPEN_GRIPPER = "open_gripper"
BUILTIN_ACTIONS = (STOP, HOLD, OPEN_GRIPPER)

_FORBIDDEN_PREFIXES = (
    "open", "os.", "shutil.", "socket.", "subprocess.", "urllib.", "http.", "ftplib.",
    "smtplib.", "ctypes.", "sqlite3.", "webbrowser.", "pty.", "fcntl.", "time.sleep",
)
_restricted = threading.local()
_hook_installed = False


class ReactiveFault(Exception):
    """Reactive code broke its rules: a forbidden operation, an exception, an overrun of
    the tick budget, or a signal that is missing or not a finite number. `source` names the
    guard or controller, `reason` says what happened."""

    def __init__(self, source: str, reason: str):
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason


def _audit(event: str, _args) -> None:
    if getattr(_restricted, "active", False) and event.startswith(_FORBIDDEN_PREFIXES):
        raise PermissionError(f"reactive code may not perform '{event}'")


def _install_hook() -> None:
    global _hook_installed
    if not _hook_installed:
        sys.addaudithook(_audit)
        _hook_installed = True


class _Overrun(BaseException):
    """Raised by the timer; a BaseException so broad excepts in user code miss it."""


@contextmanager
def _deadline(seconds: float):
    use_timer = (threading.current_thread() is threading.main_thread()
                 and hasattr(signal, "setitimer"))
    if not use_timer:
        yield
        return

    def expire(_signum, _frame):
        raise _Overrun()

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def call_reactive(source: str, function: Callable, *args, budget_seconds: float) -> Any:
    """Run one reactive function under the audit restriction and time budget."""
    _install_hook()
    started = time.perf_counter()
    _restricted.active = True
    try:
        # The hard timer is looser than the budget so the soft check reports the overrun.
        with _deadline(max(budget_seconds * 4, 0.001)):
            result = function(*args)
    except _Overrun:
        raise ReactiveFault(source, f"exceeded {budget_seconds * 1000:.1f} ms tick budget") from None
    except ReactiveFault:
        raise
    except Exception as error:
        raise ReactiveFault(source, f"{type(error).__name__}: {error}") from None
    finally:
        _restricted.active = False
    elapsed = time.perf_counter() - started
    if elapsed > budget_seconds:
        raise ReactiveFault(source, f"took {elapsed * 1000:.1f} ms; budget is {budget_seconds * 1000:.1f} ms")
    return result


def read_signal(observation: dict, signal_name: str | Callable, source: str, budget_seconds: float) -> float:
    """A watched signal's value: a dotted key into `observation` (list items by index, as
    in left_ee.2) or a callable of it, run like other reactive code. A missing key or a
    value that is not a finite number is a ReactiveFault from `source`."""
    if callable(signal_name):
        value = call_reactive(source, signal_name, observation, budget_seconds=budget_seconds)
    else:
        value = observation
        for part in signal_name.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                raise ReactiveFault(source, f"signal '{signal_name}' is not in the observation")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReactiveFault(source, f"signal '{signal_name}' is not a finite number")
    return float(value)


@dataclass
class Guard:
    """A registered guard: a `when` predicate or a `watch` signal with bounds, the action it
    takes when it fires, how often it fired, and the watched signal's last value."""

    name: str
    kind: str  # "when" or "watch"
    action: str
    predicate: Callable | None = None
    signal: str | Callable | None = None
    low: float | None = None
    high: float | None = None
    triggered: int = 0
    last_value: float | None = field(default=None, repr=False)

    def describe(self) -> dict:
        """The guard as the event it raises reports it: name, kind, action and trigger
        count, and for a watch its signal, bounds and last value."""
        data = {"name": self.name, "kind": self.kind, "action": self.action,
                "triggered": self.triggered}
        if self.kind == "watch":
            data.update(signal=self.signal if isinstance(self.signal, str) else "<callable>",
                        low=self.low, high=self.high, value=self.last_value)
        return data


class ReactiveLayer:
    """An episode's guards, evaluated in registration order at every control tick of a
    bounded execution, each within `budget_seconds`."""

    def __init__(self, budget_seconds: float, actions=BUILTIN_ACTIONS):
        if not budget_seconds > 0:
            raise ValueError("Reactive budget must be positive")
        self.budget_seconds = budget_seconds
        # The guard actions the session takes itself (its embodiment's); others are events.
        self.actions = tuple(actions)
        self.guards: dict[str, Guard] = {}
        self._count = 0

    def _name(self, name, prefix):
        if name is None:
            self._count += 1
            name = f"{prefix}_{self._count}"
        if not isinstance(name, str) or not name:
            raise ValueError("Guard name must be a nonempty string")
        return name

    def when(self, predicate: Callable, action: str, name: str | None = None) -> Guard:
        """Register and return a guard that fires when `predicate(observation)` is true.
        `action` is stop, hold, open_gripper or an event name; a guard of the same name is
        replaced, and the name defaults to `when_<n>`."""
        if not callable(predicate):
            raise TypeError("when(predicate, action): predicate must be callable")
        _check_action(action, self.actions)
        guard = Guard(self._name(name, "when"), "when", action, predicate=predicate)
        self.guards[guard.name] = guard
        return guard

    def watch(self, signal_name, low=None, high=None, event=None, name=None) -> Guard:
        """Register and return a guard that fires when a signal (a dotted observation key or
        a callable, see read_signal) falls below `low` or rises above `high`. The event
        defaults to `<SIGNAL>_OUT_OF_RANGE` and the name to `watch_<n>`."""
        if low is None and high is None:
            raise ValueError("watch needs low, high, or both")
        if not (callable(signal_name) or isinstance(signal_name, str)):
            raise TypeError("watch signal must be a dotted observation key or a callable")
        action = event or f"{signal_name.upper().replace('.', '_') if isinstance(signal_name, str) else 'SIGNAL'}_OUT_OF_RANGE"
        _check_action(action, self.actions)
        guard = Guard(self._name(name, "watch"), "watch", action, signal=signal_name,
                      low=low, high=high)
        self.guards[guard.name] = guard
        return guard

    def clear(self, names=None) -> None:
        """Remove the named guards (a name or several; unknown names are ignored), or all."""
        if names is None:
            self.guards.clear()
            return
        for name in [names] if isinstance(names, str) else names:
            self.guards.pop(name, None)

    def evaluate(self, observation: dict) -> Guard | None:
        """Return the first guard that fires on this observation, in registration order."""
        for guard in list(self.guards.values()):
            if guard.kind == "when":
                fired = call_reactive(guard.name, guard.predicate, observation,
                                      budget_seconds=self.budget_seconds)
                fired = bool(fired)
            else:
                value = read_signal(observation, guard.signal, guard.name, self.budget_seconds)
                guard.last_value = value
                fired = ((guard.low is not None and value < guard.low)
                         or (guard.high is not None and value > guard.high))
            if fired:
                guard.triggered += 1
                return guard
        return None


def _check_action(action, actions=BUILTIN_ACTIONS) -> None:
    if not isinstance(action, str) or not action:
        raise ValueError(f"Guard action must be {', '.join(repr(a) for a in actions)}, or an event name")
    if action not in actions and not action.replace("_", "").isalnum():
        raise ValueError(f"Event names must be identifiers like EMPTY_GRASP, got {action!r}")
