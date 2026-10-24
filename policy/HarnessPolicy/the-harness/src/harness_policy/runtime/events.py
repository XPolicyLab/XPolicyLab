"""Events end every bounded execution and flow up the hierarchy.

An event is offered to program handlers first. Only an unhandled event
escalates to the model, as an Escalation raised out of the running program.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


COMPLETED = "COMPLETED"
PREDICATE_SATISFIED = "PREDICATE_SATISFIED"
TIMEOUT = "TIMEOUT"
REGION_EXIT = "REGION_EXIT"
GUARD_TRIGGERED = "GUARD_TRIGGERED"
SHIELD_REJECTED = "SHIELD_REJECTED"
REACTIVE_FAULT = "REACTIVE_FAULT"
SKILL_ERROR = "SKILL_ERROR"
# A motion's path was executed but the arm did not arrive: contact (the table, an
# object) or a load is holding it back.
BLOCKED = "BLOCKED"
EPISODE_ENDED = "EPISODE_ENDED"

# A program may continue after these without consulting a handler.
NOMINAL = frozenset({COMPLETED, PREDICATE_SATISFIED})
BUILTIN = frozenset({
    COMPLETED, PREDICATE_SATISFIED, TIMEOUT, REGION_EXIT, GUARD_TRIGGERED,
    SHIELD_REJECTED, REACTIVE_FAULT, SKILL_ERROR, EPISODE_ENDED, BLOCKED,
})


@dataclass
class Event:
    """Why an execution ended: its kind (a built-in kind above or a guard's own action),
    the source that raised it (a skill, a guard, "shield", "environment"), details, and
    the robot step it happened at."""

    kind: str
    source: str
    detail: dict[str, Any] = field(default_factory=dict)
    tick: int | None = None

    def to_dict(self) -> dict:
        """The event as a plain dict."""
        return asdict(self)


@dataclass
class ExecutionTrace:
    """What one bounded execution did, and why it ended."""

    skill: str
    parameters: dict[str, Any]
    start_tick: int
    end_tick: int
    event: Event
    samples: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the execution ended nominally (COMPLETED or PREDICATE_SATISFIED)."""
        return self.event.kind in NOMINAL

    @property
    def ticks(self) -> int:
        """Robot steps the execution took."""
        return self.end_tick - self.start_tick

    def to_dict(self) -> dict:
        """The trace as a plain dict, with `ok`."""
        data = asdict(self)
        data["ok"] = self.ok
        return data

    def __repr__(self) -> str:
        return (f"Trace({self.skill}, {self.event.kind} from {self.event.source}, "
                f"ticks={self.ticks})")


class Escalation(Exception):
    """An event no handler resolved. It ends the current program."""

    def __init__(self, trace: ExecutionTrace):
        super().__init__(f"{trace.event.kind} in {trace.skill}")
        self.trace = trace


class ProgramFinished(Exception):
    """A program called finish(): its own checks say the task is done. The episode
    ends without another model call."""

    def __init__(self, summary: str = ""):
        super().__init__(summary)
        self.summary = str(summary)


class EpisodeEnded(Exception):
    """The environment terminated or truncated the episode during a program."""

    def __init__(self, trace: ExecutionTrace):
        super().__init__("episode ended")
        self.trace = trace
