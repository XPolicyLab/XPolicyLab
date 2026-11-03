"""Concurrent dual-arm execution.

A *track* is a program bound to one or both arms. Programs are generators that
yield one command per tick for their arms and return a result dict; the
executor merges the commands of all running tracks into a single backend tick,
holding every idle arm at its last command. Two single-arm tracks therefore
move simultaneously, which is how asynchronous dispatch and bimanual execution
work in a simulator whose clock only advances when we step it.

Admission follows the per-arm lanes of KRA, our team's in-house ReAct robot-agent
framework: an arm runs one track at a time, later tracks on that arm either
fail fast (`ArmBusy`) or wait in FIFO order, and a queued two-arm track is a
barrier that later tracks cannot overtake.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generator, Iterable, Optional

from knowin_agent.core.events import EventBus, EventKind
from knowin_agent.runtime.backend import ARMS, ArmCommand, Observation, RobotBackend, StepOutcome

Program = Generator[dict, Any, dict]


class ArmBusy(RuntimeError):
    pass


class EpisodeOver(RuntimeError):
    pass


class TrackStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


FINAL = {TrackStatus.DONE, TrackStatus.FAILED, TrackStatus.CANCELLED}


class ProgramContext:
    """What a running program can see: the latest observation and the robot profile."""

    def __init__(self, executor: "Executor", track: "Track"):
        self._executor = executor
        self.track = track

    @property
    def obs(self) -> Observation:
        return self._executor.observation

    @property
    def profile(self):
        return self._executor.backend.profile

    def hold(self, arm: str) -> ArmCommand:
        return self._executor.hold_command(arm)


@dataclass
class Track:
    id: str
    arms: tuple
    label: str
    owner: str
    factory: Callable[[ProgramContext], Program]
    status: TrackStatus = TrackStatus.QUEUED
    result: Optional[dict] = None
    created_tick: int = 0
    started_tick: Optional[int] = None
    finished_tick: Optional[int] = None
    pending: dict = field(default_factory=dict)
    _gen: Optional[Program] = None

    @property
    def finished(self) -> bool:
        return self.status in FINAL

    def summary(self) -> dict:
        out = {"track": self.id, "label": self.label, "arms": list(self.arms), "owner": self.owner,
               "status": self.status.value}
        if self.result is not None:
            out["result"] = compact_result(self.result)
        if self.started_tick is not None:
            end = self.finished_tick if self.finished_tick is not None else None
            out["ticks"] = None if end is None else end - self.started_tick
        return out


def compact_result(result: dict, depth: int = 0) -> dict:
    """Keep results readable in model context: drop bulky nested receipts."""
    out = {}
    for key, value in result.items():
        if isinstance(value, dict):
            if depth >= 1:
                if "status" in value:
                    out[key] = value["status"]
                continue
            out[key] = compact_result(value, depth + 1)
        elif isinstance(value, list) and len(value) > 8:
            out[key] = f"<{len(value)} items>"
        else:
            out[key] = value
    return out


class Executor:
    def __init__(self, backend: RobotBackend, bus: EventBus):
        self.backend = backend
        self.bus = bus
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._tracks: dict[str, Track] = {}
        self._obs = backend.observe()
        self._hold = {arm: ArmCommand(state.ee_pose, float(state.gripper_command), joints=state.joints)
                      for arm, state in self._obs.arms.items()}
        self.episode_over = False
        self.over_reason: Optional[str] = None
        self.tick_listeners: list[Callable[[Observation], None]] = []

    @property
    def observation(self) -> Observation:
        return self._obs

    @property
    def tick_count(self) -> int:
        return self._obs.tick

    def rebind(self) -> None:
        """After a backend reset: fresh observation, holds at the measured state, episode open again."""
        with self._lock:
            self._obs = self.backend.observe()
            self._hold = {arm: ArmCommand(state.ee_pose, float(state.gripper_command), joints=state.joints)
                          for arm, state in self._obs.arms.items()}
            self.episode_over, self.over_reason = False, None

    def refresh(self) -> Observation:
        """Re-read sensors without advancing time (real robots: fresh images)."""
        with self._lock:
            self._obs = self.backend.observe()
            return self._obs

    def hold_command(self, arm: str) -> ArmCommand:
        return self._hold[arm]

    def set_hold(self, arm: str, command: ArmCommand) -> None:
        self._hold[arm] = command

    # -- admission -----------------------------------------------------------------

    def busy_arms(self) -> dict[str, Track]:
        busy = {}
        for track in self._tracks.values():
            if track.status in (TrackStatus.RUNNING, TrackStatus.QUEUED):
                for arm in track.arms:
                    busy.setdefault(arm, track)
        return busy

    def submit(self, arms: Iterable[str], factory: Callable[[ProgramContext], Program], *, label: str,
               owner: str, queue: bool = False) -> Track:
        arms = tuple(dict.fromkeys(arms))
        if not arms or any(arm not in ARMS for arm in arms):
            raise ValueError(f"arms must be a non-empty subset of {ARMS}")
        with self._lock:
            if self.episode_over:
                raise EpisodeOver(f"episode is over ({self.over_reason})")
            busy = self.busy_arms()
            conflicts = {arm: busy[arm] for arm in arms if arm in busy}
            if conflicts and not queue:
                detail = ", ".join(f"{arm} arm runs track {t.id} ({t.label})" for arm, t in conflicts.items())
                raise ArmBusy(f"{detail}; wait for it, cancel it, or submit with queue=true")
            track = Track(id=f"t{next(self._ids)}", arms=arms, label=label, owner=owner, factory=factory,
                          created_tick=self.tick_count)
            self._tracks[track.id] = track
            self._admit()
            return track

    def _admit(self) -> None:
        running = {arm for t in self._tracks.values() if t.status is TrackStatus.RUNNING for arm in t.arms}
        blocked: set = set()
        for track in list(self._tracks.values()):
            if track.status is not TrackStatus.QUEUED:
                continue
            if running.intersection(track.arms) or blocked.intersection(track.arms):
                blocked.update(track.arms)
                if len(track.arms) > 1:
                    break
                continue
            self._start(track)
            if track.status is TrackStatus.RUNNING:
                running.update(track.arms)

    def _start(self, track: Track) -> None:
        track.status = TrackStatus.RUNNING
        track.started_tick = self.tick_count
        self.bus.publish(EventKind.TRACK_STARTED, track.owner, f"{track.label} started on {'+'.join(track.arms)}",
                         tick=self.tick_count, track=track.id)
        try:
            track._gen = track.factory(ProgramContext(self, track))
            track.pending = self._checked(track, next(track._gen))
        except StopIteration as stop:
            self._finish(track, TrackStatus.DONE, stop.value)
        except Exception as exc:  # noqa: BLE001 - a program error ends only its own track
            self._finish(track, TrackStatus.FAILED, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    @staticmethod
    def _checked(track: Track, commands: Optional[dict]) -> dict:
        commands = commands or {}
        for arm, command in commands.items():
            if arm not in track.arms:
                raise RuntimeError(f"track {track.id} commanded {arm} arm outside its lease {track.arms}")
            if not isinstance(command, ArmCommand):
                raise TypeError("programs must yield {arm: ArmCommand}")
        return commands

    # -- stepping ------------------------------------------------------------------

    def tick(self) -> StepOutcome:
        with self._lock:
            if self.episode_over:
                raise EpisodeOver(f"episode is over ({self.over_reason})")
            running = [t for t in self._tracks.values() if t.status is TrackStatus.RUNNING]
            commands = dict(self._hold)
            for track in running:
                commands.update(track.pending)
            outcome = self.backend.step(commands)
            self._obs = outcome.observation
            self._hold.update(commands)
            for track in running:
                try:
                    track.pending = self._checked(track, track._gen.send(self._obs))
                except StopIteration as stop:
                    self._finish(track, TrackStatus.DONE, stop.value)
                except Exception as exc:  # noqa: BLE001
                    self._finish(track, TrackStatus.FAILED, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            for listener in self.tick_listeners:
                listener(self._obs)
            if outcome.episode_over:
                self._end_episode(outcome.reason or "ended")
            else:
                self._admit()
            return outcome

    def _end_episode(self, reason: str) -> None:
        self.episode_over = True
        self.over_reason = reason
        for track in self._tracks.values():
            if not track.finished:
                if track._gen is not None:
                    track._gen.close()
                self._finish(track, TrackStatus.CANCELLED, {"status": "episode_over", "reason": reason})
        self.bus.publish(EventKind.EPISODE_OVER, "runtime", f"episode over: {reason}", tick=self.tick_count, reason=reason)

    def _finish(self, track: Track, status: TrackStatus, result: Optional[dict]) -> None:
        track.status = status
        track.result = dict(result or {"status": "done"})
        track.finished_tick = self.tick_count
        track.pending = {}
        for arm in track.arms:
            measured = self._obs.arms[arm]
            self._hold[arm] = ArmCommand(measured.ee_pose, self._hold[arm].gripper, joints=measured.joints)
        kind = {TrackStatus.DONE: EventKind.TRACK_FINISHED, TrackStatus.FAILED: EventKind.TRACK_FAILED,
                TrackStatus.CANCELLED: EventKind.TRACK_CANCELLED}[status]
        self.bus.publish(kind, track.owner, f"track {track.id} {track.label}: {track.result.get('status', status.value)}",
                         tick=self.tick_count, **track.summary())

    def run(self, tracks: Optional[Iterable[Track]] = None, *, until: str = "all",
            max_ticks: Optional[int] = None) -> list[Track]:
        """Advance time until the given tracks (default: all active) finish."""
        with self._lock:
            targets = list(tracks) if tracks is not None else [t for t in self._tracks.values() if not t.finished]
        ticks = 0
        while targets and not self.episode_over:
            done = [t.finished for t in targets]
            if all(done) or (until == "any" and any(done)):
                break
            if max_ticks is not None and ticks >= max_ticks:
                break
            self.tick()
            ticks += 1
        return targets

    def advance(self, ticks: int) -> int:
        """Advance a fixed number of ticks (running tracks progress; idle arms hold)."""
        done = 0
        while done < ticks and not self.episode_over:
            self.tick()
            done += 1
        return done

    def cancel(self, track_id: str, reason: str = "cancelled") -> Track:
        with self._lock:
            track = self.track(track_id)
            if not track.finished:
                if track._gen is not None:
                    track._gen.close()
                self._finish(track, TrackStatus.CANCELLED, {"status": "cancelled", "reason": reason})
                self._admit()
            return track

    def track(self, track_id: str) -> Track:
        if track_id not in self._tracks:
            raise KeyError(f"unknown track {track_id!r}")
        return self._tracks[track_id]

    def active(self) -> list[Track]:
        return [t for t in self._tracks.values() if not t.finished]

    def lanes(self, recent: int = 4) -> dict:
        """KRA-style queue view for model context: per-arm running/queued plus recent finishes."""
        view = {"left": [], "right": [], "both": [], "recently_finished": []}
        for track in self._tracks.values():
            if track.finished:
                continue
            lane = "both" if len(track.arms) > 1 else track.arms[0]
            view[lane].append({"track": track.id, "label": track.label, "status": track.status.value})
        finished = [t for t in self._tracks.values() if t.finished]
        view["recently_finished"] = [t.summary() for t in finished[-recent:]]
        return view
