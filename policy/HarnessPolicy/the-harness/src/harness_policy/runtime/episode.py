"""The episode core: what every kernel session has, whatever the embodiment and its skills.

A kernel session connects to a worker (profile, state, tick, observe, ...), keeps the
state the worker reports, the model clients of the skills (VLM, pointer, verifier,
segmenter), the learned policies and the episode's events, and runs bounded executions:
at each step the reactive layer evaluates guards, the termination conditions are
checked, and one command goes to the worker, whose shield passes it unchanged or
rejects it. An execution ends with an event; events that are not nominal go to
registered handlers, and an event no handler resolves escalates to the model.

Nothing here knows what the commands move. An embodiment kind (embodiments.KINDS)
subclasses EpisodeCore and says it: what a step's command is and how it reaches the
worker (_tick), the command that leaves everything as it is (_base_command), what
reactive code observes (_reactive_view), when a command leaves an execution's scope
(_outside), what a trace samples, and which guard actions it takes itself. The robot's
is RobotSession (runtime/session.py).

The skills (skills/<name>/) add the API on top: the kernel's session class is the
embodiment's session plus the mixin of every skill for that embodiment
(session_class()), and its namespace holds the exports of the skills enabled for the run
(runtime/skills.py).

Bootstrap: harness_policy.runtime.kernel --bootstrap harness_policy.runtime.session:create_session
Options: {"endpoint", "episode_id", "embodiment", "skills", "reactive_budget_ms", "default_max_seconds",
          "vlm", "pointer", "verifier", "segmenter", "policies", "world_models", "mcp", "memory"}
"""

import importlib
import math

from harness_policy.core.robot import RemoteEnvironment
from harness_policy.runtime.events import EpisodeEnded, Escalation, Event, ExecutionTrace
from harness_policy.runtime import events as ev
from harness_policy.runtime.reactive import HOLD, STOP, ReactiveFault, ReactiveLayer, call_reactive


# The session of each embodiment kind, as "module:Class" (embodiments.KINDS).
SESSIONS = {"robot": "harness_policy.runtime.session:RobotSession"}


class EpisodeCore:
    """The part of a kernel session every embodiment and skill builds on: the worker
    connection and profile, the state, the model clients and learned policies, handlers
    and the reactive layer, and bounded executions. An embodiment's session subclasses it
    (see the module docstring); its `namespace` holds the enabled skills' exports."""

    # The embodiment kind of this session (embodiments.KINDS).
    embodiment = None
    # Guard actions the session takes itself (when(predicate, action)); any other action
    # is an event name.
    guard_actions = (STOP, HOLD)
    # Executions that change nothing: they run before start(strategy, kind) too.
    still_executions = ("hold",)
    # The skill whose start(strategy, kind) must come before the first change, when the
    # run asks for it (strategy_first), and what the model is told until then.
    strategy_skill = None
    strategy_hint = "before start only looking runs"

    def __init__(self, options):
        self.env = RemoteEnvironment(options["endpoint"], timeout=float(options.get("timeout_seconds", 300)))
        self.episode_id = options["episode_id"]
        self.profile = self.env.call("profile")
        self.hz = float(self.profile["control_hz"])
        self.default_max_seconds = float(options.get("default_max_seconds", 30.0))
        self.reactive = ReactiveLayer(float(options.get("reactive_budget_ms", 5.0)) / 1000.0, self.guard_actions)
        self.handlers = {}
        self.policies = dict(options.get("policies") or {})
        # World-action models for predict_future: {name: {"url", "description"}}.
        self.world_models = dict(options.get("world_models") or {})
        # MCP tool servers for the mcp skill: {name: {"url", "description"}}.
        self.mcp_servers = dict(options.get("mcp") or {})
        self._policy_clients = {}
        self._attachments = []
        self._events = []
        # Latest views of a running controller that servos on images.
        self._live = None
        self.state = self.env.call("state", episode_id=self.episode_id)
        self._embodiment_setup(options)
        # Success verifier (App. E, Tab. 5): a separate model a program may ask.
        self.verifier = None
        if options.get("verifier"):
            from harness_policy.services.clients import VisionModel
            self.verifier = VisionModel(options["verifier"])
        self.vision = None
        if options.get("vlm"):
            from harness_policy.services.clients import VisionModel
            self.vision = VisionModel(options["vlm"])
        # Open-vocabulary pointing (Molmo2 in the paper); without it point_to uses the vlm.
        self.pointer = None
        if options.get("pointer"):
            from harness_policy.services.clients import VisionModel
            self.pointer = VisionModel(options["pointer"])
        self.segmenter = None
        if options.get("segmenter"):
            from harness_policy.services.clients import Segmenter
            self.segmenter = Segmenter(options["segmenter"]["url"],
                                       timeout=float(options["segmenter"].get("timeout_seconds", 120.0)),
                                       on_demand=options["segmenter"].get("on_demand"))
        # Library memory entries ({"name", "text"}), fixed for the episode (read-only).
        self.memory = [dict(entry) for entry in options.get("memory") or []]
        # The skills this kernel exposes: the driver passes the resolved list; without
        # it, every skill of the enabled interface layers for this embodiment.
        from .skills import namespace, reporters, skills_for_layers

        names = options.get("skills")
        self.skills = list(skills_for_layers(options.get("layers"), embodiment=self.embodiment)
                           if names is None else names)
        self.namespace = {"__name__": "__program__", **namespace(self, self.skills)}
        self._reporters = reporters(self.skills)
        # With learned policies the first change is start(strategy, kind) (skills/vla):
        # until then only executions that change nothing run.
        self._strategy_first = bool(options.get("strategy_first")) and self.strategy_skill in self.skills
        # A run of a learned policy ends when the objects the last locate tracks have not
        # moved for this many seconds (skills/vla run_policy); 0: only when it stops.
        self._policy_progress_seconds = float(options.get("policy_progress_seconds") or 0.0)

    def _embodiment_setup(self, options):
        """What the embodiment's session keeps besides the core's, set before the skills'
        namespace is built (the profile and the first state are there)."""

    def drain(self):
        """What the last cell produced, for the kernel's reply: attachments and events
        gathered since the previous drain (then cleared), the state, model usage, and the
        enabled skills' turn report."""
        report = self._turn_report()
        data = {"attachments": self._attachments, "events": self._events,
                "state": self.state, "usage": self._drain_usage()}
        if report is not None:
            data["report"] = report
        if self.__dict__.get("_strategy"):
            data["strategy"] = dict(self._strategy)
        self._attachments, self._events = [], []
        return data

    def _turn_report(self):
        """What the enabled skills add to the model's next turn after every program (each
        skill's turn_report: text, and camera images to show in place of the plain ones),
        joined; None when none adds anything or the episode has ended. A report that fails
        becomes a note, never a failed program."""
        if self.state.get("ended"):
            return None
        texts, images = [], {}
        for report in getattr(self, "_reporters", ()):
            try:
                part = report(self)
            except Exception as error:
                texts.append(f"({report.__module__} could not report this turn: {type(error).__name__}: {error})")
                continue
            if part:
                if part.get("text"):
                    texts.append(part["text"])
                images.update(part.get("images") or {})
        if not texts and not images:
            return None
        return {"text": "\n".join(texts), "images": images}

    def close(self):
        """Close the worker connection."""
        self.env.close()

    def _drain_usage(self):
        """Model usage of the perception and verifier calls since the last drain."""
        usage = {}
        for name, model in (("vision", self.vision), ("pointer", self.pointer), ("verifier", self.verifier)):
            if model is not None and model.usage["calls"]:
                usage[name] = dict(model.usage)
                model.usage.update(calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0)
        return usage

    def _vision(self):
        if self.vision is None:
            raise RuntimeError("No vision model is configured for this harness")
        return self.vision

    def _execute(self, skill, parameters, command_fn, max_seconds, region, until, timeout_is_nominal=False,
                 cameras_at=None):
        """One bounded execution: command_fn(state, tick) gives each step's command (None
        when the skill is done, or an Event that ends it), for at most `max_seconds`,
        inside `region` (the embodiment's scope), until `until(view)` holds."""
        if (getattr(self, "_strategy_first", False) and not self.__dict__.get("_strategy")
                and skill not in self.still_executions and not self.__dict__.get("_wrapping_up")):
            step = self.state["step"]
            return self._finish(ExecutionTrace(skill, parameters, step, step, Event(
                ev.SKILL_ERROR, skill, {"reason": "call start(strategy, kind) first",
                                        "hint": self.strategy_hint}, step)))
        if (self.__dict__.get("_handover_locked") and skill not in self.still_executions
                and not self.__dict__.get("_wrapping_up")):
            step = self.state["step"]
            return self._finish(ExecutionTrace(skill, parameters, step, step, Event(
                ev.SKILL_ERROR, skill, {"reason": "the policy has this task",
                                        "hint": "do not move the arms after handing the task over"}, step)))
        max_seconds = self.default_max_seconds if max_seconds is None else float(max_seconds)
        max_ticks = max(1, math.ceil(max_seconds * self.hz))
        start = self.state["step"]
        samples = []
        tick = 0
        event = None
        self._motion_started(self.state)
        while event is None:
            state = self.state
            view = self._reactive_view(state, tick)
            if cameras_at is not None:
                view["observation"] = self._live
            try:
                guard = self.reactive.evaluate(view)
                if guard is None and until is not None and bool(call_reactive(
                        "until", until, view, budget_seconds=self.reactive.budget_seconds)):
                    event = Event(ev.PREDICATE_SATISFIED, "until", tick=state["step"])
                    break
            except ReactiveFault as fault:
                self._motion_stopped(state)
                held = self._tick(self._base_command(state))
                event = Event(ev.REACTIVE_FAULT, fault.source,
                              {"reason": fault.reason, **self._hold_detail(held)}, state["step"])
                break
            if guard is not None:
                self._motion_stopped(state)
                event = self._apply_guard(guard, state)
                break
            if region is not None and self._outside(region, state):
                event = Event(ev.REGION_EXIT, skill, {"region": region}, state["step"])
                break
            if tick >= max_ticks:
                kind = ev.COMPLETED if timeout_is_nominal else ev.TIMEOUT
                event = Event(kind, skill, {"max_seconds": max_seconds}, state["step"])
                break
            try:
                target = command_fn(state, tick)
            except ReactiveFault as fault:
                event = Event(ev.REACTIVE_FAULT, fault.source, {"reason": fault.reason}, state["step"])
                break
            if target is None:
                event = Event(ev.COMPLETED, skill, tick=state["step"])
                break
            if isinstance(target, Event):
                event = target
                break
            result = self._tick(target, cameras=cameras_at(tick) if cameras_at else None)
            shield = result["shield"]
            if not result["executed"]:
                event = Event(ev.SHIELD_REJECTED, "shield", {"reasons": shield["reasons"]}, state["step"])
                break
            self._motion_ticked(shield["command"])
            if tick % 5 == 0:
                samples.append(self._sample(self.state))
            tick += 1
            if self.state["ended"]:
                trace = ExecutionTrace(skill, parameters, start, self.state["step"],
                                       Event(ev.EPISODE_ENDED, "environment", tick=self.state["step"]),
                                       samples)
                raise EpisodeEnded(trace)
        self._motion_stopped(self.state, arrived=event.kind == ev.COMPLETED)
        trace = ExecutionTrace(skill, parameters, start, self.state["step"], event, samples)
        return self._finish(trace)

    def _finish(self, trace):
        self._events.append({"skill": trace.skill, "event": trace.event.to_dict(), "ticks": trace.ticks})
        if trace.ok:
            return trace
        handler = self.handlers.get(trace.event.kind)
        if handler is None:
            raise Escalation(trace)
        verdict = handler(trace)
        if verdict is False:
            raise Escalation(trace)
        self._events[-1]["handled"] = True
        return trace

    def _failed(self, skill, parameters, reason):
        step = self.state["step"]
        return ExecutionTrace(skill, parameters, step, step, Event(ev.SKILL_ERROR, skill, {"reason": reason}, step))

    def _apply_guard(self, guard, state):
        """The event of a guard that fired, after the action it takes when the session
        takes it (every guard action but stop sends one command)."""
        held = None
        if guard.action in self.guard_actions and guard.action != STOP:
            held = self._tick(self._guard_command(guard.action, state))
        detail = {"guard": guard.describe(), **self._hold_detail(held)}
        if guard.action in self.guard_actions:
            return Event(ev.GUARD_TRIGGERED, guard.name, {**detail, "action": guard.action}, state["step"])
        return Event(guard.action, guard.name, detail, state["step"])

    def _guard_command(self, action, state):
        """The command a guard action other than stop sends: hold keeps everything as it
        is."""
        return self._base_command(state)

    @staticmethod
    def _hold_detail(result):
        """The shield's reasons when the hold a guard or fault sent was rejected."""
        if result is None or result["executed"]:
            return {}
        return {"hold_rejected": result["shield"]["reasons"]}

    # What the embodiment's session defines.

    def _base_command(self, state):
        """The command that leaves everything as it is."""
        raise NotImplementedError

    def _tick(self, command, cameras=None):
        """Send one step's command to the worker; the worker's reply {"executed",
        "shield", "state", ...}, with self.state updated."""
        raise NotImplementedError

    def _motion_started(self, state):
        """An execution begins (what an embodiment holds afterwards may depend on it)."""

    def _motion_ticked(self, command):
        """The shield passed this command of the running execution."""

    def _motion_stopped(self, state, arrived=False):
        """The running execution was interrupted (a guard, a fault) or ended; `arrived`
        when it completed."""

    def _reactive_view(self, state, tick):
        """What guards, controllers and until predicates observe at this step."""
        raise NotImplementedError

    def _outside(self, region, state):
        """Whether the state has left an execution's scope."""
        raise NotImplementedError

    def _sample(self, state):
        """What a trace keeps of a step (every fifth)."""
        return {"step": state["step"]}


_SESSION_CLASSES = {}


def _import(spec: str):
    module, _, attribute = spec.partition(":")
    return getattr(importlib.import_module(module), attribute)


def session_class(embodiment: str = "robot"):
    """The session of an embodiment kind with the mixin of every skill for that kind (the
    skills enabled for a run only decide the namespace; a skill's methods may use
    another's)."""
    if embodiment not in _SESSION_CLASSES:
        from .skills import mixins

        if embodiment not in SESSIONS:
            raise ValueError(f"No session for the embodiment kind {embodiment!r}; sessions exist for "
                             f"{sorted(SESSIONS)}")
        base = _import(SESSIONS[embodiment])
        _SESSION_CLASSES[embodiment] = type("Session", (*mixins(embodiment=embodiment), base),
                                            {"__module__": base.__module__, "embodiment": embodiment})
    return _SESSION_CLASSES[embodiment]


def create_session(options):
    """Kernel bootstrap: the session of options["embodiment"] (default robot)."""
    return session_class(options.get("embodiment", "robot"))(options)
