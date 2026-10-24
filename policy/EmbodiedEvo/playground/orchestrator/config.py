"""Reading ``task.toml``.

Every task uses ``workspace-session-v1``. Baselines start Test directly; other
tasks freeze PlayGround knowledge first. Each Test scene gets a fresh session.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("EMBODIEDEVO_REPO", CODE_ROOT)).resolve()
SESSION_PROTOCOL = "workspace-session-v1"


@dataclass(frozen=True)
class Simulator:
    """Which native task to construct and which interpreter owns it.

    ``backend`` is a construction detail for the host. Regardless of its
    value, callers receive the same Gymnasium environment and submit code
    through ``step(code)``. The implementations have incompatible dependency
    sets and therefore use separate virtualenvs, named by ``interpreter``.
    """

    backend: str
    api_surface: str
    interpreter: str
    env_task: str = ""
    task_config: str = ""  # Reserved configuration field for serialized task compatibility.
    solve_articulation_contact_last: bool = False  # Explicit RoboDojo stability profile.
    min_position_iterations: int | None = None
    min_velocity_iterations: int | None = None


@dataclass(frozen=True)
class Agent:
    preset: str  # codex | claude
    model: str
    reasoning_effort: str
    # ``None`` means that the Playground agent session has no wall-clock limit.
    timeout_sec: float | None
    isolation: str  # docker
    image: str
    network: str
    cpus: int
    memory_mb: int


@dataclass(frozen=True)
class Playground:
    """Fixed scene, an execution budget, and public success after each call."""

    seed: int
    budget: int
    submission_timeout_sec: int
    feedback_fps: int
    expose_success_feedback: bool


@dataclass(frozen=True)
class Test:
    """Test seeds are sampled and saved by the host before Test starts."""

    scenes: int
    submission_timeout_sec: int
    # ``None`` means that a Test agent scene also has no wall-clock limit.
    generation_timeout_sec: float | None = None
    budget: int = 100
    feedback_fps: int | None = None
    expose_success_feedback: bool | None = None
    shared_plan: str | None = None  # Host-only random scene plan shared by paired groups.
    excluded_seed: int | None = None  # Keep the paired Playground scene out of Baseline too.


@dataclass(frozen=True)
class TaskConfig:
    root: Path
    name: str
    description: str
    simulator: Simulator
    agent: Agent
    playground: Playground | None
    test: Test
    workspace_source: str = "Workspace"
    protocol: str = SESSION_PROTOCOL
    baseline: bool = False

    @property
    def feedback_fps(self) -> int:
        if self.test.feedback_fps is not None:
            return self.test.feedback_fps
        return self.playground.feedback_fps if self.playground is not None else 15

    @property
    def expose_success_feedback(self) -> bool:
        if self.test.expose_success_feedback is not None:
            return self.test.expose_success_feedback
        return self.playground.expose_success_feedback if self.playground is not None else True

    @property
    def workspace(self) -> Path:
        """Canonical benchmark-specific source for runtime workspaces."""
        return self.root / self.workspace_source

    @property
    def api_doc(self) -> Path:
        return self.workspace / "primitives" / "primitives.md"

    @property
    def python(self) -> Path:
        """Interpreter that can import this task's simulator."""
        return REPO / self.simulator.interpreter / "bin" / "python"


def resolve_task(path: Path | str) -> Path:
    """Resolve explicit paths or benchmark names directly under tasks/."""
    given = Path(path)
    candidates = [given]
    if not given.is_absolute():
        candidates += [REPO / given, REPO / "tasks" / given]
    for candidate in candidates:
        if (candidate / "task.toml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"no task.toml at {path!s}; use RoboDojo or an explicit task path"
    )


def load_task(root: Path | str, *, env_task: str | None = None) -> TaskConfig:
    root = resolve_task(root)
    data = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
    protocol = data.get("task", {}).get("protocol")
    if protocol != SESSION_PROTOCOL:
        raise ValueError(f"{root / 'task.toml'}: protocol must be {SESSION_PROTOCOL!r}")

    sim, agent = dict(data["simulator"]), data["agent"]
    baseline = data["task"].get("baseline", False)
    if type(baseline) is not bool:
        raise ValueError("[task].baseline must be a boolean")
    p1, p2 = data.get("playground"), dict(data["test"])
    if baseline and p1 is not None:
        raise ValueError("Baseline tasks start Test directly; remove [playground]")
    if not baseline and p1 is None:
        raise ValueError("Non-baseline tasks require [playground]")
    if "feedback_views" in (p1 or {}) or "feedback_views" in p2:
        raise ValueError(
            "feedback_views is no longer supported; feedback uses all native RGB cameras"
        )
    if "environments" in data:
        raise ValueError(
            "[environments] is no longer supported; select a native task with "
            "[simulator].env_task or --env-task"
        )
    for field in ("scene_seed_base", "seeds_file"):
        if field in p2:
            raise ValueError(
                f"{field} is no longer supported; Test seeds are sampled by the host"
            )
    selected = env_task or sim.get("env_task", "")

    return TaskConfig(
        root=root,
        name=data["task"]["name"],
        description=data["task"].get("description", "").strip(),
        simulator=Simulator(
            backend=sim["backend"],
            api_surface=sim["api_surface"],
            interpreter=sim.get("interpreter", ".venv"),
            env_task=selected,
            task_config=sim.get("task_config", ""),
            solve_articulation_contact_last=bool(sim.get("solve_articulation_contact_last", False)),
            min_position_iterations=sim.get("min_position_iterations"),
            min_velocity_iterations=sim.get("min_velocity_iterations"),
        ),
        agent=Agent(
            preset=agent.get("preset", "codex"),
            model=agent.get("model", ""),
            reasoning_effort=agent.get("reasoning_effort", ""),
            timeout_sec=(
                None
                if agent.get("timeout_sec", 0) in (None, 0)
                else float(agent["timeout_sec"])
            ),
            isolation=agent.get("isolation", "docker"),
            image=agent.get("image", "embodiedevo-agent:xpolicy"),
            network=agent.get("network", "filtered"),
            cpus=int(agent.get("cpus", 4)),
            memory_mb=int(agent.get("memory_mb", 8192)),
        ),
        playground=None if p1 is None else Playground(
            seed=int(p1["seed"]),
            budget=int(p1["budget"]),
            submission_timeout_sec=int(p1.get("submission_timeout_sec", 1200)),
            feedback_fps=int(p1.get("feedback_fps", 15)),
            expose_success_feedback=bool(p1.get("expose_success_feedback", False)),
        ),
        test=Test(
            scenes=int(p2["scenes"]),
            submission_timeout_sec=int(p2.get("submission_timeout_sec", 1200)),
            generation_timeout_sec=(
                None
                if p2.get("generation_timeout_sec", 0) in (None, 0)
                else float(p2["generation_timeout_sec"])
            ),
            budget=int(p2.get("budget", 100)),
            feedback_fps=int(p2["feedback_fps"]) if "feedback_fps" in p2 else None,
            expose_success_feedback=(
                bool(p2["expose_success_feedback"]) if "expose_success_feedback" in p2 else None
            ),
            shared_plan=p2.get("shared_plan"),
            excluded_seed=p2.get("excluded_seed"),
        ),
        workspace_source=str(data.get("workspace", {}).get("source", "Workspace")),
        protocol=SESSION_PROTOCOL,
        baseline=baseline,
    )


def config_dict(cfg: TaskConfig) -> dict:
    """Private resolved trial configuration; never mounted into the workspace."""
    data = asdict(cfg)
    data["root"] = str(cfg.root)
    if not data["baseline"]:
        del data["baseline"]  # Keep existing pinned runs compatible.
    for key in ("feedback_fps", "expose_success_feedback", "shared_plan", "excluded_seed"):
        if data["test"][key] is None:
            del data["test"][key]
    if not data["simulator"]["task_config"]:
        del data["simulator"]["task_config"]  # Keep existing pinned runs compatible.
    if not data["simulator"]["solve_articulation_contact_last"]:
        del data["simulator"]["solve_articulation_contact_last"]
    for key in ("min_position_iterations", "min_velocity_iterations"):
        if data["simulator"][key] is None:
            del data["simulator"][key]
    return json.loads(json.dumps(data))


def config_from_dict(data: dict) -> TaskConfig:
    data = dict(data)
    data["root"] = Path(data["root"])
    data["simulator"] = Simulator(**data["simulator"])
    data["agent"] = Agent(**data["agent"])
    if data["playground"] is not None:
        data["playground"] = Playground(**data["playground"])
    data["test"] = Test(**data["test"])
    return TaskConfig(**data)
