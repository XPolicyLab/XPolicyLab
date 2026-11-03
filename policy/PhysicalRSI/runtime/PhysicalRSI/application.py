"""Composition root shared by CLI commands and Python consumers."""

import math
import uuid
from dataclasses import asdict
from pathlib import Path

from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from PhysicalRSI.Embodied_Harness.skills.composition import identity, sequence
from PhysicalRSI.Embodied_Harness.tools.registry import Registry
from PhysicalRSI_core.contracts import Context, Contract, Operation
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.executor import Executor
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json


class Application:
    """A small local workbench; real integrations register their own Operations."""

    def __init__(self, workspace, *, workers=2):
        self.workspace = Path(workspace).resolve()
        self.registry = Registry()
        self.memory = MemoryStore(self.workspace / "memory")
        self.executor = Executor(workers)
        self.execution = Execution(self.workspace / "steps")
        self.recipe = None
        self.conversation = None
        self.task_binding = (
            read_json(self.workspace / "task.json")
            if (self.workspace / "task.json").exists()
            else None
        )
        self.task_session = None
        from .rsi import make_plan, verify_plan

        plan_path = self.workspace / "rsi-plan.json"
        if plan_path.exists():
            self._rsi_plan = verify_plan(read_json(plan_path))
        else:
            self._rsi_plan = make_plan()

    def load_task(self, path):
        from .tasks import load_spec

        binding = load_spec(path)
        return self.bind_task(binding)

    def bind_task(self, binding):
        from .tasks import TaskSession

        session = TaskSession(
            binding,
            self.workspace / "tasks" / binding["specification"]["name"],
            language_model=self.language_model,
            rsi_plan=self._rsi_plan,
            harness=self._rsi_plan.get("harness"),
        )
        atomic_json(self.workspace / "task.json", binding)
        self.task_binding, self.task_session = binding, session
        self.recipe = None
        return session.status()

    def task_draft(self, action="show", value=None):
        from copy import deepcopy

        from .tasks import load_spec, merge_configuration, configuration_schema

        target = self.workspace / "task-draft.json"
        if action == "start":
            binding = load_spec(value)
        else:
            if not target.exists():
                raise ValueError("Start a draft with /draft start path.json")
            binding = read_json(target)
        if action == "set":
            binding = deepcopy(binding)
            spec = binding["specification"]
            spec["configuration"] = merge_configuration(spec["configuration"], value)
            spec_schema = configuration_schema(binding)
            if spec_schema is not None:
                from jsonschema import validate, ValidationError

                try:
                    validate(spec["configuration"], spec_schema)
                except ValidationError as error:
                    raise ValueError(
                        "Invalid task configuration: " + error.message
                    ) from None
            binding["specification_sha256"] = digest(spec)
        elif action == "apply":
            return self.bind_task(binding)
        elif action not in {"start", "show"}:
            raise ValueError("Draft action must be start, show, set or apply")
        schema = configuration_schema(binding)
        if action in {"start", "set"}:
            atomic_json(target, binding)
        return dict(
            specification=binding["specification"],
            configuration_schema=schema,
            source_directory=binding["source_directory"],
            scope="configuration draft; no execution or qualification",
            next="/draft apply then /check; applying does not execute RSI",
        )

    def active_task(self):
        from .tasks import TaskSession

        if self.task_binding is None:
            raise ValueError("Load a task package with /task path.json first")
        if self.task_session is None:
            self.task_session = TaskSession(
                self.task_binding,
                self.workspace / "tasks" / self.task_binding["specification"]["name"],
                language_model=self.language_model,
                rsi_plan=self._rsi_plan,
                harness=self._rsi_plan.get("harness"),
            )
        return self.task_session

    def rsi_plan(self, action="show", value=None):
        """Show or persist the selected RSI scheme and data-only harness."""
        from .rsi import make_plan

        if action == "show":
            return dict(self._rsi_plan)
        if action != "set":
            raise ValueError("RSI plan action must be show or set")
        plan = make_plan(value)
        atomic_json(self.workspace / "rsi-plan.json", plan)
        self._rsi_plan = plan
        # A plan is part of adapter construction and must not mutate an
        # already-created backend in place.
        self.task_session = None
        return dict(plan)

    def harness(self, action="show", value=None):
        """Design/import a harness through the same persisted RSI plan."""
        from .rsi import import_harness, make_plan, design_harness

        if action == "show":
            return self._rsi_plan.get("harness")
        if action == "design":
            harness = design_harness(value)
        elif action == "import":
            harness = import_harness(value)
        else:
            raise ValueError("Harness action must be show, design or import")
        plan = make_plan(self._rsi_plan, harness=harness)
        atomic_json(self.workspace / "rsi-plan.json", plan)
        self._rsi_plan = plan
        self.task_session = None
        return harness

    def check_task(self):
        return self.active_task().backend.check()

    def evolve(self):
        if self.task_binding is not None:
            return self.active_task().backend.evolve()
        return self.evolve_example()

    def run_baseline(self, value=None):
        """Check and run the active task in one explicit, auditable action."""
        if self.task_binding is None:
            raise ValueError("Load a task package with /task path.json first")
        check = self.check_task()
        if not isinstance(check, dict) or check.get("ready") is not True:
            raise ValueError("Baseline preflight failed: " + str(check))
        result = self.active_task().backend.run(value)
        return {
            "schema": "physicalrsi.baseline-run/v1",
            "baseline": self.task_binding["specification"]["name"],
            "rsi_plan": dict(self._rsi_plan),
            "check": check,
            "result": result,
            "qualification": (
                result.get("qualification") if isinstance(result, dict) else None
            ),
            "scope": "task backend result; physical qualification remains adapter-defined",
        }

    def language_model(self):
        from PhysicalRSI_core.infra.language import LanguageModel, ModelConfig

        path = self.workspace / "model.json"
        if not path.exists():
            raise ValueError(
                "Configure conversation with /model config.json; /help lists offline commands"
            )
        return LanguageModel(ModelConfig(**read_json(path)))

    def command(self, line):
        from .commands import dispatch

        return dispatch(self, line)

    def configure_model(self, path=None):
        from PhysicalRSI_core.infra.language import ModelConfig

        target = self.workspace / "model.json"
        if path is not None:
            config = ModelConfig(**read_json(Path(path).expanduser()))
            atomic_json(target, config.public())
            self.conversation = None
        if not target.exists():
            return {"model": None, "setup": "/model /path/to/config.json"}
        return ModelConfig(**read_json(target)).public()

    def chat(self, text):
        from .conversation import Conversation

        if text.strip().casefold() == "hello world" and not (
            self.workspace / "model.json"
        ).exists():
            return {
                "message": "Hello world!",
                "model": None,
                "configuration": {
                    "command": "/model configs/model.example.json",
                    "api_key_env": "PHYSICALRSI_API_KEY",
                    "note": "Set the named environment variable before using model-backed chat.",
                },
                "next": "Use /help for offline commands, or configure a model with /model.",
            }
        if self.conversation is None:
            self.conversation = Conversation(self, self.language_model())
        return self.conversation.ask(text)

    def demo(self) -> dict:
        """Install explicit software examples, without implicit robot connections."""
        (self.workspace / "task.json").unlink(missing_ok=True)
        self.task_binding = self.task_session = None
        degrees = Contract("joint-path", "degree", "joint", "example-2dof")
        radians = Contract("joint-path", "radian", "joint", "example-2dof")
        snapshot = self.memory.snapshot(
            {"approach": [[0.0, 30.0], [20.0, 45.0], [30.0, 60.0]]}
        )
        revision = file_digest(Path(__file__))
        operations = [
            snapshot.reader("approach", degrees),
            Operation(
                "tool.to_radians",
                revision,
                degrees,
                radians,
                lambda path, _: [[math.radians(v) for v in row] for row in path],
            ),
            Operation(
                "skill.reverse",
                revision,
                radians,
                radians,
                lambda path, _: list(reversed(path)),
            ),
        ]
        for operation in operations:
            if operation.name not in {op.name for op in self.registry.list()}:
                self.registry.register(operation)
        self.compose("return_path", [op.name for op in operations])
        return {"scope": "local software example", "recipe": self.describe()}

    def compose(self, name: str, steps: list[str]) -> dict:
        operations = {op.name: op for op in self.registry.list()}
        try:
            selected = [operations[step] for step in steps]
        except KeyError as error:
            raise ValueError(f"Unknown capability: {error.args[0]}") from error
        composed = sequence(name, *selected)
        (self.workspace / "task.json").unlink(missing_ok=True)
        self.task_binding = self.task_session = None
        self.recipe = (composed, tuple(selected))
        return self.describe()

    def describe(self) -> dict:
        if self.recipe is None:
            return {"recipe": None}
        composed, steps = self.recipe
        return {
            "name": composed.name,
            "revision": composed.revision,
            "steps": [identity(step) for step in steps],
            "input": asdict(composed.input),
            "output": asdict(composed.output),
        }

    def skill_memory_demo(self) -> dict:
        """Persist and show the core skill/task exploration-memory catalog."""
        from PhysicalRSI_demos.skill_memory import build_catalog

        return build_catalog(self.workspace / "skill-memory")

    def save(self, path) -> dict:
        if self.recipe is None:
            raise ValueError("Compose a recipe first")
        path = Path(path).expanduser().resolve()
        atomic_json(path, self.describe())
        return {"saved": str(path)}

    def load(self, path) -> dict:
        recipe = read_json(Path(path).expanduser())
        steps = [
            self.registry.resolve(step["name"], step["revision"])
            for step in recipe["steps"]
        ]
        composed = sequence(recipe["name"], *steps)
        expected = dict(
            name=composed.name,
            revision=composed.revision,
            steps=[identity(step) for step in steps],
            input=asdict(composed.input),
            output=asdict(composed.output),
        )
        if expected != recipe:
            raise ValueError("Recipe contracts or versions changed")
        (self.workspace / "task.json").unlink(missing_ok=True)
        self.task_binding = self.task_session = None
        self.recipe = (composed, tuple(steps))
        return self.describe()

    def run(self, value=None) -> dict:
        if self.task_binding is not None:
            return self.active_task().backend.run(value)
        if self.recipe is None:
            raise ValueError("Compose a recipe first; /demo loads the local example")
        operation = self.recipe[0]
        run_id = uuid.uuid4().hex
        path = self.workspace / "runs" / (run_id + ".json")
        record = dict(
            id=run_id,
            recipe=self.describe(),
            input_sha256=digest(value),
            state="started",
        )
        atomic_json(path, record)
        try:
            context = Context(
                run_id, execution=self.execution, harness_revision=operation.revision
            )
            result = self.executor.map(
                lambda _: operation(value, context), [None], context=context
            )[0]
        except BaseException as error:
            atomic_json(
                path,
                dict(
                    record,
                    state="failed",
                    error=type(error).__name__ + ": " + str(error),
                ),
            )
            raise
        receipt = dict(record, state="completed", output=result, qualification=None)
        atomic_json(path, receipt)
        return receipt

    def status(self) -> dict:
        return {
            "task": self.active_task().status()
            if self.task_binding is not None
            else None,
            "workspace": str(self.workspace),
            "recipe": self.describe(),
            "rsi_plan": dict(self._rsi_plan),
            "resources": self.executor.pool.status(),
            "baselines": "RoboDojo software baseline implemented; native reproduction not qualified",
        }

    def evaluation_status(self) -> dict:
        """Read separately recorded evaluation runs without merging candidates."""
        root = Path.cwd() / ".release-staging" / "archive-dsw-eval"
        comparison = root / "LIVE_COMPARISON.json"
        try:
            snapshot = read_json(comparison)
            if not isinstance(snapshot.get("runs"), dict):
                raise ValueError("Evaluation snapshot has no separate run records")
        except (OSError, ValueError, AttributeError) as error:
            return {
                "status": "unavailable",
                "source": str(comparison),
                "error": str(error),
                "scope": "local diagnostic status; not a leaderboard score",
            }
        runs = {}
        for name, record in snapshot["runs"].items():
            rows = record.get("results", [])
            runs[name] = {
                "completed": sum(row.get("status") == "complete" for row in rows),
                "unresolved": [row["task"] for row in rows
                               if row.get("status") != "complete"],
                "results": rows,
                "score_review": record.get("score_review", []),
            }
        return {
            "source": str(comparison),
            "runs": runs,
            "reference": snapshot.get("reference"),
            "scope": "Separate candidate diagnostics; unresolved records do not establish live workers or zero scores.",
        }

    def tools(self):
        from PhysicalRSI.Embodied_Harness.tools.gateway import ToolBinding, ToolGateway

        operations = list(self.registry.list())
        if self.recipe is not None:
            operations.append(self.recipe[0])
        return ToolGateway(
            [ToolBinding(operation, operation.name) for operation in operations],
            execution=self.execution,
        )

    def evolve_example(self) -> dict:
        from PhysicalRSI_demos.runtime_evolution import run

        return run(self.workspace / "self_harness_example")
