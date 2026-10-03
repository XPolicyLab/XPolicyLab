"""One RoboDojo RSI loop over fixed VLAs, skill_selection memory and code-policy skills.

Runtime adapters own native simulation, perception/planning and policy isolation.
This module never imports or executes generated Python in the harness process.
"""

import ast
import importlib
import shutil
import uuid
from pathlib import Path

from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from PhysicalRSI.Embodied_Harness.skills.skill_selection import skill_selection_skill
from PhysicalRSI_core.contracts import Context, Contract, Operation
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    locked,
    read_json,
)
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, now, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness

from . import evidence, skill_proposal
from .layouts import TASKS
from .skill_selection import VLA_EXPERTS, assess, propose_skill_choice
from PhysicalRSI.rsi import canonical_scheme


def manifest(root, name, **metadata):
    root = Path(root).resolve()
    components = {
        kind: {kind + ".json": file_digest(root / (kind + ".json"))} for kind in CLOSURE
    }
    for skill in read_json(root / "skills.json").values():
        components["skills"][skill["source"]] = file_digest(root / skill["source"])
    return dict(id=name, root=str(root), components=components, **metadata)


class Selector:
    def identity(self):
        return {"implementation": file_digest(Path(selection.__file__))}

    select = staticmethod(selection.select_survivor)


class RoboDojoTask:
    protocol_version = 1

    def __init__(self, configuration, workspace, runtime, code_proposer, *, rsi_plan=None, harness=None):
        self.config = configuration
        self.root = Path(workspace).resolve()
        self.runtime, self.code_proposer = runtime, code_proposer
        self.rsi_plan = dict(rsi_plan or {"scheme": "hybrid"})
        self.rsi_scheme = canonical_scheme(self.rsi_plan.get("scheme", "hybrid"))
        self.harness = harness
        self.state = HarnessState(self.root / "state")
        self.tasks = configuration["tasks"]
        if (
            not self.tasks
            or len(set(self.tasks)) != len(self.tasks)
            or not set(self.tasks) <= set(TASKS)
        ):
            raise ValueError("Unique task names required")
        if set(configuration["models"]) != set(VLA_EXPERTS) or any(
            not v for v in configuration["models"].values()
        ):
            raise ValueError("Pin both trained VLA revisions")
        for name in ("development_episodes", "validation_episodes"):
            if type(configuration[name]) is not int or configuration[name] < 1:
                raise ValueError("Positive explicit episode budgets required")
        if not 0 <= configuration["minimum_vla_score"] <= 1:
            raise ValueError("VLA threshold must lie in [0, 1]")
        self.batch_size = configuration.get("rollout_batch_size", 1)
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 10:
            raise ValueError("Rollout batch size must be one to ten")
        self.memory = MemoryStore(self.root / "memory")

    def identity(self):
        project = Path(__file__).resolve().parents[2]
        paths = [
            *Path(__file__).parent.rglob("*.py"),
            *project.joinpath("PhysicalRSI/Embodied_Harness").rglob("*.py"),
            *project.joinpath("PhysicalRSI_core").rglob("*.py"),
        ]
        return dict(
            implementation={
                str(path.relative_to(project)): file_digest(path) for path in paths
            },
            evidence=file_digest(Path(evidence.__file__)),
            runtime=self.runtime.identity(),
            proposal=self.code_proposer.identity(),
            configuration=digest(self.config),
            rsi_plan=self.rsi_plan,
            harness=self.harness,
        )

    def check(self):
        return self.runtime.check()

    def _ready(self):
        report = self.check()
        if report.get("ready") is not True:
            raise ValueError("Runtime is not ready: " + str(report))

    def _seed(self):
        root = self.root / "seed"
        if not root.exists():
            root.mkdir(parents=True)
            skill_choices = {
                task: dict(name="pi05", revision=self.config["models"]["pi05"])
                for task in self.tasks
            }
            values = dict(
                foundation=self.config["models"],
                skill_selection={"skill_choices": skill_choices},
                skills={},
                tools=self.runtime.skill_api(),
                control={"entry": "skill.skill_selection"},
                prompts={"proposal": self.code_proposer.identity()},
                memory_rules={},
                dependencies=self.identity(),
                weights=self.config["models"],
                assets=self.runtime.identity(),
                configuration=self.config,
                rsi_plan=self.rsi_plan,
                harness=self.harness,
            )
            for kind, value in values.items():
                atomic_json(root / (kind + ".json"), value)
        return manifest(root, "initial")

    def _loop(self, directory):
        profile = dict(
            tasks={
                task: dict(
                    weight=1,
                    episodes=self.config["validation_episodes"],
                    score_range=[0, 1],
                    maximum_regression=self.config.get("maximum_regression", 0),
                )
                for task in self.tasks
            },
            minimum_gain=self.config.get("minimum_gain", 0),
            tie_tolerance=1e-10,
        )
        loop = SelfHarness(
            directory,
            state=self.state,
            proposer=Proposal(self),
            evaluator=Evaluation(self),
            selector=Selector(),
            profile=profile,
            protocol=dict(
                identity=digest(self.identity()),
                evaluation_kind="robodojo_episode_evaluation",
            ),
            scope=self.runtime.identity()["scope"],
            max_candidates=1,
        )
        self.state.initialize(
            self._seed(), policy=loop.config, scope=loop.config["scope"]
        )
        return loop

    def evolve(self):
        self._ready()
        with locked(self.root / ".iteration.lock"):
            rounds = sorted((self.root / "rounds").glob("round-*"))
            if rounds and not any(
                (rounds[-1] / name).exists()
                for name in ("commit.json", "retained.json")
            ):
                directory = rounds[-1]
            else:
                directory = self.root / "rounds" / f"round-{len(rounds):06d}"
            result = self._loop(directory).run()
            return dict(
                revision=result["revision"],
                harness=result["harness"]["id"],
                action=result["action"],
                round=str(directory),
                qualification=None,
                scope=self.runtime.identity()["scope"],
            )

    def status(self):
        if not (self.state.root / "current.json").exists():
            return dict(state="not_initialized", tasks=self.tasks, qualification=None)
        current = self.state.resolve()
        return dict(
            state=current["action"],
            revision=current["revision"],
            skill_selection=read_json(Path(current["harness"]["root"]) / "skill_selection.json"),
            rsi_scheme=self.rsi_scheme,
            custom_harness=self.harness,
            qualification=None,
        )

    def execute(self, harness, case, cohort_root, output, *, expert=None):
        return self.execute_batch(harness, [case], cohort_root, output, expert=expert)[
            0
        ]

    def execute_batch(self, harness, cases, cohort_root, output, *, expert=None):
        if (
            not 1 <= len(cases) <= 10
            or len({(case["task"], case["split"]) for case in cases}) != 1
            or len({case["layout_sha256"] for case in cases}) != len(cases)
        ):
            raise ValueError(
                "One task/split and one to ten distinct cases per batch required"
            )
        if expert is not None and cases[0]["split"] != "development":
            raise ValueError("Expert overrides are only allowed during development")
        root = Path(harness["root"])
        frozen = verify_harness(harness)
        for case in cases:
            evidence.verify_case(case, cohort_root)
        skill_choices = read_json(root / "skill_selection.json")
        if expert is not None:
            skill_choices = {"skill_choices": {cases[0]["task"]: expert}}
        snapshot = self.memory.snapshot(skill_choices)
        bindings = {}
        for spec in skill_choices["skill_choices"].values():

            def call(request, context, spec=spec):
                output.mkdir(parents=True, exist_ok=False)
                common = dict(candidate=harness, expert=spec, layout_root=cohort_root)
                batch = getattr(self.runtime, "rollouts", None)
                if callable(batch):
                    receipts = batch(cases=request["cases"], output=output, **common)
                    directories = [output] * len(cases)
                else:
                    # Existing single-episode adapters retain their contract.
                    receipts, directories = [], []
                    for index, case in enumerate(request["cases"]):
                        directory = output if len(cases) == 1 else output / str(index)
                        directory.mkdir(parents=True, exist_ok=True)
                        receipts.append(
                            self.runtime.rollout(case=case, output=directory, **common)
                        )
                        directories.append(directory)
                if not isinstance(receipts, (list, tuple)) or len(receipts) != len(
                    cases
                ):
                    raise ValueError(
                        "Batch receipt coverage differs from requested cases"
                    )
                return [
                    evidence.episode_receipt(
                        receipt,
                        directory=directory,
                        task=case["task"],
                        split=case["split"],
                        layout_sha256=case["layout_sha256"],
                        expert=spec,
                        harness_sha256=frozen,
                    )
                    for receipt, directory, case in zip(receipts, directories, cases)
                ]

            bindings[spec["name"]] = Operation(
                spec["name"],
                spec["revision"],
                Contract("robodojo-episode-batch/v1"),
                Contract("robodojo-result-batch/v1"),
                call,
                effects=frozenset({"robodojo-runtime"}),
            )
        skill_selector = skill_selection_skill("skill.skill_selection", snapshot, bindings)
        result = skill_selector(
            dict(task=cases[0]["task"], cases=cases),
            Context(
                uuid.uuid4().hex,
                execution=Execution(output.parent / "steps"),
                harness_revision=frozen,
            ),
        )
        if verify_harness(harness) != frozen:
            raise ValueError("Harness changed during rollout")
        for case in cases:
            evidence.verify_case(case, cohort_root)
        return result

    def execute_cases(self, harness, cases, cohort_root, output, *, expert=None):
        """Group homogeneous batches while returning results in input order."""
        groups = {}
        for index, case in enumerate(cases):
            groups.setdefault((case["task"], case["split"]), []).append((index, case))
        results = [None] * len(cases)
        batch_index = 0
        for group in groups.values():
            for start in range(0, len(group), self.batch_size):
                chunk = group[start : start + self.batch_size]
                outcomes = self.execute_batch(
                    harness,
                    [case for _, case in chunk],
                    cohort_root,
                    output / f"batch-{batch_index:06d}",
                    expert=expert,
                )
                for (index, _), outcome in zip(chunk, outcomes):
                    results[index] = outcome
                batch_index += 1
        return results

    def run(self, value=None):
        self._ready()
        with locked(self.root / ".iteration.lock"):
            if not (self.state.root / "current.json").exists():
                self._loop(self.root / "rounds/round-000000")
        current = self.state.resolve()  # Never execute a pending proposal.
        task = (value or {}).get("task")
        if task is None and len(self.tasks) == 1:
            task = self.tasks[0]
        if task not in self.tasks:
            raise ValueError("Specify one configured task in /run JSON")
        output = self.root / "runs" / uuid.uuid4().hex
        rows = evidence.cases(
            self.runtime,
            tasks=[task],
            count=1,
            split="development",
            output=output / "layouts",
        )
        return self.execute(
            current["harness"], rows[0], output / "layouts", output / "episode"
        )


class Proposal:
    def __init__(self, task):
        self.task = task

    def identity(self):
        return self.task.identity()

    def develop(self, parent, output):
        task = self.task
        rows = evidence.cases(
            task.runtime,
            tasks=task.tasks,
            count=task.config["development_episodes"],
            split="development",
            output=output / "development-layouts",
        )
        atomic_json(output / "development-cases.json", rows)
        results = []
        for expert in VLA_EXPERTS:
            results.extend(
                task.execute_cases(
                    parent,
                    rows,
                    output / "development-layouts",
                    output / "development" / expert,
                    expert=dict(name=expert, revision=task.config["models"][expert]),
                )
            )
        skill_choices = read_json(Path(parent["root"]) / "skill_selection.json")["skill_choices"]
        incumbent_cases = [
            case for case in rows if skill_choices[case["task"]]["name"] not in VLA_EXPERTS
        ]
        results.extend(
            task.execute_cases(
                parent,
                incumbent_cases,
                output / "development-layouts",
                output / "development/incumbent",
            )
        )
        atomic_json(output / "development-results.json", results)
        return dict(
            split="evolve",
            evidence={
                "development-results.json": file_digest(
                    output / "development-results.json"
                )
            },
            costs={"episodes": len(results)},
        )

    def propose(self, parent, feedback, output):
        task = self.task
        results = read_json(output / "development-results.json")
        if (
            file_digest(output / "development-results.json")
            != feedback["evidence"]["development-results.json"]
        ):
            raise ValueError("Development evidence changed before proposing")
        original = Path(parent["root"])
        for result in results:
            for name, sha in result["evidence"].items():
                artifact = Path(name).resolve()
                if not artifact.is_relative_to(output) or file_digest(artifact) != sha:
                    raise ValueError(
                        "Development execution evidence changed before proposing"
                    )
        skill_choices = read_json(original / "skill_selection.json")
        skills = read_json(original / "skills.json")
        memories = read_json(original / "memory_rules.json")
        child = output / "candidate"
        shutil.copytree(original, child)
        changes = {}
        proposal_evidence = {}
        for name in task.tasks:
            assessments = [
                assess(
                    name,
                    expert,
                    task.config["models"][expert],
                    [
                        row
                        for row in results
                        if row["task"] == name and row["expert"] == expert
                    ],
                    minimum_episodes=task.config["development_episodes"],
                )
                for expert in VLA_EXPERTS
            ]
            skill_choice = propose_skill_choice(
                name, assessments, minimum_score=task.config["minimum_vla_score"]
            )
            # The application-level plan changes the search policy while
            # retaining the same paired evidence and frozen Self-Harness.
            # learned_skill deliberately keeps the best learned skill_choice even
            # below threshold; code_policy always asks for a code proposal.
            if task.rsi_scheme == "learned_skill" and skill_choice["state"] == "needs_code_policy_development":
                selected = max(
                    assessments,
                    key=lambda item: (item["mean_score"], -VLA_EXPERTS.index(item["name"])),
                )
                skill_choice = dict(
                    task=name,
                    state="skill_choice_candidate",
                    name=selected["name"],
                    revision=selected["revision"],
                    evidence=assessments,
                    reason="learned_skill_plan_selected_best_paired_vla",
                )
            if task.rsi_scheme == "code_policy":
                skill_choice = dict(skill_choice, state="needs_code_policy_development")
            if skill_choice["state"] == "needs_code_policy_development":
                skill_name = "code." + name
                old_skill = skills.get(skill_name)
                source = (
                    (original / old_skill["source"]).read_text() if old_skill else ""
                )
                directory = output / "proposals" / name
                directory.mkdir(parents=True)
                proposal = skill_proposal.propose(
                    task=name,
                    source=source,
                    memory=memories.get(name, {}),
                    feedback=[row for row in results if row["task"] == name],
                    skill_api=task.runtime.skill_api(),
                    proposer=task.code_proposer,
                    output=directory,
                )
                skill_proposal.verify(proposal, task=name)
                ast.parse(proposal["source"])
                for artifact_name in ("skill-request.json", "skill-proposal.json"):
                    artifact = directory / artifact_name
                    proposal_evidence[str(artifact.relative_to(output))] = file_digest(
                        artifact
                    )
                # Keep one read-only compatibility receipt for workspaces made
                # before the public terminology was cleaned up.
                for new_name, old_name in (
                    ("skill-request.json", "program-request.json"),
                    ("skill-proposal.json", "program-proposal.json"),
                ):
                    artifact = directory / new_name
                    legacy = directory / old_name
                    legacy.write_bytes(artifact.read_bytes())
                    proposal_evidence[str(legacy.relative_to(output))] = file_digest(
                        legacy
                    )
                code = Path("code") / (name + ".py")
                (child / code).parent.mkdir(exist_ok=True)
                (child / code).write_text(proposal["source"])
                revision = file_digest(child / code)
                skills[skill_name] = dict(source=str(code), revision=revision)
                memories[name] = proposal["memory"]
                skill_choice = propose_skill_choice(
                    name,
                    assessments,
                    minimum_score=task.config["minimum_vla_score"],
                    code_policy=dict(name=skill_name, revision=revision),
                )
            selected = {key: skill_choice[key] for key in ("name", "revision")}
            if skill_choices["skill_choices"][name] != selected:
                changes[name] = dict(before=skill_choices["skill_choices"][name], after=selected)
            skill_choices["skill_choices"][name] = selected
        atomic_json(child / "skill_selection.json", skill_choices)
        atomic_json(child / "skills.json", skills)
        memory = task.memory.snapshot(memories)
        atomic_json(child / "memory_rules.json", memory.read())
        candidate = manifest(
            child,
            "candidate_" + output.name,
            parent_sha256=verify_harness(parent),
            method="paired_vla_skill_selection_and_development_code_memory",
            changes=changes or {"code_policy": "code or memory refinement"},
            environment=task.runtime.identity(),
            evidence=dict(feedback["evidence"], **proposal_evidence),
            costs=feedback["costs"],
        )
        if verify_harness(candidate) == verify_harness(parent):
            return []
        return [candidate]


class Evaluation:
    def __init__(self, task):
        self.task = task

    def identity(self):
        return self.task.identity()

    def admit(self, candidate, output):
        frozen = verify_harness(candidate)
        root = Path(candidate["root"])
        skills = read_json(root / "skills.json")
        for skill in skills.values():
            if file_digest(root / skill["source"]) != skill["revision"]:
                raise ValueError("Code skill revision differs from frozen source")
        report = self.task.runtime.admit(
            candidate, output / ("runtime-admission-" + candidate["id"])
        )
        path = output / ("admission-evidence-" + candidate["id"] + ".json")
        atomic_json(path, report)
        return dict(
            accepted=report["accepted"],
            freeze_sha256=frozen,
            evidence={path.name: file_digest(path)},
            reason=report.get("reason", "runtime admission"),
        )

    def validation(self, comparison, output):
        development = read_json(output / "development-cases.json")
        rows = evidence.cases(
            self.task.runtime,
            tasks=self.task.tasks,
            count=self.task.config["validation_episodes"],
            split="validation",
            output=output / "validation-layouts",
            comparison=comparison,
            excluded=[row["layout_sha256"] for row in development],
        )
        atomic_json(output / "validation-cases.json", rows)
        return dict(
            split="validation",
            comparison_sha256=digest(comparison),
            generated_at=now(),
            layouts={
                name: [row["layout_sha256"] for row in rows if row["task"] == name]
                for name in self.task.tasks
            },
            admission_evidence={
                "validation-cases.json": file_digest(output / "validation-cases.json")
            },
        )

    def evaluate(self, candidate, comparison, cohort, output):
        if (
            file_digest(output / "validation-cases.json")
            != cohort["admission_evidence"]["validation-cases.json"]
        ):
            raise ValueError("Frozen validation cases changed")
        rows = []
        cases = read_json(output / "validation-cases.json")
        results = self.task.execute_cases(
            candidate,
            cases,
            output / "validation-layouts",
            output / "evaluation" / candidate["id"],
        )
        for case, result in zip(cases, results):
            retained = dict(result["evidence"])
            for development in read_json(output / "development-results.json"):
                retained.update(development["evidence"])
            for split in ("development", "validation"):
                for layout in read_json(output / (split + "-cases.json")):
                    cohort_root = output / (split + "-layouts")
                    artifact = cohort_root / layout["file"]
                    retained[str(artifact)] = file_digest(artifact)
                    retained.update(
                        {
                            str(cohort_root / name): sha
                            for name, sha in layout["validation_artifacts"].items()
                        }
                    )
            rows.append(
                dict(
                    task=case["task"],
                    layout_sha256=case["layout_sha256"],
                    state="completed",
                    score=result["episode_score"],
                    success=result["success"],
                    evidence_sha256={
                        str(Path(path).relative_to(output)): sha
                        for path, sha in retained.items()
                    },
                )
            )
        return dict(
            candidate_id=candidate["id"],
            kind="robodojo_episode_evaluation",
            native_exit_code=0,
            comparison_sha256=digest(comparison),
            cohort_sha256=digest(cohort),
            freeze_sha256=verify_harness(candidate),
            evaluator_revision=digest(self.task.identity()),
            episodes=rows,
        )


def create(*, configuration, workspace, source_directory, language_model, rsi_plan=None, harness=None):
    from .proposal import CodeProposer, OfflineProposer

    module, name = configuration["runtime_factory"].split(":")
    runtime = getattr(importlib.import_module(module), name)(
        configuration=configuration["runtime"], source_directory=source_directory
    )
    mode = configuration.get("proposal_mode", "api")
    if mode not in {"api", "offline"}:
        raise ValueError("Unknown proposal mode: " + str(mode))
    proposer = (
        OfflineProposer()
        if mode == "offline"
        else CodeProposer(model_factory=language_model)
    )
    return RoboDojoTask(
        configuration,
        workspace,
        runtime,
        proposer,
        rsi_plan=rsi_plan,
        harness=harness,
    )


create.configuration_schema = {
    "type": "object",
    "required": [
        "tasks",
        "models",
        "development_episodes",
        "validation_episodes",
        "minimum_vla_score",
        "runtime_factory",
        "runtime",
    ],
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"enum": sorted(TASKS)},
            "description": "Tasks to develop and independently validate",
        },
        "models": {
            "type": "object",
            "required": list(VLA_EXPERTS),
            "additionalProperties": False,
            "properties": {
                name: {"type": "string", "minLength": 1} for name in VLA_EXPERTS
            },
        },
        "development_episodes": {
            "type": "integer",
            "minimum": 1,
            "description": "Development episodes per task",
        },
        "validation_episodes": {
            "type": "integer",
            "minimum": 1,
            "description": "Fresh validation episodes per task",
        },
        "minimum_vla_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Develop code policy when both VLAs fall below this score",
        },
        "rollout_batch_size": {"type": "integer", "minimum": 1, "maximum": 10},
        "runtime_factory": {"type": "string", "minLength": 1},
        "runtime": {"type": "object"},
        "proposal_mode": {
            "enum": ["api", "offline"],
            "description": "Use the configured model or the offline proposal adapter",
        },
    },
}
