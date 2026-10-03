"""Model-written code/memory proposals; validation inputs are never supplied."""

import ast
import json
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, file_digest

from .primitive_api import for_task
from .skill_proposal import validate_policy_source


class CodeProposer:
    def __init__(self, model=None, *, model_factory=None):
        if (model is None) == (model_factory is None):
            raise ValueError("Provide one model or one model factory")
        if model_factory is not None and not callable(model_factory):
            raise ValueError("Model factory must be callable")
        self._model = model
        self._model_factory = model_factory

    @property
    def model(self):
        # Loading/checking a task is offline. Bind once when the experiment
        # first requests identity, before development or comparison starts.
        # Later model configuration changes cannot mutate this frozen proposer.
        if self._model is None:
            self._model = self._model_factory()
        return self._model

    def identity(self):
        return dict(
            model=self.model.config.public(), implementation=file_digest(Path(__file__))
        )

    def propose(self, *, task, source, memory, feedback, skill_api, output):
        if not feedback or any(
            row.get("split") != "development" or row.get("task") != task
            for row in feedback
        ):
            raise ValueError(
                "Code proposal accepts only this task's development feedback"
            )
        request = dict(
            task=task,
            current_code=source,
            memory=memory,
            development_feedback=feedback,
            primitive_api=for_task(skill_api, task),
        )
        atomic_json(output / "request.json", request)
        instruction = (
            "Develop a RoboDojo code policy using only the supplied robot primitive API. "
            "Return one JSON object with source (Python defining policy(robot, memory)), "
            "memory (JSON object of reusable lessons), and rationale (string). "
            "Use development feedback to repair failures. Do not access files, networks, "
            "simulator state, benchmark layouts or evaluation answers. Do not claim success; "
            "the independent environment evaluator decides. The runtime will isolate execution."
        )
        answer, calls, _ = self.model.complete(
            [
                dict(role="system", content=instruction),
                dict(role="user", content=json.dumps(request, ensure_ascii=False)),
            ],
            [],
        )
        atomic_json(output / "response.json", dict(text=answer, calls=calls))
        if calls:
            raise ValueError("Code proposal must be JSON, not a tool call")
        candidate = json.loads(answer)
        if set(candidate) != {"source", "memory", "rationale"} or not isinstance(
            candidate["memory"], dict
        ):
            raise ValueError("Expected source, memory and rationale in code proposal")
        tree = validate_policy_source(candidate["source"])
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "policy"
            for node in tree.body
        ):
            raise ValueError("Code proposal must define policy(robot, memory)")
        # Parsing is a syntax gate, never a security sandbox or qualification.
        return candidate


class OfflineProposer:
    """Deterministic, API-free proposal adapter for offline development.

    This is deliberately a conservative proposal seed, not a claim that a
    software template has solved a physical task.  It writes the same request
    and response artifacts as the model adapter so the surrounding harness can
    audit provenance and retain failed evidence.
    """

    def identity(self):
        return {
            "kind": "offline",
            "mode": "offline_template",
            "implementation": file_digest(Path(__file__)),
        }

    def propose(self, *, task, source, memory, feedback, skill_api, output):
        if not feedback or any(
            row.get("split") != "development" or row.get("task") != task
            for row in feedback
        ):
            raise ValueError(
                "Code proposal accepts only this task's development feedback"
            )
        task_api = for_task(skill_api, task)
        request = dict(
            task=task,
            current_code=source,
            memory=memory,
            development_feedback=feedback,
            primitive_api=task_api,
            adapter=self.identity(),
        )
        atomic_json(output / "request.json", request)
        # Preserve an existing syntactically valid policy across iterations.
        # A failed development episode is evidence for the next proposal, not
        # permission to erase a previously reviewed policy.  The first seed
        # only calls primitives explicitly present in the supplied API.
        try:
            tree = ast.parse(source) if source.strip() else None
            has_policy = tree is not None and any(
                isinstance(node, ast.FunctionDef) and node.name == "policy"
                for node in tree.body
            )
        except SyntaxError:
            has_policy = False
        if not has_policy:
            methods = task_api.get("methods", {})
            if "reobserve" in methods:
                observe = "    return robot.reobserve()\n"
            elif "observe" in methods:
                observe = "    return robot.observe()\n"
            else:
                # A task may expose only action primitives.  The seed must
                # still be executable in that declared API; qualification
                # remains the independent evaluator's responsibility.
                observe = "    return None\n"
            if "scene" in methods:
                observe = "    robot.scene()\n" + observe
            source = "def policy(robot, memory):\n" + observe
        lessons = dict(memory) if isinstance(memory, dict) else {}
        lessons.update(
            {
                "proposal_adapter": "offline",
                "task": task,
                "feedback_episodes": len(feedback),
                "scope": "development feedback only; qualification remains independent",
            }
        )
        candidate = {
            "source": source,
            "memory": lessons,
            "rationale": "Offline proposal seed: observe scene and state before the next reviewed revision.",
        }
        atomic_json(output / "response.json", candidate)
        validate_policy_source(source)
        return candidate
