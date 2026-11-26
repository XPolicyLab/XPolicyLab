"""Read-only, versioned code-skill inputs; never store evaluation feedback here."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ..evaluation.storage import atomic_json, digest


class CodeBundle:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.files = {"manifest": self.path}
        self.content = {"manifest": self.path.read_bytes()}
        document = json.loads(self.content["manifest"])
        self.controller_only = document.get("schema") == "orimapi.code-controller-skill/v1"
        # Both runtime-owned formats retain their configured controller when
        # the router advertises tools; neither needs a placeholder program.
        self.tool_only = self.controller_only or document.get("schema") == "orimapi.code-tool-skill/v1"
        if document.get("schema") not in {"orimapi.code-skill/v1", "orimapi.code-tool-skill/v1", "orimapi.code-controller-skill/v1"}:
            raise ValueError("Unsupported code skill bundle schema")
        self.skill_id = document.get("skill_id")
        self.tasks = document.get("supported_tasks")
        self.task_spec = deepcopy(document.get("task_spec"))
        if not isinstance(self.skill_id, str) or not self.skill_id.isidentifier():
            raise ValueError("Invalid code skill ID")
        if (
            not isinstance(self.tasks, list)
            or not self.tasks
            or any(
                not isinstance(task, str) or not task.isidentifier()
                for task in self.tasks
            )
        ):
            raise ValueError("Declare supported_tasks in the code skill bundle")
        if self.task_spec is not None and (
            not isinstance(self.task_spec, dict)
            or self.task_spec.get("task") not in self.tasks
            or type(self.task_spec.get("budget")) is not int
            or not 1 <= self.task_spec["budget"] <= 10000
            or (not self.tool_only and (not isinstance(self.task_spec.get("roles"), list)
            or not self.task_spec["roles"]
            or any(
                not isinstance(role, str) or not role.isidentifier()
                for role in self.task_spec["roles"]
            )))
            or not isinstance(self.task_spec.get("requirements"), str)
        ):
            raise ValueError("Invalid frozen task_spec in the code skill bundle")
        prompts = document.get("prompts", {})
        if self.tool_only and (prompts or document.get("program") is not None or self.task_spec is None):
            raise ValueError("Tool-only bundles need a task_spec and no placeholder program/prompts")
        if not self.tool_only and set(prompts) != {"plan", "generate"}:
            raise ValueError("Declare plan and generate prompt paths")
        paths = {
            "experience": document.get("experience"),
        }
        if not self.tool_only:
            paths["program"] = document.get("program")
        paths.update({"prompt_" + key: value for key, value in prompts.items()})
        sources = document.get("sources", {})
        if not isinstance(sources, dict) or not sources:
            raise ValueError("Declare the skill's execution sources")
        paths.update({"source_" + key: value for key, value in sources.items()})
        recipes = document.get("tool_recipes", {})
        if not isinstance(recipes, dict) or len(recipes) > 16:
            raise ValueError(
                "Optional tool_recipes must map names to recipe JSON paths"
            )
        if self.tool_only and not self.controller_only and not recipes:
            raise ValueError("Tool-only bundles require frozen tool recipes")
        self.tool_limits = None
        if self.tool_only:
            from .code_tools import settings

            if not isinstance(document.get("tool_limits"), dict):
                raise ValueError("Tool-only bundles require frozen tool_limits")
            self.tool_limits = settings({"tool_limits": document["tool_limits"]})
        paths.update({"tool_recipe_" + key: value for key, value in recipes.items()})
        for key, value in paths.items():
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                raise ValueError("Bundle file paths must be nonempty relative paths")
            resource = (self.path.parent / value).resolve(strict=True)
            if not resource.is_file() or resource.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Bundle inputs must be bounded regular files")
            self.files[key] = resource
            self.content[key] = resource.read_bytes()
        self.program = None if self.tool_only else self.content["program"].decode("utf-8")
        self.tool_recipes = {}
        self.frozen_tools = None
        if "source_shared_tools" in self.files:
            if not self.tool_only or self.controller_only:
                raise ValueError("Frozen shared tools require a tool-only bundle")
            from .frozen_tools import FrozenTools

            self.frozen_tools = FrozenTools(self.files["source_shared_tools"])
            if document.get("shared_tool_sha256") != self.frozen_tools.identity["sha256"]:
                raise ValueError("Skill manifest does not pin this shared tool version")
            for name, path in self.frozen_tools.files.items():
                key = "shared_tool_" + name
                self.files[key] = path
                self.content[key] = path.read_bytes()
        if recipes:
            from .code_tools.recipes import RecipeLibrary

            self.tool_recipes = RecipeLibrary(
                {
                    name: json.loads(self.content["tool_recipe_" + name])
                    for name in recipes
                }
            ).documents
        self.experience = json.loads(self.content["experience"])
        if not isinstance(self.experience, list):
            raise ValueError("Frozen experience must be a JSON list")
        self.prompts = {
            key: self.content["prompt_" + key].decode("utf-8") for key in prompts
        }
        self.hashes = {
            key: hashlib.sha256(value).hexdigest()
            for key, value in self.content.items()
        }
        self.sha256 = digest(self.hashes)

    def verify(self):
        if any(
            path.read_bytes() != self.content[key] for key, path in self.files.items()
        ):
            raise ValueError("Frozen skill bundle changed during evaluation")
        if self.frozen_tools is not None:
            self.frozen_tools.verify()

    def parameters(self):
        self.verify()
        return {
            **(
                {"task_spec": deepcopy(self.task_spec)}
                if self.task_spec is not None
                else {}
            ),
            "skill_id": self.skill_id,
            "bundle_sha256": self.sha256,
            "experience": deepcopy(self.experience),
            "best_program": self.program,
            "prompts": dict(self.prompts),
            **({"tool_limits": deepcopy(self.tool_limits)} if self.tool_limits else {}),
            **(
                {"tool_recipes": deepcopy(self.tool_recipes)}
                if self.tool_recipes
                else {}
            ),
        }

    def snapshot(self, root):
        folder = Path(root) / "code_bundles" / self.sha256
        folder.mkdir(parents=True, exist_ok=True)
        for key, content in self.content.items():
            target = folder / (self.hashes[key] + self.files[key].suffix)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError("Code bundle snapshot changed")
            else:
                with target.open("xb") as stream:
                    stream.write(content)
        atomic_json(
            folder / "index.json",
            {
                "skill_id": self.skill_id,
                "sha256": self.sha256,
                "files": {
                    key: {
                        "original_path": str(path),
                        "sha256": self.hashes[key],
                        "snapshot": self.hashes[key] + path.suffix,
                    }
                    for key, path in self.files.items()
                },
            },
        )
        return str(folder.relative_to(root))


def code_identity(policy):
    if policy.code_skills:
        skills = {
            key: code_identity(row.policy_config(policy))
            for key, row in policy.code_skills.items()
        }
        return {
            "selection_prompt_sha256": hashlib.sha256(
                policy.code_selection_prompt.read_bytes()
            ).hexdigest(),
            "skills": skills,
            "supported_tasks": sorted(
                {task for row in skills.values() for task in row["supported_tasks"]}
            ),
        }
    if policy.code_path is None or not policy.code_path.is_file():
        raise ValueError("Code policy requires an existing code_path")
    source_sha = hashlib.sha256(policy.code_path.read_bytes()).hexdigest()
    if policy.code_sha256 is not None and policy.code_sha256 != source_sha:
        raise ValueError("Code policy source SHA-256 does not match")
    result = {"source_sha256": source_sha}
    from .code_tools import depth_availability, depth_identity, identity, settings

    tools = settings(policy.code_parameters)
    if tools.get("sam_proposals"):
        from .code_tools.segmentation import resource_identity

        result["segmentation_model"] = resource_identity(tools["sam_proposals"])
    if policy.code_bundle is not None:
        result["code_tools"] = identity()
        if tools.get("depth"):
            availability = depth_availability(tools["depth"])
            result["depth_model"] = {
                **availability,
                **(
                    depth_identity(tools["depth"])
                    if availability["available"]
                    else tools["depth"]
                ),
            }
    if policy.code_bundle is not None:
        bundle = CodeBundle(policy.code_bundle)
        from .initial_experience import input_identity

        result["initial_skill_experience"] = input_identity()
        if bundle.tool_limits and bundle.tool_limits.get("sam_proposals"):
            from .code_tools.segmentation import resource_identity

            result["segmentation_model"] = resource_identity(bundle.tool_limits["sam_proposals"])
        if bundle.frozen_tools is not None:
            result["code_tools"] = deepcopy(bundle.frozen_tools.identity)
        result.update(
            bundle_sha256=bundle.sha256,
            files=bundle.hashes,
            skill_id=bundle.skill_id,
            supported_tasks=bundle.tasks,
        )
        geometry = policy.code_parameters.get("geometry", {})
        result["geometry"] = {
            key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
            for key, value in geometry.items()
        }
    return result
