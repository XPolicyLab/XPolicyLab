"""Episode-local, bounded historical experience for a skill's first LLM request.

This is a runtime input adapter, not a controller or a new interaction loop.
Only files already declared in the selected frozen bundle are eligible.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

from ..evaluation.storage import atomic_json

SCHEMA = "orimapi.initial-skill-experience/v1"
CONTEXT_KEY = "initial_skill_experience"
MAX_EXPERIENCE_BYTES = 128_000
_CURRENT = ContextVar("orimapi_initial_skill_experience", default=None)
_RESOURCE_ROLES = {
    "failure_experience.json": "failure_experience",
    "success_experience.json": "success_experience",
    "successful_code_examples.json": "successful_code_examples",
    "frozen_knowledge.json": "frozen_knowledge",
}
SCOPE = (
    "Frozen historical experience for the selected skill, not current scene evidence. "
    "Use relevant lessons in this initial decision. Preserve each record's success, "
    "failure, partial-score and unverified-hypothesis labels; a filename or existing "
    "implementation alone does not establish native success. Do not invent missing "
    "successful examples or reuse historical pixels, poses or scores as current truth. "
    "Follow the live instruction and unchanged tool, physics and budget constraints. "
    "Subsequent interaction remains controlled by this skill."
)


def _json(value):
    # Match APISupervisor's context serialization, including ASCII escaping.
    return json.dumps(value, allow_nan=False)


def input_identity():
    """Record this input change separately from old frozen native qualification."""
    root = Path(__file__).parent
    return {
        "schema": SCHEMA,
        "max_bytes": MAX_EXPERIENCE_BYTES,
        "sources": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("initial_experience.py", "code_llm.py", "code_policy.py")
        },
    }


def bundle_experience(bundle):
    bundle.verify()
    resources = []
    seen = set()
    # The main experience may have a custom filename. Other roles are explicit
    # declared JSON resources, not neighboring files discovered from disk.
    for key, path in bundle.files.items():
        role = "experience" if key == "experience" else _RESOURCE_ROLES.get(path.name)
        if role is None or path in seen:
            continue
        seen.add(path)
        resources.append(
            {
                "role": role,
                "bundle_key": key,
                "source": path.name,
                "sha256": bundle.hashes[key],
                "data": json.loads(bundle.content[key]),
                "truncated": False,
            }
        )
    result = {
        "schema": SCHEMA,
        "skill_id": bundle.skill_id,
        "bundle_sha256": bundle.sha256,
        "scope": SCOPE,
        "resources": resources,
    }
    if len(_json(result).encode()) <= MAX_EXPERIENCE_BYTES:
        return result

    # Keep every resource category and its provenance. Oversized resources get
    # an explicitly labeled excerpt, never a silently incomplete success record.
    allowance = (MAX_EXPERIENCE_BYTES - 4096) // max(1, len(resources))
    for row in resources:
        if len(_json(row).encode()) <= allowance:
            continue
        text = _json(row.pop("data"))
        row.update(
            truncated=True, original_json_bytes=len(text.encode()), json_excerpt=""
        )
        if len(_json(row).encode()) > allowance:
            raise ValueError("Initial skill experience metadata exceeds its budget")
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            row["json_excerpt"] = text[:middle]
            if len(_json(row).encode()) <= allowance:
                low = middle
            else:
                high = middle - 1
        row["json_excerpt"] = text[:low]
    if len(_json(result).encode()) > MAX_EXPERIENCE_BYTES:
        raise ValueError("Initial skill experience exceeds its budget")
    return result


class InitialSkillExperience:
    def __init__(self, bundle, *, env_id, trace=None, policy_id=None):
        self.bundle = bundle
        self.env_id, self.trace, self.policy_id = env_id, trace, policy_id
        self.id = uuid.uuid4().hex
        self.delivered = False
        self.attempts = 0

    def prepare(self, context, images, prompt):
        if CONTEXT_KEY in context:
            raise ValueError("Initial skill experience is owned by the runtime")
        augmented = {**deepcopy(context), CONTEXT_KEY: bundle_experience(self.bundle)}
        self.attempts += 1
        if self.trace is not None:
            path = (
                Path("initial_skill_experience")
                / self.id
                / f"attempt-{self.attempts:03d}.json"
            )
            atomic_json(
                self.trace.root / path,
                {
                    "input_identity": input_identity(),
                    "env_id": self.env_id,
                    "skill_id": self.bundle.skill_id,
                    "context": augmented,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "current_cameras": list(images),
                },
            )
            self.trace.write(
                "code_skill_initial_experience",
                policy_id=self.policy_id,
                env_id=self.env_id,
                skill_id=self.bundle.skill_id,
                injection_id=self.id,
                attempt=self.attempts,
                request_file=str(path),
                status="prepared",
            )
        return augmented

    def delivered_request(self):
        self.delivered = True
        if self.trace is not None:
            self.trace.write(
                "code_skill_initial_experience",
                policy_id=self.policy_id,
                env_id=self.env_id,
                skill_id=self.bundle.skill_id,
                injection_id=self.id,
                attempt=self.attempts,
                status="delivered",
            )


def current_initial_experience():
    return _CURRENT.get()


@contextmanager
def initial_experience_scope(experience):
    token = _CURRENT.set(experience)
    try:
        yield
    finally:
        _CURRENT.reset(token)
