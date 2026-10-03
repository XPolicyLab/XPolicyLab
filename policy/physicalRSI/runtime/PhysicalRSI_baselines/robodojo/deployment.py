"""One fixed XPolicyLab-compatible expert model with an agent planner around it."""

from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, read_json
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness.artifacts import verify_harness


class CommittedModel:
    def __init__(self, *, state_root, task, factories, output):
        current = HarnessState(state_root).resolve()
        if len(factories) != 1:
            raise ValueError("Exactly one fixed policy factory required")
        name, revision = next(iter(factories))
        self._initialize(current=current, task=task, factories=factories, output=output,
                         expert=dict(name=name, revision=revision))

    def _initialize(self, *, current, task, factories, output, expert=None):
        if expert is None:
            raise ValueError("A fixed primary expert binding is required")
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.harness = deepcopy(current["harness"])
        self.freeze = current["freeze_sha256"]
        self.model = None
        self.closed = False
        self.failed = False
        key = (expert["name"], expert["revision"])
        if key not in factories:
            raise ValueError("Exact fixed expert factory unavailable: " + str(key))
        self.binding = dict(task=task, expert=deepcopy(expert), lineage_revision=current["revision"],
                            harness_sha256=self.freeze, system_mode="single_fixed_system",
                            physical_qualification=False, execution_mode=current.get("execution_mode", "committed"))
        atomic_json(self.root / "binding.json", self.binding)
        self._verify()
        try:
            self.model = factories[key](harness=deepcopy(self.harness), binding=deepcopy(self.binding), output=self.root / "expert")
            for name in ("update_obs","update_obs_batch","get_action","get_action_batch","reset","close"):
                if not callable(getattr(self.model, name, None)): raise ValueError("Expert must implement " + name)
            self._verify()
        except BaseException:
            if self.model is not None: self.model.close()
            raise

    def _verify(self):
        if verify_harness(self.harness) != self.freeze:
            raise ValueError("Frozen deployment harness changed")

    def _call(self, name, *args):
        if self.closed or self.failed:
            raise RuntimeError("Deployment closed or interrupted; reset required")
        try:
            self._verify()
            result = getattr(self.model, name)(*args)
            self._verify()
            return result
        except BaseException:
            self.failed = True
            raise

    def update_obs(self, obs):
        return self._call("update_obs", obs)

    def update_obs_batch(self, obs_list):
        return self._call("update_obs_batch", obs_list)

    def get_action(self):
        return self._call("get_action")

    def get_action_batch(self, env_idx_list=None):
        return self._call("get_action_batch", env_idx_list)

    def reset(self):
        if self.closed:
            raise RuntimeError("Closed deployment cannot reset")
        # Reset cleans the selected expert, never re-reads a moving current pointer.
        self.failed = True
        self.model.reset()
        self._verify()
        self.failed = False

    def on_trial_end(self, result=None):
        self.reset()

    def close(self):
        if not self.closed:
            self.failed = True
            self.model.close()
            self.closed = True


class FrozenModel(CommittedModel):
    """Evaluate an exact frozen candidate without resolving or updating lineage.

    An explicit expert override is for paired development comparison only. It must
    already exist at the exact revision in this candidate's foundation or skills.
    Validation uses the candidate's fixed primary expert.
    """

    def __init__(
        self, *, harness, expected_sha256, task, factories, output, expert=None, split
    ):
        frozen = deepcopy(harness)
        if split not in {"development", "validation"}:
            raise ValueError(
                "Candidate execution requires an explicit experiment split"
            )
        if verify_harness(frozen) != expected_sha256:
            raise ValueError("Candidate differs from expected frozen identity")
        if expert is not None:
            if split != "development":
                raise ValueError("Expert override is only allowed in development")
            root = Path(frozen["root"])
            foundation = read_json(root / "foundation.json")
            skills = read_json(root / "skills.json")
            name, revision = expert["name"], expert["revision"]
            if (
                foundation.get(name) != revision
                and skills.get(name, {}).get("revision") != revision
            ):
                raise ValueError("Expert override is not pinned in candidate")
        self._initialize(
            current=dict(
                harness=frozen,
                freeze_sha256=expected_sha256,
                revision=None,
                execution_mode=split,
            ),
            task=task,
            factories=factories,
            output=output,
            expert=expert,
        )
