"""Configuration-only model entrypoint for pinned VLA/code-policy execution.

Pass the physicalRSI configuration under the deploy config's `physicalrsi` key.
This is trusted local deployment configuration, not generated policy input.
"""

from copy import deepcopy

from PhysicalRSI_core.infra.resources import ResourcePool
from PhysicalRSI_core.infra.storage import atomic_json, digest

from .isolated_factory import isolated_factory
from .deployment import CommittedModel, FrozenModel
from .expert_model import expert_factory
from .agent_planner import AgentPlanner, AgentLoop


class Model:
    def __init__(self, config):
        settings = deepcopy(config["physicalrsi"])
        if settings.get("schema") != "physicalrsi.robodojo.policy/v1":
            raise ValueError("Versioned physicalRSI policy configuration required")
        mode = settings["mode"]
        if mode not in {"committed", "candidate"}:
            raise ValueError("Policy mode must be committed or candidate")
        if mode == "committed" and any(
            k in settings for k in ("harness", "expected_sha256", "split", "expert")
        ):
            raise ValueError("Committed policy cannot accept candidate selectors")
        if mode == "candidate" and "state_root" in settings:
            raise ValueError("Candidate policy cannot select a moving lineage")
        if len(settings["factories"]) != 1:
            raise ValueError("Evaluation requires exactly one fixed policy factory")
        planner = AgentPlanner(**settings.get("agent", {}))
        reference_interval = settings.get("reference_agent_chunks_per_call", 1)
        if type(reference_interval) is not int or reference_interval < 1:
            raise ValueError("Positive integer reference_agent_chunks_per_call required")
        self.agent_interval = 100 * reference_interval
        self.plan_count = 0
        self.observed_indices = []
        self.pool = ResourcePool(settings["resources"])
        factories = {}
        for entry in settings["factories"]:
            expert = entry["expert"]
            key = (expert["name"], expert["revision"])
            if not all(isinstance(v, str) and v for v in key) or key in factories:
                raise ValueError("Unique exact expert factory identities required")
            if entry["kind"] in {"code_policy"}:
                if not key[0].startswith("code."):
                    raise ValueError("Code-policy factory must select a code skill")
                factories[key] = isolated_factory(entry["spec"], pool=self.pool)
            elif entry["kind"] == "vla":
                if entry["spec"]["expert"] != expert:
                    raise ValueError(
                        "VLA factory identity differs from its specification"
                    )
                factories[key] = expert_factory(spec=entry["spec"], **entry["service"])
            else:
                raise ValueError("Unknown expert factory kind")
        if not factories:
            raise ValueError("At least one exact expert factory required")
        common = dict(
            task=settings["task"], factories=factories, output=settings["output"]
        )
        if mode == "committed":
            self.deployment = CommittedModel(
                state_root=settings["state_root"], **common
            )
        else:
            self.deployment = FrozenModel(
                harness=settings["harness"],
                expected_sha256=settings["expected_sha256"],
                split=settings["split"],
                expert=settings.get("expert"),
                **common,
            )
        self.agent_loop = AgentLoop(planner, settings.get("task_specific_policy_memory", {}), self._record_plan, self.agent_interval, settings.get("agent_max_calls_per_episode", 1))
        try:
            self.identity = dict(
                configuration_sha256=digest(settings),
                **deepcopy(self.deployment.binding),
            )
            atomic_json(self.deployment.root / "policy-configuration.json", settings)
            atomic_json(self.deployment.root / "policy-identity.json", self.identity)
        except BaseException:
            self.deployment.close()
            raise

    def physicalrsi_identity(self):
        return deepcopy(self.identity)

    def _record_plan(self, receipt):
        self.plan_count += 1
        atomic_json(self.deployment.root / "agent-plans" / f"{self.plan_count:06d}.json", receipt)

    def update_obs(self, obs):
        prepared = self.agent_loop.prepare([obs])
        self.observed_indices = [obs.get("env_idx", 0)]
        try:
            return self.deployment.update_obs(prepared[0])
        except BaseException:
            self.agent_loop.ready.clear()
            raise

    def update_obs_batch(self, obs_list):
        rows = list(obs_list)
        prepared = self.agent_loop.prepare(rows)
        self.observed_indices = [obs.get("env_idx", 0) for obs in rows]
        try:
            return self.deployment.update_obs_batch(prepared)
        except BaseException:
            self.agent_loop.ready.clear()
            raise

    def get_action(self):
        self.agent_loop.consume(self.observed_indices)
        return self.deployment.get_action()

    def get_action_batch(self, env_idx_list=None):
        self.agent_loop.consume(self.observed_indices if env_idx_list is None else env_idx_list)
        return self.deployment.get_action_batch(env_idx_list)

    def reset(self):
        self.agent_loop.reset()
        return self.deployment.reset()

    def on_trial_end(self, result=None):
        self.agent_loop.reset()
        return self.deployment.on_trial_end(result)

    def close(self):
        return self.deployment.close()
