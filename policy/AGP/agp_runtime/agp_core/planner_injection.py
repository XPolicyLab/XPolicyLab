"""Route the migration Worker's planner calls to agp_core.policy_planner.PolicyPlanner (switch OURS_POLICY_PLANNER).

migration/worker.py (shared with the other variant, unchanged) reaches a planner only through
`self.rm.planner[robot.robot_name]`, in four places:
    _prepare_from_state  home -> plan_joint; move_ee plan -> solve_ik_to_joint + plan_joint;
                         move_ee / move_delta linear -> solve_ik_to_joint for every sample
    _validate_motion     validate_joint_trajectory (preflight before any physics step)
    _predicted_tcp       forward_ee_pose_base (Cartesian rate limits, estimate)
With the switch on, `worker_class()` returns the worker class with PolicyPlannerMixin in front. Its constructor
builds one PolicyPlanner per robot type of the two target arms (both arms are 'x5' and share one, as they share the
benchmark's planner), runs the unchanged worker constructor, then replaces `self.rm` by RobotManagerProxy:
`planner` and `ik_solver` (the benchmark's alias of the same objects) map robot names to the PolicyPlanners, every
other attribute is the benchmark robot manager's own. The environment keeps its robot manager; only the worker sees
the proxy, so the worker never calls the benchmark's planner.
With the switch off (default) nothing is imported or replaced: worker_class() returns the class it was given.

agp_core/sim_main.py calls install(migration.worker) after the worker_ext selection, so both compose.
"""
import os

from agp_core import features

FLAG = "policy_planner"


def target_arm_robots(robot_manager):
    """The worker's two arms, selected exactly as migration.worker.Worker.__init__ does."""
    return {arm: next(r for r in robot_manager.robot_list if r.type == "target" and r.arm_name.startswith(arm))
            for arm in ("left", "right")}


def default_planner_factory(robot, env):
    """PolicyPlanner for a live benchmark robot: public cuRobo yml, table of the task's scene, the worker's dt."""
    from agp_core.policy_planner import PolicyPlanner   # imported only when the switch is on (cuRobo, torch)
    return PolicyPlanner.from_public_config(
        robot.robot_name, list(robot.entity_origin_pose), os.environ.get("AGENT_TASK"),
        active_joints_name=list(robot.arm_joints_name), interpolation_dt=1 / env.obs_manager.collect_freq)


def build_planners(env, factory=default_planner_factory):
    planners = {}
    for robot in target_arm_robots(env.robot_manager).values():
        if robot.robot_name not in planners:
            planners[robot.robot_name] = factory(robot, env)
    return planners


class PlannerMap:
    """Read-only robot_name -> planner mapping; an unknown robot is a KeyError, never the benchmark planner."""

    def __init__(self, planners):
        self._planners = dict(planners)

    def __getitem__(self, name):
        return self._planners[name]

    def get(self, name, default=None):
        return self._planners.get(name, default)

    def __contains__(self, name):
        return name in self._planners

    def __iter__(self):
        return iter(self._planners)

    def __len__(self):
        return len(self._planners)

    def keys(self):
        return self._planners.keys()

    def values(self):
        return self._planners.values()

    def items(self):
        return self._planners.items()


class RobotManagerProxy:
    """The benchmark robot manager with `planner` / `ik_solver` replaced; every other attribute is forwarded."""

    def __init__(self, robot_manager, planners):
        mapping = PlannerMap(planners)
        object.__setattr__(self, "_robot_manager", robot_manager)
        object.__setattr__(self, "planner", mapping)
        object.__setattr__(self, "ik_solver", mapping)

    def __getattr__(self, name):   # only reached for names the proxy itself does not define
        return getattr(object.__getattribute__(self, "_robot_manager"), name)

    def __setattr__(self, name, value):
        raise AttributeError(f"RobotManagerProxy is read-only (tried to set {name!r})")

    def __delattr__(self, name):
        raise AttributeError(f"RobotManagerProxy is read-only (tried to delete {name!r})")


class PolicyPlannerMixin:
    """Put in front of a migration Worker class: the worker plans with PolicyPlanner instances."""

    planner_factory = staticmethod(default_planner_factory)

    def __init__(self, env, session):
        import torch

        def allocated():   # never initialises CUDA itself (stub planners in tests do not use it)
            return torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0

        before = allocated()
        # Built before the worker constructor: a failure leaves no worker_ready.json behind.
        planners = build_planners(env, type(self).planner_factory)
        after = allocated()
        super().__init__(env, session)
        self.rm = RobotManagerProxy(self.rm, planners)
        self.policy_planners = planners
        from migration.worker import atomic
        info = {name: (p.info() if hasattr(p, "info") else {"class": type(p).__qualname__})
                for name, p in planners.items()}
        atomic(self.session / "policy_planner.json", {
            "switch": features.FLAGS[FLAG], "planners": info, "torch_cuda_allocated_delta_bytes": after - before})


def worker_class(base_cls, environ=None, factory=None):
    """`base_cls` unchanged when the switch is off; else a subclass with PolicyPlannerMixin in front."""
    if not features.enabled(FLAG, environ):
        return base_cls
    if issubclass(base_cls, PolicyPlannerMixin) and factory is None:
        return base_cls
    attributes = {"__module__": __name__, "__doc__": f"{base_cls.__qualname__} planning with PolicyPlanner."}
    if factory is not None:
        attributes["planner_factory"] = staticmethod(factory)
    return type(f"PolicyPlanner{base_cls.__name__}", (PolicyPlannerMixin, base_cls), attributes)


def install(worker_module, environ=None, factory=None):
    """Replace `worker_module.Worker` when OURS_POLICY_PLANNER=1; a no-op otherwise. Returns the class in use."""
    cls = worker_class(worker_module.Worker, environ, factory)
    if cls is not worker_module.Worker:
        worker_module.Worker = cls
        print(f"[ours] policy_planner: {cls.__name__} (planner calls -> agp_core.policy_planner.PolicyPlanner)",
              flush=True)
    return cls
