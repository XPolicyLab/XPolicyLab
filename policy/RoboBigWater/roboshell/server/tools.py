"""Tool registry: extra robo commands loaded from tools/<name>/tool.py and tasks/<task>/tools/<name>/tool.py.

A tool module defines

    TOOL = {
        "name": "pick",
        "commands": [
            {"name": "pick", "budget": True, "args": [
                {"name": "arm", "positional": True, "choices": ["left", "right"]},
                {"name": "x", "type": "float", "required": True, "help": "..."},
            ]},
        ],
    }

    def run(api, command, args):  # -> (feedback dict, exit code)
        ...

`api` is the EpisodeAPI below: the robot state, the motion primitives and the
observations, nothing else. A tool never sees the simulator, so it works in
both modes. The agent-facing text of a tool is tools/<name>/interface.md.
"""

import importlib.util
import os

from roboshell.contract import v0 as C

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def base_task(task):
    """`stack_bowls_random` is evaluated as part of `stack_bowls` and shares its tools and manual."""
    if task and task.endswith("_random"):
        return task[: -len("_random")]
    return task


def tool_dirs(task):
    """Directories that may hold tools, task-specific ones last so they can shadow shared ones."""
    dirs = [os.path.join(ROOT, "tools")]
    if task:
        dirs.append(os.path.join(ROOT, "tasks", base_task(task), "tools"))
    return dirs


def enabled_tools(task):
    """tasks/<task>/enabled_tools.txt: one tool name per line. Missing file: no extra tools.

    ROBOSHELL_TOOLS overrides the file: "none" for the base tools only, or a comma-separated list.
    """
    override = os.environ.get("ROBOSHELL_TOOLS")
    if override is not None:
        return [] if override.strip() in ("", "none") else [t.strip() for t in override.split(",") if t.strip()]
    path = os.path.join(ROOT, "tasks", base_task(task) or "", "enabled_tools.txt")
    if not task or not os.path.exists(path):
        return []
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def find_tool(name, task):
    for directory in reversed(tool_dirs(task)):
        path = os.path.join(directory, name, "tool.py")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"tool {name!r} not found for task {task!r}")


def load_tools(task):
    registry = {}
    for name in enabled_tools(task):
        path = find_tool(name, task)
        spec = importlib.util.spec_from_file_location(f"roboshell_tool_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        info = module.TOOL
        for command in info["commands"]:
            if command["name"] in C.BUDGETED + C.FREE or command["name"] in registry:
                raise ValueError(f"tool {name}: command {command['name']!r} clashes with an existing command")
            registry[command["name"]] = {"tool": name, "module": module, "spec": command, "path": path}
    return registry


def schema(registry):
    """What the robo client needs to build the sub-commands."""
    return [dict(entry["spec"], tool=entry["tool"]) for entry in registry.values()]


class EpisodeAPI:
    """The surface a tool may use."""

    def __init__(self, episode):
        self._episode = episode
        self.arms = episode.arms
        self.geometry = __import__("roboshell.server.geometry", fromlist=["x"])
        self.motion = __import__("roboshell.server.motion", fromlist=["x"])

    def arm(self, tag):
        return self.arms[tag]

    def move_tcp(self, arm, target, feedback):
        """Straight-line move of one arm's TCP to a 4x4 target pose; fills the motion feedback, returns the exit code."""
        return self._episode.move_tcp(arm, target, feedback)

    def set_gripper(self, arm, value):
        arm.gripper_target = float(min(1.0, max(0.0, value)))
        return self._episode.hold(self._episode.GRIPPER_STEPS)

    def hold(self, steps):
        return self._episode.hold(int(steps))

    def run(self, sequences):
        return self._episode.run(sequences)

    def observe(self):
        """The latest observation: PNG bytes per camera, depth arrays (if any), camera matrices."""
        return self._episode.executor.observe()

    def planner(self, tag):
        return self._episode.executor.planner(tag)

    @property
    def over(self):
        return self._episode.over

    def sim_time_left(self):
        return self._episode.sim_time_left()

    def estimate_tcp_chain(self, arm, stages):
        """Read-only cost estimate. Internal plans are discarded, never executed/exported.

        Includes existing settle, close/open and final home costs. Reachability
        is IK-only, not a collision, contact or grasp-success certificate.
        """
        import hashlib
        import numpy as np
        from roboshell.server.core import SETTLE_STEPS, GRIPPER_STEPS
        episode = self._episode
        current = arm.joints().copy()
        ee = arm.ee().copy()
        rows = []
        scene = hashlib.sha256(
            f"{episode.directory}:{episode.steps}:{episode.commands}".encode()
        ).hexdigest()[:24]
        for name, target_tcp in stages:
            try:
                sequence, _ = self.motion.plan(
                    episode.executor.planner(arm.tag), episode.executor.robot(arm.tag),
                    current, ee, target_tcp @ arm.tcp_to_ee,
                    episode.executor.dt(), episode.executor.limits(arm.tag))
            except self.motion.PlanFailure as exc:
                return dict(estimate_ok=False, reason=exc.reason, failed_stage=name,
                            stage_costs=rows, scene_ref=scene, estimate_only=True)
            rows.append(dict(stage=name, action_steps=len(sequence) + SETTLE_STEPS))
            current = sequence[-1].copy()
            ee = target_tcp @ arm.tcp_to_ee
        home = max(self.motion.MIN_STEPS, int(np.ceil(
            float(np.abs(arm.home_joints - current).max()) / 1.2 * self.motion.CONTROL_HZ
        ))) + SETTLE_STEPS
        transfer = sum(row['action_steps'] for row in rows) + 2 * GRIPPER_STEPS
        return dict(estimate_ok=True, scene_ref=scene, estimate_only=True,
                    stage_costs=rows, gripper_action_steps=2 * GRIPPER_STEPS,
                    transfer_action_steps=transfer, home_action_steps=home,
                    total_action_steps=transfer + home,
                    remaining_action_steps=max(0, int(episode.executor.step_lim()) - episode.steps),
                    charged_commands=2, collision_checked=False,
                    holding_verified=False,
                    note='Read-only IK cost estimate; not an executable plan or holding evidence. '
                         'Discard after every physical command/failure/reset and measure again.')
