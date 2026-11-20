"""The official RoboDojo TASK_ENV as the AGP adapter may use it.

Three layers sit between the benchmark and the migration worker:

    TaskEnvWhitelist   wraps the real TASK_ENV. It forwards only the whitelisted interface and logs every access
                       (allowed or refused) to task_env_access.json in the session:
                         setup phase (adapter, before the agent starts): read task_name, env_seeds, eval_seed,
                             num_envs (episode bookkeeping only; the evaluator's success_nums / fail_nums are not
                             read: the episode index is the adapter's own count)
                         policy phase: get_obs(), take_action(action), is_episode_end(); read take_action_cnt
                         finalize phase: the same, plus mark_failed() = `success[0] = False`, used only to end an
                             unfinished episode as a failure before is_episode_end() (EvalEnv.reset starts with
                             success True, so an early return would otherwise count as a success)
                       Anything else (robot_manager, planner, sim, obs_manager, end_flag, reward_manager, success
                       reads, attribute writes, ...) is refused with WhitelistViolation and logged. step_lim is not
                       read at all: the episode's step limit is the public task code's value for the configuration
                       TASK_ENV.task_name names (public_step_lim, agp_core/task_step_limits.json).
    OfficialEnv        the environment object the worker sees. Everything it offers comes from those calls plus
                       public configuration: robot_manager is ObservedRobotManager (joints and link6 pose from
                       get_obs, arm descriptors from the public robot configs, planners injected by the worker),
                       obs_manager.collect_freq from env_cfg, the instruction from the observation, sim time and
                       take_action_cnt from its own count of take_action calls, end_flag from is_episode_end().
                       It fetches at most one observation per take_action: the first state read after a step
                       calls get_obs() and every later read until the next step reuses it (so the official video
                       gets one frame per control step and an observe() right after a step renders nothing new).
    AgentInfrastructureFailure (BaseException): agent, MCP server, codex authentication, worker start-up or a
                       whitelist violation. Deliberately not an Exception: RoboDojo's main.py wraps every episode in
                       `except Exception` and moves on to the next layout, which would fold an outage into the
                       success rate (the RoboDojo team's Astra adapter does the same).
    SimulatorFault (BaseException) carries a PhysX error raised by an official call through the migration worker's
                       `except Exception` request handler, so the adapter can re-raise the original and main.py's
                       PhysX recovery (abandon the seed, refill, restart) works as designed.

The finger opening is not observed under these conditions (the observation has only the last gripper command), so
the worker built on OfficialEnv reports gripper outcomes from the command (agp_core/official_worker.py).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ARMS = ("left", "right")
CALLS = ("get_obs", "take_action", "is_episode_end")
READS = ("take_action_cnt",)
METADATA = ("task_name", "env_seeds", "eval_seed", "num_envs")
PHASES = ("setup", "policy", "finalize")
WHITELIST = {
    "setup phase (adapter bookkeeping before the agent starts), reads": list(METADATA),
    "policy and finalize phases, calls": [f"{name}()" for name in CALLS],
    "policy and finalize phases, reads": list(READS),
    "finalize phase, write": ["success[0] = False (mark_failed: end an unfinished episode as a failure)"],
}
PHYSX_FAULTS = {"PhysXBrokenError", "PhysXFatalError"}
# Step limit of every configuration from the public task code (task/RoboDojo/tasks/<name>.py `self.step_lim = N`),
# written offline from that code.
STEP_LIMITS = Path(__file__).with_name("task_step_limits.json")


FAILURE_KINDS = ("infrastructure", "external_kill", "freeze_mismatch")


class AgentInfrastructureFailure(BaseException):
    """The agent side could not run the episode (agent, MCP, codex auth, worker start-up, whitelist violation).

    Derived from BaseException on purpose: RoboDojo's main.py catches `Exception` per layout and continues with the
    next one, which would silently count an outage as a failed or skipped layout. This aborts the run instead.

    `kind` names the class of failure for the launcher (written to the session's infrastructure_failure.json):
        infrastructure   the agent side could not run or broke down (default). A crash of our own policy code
                         mid-episode (the MCP server or agp_core/run_agent.py dying by itself after codex started) is not
                         raised: it is a counted policy failure of kind mcp_crash (agp_core/official_deploy.py
                         policy_crash); a failure to start stays an infrastructure failure.
        external_kill    a policy process (codex, the MCP server, the agent launcher or the scripted driver) was ended
                         by a signal our own code did not send (for example the kernel OOM killer) before the official
                         episode ended. It does not count in the score: the launcher re-runs the layout with a new
                         attempt number and stops after two such kills in a row.
                         A fault signal (SIGSEGV, SIGABRT, ...) of our own processes (the MCP server, the agent
                         launcher, the scripted driver) is that process crashing by itself, not an external kill.
        freeze_mismatch  the system about to run is not the frozen one: the freeze manifest (agp_core/freeze.py)
                         does not verify against the current state, or a frozen measurement got a test-only run control
                         or a pinned one with another value. Refused before the worker or the policy starts; the
                         launcher stops the batch instead of retrying.
    `details` identifies it (process, signal, return code; for freeze_mismatch the differing entries)."""

    def __init__(self, message="", *, kind="infrastructure", details=None):
        super().__init__(message)
        if kind not in FAILURE_KINDS:
            raise ValueError(f"unknown infrastructure failure kind {kind!r}")
        self.kind = kind
        self.details = dict(details or {})


class WhitelistViolation(AttributeError):
    """An access outside the AGP TASK_ENV whitelist (logged before it is raised)."""


class SimulatorFault(BaseException):
    """A PhysX error from an official call, carried through `except Exception` handlers; `.original` is re-raised."""

    def __init__(self, original):
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original


def is_physx_fault(exc):
    return type(exc).__name__ in PHYSX_FAULTS


# ----------------------------------------------------------------------------- access log
def _caller(skip_files):
    """file:line of the innermost frame outside this module (where the access came from)."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        if frame.filename not in skip_files:
            return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return "unknown"


class AccessLog:
    """Counts of every access by (layer, name, kind, phase, allowed); refused accesses are listed with their caller."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.counts = {}
        self.first_s = {}
        self.refused = []
        self.introspection = []
        self.path = None

    def allow(self, layer, name, kind, phase):
        key = (layer, name, kind, phase, True)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.first_s.setdefault(key, round(time.monotonic() - self.t0, 3))

    def refuse(self, layer, name, kind, phase):
        key = (layer, name, kind, phase, False)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.first_s.setdefault(key, round(time.monotonic() - self.t0, 3))
        self.refused.append({"layer": layer, "name": name, "kind": kind, "phase": phase,
                             "t_s": round(time.monotonic() - self.t0, 3), "where": _caller({__file__})})
        self.write()

    def note_introspection(self, layer, name):
        if len(self.introspection) < 50:
            self.introspection.append({"layer": layer, "name": name})

    def report(self):
        accesses = [{"layer": layer, "name": name, "kind": kind, "phase": phase, "allowed": allowed,
                     "count": count, "first_t_s": self.first_s[(layer, name, kind, phase, allowed)]}
                    for (layer, name, kind, phase, allowed), count in sorted(self.counts.items(), key=str)]
        task_env = [a for a in accesses if a["layer"] == "TASK_ENV"]
        return {
            "whitelist": WHITELIST,
            "task_env_names_accessed": sorted({a["name"] for a in task_env}),
            "accesses": accesses,
            "refused": self.refused,
            "only_whitelisted": not self.refused and all(a["allowed"] for a in accesses),
            "introspection_not_forwarded": self.introspection,
            "meaning": ("layer TASK_ENV = calls that reached the benchmark object; layer facade / robot_manager = "
                        "attributes the worker asked of the adapter-side environment that it does not offer"),
        }

    def write(self, path=None):
        path = Path(path) if path is not None else self.path
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.report(), indent=1, default=str))
        tmp.replace(path)


# ----------------------------------------------------------------------------- the whitelisted TASK_ENV
class TaskEnvWhitelist:
    """The benchmark TASK_ENV behind the AGP whitelist (see the module docstring). Every access is logged."""

    def __init__(self, task_env, log=None):
        object.__setattr__(self, "_env", task_env)
        object.__setattr__(self, "_log", log if log is not None else AccessLog())
        object.__setattr__(self, "_phase", "setup")

    # -- phase control and bookkeeping (attributes of the proxy, not of TASK_ENV)
    @property
    def access_log(self):
        return object.__getattribute__(self, "_log")

    @property
    def phase(self):
        return object.__getattribute__(self, "_phase")

    def enter(self, phase):
        if phase not in PHASES or PHASES.index(phase) < PHASES.index(self.phase):
            raise ValueError(f"cannot go from phase {self.phase!r} to {phase!r}")
        object.__setattr__(self, "_phase", phase)

    def episode_metadata(self):
        """Episode bookkeeping, read once in the setup phase (task, seed, layout), and the official step limit of the
        task from the public task code (public_step_lim; TASK_ENV.step_lim is not read). The episode index is the
        adapter's own count (agp_core/official_deploy.py), not the evaluator's success / fail counters."""
        if self.phase != "setup":
            self.access_log.refuse("TASK_ENV", "episode metadata", "read", self.phase)
            raise WhitelistViolation("episode metadata is read only before the agent starts")
        values = {name: getattr(self, name) for name in METADATA}
        seeds = list(values["env_seeds"])
        return {"task": str(values["task_name"]), "seed": int(values["eval_seed"]),
                "layout_id": int(seeds[0]), "env_seeds": [int(s) for s in seeds],
                "official_step_lim": public_step_lim(str(values["task_name"])), "num_envs": int(values["num_envs"])}

    def mark_failed(self):
        """`success[0] = False` on the benchmark object: only to end an unfinished episode as a failure."""
        log, phase = self.access_log, self.phase
        if phase != "finalize":
            log.refuse("TASK_ENV", "success[0] = False", "write", phase)
            raise WhitelistViolation("success may only be set to False while finalising the episode")
        log.allow("TASK_ENV", "success[0] = False", "write", phase)
        object.__getattribute__(self, "_env").success[0] = False

    # -- the whitelist
    def __getattr__(self, name):   # reached only for names the proxy itself does not define
        log, phase = self.access_log, self.phase
        if name.startswith("__") and name.endswith("__"):
            log.note_introspection("TASK_ENV", name)
            raise AttributeError(name)
        env = object.__getattribute__(self, "_env")
        if phase == "setup" and name in METADATA:
            log.allow("TASK_ENV", name, "read", phase)
            return deepcopy(getattr(env, name))
        if phase in ("policy", "finalize"):
            if name in CALLS:
                method = getattr(env, name)

                def call(*args, **kwargs):
                    log.allow("TASK_ENV", name, "call", phase)
                    return method(*args, **kwargs)
                call.__name__ = name
                return call
            if name in READS:
                log.allow("TASK_ENV", name, "read", phase)
                return tuple(int(v) for v in getattr(env, name))
        log.refuse("TASK_ENV", name, "read", phase)
        raise WhitelistViolation(f"TASK_ENV.{name} is outside the AGP whitelist (phase {phase})")

    def __setattr__(self, name, value):
        self.access_log.refuse("TASK_ENV", name, "write", self.phase)
        raise WhitelistViolation(f"TASK_ENV.{name} may not be set")

    def __delattr__(self, name):
        self.access_log.refuse("TASK_ENV", name, "delete", self.phase)
        raise WhitelistViolation(f"TASK_ENV.{name} may not be deleted")


# ----------------------------------------------------------------------------- public task and robot description
def public_step_lim(task):
    """The step limit of configuration `task` (TASK_ENV.task_name) from the public task code: agp_core/task_step_limits.json
    (`self.step_lim = N` of task/RoboDojo/tasks/<task>.py). A configuration missing from the
    table is an infrastructure failure at start-up; there is no fallback to TASK_ENV.step_lim."""
    entry = json.loads(STEP_LIMITS.read_text())["limits"].get(task)
    if entry is None:
        raise AgentInfrastructureFailure(f"no public step limit for task configuration {task!r} in {STEP_LIMITS}")
    return int(entry["step_lim"])


def _task_setting(task, key, default, root):
    """A task's setting as utils.pipeline_utils.process_config resolves it (task, then common, then default)."""
    info = yaml.safe_load((Path(root) / "task" / "RoboDojo" / "config" / "_task.yml").read_text()) or {}
    task_info = ((info.get("tasks") or {}).get(task) or {}) if task else {}
    return task_info.get(key, (info.get("common") or {}).get(key, default))


@dataclass(frozen=True)
class PublicArm:
    """One target arm as the benchmark's RobotManager names it, built from public config files only."""
    arm_name: str
    gripper_name: str
    robot_name: str
    entity_origin_pose: tuple
    gripper_bias: float
    gripper_scale: tuple
    gripper_move: dict = field(hash=False)
    gripper_joints_name: tuple = ()
    arm_joints_name: tuple = ()
    gripper_limits_m: tuple = ()
    ee_link_name: str = "link6"
    ee_type: str = "gripper"
    base_link: str = "base_link"
    type: str = "target"
    robot_type: str = "arm"

    @property
    def side(self):
        return self.arm_name.split("_")[0]


def _urdf_limits(urdf, names):
    limits = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        if joint.get("name") in names:
            limit = joint.find("limit")
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    missing = set(names) - set(limits)
    if missing:
        raise ValueError(f"{urdf}: no limits for {sorted(missing)}")
    return tuple(limits[name] for name in names)


def public_arms(task, root):
    """The two target arms of a task (left first, as RobotManager._set_robot_obs_name names them).

    Sources: env_cfg/arx_x5.yml (default robot config), task/RoboDojo/config/_task.yml (per-task robot_config),
    env_cfg/robot/<config>.yml (base poses; entity_origin_pose = default_root_pos + default_root_rot as in the X5
    robot class), Assets/Robots/<robot>/robot_config.yml (joints, gripper mapping, gripper_bias) and its URDF
    (finger joint limits)."""
    root = Path(root)
    default = yaml.safe_load((root / "env_cfg" / "arx_x5.yml").read_text())["config"]["robot"]
    config = _task_setting(task, "robot_config", default, root)
    robots = yaml.safe_load((root / "env_cfg" / "robot" / f"{config}.yml").read_text())["robots"]
    targets = [r for r in robots if r.get("robot_type") == "arm" and not r.get("coupled", False)
               and r.get("type", "target") == "target"]
    if len(targets) != 2:
        raise NotImplementedError(f"robot config {config!r} has {len(targets)} uncoupled target arms, expected 2")
    arms = []
    for side, cfg in zip(ARMS, targets):
        robot_dir = root / "Assets" / "Robots" / cfg["robot_name"]
        rc = yaml.safe_load((robot_dir / "robot_config.yml").read_text())
        fingers = tuple(rc["gripper_joints_name"])
        arms.append(PublicArm(
            arm_name=f"{side}_arm", gripper_name=f"{side}_ee", robot_name=str(cfg["robot_name"]),
            entity_origin_pose=tuple(float(v) for v in list(cfg["default_root_pos"]) + list(cfg["default_root_rot"])),
            gripper_bias=float(rc["gripper_bias"]), gripper_scale=tuple(float(v) for v in rc["gripper_scale"]),
            gripper_move=dict(rc["gripper_move"]), gripper_joints_name=fingers,
            arm_joints_name=tuple(rc["arm_joints_name"]),
            gripper_limits_m=_urdf_limits(robot_dir / rc["urdf_path"], fingers),
            ee_link_name=rc["ee_link"], ee_type=rc.get("ee_type", "gripper"),
            base_link=rc.get("base_link", "base_link")))
    return arms


def public_collect_freq(root):
    return float(yaml.safe_load((Path(root) / "env_cfg" / "arx_x5.yml").read_text())["observation"]["collect_freq"])


# ----------------------------------------------------------------------------- what the worker sees
class _ControlClock:
    """sim.sim.current_time for the worker: own count of control steps times the control period."""

    def __init__(self, env):
        self._env = env

    @property
    def current_time(self):
        return self._env.steps_sent / self._env.collect_freq


def _refuse(log, layer, name):
    log.refuse(layer, name, "read", "policy")
    raise WhitelistViolation(f"{layer}.{name} is not offered under official conditions")


class ObservedRobotManager:
    """The robot-manager interface the migration worker uses, from the official observation and public config."""

    def __init__(self, env, arms):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "robot_list", list(arms))
        object.__setattr__(self, "planner", {})     # filled by planner_injection.RobotManagerProxy
        object.__setattr__(self, "ik_solver", {})

    def _state(self):
        return object.__getattribute__(self, "_env").get_obs()["state"]

    def get_joint(self, robot, env_idx_list=None):
        return {0: np.asarray(self._state()[f"{robot.arm_name}_joint_state"], dtype=float).ravel().copy()}

    def get_real_endpose(self, robot, env_idx_list=None, is_relative=True):
        # Official <arm>_ee_pose is link6 relative to the env origin; env 0's origin is the world origin
        # (verified for the camera model, agp_core/check_public_camera.py), so both frames give the same pose.
        return {0: np.asarray(self._state()[f"{robot.side}_ee_pose"], dtype=float).ravel().copy()}

    @staticmethod
    def process_name(name):
        return name if name.endswith("_state") else name + "_joint_state"

    def __getattr__(self, name):
        log = object.__getattribute__(self, "_env").access_log
        if name.startswith("__") and name.endswith("__"):
            log.note_introspection("robot_manager", name)
            raise AttributeError(name)
        _refuse(log, "robot_manager", name)

    def __setattr__(self, name, value):
        raise AttributeError(f"ObservedRobotManager is read-only (tried to set {name!r})")


class OfficialEnv:
    """The environment of an AGP episode under official conditions (see the module docstring)."""

    def __init__(self, proxy, arms, *, task, collect_freq, step_lim=None):
        """step_lim: the episode's step limit (the adapter passes episode_metadata's official_step_lim); by default
        public_step_lim(task). It is never read from TASK_ENV."""
        self._proxy = proxy
        self.access_log = proxy.access_log
        self.task = task
        self.collect_freq = float(collect_freq)
        self.steps_sent = 0
        self._version = 0
        self._obs, self._obs_version = None, -1
        self._ended, self._end_version = None, -1
        self.observations_fetched = 0
        self.end_checks = 0
        self.step_lim = public_step_lim(task) if step_lim is None else int(step_lim)
        first = self.get_obs()
        frequency = float((first.get("additional_info") or {}).get("frequency") or 0.0)
        if frequency and abs(frequency - self.collect_freq) > 1e-9:
            raise AgentInfrastructureFailure(f"observation frequency {frequency} differs from the public "
                                             f"collect_freq {self.collect_freq}")
        self.first_observation_schema = observation_schema(first)
        self.obs_manager = SimpleNamespace(collect_freq=self.collect_freq, instruction=[str(first.get("instruction"))])
        self.robot_manager = ObservedRobotManager(self, arms)
        self.sim = SimpleNamespace(sim=_ControlClock(self))

    def _official(self, name, *args):
        try:
            attr = getattr(self._proxy, name)
            return attr(*args) if callable(attr) else attr
        except Exception as exc:
            if is_physx_fault(exc):
                raise SimulatorFault(exc) from exc
            raise

    # -- the official calls
    def get_obs(self):
        if self._obs_version != self._version:
            self._obs = self._official("get_obs")
            self._obs_version = self._version
            self.observations_fetched += 1
        return self._obs

    def take_action(self, action):
        # The official take_action ignores a call after the episode end without counting it; the worker always
        # checks is_episode_end() first, so such a call is a bug here and the own count must stay exact.
        if self.is_episode_end():
            raise RuntimeError("take_action after the official episode end")
        self._version += 1          # the state changes even if the call fails half-way
        result = self._official("take_action", action)
        self.steps_sent += 1
        return result

    def is_episode_end(self):
        if self._end_version != self._version:
            self._ended = bool(self._official("is_episode_end"))
            self._end_version = self._version
            self.end_checks += 1
        return self._ended

    # -- derived views
    @property
    def take_action_cnt(self):
        return [self.steps_sent]

    @property
    def end_flag(self):
        return [self.is_episode_end()]

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            log = object.__getattribute__(self, "access_log")
        except AttributeError:
            raise AttributeError(name) from None
        _refuse(log, "facade", name)


def observation_schema(obs):
    """Key structure of an observation (no values), to record which fields the official observation carries."""
    if isinstance(obs, dict):
        return {k: observation_schema(v) for k, v in sorted(obs.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obs, np.ndarray):
        return f"array{list(obs.shape)} {obs.dtype}"
    if isinstance(obs, (list, tuple)):
        return f"{type(obs).__name__}[{len(obs)}]"
    return type(obs).__name__


def depth_fields(schema, path=""):
    """Paths of any depth, calibration or intrinsic/extrinsic entries in an observation schema."""
    found = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            here = f"{path}.{key}" if path else str(key)
            if any(word in str(key).lower() for word in ("depth", "intrinsic", "extrinsic", "calib")):
                found.append(here)
            found += depth_fields(value, here)
    return found


def module_provenance(names=("env", "utils", "task", "src", "XPolicyLab", "client_server", "isaaclab", "curobo")):
    """Where the benchmark packages of this process were imported from (file or namespace portions)."""
    out = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            out[name] = None
            continue
        paths = list(getattr(module, "__path__", []) or [])
        out[name] = {"file": getattr(module, "__file__", None), "path": [str(p) for p in paths]}
    return out


BENCHMARK_PACKAGES = ("env", "utils", "task", "src", "XPolicyLab", "client_server")


def provenance_problems(provenance, checkout, forbidden):
    """(problems, notes) for where the benchmark was imported from.

    Problems: a benchmark package (env, utils, task, src, XPolicyLab, client_server) with any file or namespace
    portion inside the forbidden tree (the workspace's patched copy), or env / src / XPolicyLab without a portion
    inside the checkout. Notes: portions elsewhere, and third-party packages (isaaclab, curobo) outside the checkout
    (the documented fallback to the workspace's byte-identical copies)."""
    checkout, forbidden = os.path.realpath(checkout), os.path.realpath(forbidden)
    under = lambda path, root: path == root or path.startswith(root + os.sep)  # noqa: E731
    problems, notes = [], []
    for name, info in provenance.items():
        if info is None:
            continue
        places = [p for p in [info.get("file")] + info.get("path", []) if p]
        inside = [p for p in places if under(os.path.realpath(p), checkout)]
        for place in places:
            real = os.path.realpath(place)
            if name in BENCHMARK_PACKAGES and under(real, forbidden):
                problems.append(f"{name}: {place}")
            elif not under(real, checkout):
                notes.append(f"{name}: {place}")
        if name in ("env", "src", "XPolicyLab") and not inside:
            problems.append(f"{name}: not imported from the checkout ({places})")
    return problems, notes
