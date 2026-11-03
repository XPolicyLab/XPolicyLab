"""Read robot limits and control cadence without importing evaluation drivers."""
from pathlib import Path
from xml.etree import ElementTree
import os
import numpy as np
import yaml
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure, RoboDojoActionSpec
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.embodiment_docs import ARX_X5_JOINT_DOCS

_GRIPPER_EPS_PATTERN = "gripper_eps="
# Conservative control-tracking limits, shared by every instruction.
_TRACKED_ARM_STEP_RAD = 0.05
_TRACKED_GRIPPER_STEP = 0.25

def _gripper_eps(root: Path) -> float:
    text = (root / "env/robot_manager/control_manager.py").read_text(encoding="utf-8")
    for token in text.split():
        if token.startswith(_GRIPPER_EPS_PATTERN):
            return float(token.split("=", 1)[1].rstrip(",)"))
    raise RuntimeError("RoboDojo MetaControl gripper_eps was not found")


def _declared_step(span: float, per_step: float) -> float | None:
    """The per-step budget to declare for one arm joint, in radians.

    ``per_step`` is the actuator ceiling; what gets declared is the smaller of
    that and what the controller demonstrably tracks.
    """
    if not np.isfinite(per_step) or per_step <= 0:
        return None
    per_step = min(per_step, _TRACKED_ARM_STEP_RAD)
    if per_step >= span:
        return None
    return float(per_step)


def _gripper_step(span: float, per_step: float) -> float | None:
    if not np.isfinite(per_step) or per_step <= 0:
        return None
    return float(min(span, per_step, _TRACKED_GRIPPER_STEP))


def _arm_velocity_limit_sim(root: Path) -> float:
    text = (root / "env/robot_manager/robot_config/x5.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "velocity_limit_sim" in line:
            return float(line.split("=", 1)[1].strip().rstrip(","))
    raise RuntimeError("RoboDojo X5 velocity_limit_sim was not found")


def _positive_rate(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise InfrastructureFailure(
            f"RoboDojo reported {name}={value!r}; it must be finite and > 0. "
            "Check the observation section of the RoboDojo env config."
        )
    return float(value)


def actuator_command_limits(spec):
    """WAM setpoint budget from the actual simulator actuator and observation Hz.

    Native planner interpolation keeps its smaller tracking budget. This does
    not change joint bounds, execution feedback, checkpoint actions or cadence.
    """
    root = Path(os.environ['ROBODOJO_ROOT'])
    velocity = _positive_rate('velocity_limit_sim', _arm_velocity_limit_sim(root))
    rate = _positive_rate('collect_freq', spec.control_hz)
    limits = np.asarray(spec.max_step, dtype=float).copy()
    for start in (0, 7):
        limits[start:start+6] = np.minimum(
            spec.high[start:start+6] - spec.low[start:start+6], velocity / rate)
    return limits


def load_robot_spec() -> RoboDojoActionSpec:
    root_value = os.environ.get("ROBODOJO_ROOT")
    if not root_value:
        raise InfrastructureFailure(
            "ROBODOJO_ROOT is unset; the robot adapter reads the joint "
            "limits from the RoboDojo checkout."
        )
    root = Path(root_value)
    urdf = ElementTree.parse(root / "Assets/Robots/x5/X5A.urdf")
    arm_joints = []
    for joint in urdf.getroot().findall("joint"):
        name = joint.get("name", "")
        if name not in {f"joint{i}" for i in range(1, 7)}:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"URDF joint {name} has no limit")
        arm_joints.append(
            (name, float(limit.get("lower", "")), float(limit.get("upper", "")))
        )
    arm_joints.sort(key=lambda item: int(item[0].removeprefix("joint")))
    if len(arm_joints) != 6:
        raise RuntimeError(f"expected six X5 arm joints in URDF, got {arm_joints}")
    env_config = yaml.safe_load(
        (root / "env_cfg/arx_x5.yml").read_text(encoding="utf-8")
    )
    sim_config = yaml.safe_load(
        (root / "env_cfg/sim/sim_config.yml").read_text(encoding="utf-8")
    )
    control_hz = _positive_rate(
        "collect_freq", float(env_config["observation"]["collect_freq"])
    )
    collect_interval = _positive_rate(
        "collect_interval", 1.0 / (float(sim_config["dt"]) * control_hz)
    )
    arm_velocity = _arm_velocity_limit_sim(root)
    gripper_step = _gripper_step(1.0, _gripper_eps(root) * collect_interval)
    labels: list[str] = []
    low: list[float] = []
    high: list[float] = []
    max_step: list[float | None] = []
    for side in ("left", "right"):
        for name, lower, upper in arm_joints:
            labels.append(f"{side}_{name}")
            low.append(lower)
            high.append(upper)
            max_step.append(_declared_step(upper - lower, arm_velocity / control_hz))
        labels.append(f"{side}_gripper")
        low.append(0.0)
        high.append(1.0)
        max_step.append(gripper_step)
    return RoboDojoActionSpec(
        labels=tuple(labels),
        low=np.asarray(low, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        control_hz=control_hz,
        docs=ARX_X5_JOINT_DOCS,
        max_step=tuple(max_step),
    )
