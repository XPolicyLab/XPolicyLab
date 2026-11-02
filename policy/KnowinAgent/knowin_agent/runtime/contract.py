"""Interaction contract: what the Local Interaction Agent may change.

The Task-Level Agent opens a contract through `interact`. It fixes the goal and
the arm roles; Local may only use the permitted actions within the limits, and
must return to Task when progress needs a change outside the contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from knowin_agent.runtime.backend import ARMS

ROLES = {
    "actor": "manipulates within the goal (moves, rotates, advances, gripper if permitted)",
    "observer": "moves only to acquire views (observe); never touches the scene",
    "holder": "keeps its grasp and pose; may not move or open",
    "idle": "does nothing",
}
# A described role ("idle at saved origin", "holding the cup") is read by its first keyword: 177 of 247 failed
# interact calls in the 10-30/31 campaigns were refused only for such a wording.
ROLE_WORDS = (("hold", "holder"), ("observ", "observer"), ("watch", "observer"), ("camera", "observer"),
              ("view", "observer"), ("act", "actor"), ("work", "actor"), ("manipulat", "actor"),
              ("idle", "idle"), ("rest", "idle"), ("park", "idle"), ("home", "idle"), ("origin", "idle"),
              ("none", "idle"), ("stay", "idle"))
LOCAL_ACTIONS = ("shift", "rotate", "advance", "gripper", "observe", "knob_cycle")
MOVING_ACTIONS = ("shift", "rotate", "advance", "knob_cycle")


class ContractViolation(PermissionError):
    pass


def _role(role) -> str:
    text = str(role).strip().lower()
    if text in ROLES:
        return text
    hits = [(text.find(word), name) for word, name in ROLE_WORDS if word in text]
    return min(hits)[1] if hits else text


@dataclass
class InteractionContract:
    goal: str
    completion: str
    arm_roles: dict
    permitted: list
    reference_view: Optional[str] = None
    advance_direction: Optional[list] = None
    max_step_translation_m: float = 0.03
    max_total_translation_m: float = 0.12
    max_step_rotation_deg: float = 30.0
    max_rounds: int = 16
    release_allowed: bool = False
    notes: str = ""
    anchors: dict = field(default_factory=dict)

    @staticmethod
    def from_args(args: dict) -> "InteractionContract":
        roles = {arm: _role(role) for arm, role in dict(args.get("arm_roles") or {}).items()}
        for arm in ARMS:
            roles.setdefault(arm, "idle")
        unknown = {a: r for a, r in roles.items() if a not in ARMS or r not in ROLES}
        if unknown:
            raise ValueError(f"arm_roles must map left/right to one of {sorted(ROLES)}; got {unknown}")
        if not any(role in ("actor", "observer") for role in roles.values()):
            raise ValueError("a contract needs at least one actor or observer arm")
        permitted = list(args.get("permitted") or ["shift", "rotate", "advance", "observe"])
        bad = [p for p in permitted if p not in LOCAL_ACTIONS]
        if bad:
            raise ValueError(f"permitted actions must be among {LOCAL_ACTIONS}; got {bad}")
        direction = args.get("advance_direction")
        if direction is not None:
            direction = np.asarray(direction, dtype=float)
            if direction.shape != (3,) or np.linalg.norm(direction) < 1e-6:
                raise ValueError("advance_direction must be a non-zero [x, y, z]")
            direction = (direction / np.linalg.norm(direction)).round(4).tolist()
        contract = InteractionContract(
            goal=str(args.get("goal") or "").strip(),
            completion=str(args.get("completion") or "").strip(),
            arm_roles=roles,
            permitted=permitted,
            reference_view=args.get("reference_view"),
            advance_direction=direction,
            release_allowed=bool(args.get("release_allowed", False)),
            notes=str(args.get("notes") or ""),
        )
        for key in ("max_step_translation_m", "max_total_translation_m", "max_step_rotation_deg"):
            if args.get(key) is not None:
                value = float(args[key])
                if key == "max_step_rotation_deg" and value == 0:
                    # "no rotation" (68 refused calls): drop rotate instead of refusing the contract
                    contract.permitted = [a for a in contract.permitted if a != "rotate"]
                    continue
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be positive")
                setattr(contract, key, value)
        if args.get("max_rounds") is not None:
            contract.max_rounds = max(1, min(40, int(args["max_rounds"])))
        if not contract.goal or not contract.completion:
            raise ValueError("a contract needs a goal and a completion condition")
        return contract

    def open(self, flange_positions: dict) -> None:
        self.anchors = {arm: [float(v) for v in xyz] for arm, xyz in flange_positions.items()}

    def describe(self) -> dict:
        out = asdict(self)
        out.pop("anchors", None)
        return out

    def authorize(self, action: str, arm: Optional[str], *, translation_m: float = 0.0,
                  rotation_deg: float = 0.0, new_flange_xyz=None, direction=None,
                  opening_gripper: bool = False) -> None:
        if action not in self.permitted:
            raise ContractViolation(f"{action} is not permitted by this contract (permitted: {self.permitted}); "
                                    "return_to_task if it is needed")
        if arm is None:
            return
        role = self.arm_roles.get(arm, "idle")
        if action == "observe":
            if role not in ("observer", "actor"):
                raise ContractViolation(f"{arm} arm is '{role}'; only observer or actor arms may move to observe")
            return
        if role != "actor":
            raise ContractViolation(f"{arm} arm is '{role}' in this contract; {action} needs an actor arm")
        if action == "gripper" and opening_gripper and not self.release_allowed:
            raise ContractViolation("opening the gripper (release) is not allowed by this contract")
        if translation_m > self.max_step_translation_m + 1e-9:
            raise ContractViolation(f"step of {translation_m * 1000:.1f} mm exceeds the "
                                    f"{self.max_step_translation_m * 1000:.0f} mm per-call limit")
        if rotation_deg > self.max_step_rotation_deg + 1e-9:
            raise ContractViolation(f"rotation of {rotation_deg:.1f} deg exceeds the "
                                    f"{self.max_step_rotation_deg:.0f} deg per-call limit")
        if new_flange_xyz is not None and arm in self.anchors:
            drift = float(np.linalg.norm(np.asarray(new_flange_xyz) - np.asarray(self.anchors[arm])))
            if drift > self.max_total_translation_m + 1e-9:
                raise ContractViolation(f"{arm} arm would end {drift * 1000:.0f} mm from where the contract started "
                                        f"(limit {self.max_total_translation_m * 1000:.0f} mm); return_to_task to re-plan")
        if action == "advance" and self.advance_direction is not None and direction is not None:
            d = np.asarray(direction, dtype=float)
            d = d / max(np.linalg.norm(d), 1e-9)
            cos = float(np.dot(d, self.advance_direction))
            if cos < math.cos(math.radians(30)):
                raise ContractViolation("advance must stay within 30 deg of the contract's advance_direction "
                                        f"{self.advance_direction}")
