"""Bounded tool compositions, checked against a fresh observation at every stage."""

from copy import deepcopy
import json
import math

import numpy as np
from .geometry import relative_pose, vector, unit

PLANNERS = {"plan_pose", "plan_relative_pose"}
ALLOWED = PLANNERS | {"gripper", "wait", "verify", "verify_attachment"}


def shape(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise ValueError("Invalid sequence tool argument fields")


def identifier(value):
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("Sequence references must be bounded identifiers")


def validate_arguments(name, args):
    if name in PLANNERS or name == "gripper":
        if args.get("arm") not in ("left", "right"):
            raise ValueError("Sequence arm must be left or right")
    if name == "plan_relative_pose":
        shape(
            args, ("arm", "translation", "rotation_vector", "pivot_local"), ("steps",)
        )
        relative_pose(
            np.eye(4), args["translation"], args["rotation_vector"], args["pivot_local"]
        )
    elif name == "plan_pose":
        shape(args, ("arm", "constraints"), ("pivot_local", "steps"))
        if np.linalg.norm(vector(args.get("pivot_local", [0, 0, 0]))) > 0.3:
            raise ValueError("Sequence rotation pivot exceeds 0.3 m")
        if (
            not isinstance(args["constraints"], list)
            or not 1 <= len(args["constraints"]) <= 8
        ):
            raise ValueError("Use 1..8 pose constraints")
        for row in args["constraints"]:
            if not isinstance(row, dict):
                raise ValueError("Constraint must be an object")
            kind = row.get("kind")
            if kind == "point":
                shape(row, ("kind", "local", "target"), ("offset",))
                identifier(row["target"])
                vector(row.get("offset", [0, 0, 0]))
            elif kind == "axis":
                if "direction" in row:
                    shape(row, ("kind", "local", "direction"))
                    unit(row["direction"])
                else:
                    shape(row, ("kind", "local", "from", "to"))
                    identifier(row["from"])
                    identifier(row["to"])
                unit(row["local"])
            elif kind == "plane":
                shape(row, ("kind", "local", "points"))
                if not isinstance(row["points"], list) or len(row["points"]) != 3:
                    raise ValueError("Plane constraint requires three point IDs")
                for point in row["points"]:
                    identifier(point)
            else:
                raise ValueError("Unknown sequence pose constraint")
            vector(row["local"])
    elif name in {"wait", "gripper"}:
        shape(args, ("steps",) if name == "wait" else ("arm", "value", "steps"))
        minimum = 1
        if name == "gripper":
            if type(args["value"]) not in (float, int) or not 0 <= args["value"] <= 1:
                raise ValueError("Gripper value must be 0..1")
            minimum = 8 if args["value"] < 0.8 else 6
        if type(args["steps"]) is not int or not minimum <= args["steps"] <= 120:
            raise ValueError("Sequence stage steps outside bounded tool contract")
    elif name == "verify":
        shape(args, ("first", "second", "tolerance_m"))
        identifier(args["first"])
        identifier(args["second"])
        tolerance = args["tolerance_m"]
        if (
            type(tolerance) not in (float, int)
            or not math.isfinite(tolerance)
            or not 0.001 <= tolerance <= 0.1
        ):
            raise ValueError("Invalid verification tolerance")
    elif name == "verify_attachment":
        shape(args, ("attachment_id",))
        identifier(args["attachment_id"])
    if (
        name in PLANNERS
        and "steps" in args
        and (type(args["steps"]) is not int or not 1 <= args["steps"] <= 120)
    ):
        raise ValueError("Motion steps must be an integer in1..120")


def validate_sequence(value):
    if not isinstance(value, dict) or set(value) != {"stages", "reserve_actions"}:
        raise ValueError("Sequence requires stages and reserve_actions")
    rows, reserve = value["stages"], value["reserve_actions"]
    if type(reserve) is not int or not 0 <= reserve <= 10000:
        raise ValueError("reserve_actions must be a nonnegative bounded integer")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError("Use 1..8 sequence stages")
    if len(json.dumps(value, allow_nan=False)) > 24000:
        raise ValueError("Sequence exceeds its size limit")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != {"tool", "arguments"}
            or not isinstance(row["tool"], str)
            or row["tool"] not in ALLOWED
            or not isinstance(row["arguments"], dict)
        ):
            raise ValueError("Sequence contains an unsupported tool or argument shape")
        if row["tool"] == "gripper" and index != len(rows) - 1:
            raise ValueError(
                "Gripper changes must end the sequence for a fresh visual review"
            )
        validate_arguments(row["tool"], row["arguments"])
    return deepcopy(value)


class CheckedSequence:
    def __init__(self, value):
        data = validate_sequence(value)
        self.stages, self.reserve = data["stages"], data["reserve_actions"]
        self.index, self.waiting, self.results = 0, False, []

    def _finish(self, session, status, error=None):
        result = {
            "status": status,
            "error": error,
            "stages": deepcopy(self.results),
            "completed_stages": self.index,
            "reserve_actions": self.reserve,
            "official_success": None,
            "scope": "Checked robot motion only, not grasp or task success",
        }
        session.sequence = None
        session.last_result = result
        session.emit("sequence_finished", step=session.observation.step, **result)
        return None

    def action(self, session):
        while self.index < len(self.stages):
            row = self.stages[self.index]
            try:
                if self.waiting:
                    action = session._motion_action()
                    if action is not None:
                        return action
                    result = deepcopy(session.last_result)
                    self.results.append(result)
                    self.waiting = False
                    if result["status"] != "completed" or (
                        row["tool"] in PLANNERS
                        and result["verification"]["status"] != "pass"
                    ):
                        return self._finish(
                            session,
                            "failed",
                            result.get("error") or "TCP verification failed",
                        )
                    self.index += 1
                    continue
                if session.remaining is not None and session.remaining <= self.reserve:
                    return self._finish(
                        session, "failed", "Sequence recovery reserve reached"
                    )
                result = session._call(row["tool"], deepcopy(row["arguments"]))
                if row["tool"] in PLANNERS:
                    result = session._call(
                        "execute_plan", {"plan_id": result["plan_id"]}
                    )
                session.emit(
                    "sequence_stage",
                    step=session.observation.step,
                    index=self.index,
                    request=row,
                    result=result,
                )
                if session.active is not None:
                    self.waiting = True
                    continue
                self.results.append(result)
                if result.get("status") not in {"pass", "visually_consistent"}:
                    return self._finish(
                        session,
                        "failed",
                        "Verification is failed, estimated or unknown",
                    )
                self.index += 1
            except (ValueError, TypeError, KeyError) as error:
                if session.active is not None:
                    session.motion.failure = str(error)
                    session._motion_action()
                return self._finish(session, "failed", str(error))
        status = (
            "needs_visual_review"
            if self.stages[-1]["tool"] == "gripper"
            else "completed"
        )
        return self._finish(session, status)
