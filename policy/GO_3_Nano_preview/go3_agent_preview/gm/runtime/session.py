
from __future__ import annotations

import json
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from XPolicyLab.utils.process_data import encode_image_bit
from ..lib.kinematics import chain as kin
from .switch_sources import SWITCH_SOURCES

ARMS = ("left", "right")

class SwitchReason(str, Enum):
    """Why a program stopped driving the episode.

    Every value is a statement about the program, not about the outcome: the
    controller records it and the learned branch takes over regardless.
    """

    NO_CODE_PLAN = "no_code_plan"
    CODE_ABORTED = "code_aborted"
    PLAN_EXHAUSTED = "plan_exhausted"
    RETRIES_EXHAUSTED = "retries_exhausted"
    CONTROL_BUDGET_EXHAUSTED = "control_budget_exhausted"




@dataclass(frozen=True)
class SwitchRequest:
    reason: SwitchReason
    detail: str
    source: str
    sequence: int

    def record(self):
        return {**asdict(self), "reason": self.reason.value}

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("skill switch requires exact typed request fields")
        if (not isinstance(data["source"],str) or not data["source"]
                or not isinstance(data["detail"],str) or not data["detail"]
                or type(data["sequence"]) is not int or data["sequence"] < 1):
            raise ValueError("invalid switch request identity")
        return cls(SwitchReason(data["reason"]), data["detail"], data["source"], data["sequence"])


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return np.round(x, 5).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x

class Ctx:
    def __init__(self, task, out_dir: Path | None):
        self.task = task
        self.out = out_dir
        self.obs = None
        self.step = 0
        self.cmd_q = {}
        self.cmd_g = {a: 1.0 for a in ARMS}
        self.home_q = {}
        self.memory = {}
        self.done = False
        self.failure = None
        self._tape = None
        self.online_events = deque(maxlen=256)
        self.event_sequence = 0
        self.stage = None
        self.switch_request = None
        self.switch_sequence = 0
        if out_dir is not None:
            (out_dir / "frames").mkdir(parents=True, exist_ok=True)
            self._tape = (out_dir / "tape.jsonl").open("a", buffering=1)

    def q(self, arm):
        return np.asarray(self.obs["state"][f"{arm}_arm_joint_state"], float)

    def g(self, arm):
        return float(np.asarray(self.obs["state"][f"{arm}_ee_joint_state"], float).reshape(-1)[0])

    def ee(self, arm):
        return np.asarray(self.obs["state"][f"{arm}_ee_pose"], float)

    def rgb(self, cam="cam_head"):
        return np.asarray(self.obs["vision"][cam]["color"], np.uint8)

    @property
    def instruction(self):
        return self.obs.get("instruction", "")

    def switch_to_wam(self, reason: SwitchReason, *, detail: str, source: str):
        """Yield this checkpoint BEFORE opening grippers or returning home.

        A request records code abandonment, not a visually established failure.
        A pure-code validation Session can consume the marker without a control step.
        """
        if not isinstance(reason, SwitchReason):
            raise TypeError("switch_to_wam reason must be SwitchReason")
        if source not in SWITCH_SOURCES.get(str(self.task).removesuffix("_random"), set()):
            raise ValueError("switch request source is not registered for this task")
        self.switch_sequence += 1
        request = SwitchRequest(reason, detail, source, self.switch_sequence)
        SwitchRequest.parse(request.record())
        self.switch_request = request
        self.log("skill_switch_requested", **request.record())
        return request

    def action(self):
        a = {}
        for arm in ARMS:
            a[f"{arm}_arm_joint_state"] = np.asarray(self.cmd_q[arm], np.float32)
            a[f"{arm}_ee_joint_state"] = np.array([self.cmd_g[arm]], np.float32)
        return a

    def log(self, kind, /, **data):
        self.event_sequence += 1
        rec = {"step": self.step, "t": round(time.time(), 3), **_jsonable(data),
               "kind": kind, "sequence": self.event_sequence}
        self.online_events.append(rec)
        if kind != "tracking":
            self.stage = str(data.get("stage", kind))
        if self._tape is not None:
            self._tape.write(json.dumps(rec) + "\n")

    def snap(self, tag, cams=("cam_head",)):
        """Save the current frame of each camera as a recorded buffer.

        The array is already RGB. It goes through the shared encoder so the file
        records its own channel order; writing it with a hand-rolled encoder would
        mint a buffer that only reads back correctly by accident.
        """
        if self.out is None or self.obs is None:
            return
        for cam in cams:
            if cam in self.obs["vision"]:
                path = self.out / "frames" / f"{self.step:04d}_{tag}_{cam}.jpg"
                path.write_bytes(encode_image_bit(self.rgb(cam)))

    def close(self):
        if self._tape is not None:
            self._tape.close()
            self._tape = None

class Session:
    def __init__(self, task, program, out_dir: Path | None, *, respect_switch=True):
        self.ctx = Ctx(task, out_dir)
        self.program = program
        self.gen = None
        self.finished = False
        self.respect_switch = bool(respect_switch)
        self.switch_paused = False

    def observe(self, obs):
        ctx = self.ctx
        ctx.obs = obs
        if self.gen is None:
            for arm in ARMS:
                ctx.cmd_q[arm] = ctx.q(arm).copy()
                ctx.home_q[arm] = ctx.q(arm).copy()
            ctx.log("code", program=getattr(self.program, "__module__", "").rsplit(".", 1)[-1])
            ctx.log("start", instruction=ctx.instruction,
                    q={a: ctx.q(a) for a in ARMS}, ee={a: ctx.ee(a) for a in ARMS},
                    fk={a: kin.fk(a, ctx.q(a))[:3, 3] for a in ARMS})
            ctx.snap("start", cams=("cam_head", "cam_left_wrist", "cam_right_wrist"))
            self.gen = self.program(ctx)

    def _watch_tracking(self):
        ctx = self.ctx
        if ctx.step % 5:
            return
        for arm in ARMS:
            err = float(np.max(np.abs(ctx.q(arm) - ctx.cmd_q[arm])))
            if err > 0.15:
                ctx.log("tracking", arm=arm, err=round(err, 3))

    def next_action(self):
        ctx = self.ctx
        if ctx.obs is not None and ctx.cmd_q:
            self._watch_tracking()
        if not self.finished:
            try:
                if not self.switch_paused:
                    while True:
                        yielded = next(self.gen)
                        if not isinstance(yielded, SwitchRequest):
                            break
                        SwitchRequest.parse(yielded.record())
                        if (yielded.source not in SWITCH_SOURCES.get(str(ctx.task).removesuffix("_random"), set())
                                or yielded != ctx.switch_request):
                            raise ValueError("switch marker must be emitted by this task context")
                        if self.respect_switch:
                            self.switch_paused = True
                            break
                        # No action/control-step overhead in pure-code comparison.
                if self.switch_paused:
                    for arm in ARMS:
                        ctx.cmd_q[arm] = ctx.q(arm).copy()
                        ctx.cmd_g[arm] = ctx.g(arm)
            except StopIteration:
                self.finished = True
                ctx.done = True
                ctx.log("program_end", failure=ctx.failure)
                ctx.snap("end", cams=("cam_head",))
            except Exception as exc:
                self.finished = True
                ctx.done = True
                ctx.failure = f"exception: {exc!r}"
                ctx.log("program_exception", error=traceback.format_exc())
        ctx.step += 1
        return ctx.action()
