"""RoboDojo's side of an episode: XPolicyLab delivers the observations through the :class:`Bridge`.

The robot calls are answered by :class:`hexashell.robot.host.RobotHost`; the task card is
:func:`hexashell.runtime.card.task_card`; the verdict is RoboDojo's, reported by the client.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..motion.ik import speed_named
from ..robot.env_icl import MEMORY_TASKS
from ..robot.host import RobotHost, describe_verdict
from ..runtime import card as cards
from ..runtime.session import Attach, Session
from ..state import CAMERAS, Snapshot, parse_obs
from .bridge import Bridge


class RoboDojoSession(Session):
    bench = "robodojo"
    proxy = "hexashell.robot.proxy"

    def __init__(self, bridge: Bridge, info: dict[str, Any]):
        super().__init__(info)
        self.env_icl = cards.base_task(self.info.get("task_name")) not in MEMORY_TASKS
        if self.env_icl:
            self.proxy = "hexashell.robot.env_icl"
        self.bridge = bridge
        self.host: RobotHost | None = None
        self.first: Snapshot | None = None
        self.rec = None

    # -- lifetime ------------------------------------------------------------------------------
    def open(self, settings, rec, attach: Attach) -> None:
        self.rec = rec
        seq, obs = self.bridge.latest()
        if obs is None:
            raise RuntimeError("no observation to start from")
        first = parse_obs(obs, seq)
        self._log_state(seq, obs, None)
        scene = self._scene(settings)
        self.host = RobotHost(self.bridge, first, scene=scene, log=rec.event, attach=attach,
                              on_step=self._log_state, speed=speed_named(settings.motion))
        self.first = first
        self.perception = scene is not None

    def _scene(self, settings):
        if not settings.sam3_url:
            return None
        try:
            from ..vision.scene import Scene
        except ImportError:
            return None
        try:
            return Scene(settings.sam3_url, log=self.rec.event)
        except Exception as exc:  # noqa: BLE001 - run without perception, and say so
            self.errors.append(f"perception unavailable: {exc}")
            self.rec.event({"kind": "perception_unavailable", "error": str(exc)})
            return None

    def conclude(self, timeout_s: float) -> dict | None:
        verdict = None
        if not self.bridge.over:
            self.bridge.finish()
            verdict = self.bridge.wait_verdict(timeout_s)
        return verdict or self.bridge.verdict

    # -- during the episode ----------------------------------------------------------------------
    @property
    def over(self) -> bool:
        return self.bridge.over

    @property
    def verdict(self) -> dict | None:
        return self.bridge.verdict

    def dispatch(self, name: str, args: tuple, kwargs: dict) -> Any:
        return self.host.dispatch(name, args, kwargs)

    def card(self, settings) -> tuple[str, dict]:
        return cards.task_card(self.first, with_ladder=settings.ladder, with_skill=settings.skills)

    def start_pictures(self) -> list[tuple[str, np.ndarray]]:
        return [(cam, self.first.images[cam]) for cam in CAMERAS if cam in self.first.images]

    @property
    def step(self) -> int | None:
        return self.host.snap.step

    @property
    def step_limit(self) -> int | None:
        return self.host.snap.step_limit

    @property
    def instruction(self) -> str | None:
        return self.host.snap.instruction if self.host else None

    def frames(self) -> dict[str, np.ndarray]:
        return self.host.snap.images

    def over_text(self) -> str:
        return f"The episode is over: {describe_verdict(self.bridge.verdict)}."

    def result_fields(self, verdict: dict) -> dict[str, Any]:
        return {"task": self.info.get("task_name"), "seed": self.info.get("eval_seed"),
                "layout": self.info.get("layout"), "instruction": self.instruction}

    # -- records ---------------------------------------------------------------------------------
    def _log_state(self, seq: int, obs: dict[str, Any], action: dict | None) -> None:
        state = obs.get("state") or {}
        info = ((obs.get("additional_info") or {}).get("hexashell")) or {}
        record = {"seq": seq, "step": info.get("step")}
        for key, value in state.items():
            if key.endswith(("_arm_joint_state", "_ee_pose", "_ee_joint_state")):
                record[key] = [round(float(v), 5) for v in np.asarray(value, float).reshape(-1)]
        if action is not None:
            record["action"] = {k: [round(float(v), 5) for v in vals] for k, vals in action.items()}
        self.rec.state(record)
