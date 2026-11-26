"""KnowinAgent backend over the official evaluator's environment (XPolicyLab ``deploy.eval_one_episode``).

The official RoboDojo client builds, resets and scores the episode; it hands its ``TASK_ENV`` to the policy.
This backend only reads ``get_obs()`` (official ``arx_x5.yml`` observation: RGB, arm state, instruction),
calls ``take_action`` and reads the native action counter and end flag. Camera calibration comes from the rig
constants. It never reads the reward manager, the scene or layout, the simulator cameras or the success flag,
never resets and never closes the environment or the app, which belong to the client.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from knowin_agent.backends.robodojo.env import RoboDojoBackend
from knowin_agent.runtime.backend import ArmCommand, StepOutcome


class OfficialRoboDojoBackend(RoboDojoBackend):
    supports_reset = False

    @classmethod
    def attach(cls, task_env: Any, task: Optional[str] = None, *, frame_sink=None) -> "OfficialRoboDojoBackend":
        name = task or getattr(task_env, "task_name", None) or "unknown"
        return cls.from_native(task_env, None, task=name, seed=0, depth="none", calibration="constant",
                               official_action_limit=True, frame_sink=frame_sink)

    def step(self, commands: Mapping[str, ArmCommand]) -> StepOutcome:
        if self._episode_over:
            raise RuntimeError("the official episode is over")
        action = self.native_action(commands)
        native = self.native
        for arm in self.profile.arms:
            self._grip_commands[arm] = float(commands[arm].gripper)
        before = int(native.take_action_cnt[0])
        native.take_action(action)
        after = int(native.take_action_cnt[0])
        if after != before + 1:
            raise RuntimeError(f"native take_action advanced {after - before} ticks instead of 1")
        self._last = self._fetch()
        if bool(native.end_flag[0]):
            return self._finish("action_limit" if after >= native.step_lim else "task_ended")
        self._emit(self._last)
        return StepOutcome(self._last)

    def hold_until_end(self) -> int:
        """Hold both arms still (current joints, last gripper commands) until the evaluator ends the episode.

        The official client counts an episode that the policy leaves before the evaluator ended it as a
        success (``success`` starts True and only the action limit clears it), so an agent that finishes early
        must not return: it runs the remaining actions out and the evaluator scores the tiers it reached."""
        held = 0
        while not self._episode_over:
            state = self._last.arms
            self.step({arm: ArmCommand(None, self._grip_commands.get(arm, s.gripper_command), joints=s.joints)
                       for arm, s in state.items()})
            held += 1
        return held

    def reset(self):
        raise RuntimeError("the official evaluator owns the episode; it cannot be reset by the policy")

    def non_official_conditions(self) -> list[str]:
        return []

    def evaluation(self) -> dict:
        """What the policy side may know about its own episode: no score, no success, no layout."""
        return {"task": self.task, "official": True, "episode_over": self._episode_over, "reason": self._reason,
                "error": self._error, "budget": self.budget().as_dict(), "non_official_conditions": []}

    def ground_truth(self) -> Optional[dict]:
        return None

    def _attempt_score(self) -> Optional[float]:
        return None

    def close(self) -> None:
        """The client owns the environment and the app: nothing to close (``hold_until_end`` may still run)."""
