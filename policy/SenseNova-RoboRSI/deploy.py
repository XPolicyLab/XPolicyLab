"""Execute standard observations and one-action chunks through the policy server."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def eval_one_episode(TASK_ENV, model_client):
    """Preserve each action's feedback and close the owned agent after a trial."""
    try:
        model_client.call(func_name="reset")
        while not TASK_ENV.is_episode_end():
            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action")
            while actions == []:
                # Retry only inference: another observation would append video
                # frames, and an idle joint command would spend native steps.
                actions = model_client.call(func_name="get_action")
            if not isinstance(actions, list) or len(actions) != 1:
                raise ValueError(
                    "SenseNova-RoboRSI requires exactly one action per chunk"
                )
            TASK_ENV.take_action(actions[0])

        # The final action also needs feedback, even though no next action is due.
        model_client.call(func_name="end_episode", obs=TASK_ENV.get_obs())
    finally:
        if sys.exc_info()[0] is None:
            model_client.call(func_name="close")
        else:
            try:
                model_client.call(func_name="close")
            except Exception:
                # Keep the original environment or inference failure visible.
                logger.warning("Could not close the RoboRSI agent after a trial error")


def eval_one_episode_batch(TASK_ENV, model_client):
    """Reject unsupported parallel episodes before starting any agent."""
    raise NotImplementedError("SenseNova-RoboRSI requires eval_batch=false")
