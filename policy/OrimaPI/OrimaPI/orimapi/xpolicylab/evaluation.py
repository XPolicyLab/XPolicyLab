# coding=utf-8
"""Execute one episode through the public XPolicyLab client."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def fast_trajectory_chunk_size() -> int:
    raw = os.environ.get("ORIMAPI_FAST_TRAJECTORY_CHUNKING", "").strip().lower()
    if not raw or raw in {"0", "false", "no", "off"}:
        return 0
    if raw in {"1", "true", "yes", "on"}:
        return 8
    value = int(raw)
    if value < 0 or value > 64:
        raise ValueError("ORIMAPI_FAST_TRAJECTORY_CHUNKING must be in 0..64")
    return 0 if value <= 1 else value


def _configure_request_timeout(env, client, timeout=None):
    if timeout is None:
        timeout = getattr(env, "deploy_cfg", {}).get("request_timeout_s")
    if timeout is None:
        return
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout_s must be finite and positive")
    # RoboDojo doesn't forward this deploy option to WsModelClient. Its protocol
    # client owns the mutable request configuration; no vendor files are changed.
    config = getattr(getattr(client, "_client", None), "config", None)
    if config is None or not hasattr(config, "request_timeout_s"):
        raise ValueError(
            "XPolicyLab client does not expose request timeout configuration"
        )
    config.request_timeout_s = float(timeout)


def eval_one_episode(TASK_ENV, model_client, *, request_timeout_s=None):
    """Reset the policy, then observe, request and execute actions until done."""
    env, client = TASK_ENV, model_client
    try:
        _configure_request_timeout(env, client, request_timeout_s)
        client.call(func_name="reset")
        if env.is_episode_end():
            return
        first = env.get_obs()
        index = int(first.get("env_idx", 0))
        case = {"env_idx": index}
        seeds = getattr(env, "env_seeds", None)
        if seeds is not None:
            case["layout_id"] = int(seeds[index])
        client.call(func_name="record_case", obs=case)
        client.call(func_name="update_obs", obs=first)
        executed = 0
        fast_chunking = fast_trajectory_chunk_size() > 1
        while not env.is_episode_end():
            actions = client.call(func_name="get_action")
            if (
                not isinstance(actions, list)
                or not actions
                or any(not isinstance(action, dict) or not action for action in actions)
            ):
                raise ValueError("Policy must return a nonempty action chunk")
            chunk_len = len(actions)
            for step_idx, action in enumerate(actions):
                env.take_action(action)
                executed += 1
                if env.is_episode_end():
                    # Scores come from the simulator's independent _result.json.
                    client.call(
                        func_name="on_trial_end",
                        obs={
                            "env_idx": index,
                            "status": "environment_terminated",
                            "executed_actions": executed,
                        },
                    )
                    return
                if not fast_chunking or step_idx == chunk_len - 1:
                    obs = env.get_obs()
                    if fast_chunking and chunk_len > 1:
                        obs = {**obs, "executed_chunk_steps": step_idx + 1}
                    client.call(func_name="update_obs", obs=obs)
    except BaseException as error:
        # Some simulator versions catch callback failures and still exit zero.
        # A sticky marker prevents those runs from being reported as completed.
        path = os.environ.get("ORIMAPI_EVALUATION_FAILURE")
        if path:
            with Path(path).open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"error": f"{type(error).__name__}: {error}"}) + "\n"
                )
        raise
