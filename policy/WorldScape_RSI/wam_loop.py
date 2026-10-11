"""Single-environment explicit-commit WAM loop; no second WAM websocket service."""
import time


def execute(env, client, config, capture, output):
    deadline = time.monotonic() + float(config.get("episode_budget_s", 9000))
    blocks = steps = 0
    while not env.is_episode_end():
        if time.monotonic() >= deadline:
            return {"termination_reason": "wall_budget", "decisions": blocks,
                    "wam_steps_executed": steps}
        client.call(func_name="wam_update_obs", obs=env.get_obs())
        execution_started = False
        try:
            actions = client.call(func_name="wam_get_action")
            if not actions:
                raise ValueError("WAM returned an empty action chunk")
            blocks += 1
            capture.decisions = blocks
            executed = 0
            for index, action in enumerate(actions):
                # A raised take_action can still have changed the real environment.
                execution_started = True
                env.take_action(action)
                executed += 1
                steps += 1
                finished = bool(env.is_episode_end())
                if finished or index + 1 == len(actions):
                    break
                client.call(func_name="wam_update_obs", obs=env.get_obs())
            client.call(func_name="wam_commit_action", obs={
                "executed_steps": {0: executed},
                "finished_env_idx_list": [0] if finished else [],
            })
        except BaseException as error:
            try:
                client.call(func_name="wam_discard_action", obs={"execution_started": execution_started})
            except BaseException as cleanup:
                if hasattr(error, "add_note"):
                    error.add_note("WAM discard failed: " + repr(cleanup))
            raise
    return {"termination_reason": "native", "decisions": blocks,
            "wam_steps_executed": steps}
