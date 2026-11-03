"""Official environment loop; reward/scoring/physics remain evaluator-owned."""


class AdapterFailure(BaseException):
    """Abort the run: native main must not retry policy failures on a new layout."""


def eval_one_episode(TASK_ENV, model_client):
    try:
        _eval_one_episode(TASK_ENV, model_client)
    except Exception as exc:
        raise AdapterFailure(str(exc)) from exc


def _eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    model_client.call(func_name="prepare_case", obs={"native_step_limit": int(TASK_ENV.step_lim)})
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = []
        while not actions:
            actions = model_client.call(func_name="get_action")
        if len(actions) != 1:
            raise ValueError("EmbodiedEvo requires a one-action chunk for closed-loop Python")
        TASK_ENV.take_action(actions[0])
    # Only binary completion and termination reach the policy, never shaped
    # reward, score, layout id or object state. These are standard WS hooks.
    model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs_batch([0], last_frame=True)[0])
    success = bool(TASK_ENV.end_flag[0] and TASK_ENV.success[0])
    truncated = not success and TASK_ENV.take_action_cnt[0] >= TASK_ENV.step_lim
    model_client.call(func_name="trial_end", obs={
        "success": success, "terminated": not truncated, "truncated": truncated,
    })


def eval_one_episode_batch(TASK_ENV, model_client):
    raise ValueError("Run this policy with num_envs=1 and eval_batch=false")


def playground_one_session(env, client):
    """Keep one policy worker alive across explicitly requested native resets."""
    client.call("reset")
    client.call("prepare_case", obs={"native_step_limit": int(env.step_lim)})
    initial_instruction = None
    terminal_sent = False

    def initialize_reward():
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()
        if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
            env.query_support_arm_traj(env_idx=0)

    initialize_reward()
    while True:
        ended = env.is_episode_end()
        obs = env.get_obs_batch([0], last_frame=True)[0] if ended else env.get_obs()
        if initial_instruction is None:
            initial_instruction = obs["instruction"]
        obs["instruction"] = initial_instruction
        client.call("update_obs", obs=obs)
        if ended and not terminal_sent:
            success = bool(env.end_flag[0] and env.success[0])
            truncated = not success and env.take_action_cnt[0] >= env.step_lim
            client.call("trial_end", obs={
                "success": success, "terminated": not truncated, "truncated": truncated,
            })
            terminal_sent = True
        while True:
            actions = client.call("get_action")
            control = client.call("get_control")
            if control["finished"]:
                return
            if control["reset"]:
                env.reset(seed=list(env.env_seeds))
                if env.end_flag[0]:
                    raise AdapterFailure("Playground reset failed native stability checks")
                env.obs_manager.instruction[0] = initial_instruction
                initialize_reward()
                reset_obs = env.get_obs()
                reset_obs["instruction"] = initial_instruction
                client.call("complete_reset", obs=reset_obs)
                terminal_sent = False
                break
            if actions:
                if ended or len(actions) != 1:
                    raise AdapterFailure("Invalid action after native termination")
                env.take_action(actions[0])
                break
