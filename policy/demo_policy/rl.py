# RL helpers. Model.predict_action_batch_rl and compute_logprob_value already call
# sample_actions_rl and get_log_prob_value. Fill _sample_policy and _log_prob_value.
# deploy.py does not import this module. RLinf calls those methods when required_mode
# is "ppo" or "grpo". Until RL_MODES includes that mode, eval keeps using get_action_batch.
#
# Put the trainable nn.Module on model.model or model.policy so RLinf can see its parameters.
# Log-probs have to come from the policy sample.


def sample_actions_rl(model, obs_list, *, mode="train", compute_values=True):
    # obs_list uses the same dicts as update_obs_batch.
    # mode "train" samples and records log-probs. mode "eval" is deterministic.
    # compute_values is the PPO value.
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}.")

    actions, prev_logprobs, prev_values, forward_inputs = _sample_policy(
        model, obs_list, mode=mode, compute_values=compute_values
    )
    return {
        "actions": actions,  # [B, H, D] in pack_robot_state order
        "prev_logprobs": prev_logprobs,  # required for PPO and GRPO
        "prev_values": prev_values,  # [B, 1] when compute_values, else None
        "forward_inputs": forward_inputs,  # tensors _log_prob_value needs, including action
    }


def get_log_prob_value(model, forward_inputs, compute_values=False):
    # Recompute the rollout sample. Gradients must reach model.model or model.policy.
    logprobs, values, entropy = _log_prob_value(
        model, forward_inputs, compute_values=compute_values
    )
    return {
        "logprobs": logprobs,  # same shape as prev_logprobs, float32
        "values": values,  # [B] when compute_values
        "entropy": entropy,  # [B, 1]
    }


def _sample_policy(model, obs_list, *, mode, compute_values):
    # Replace this function with the policy sample.
    raise NotImplementedError(
        f"Sample mode={mode!r} compute_values={compute_values} for {type(model).__name__} "
        f"over {len(obs_list)} observations. Return "
        "(actions [B, H, D], prev_logprobs, prev_values [B, 1] or None, forward_inputs)."
    )


def _log_prob_value(model, forward_inputs, compute_values=False):
    # Replace this function with the gradient recompute.
    raise NotImplementedError(
        f"Recompute compute_values={compute_values} for {type(model).__name__} "
        f"from keys {sorted(forward_inputs)}. Return (logprobs, values [B], entropy [B, 1])."
    )
