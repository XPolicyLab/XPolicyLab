"""Bootstrap truncated RoboDojo episodes from their terminal observations."""

from ...workers.rollout import XPolicyLabRolloutWorker


class RoboDojoRolloutWorker(XPolicyLabRolloutWorker):
    def _publish_env_part(self, env_part, policy_obs, next_rlt_obs, actor_channel):
        gamma = float(self.algorithm_cfg.gamma)

        class TimeoutBootstrapChannel:
            def put(self, part, **kwargs):
                return actor_channel.put(
                    bootstrap_robodojo_timeout(part, gamma), **kwargs
                )

        return super()._publish_env_part(
            env_part, policy_obs, next_rlt_obs, TimeoutBootstrapChannel()
        )


def bootstrap_robodojo_timeout(part, gamma):
    """v0.4 bootstraps only auto-reset envs. Dojo uses full-batch reset.

    Config validation requires episode timeouts at the final chunk of an epoch;
    EnvWorker then requests final inference. No extra model sampling is needed.
    True terminations never receive a value bonus.
    """
    from dataclasses import replace

    transition = part.transition
    truncated = transition.truncations
    if transition.rewards is None or truncated is None or not truncated.any():
        return part
    if part.bootstrap_values is None:
        raise RuntimeError(
            "RoboDojo timeout requires final-observation value inference."
        )
    rewards = transition.rewards.clone()
    mask = truncated.any(dim=1)
    indices = truncated.long().argmax(dim=1)
    rewards[mask, indices[mask]] += gamma * part.bootstrap_values[mask].reshape(-1).to(
        rewards
    )
    return replace(part, transition=replace(transition, rewards=rewards))
