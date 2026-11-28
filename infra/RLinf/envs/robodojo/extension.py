"""RoboDojo registration and validation of synchronous rollout settings."""

NAME = "robodojo"
ENV_ID = "RoboDojo-Stack-Bowls-XPolicyLab-v0"
ENV_KEY = ("isaaclab", ENV_ID)


def register():
    from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS
    from .env import RoboDojoEnv

    REGISTER_ISAACLAB_ENVS[ENV_ID] = RoboDojoEnv


def rollout_worker(cfg):
    if cfg.actor.model.get("add_value_head", False):
        from .rollout import RoboDojoRolloutWorker

        return RoboDojoRolloutWorker


def validate(cfg, model, env, mode):
    if (
        env.auto_reset
        or env.ignore_terminations
        or env.get("enable_offload", False)
        or cfg.env.get("enable_offload", False)
    ):
        raise ValueError(
            "RoboDojo requires synchronous reset, terminations and no simulator offload."
        )
    if env.video_cfg.save_video:
        raise ValueError("RoboDojo RL video wrapper is not supported yet.")

    if (
        env.max_steps_per_rollout_epoch != env.max_episode_steps
        or env.max_episode_steps % model.num_action_chunks
    ):
        raise ValueError(
            "RoboDojo timeout must equal the rollout epoch length and divide into complete action chunks."
        )
    if (
        env.get("action_type", "joint") not in ("joint", "ee")
        or model.xpolicylab.action_type != env.get("action_type", "joint")
        or model.xpolicylab.env_cfg_type != env.init_params.env_cfg_type
    ):
        raise ValueError(
            "RoboDojo requires matching policy/simulator action_type and env_cfg_type."
        )
    if env.group_size < 1 or env.total_num_envs % env.group_size:
        raise ValueError("RoboDojo total_num_envs must be a multiple of group_size.")
    if mode == "train":
        if cfg.algorithm.adv_type not in ("gae", "grpo") or cfg.reward.get(
            "use_reward_model", False
        ):
            raise ValueError(
                "RoboDojo requires GAE or group-relative advantages with simulator rewards."
            )
        if cfg.algorithm.adv_type == "gae" and not model.get("add_value_head", False):
            raise ValueError(
                "RoboDojo PPO requires add_value_head=true for timeout bootstrap."
            )
        if cfg.algorithm.reward_type != "chunk_level":
            raise ValueError("RoboDojo requires chunk-level rewards in RLinf.")
        if cfg.algorithm.loss_type == "embodied_nft":
            if (
                model.xpolicylab.required_mode != "nft"
                or env.get("action_type", "joint") != "ee"
                or cfg.algorithm.adv_type != "grpo"
            ):
                raise ValueError(
                    "RoboDojo NFT requires NFT mode, EE actions and group-relative advantages."
                )
        elif cfg.algorithm.loss_type not in ("actor", "actor_critic"):
            raise ValueError("RoboDojo RLT is not enabled in this adapter.")
