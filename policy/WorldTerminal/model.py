"""XPolicyLab adapter for the public worldterminal-sdk client."""
from typing import Any
from XPolicyLab.model_template import ModelTemplate
from worldterminal_sdk import Client, arx_x5


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        cfg = dict(model_cfg or {})
        self.robotwin = str(cfg.get("bench_name", "")).casefold() == "robotwin"
        profile = None
        if cfg.get("embodiment"):
            profile = arx_x5()
            if cfg["embodiment"] != profile["id"]:
                raise ValueError(f"Unknown embodiment: {cfg['embodiment']}")
        self.client = Client(cfg.get('api_url'), embodiment=profile, timeout=float(cfg.get('api_request_timeout_s') or 120),
                             connect_timeout=float(cfg.get('api_connect_seconds') or 30))

    def reset(self):
        return self.client.reset()

    def update_obs(self, obs):
        return self.client.update_observation(obs)

    def get_action(self):
        if self.robotwin:
            batch = self.client.poll()
            while batch.waiting:
                batch = self.client.poll()
            return list(batch.actions)
        return self.client.get_action()

    def on_trial_end(self, result=None):
        return self.client.end(result)

    def close(self):
        self.client.close()

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('one WorldTerminal client per environment; eval_batch=false')

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('one WorldTerminal client per environment; eval_batch=false')
