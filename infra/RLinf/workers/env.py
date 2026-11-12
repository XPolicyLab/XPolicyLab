"""Explicit cleanup for simulator processes owned by synchronous env workers."""

from rlinf.workers.env.env_worker import EnvWorker


class XPolicyLabEnvWorker(EnvWorker):
    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        from ..core.extensions import policy_extension, simulator_extension

        for extension in (simulator_extension(env_cfg), policy_extension(self.model_cfg)):
            if hasattr(extension, "env_class"):
                env_cls = extension.env_class(env_cfg) or env_cls
        return super()._setup_env_and_wrappers(env_cls, env_cfg, num_envs_per_stage)

    def close_environments(self):
        errors = []
        seen = set()
        for name in ("env_list", "eval_env_list"):
            for env in getattr(self, name, []):
                if id(env) in seen:
                    continue
                seen.add(id(env))
                try:
                    env.close()
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            raise RuntimeError("Environment cleanup failed: " + "; ".join(errors))
