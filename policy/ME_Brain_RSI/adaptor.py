"""The single coordinator for every ME Brain task and execution protocol."""
from .routing import configure
from .core.runtime import apply, prepare
from .core.modules import TaskModules


class ME_Brain_Adaptor:
    def __init__(self, config):
        self.config, self.route = configure(apply(config))
        self.scope = None
        if self.route['driver'] == 'vla':
            from .vla_model import Model
            self.core = Model(self.config)
            return
        self.config, self.run = prepare(self.config, self.route)
        self.scope = TaskModules(self.route['canonical_task'])
        driver = self.route['driver']
        if driver == 'step':
            from .backends.step import StepModel
            self.core = StepModel(self.config, self.route, self.scope)
        elif driver in ('native', 'equation'):
            from .backends.native import create
            self.core = create(self.config, self.route, self.scope, self.run)
        else:
            self._relocate_resources()
            self.core = self.scope.resolve(self.route['model'])(self.config)

    def _relocate_resources(self):
        from pathlib import Path
        import os
        root = self.config.get('perception_assets_root') or self.config.get('zyr_assets_root') or os.environ.get('ME_BRAIN_PERCEPTION_ASSETS') or os.environ.get('ME_BRAIN_ZYR_ASSETS')
        task = self.route['canonical_task']
        if root and task in ('general_pickup', 'pick_from_conveyor_by_image'):
            root = Path(root).expanduser().resolve(strict=True)
            module = self.scope.import_module('coap.perception' if task == 'general_pickup' else 'coap.vision')
            original = module.ASSETS
            if task == 'general_pickup':
                module.MODELS = {key: root / value.relative_to(original) for key, value in module.MODELS.items()}
            else:
                module.CLIP, module.DINO = root / module.CLIP.relative_to(original), root / module.DINO.relative_to(original)
            module.ASSETS = root

    def _check_batch(self, values):
        if values is not None and self.route['max_envs'] is not None and len(values) != self.route['max_envs']:
            raise ValueError('This strategy requires one environment per process')

    def update_obs(self, obs):
        method = getattr(self.core, 'update_obs', None)
        return method(obs) if callable(method) else self.core.update_obs_batch([obs])

    def update_obs_batch(self, obs):
        self._check_batch(obs)
        return self.core.update_obs_batch(obs)

    def get_action(self, **kwargs):
        method = getattr(self.core, 'get_action', None)
        return method(**kwargs) if callable(method) else self.core.get_action_batch([0], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        self._check_batch(env_idx_list)
        return self.core.get_action_batch(env_idx_list, **kwargs)

    def reset(self):
        return self.core.reset()

    def optional(self, name, *args):
        method = getattr(self.core, name, None)
        if callable(method):
            return method(*args)

    def runtime_info(self):
        return dict(task_name=self.route['task_name'], driver=self.route['driver'], action_type=self.route['action_type'], eval_batch=self.route['eval_batch'], env_cfg_type=self.config.get('env_cfg_type'), me_brain_backend=self.config.get('me_brain_backend', 'auto'))
