"""Keep native Lxm6wuec cores in the ordinary policy server process."""
import os


class Model:
    def __init__(self, config):
        self.core = None
        if os.environ.get('B88R42WQ_LXM6WUEC_NATIVE') == '1':
            from b88r42wq.lxm6wuec_native import create_original_model
            self.core, self.native_metadata = create_original_model(os.environ['B88R42WQ_TASK'])

    def reset(self):
        if self.core is not None:
            return self.core.reset()

    def update_obs_batch(self, observations):
        return self.core.update_obs_batch(observations)

    def get_action_batch(self, indices=None):
        return self.core.get_action_batch(indices)

    def update_obs(self, observation):
        return self.core.update_obs(observation)

    def get_action(self):
        return self.core.get_action()

    def prepare_case(self, case_meta=None):
        method = getattr(self.core, 'prepare_case', None)
        if callable(method):
            return method(case_meta)

    def on_trial_end(self, result=None):
        method = getattr(self.core, 'on_trial_end', None)
        if callable(method):
            return method(result)

    def close(self):
        method = getattr(self.core, 'close', None)
        if callable(method):
            return method()
