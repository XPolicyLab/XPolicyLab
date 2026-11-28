"""Rollout worker that passes train/eval mode to XPolicyLab policies."""

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class XPolicyLabRolloutWorker(MultiStepRolloutWorker):
    """Passes rollout mode to externally registered XPolicyLab policies.

    Upstream ``predict()`` only injects ``mode`` for a hardcoded model list.
    ``setup_sample_params()`` runs at the end of ``init_worker``, so the mode
    flag is added after that call.
    """

    def init_worker(self):
        super().init_worker()
        self._train_sampling_params = {**self._train_sampling_params, "mode": "train"}
        self._eval_sampling_params = {**self._eval_sampling_params, "mode": "eval"}


def rollout_worker(cfg):
    from ..core.extensions import simulator_extension

    if not cfg.runner.only_eval:
        extension = simulator_extension(cfg.env.train)
        if hasattr(extension, "rollout_worker"):
            return extension.rollout_worker(cfg) or XPolicyLabRolloutWorker
    return XPolicyLabRolloutWorker
