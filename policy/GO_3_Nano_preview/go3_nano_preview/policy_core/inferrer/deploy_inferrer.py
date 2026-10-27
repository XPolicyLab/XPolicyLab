import logging
from collections.abc import Mapping
import torch

logger = logging.getLogger(__name__)

class DeployInferrer:
    def __init__(self, *, run_name, infer_ckpt_cfg, infer_task_flow, infer_type="deploy"):
        if infer_type != "deploy":
            raise ValueError("WAM_policy only supports deploy inference")
        checkpoint = infer_ckpt_cfg.get("pretrain_model")
        if not checkpoint:
            raise ValueError("infer_ckpt_cfg.pretrain_model is required")
        if not isinstance(infer_task_flow, torch.nn.Module):
            raise TypeError("infer_task_flow must be a torch.nn.Module")
        self._run_name = run_name
        self._taskflow = infer_task_flow
        self._model = infer_task_flow.load_deploy_model(checkpoint)
        self._model.eval()
        logger.info("Load deployed model: %s", checkpoint)

    def infer(self, *, data):
        if not isinstance(data, Mapping) or not data:
            raise ValueError("data must be a non-empty request mapping")
        self._model.eval()
        return self._model(dict(data))
