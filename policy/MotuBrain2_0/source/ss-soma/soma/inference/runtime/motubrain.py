"""Direct-inference factory for the inference Motubrain model."""

import logging
import time

import torch
from omegaconf import DictConfig

from falcon.config import TrainerConfig
from soma.inference.models.motubrain import MotubrainInferModel
from soma.inference.runtime.checkpoint import load_model_weights

logger = logging.getLogger(__name__)


def create_model_config(kwargs: dict[str, object]) -> TrainerConfig:
    """Adapt model keyword arguments to Falcon's model constructor config."""
    return TrainerConfig.from_lazy_config(
        DictConfig({"model": kwargs, "noise_scheduler": {}})
    )


def build_model(
    checkpoint_path: str | None = None,
    **kwargs: object,
) -> MotubrainInferModel:
    """Build a Falcon Motubrain model with inference operator patches."""
    config = create_model_config(kwargs)
    with torch.device("meta"):
        model = MotubrainInferModel(config).eval().bfloat16()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to_empty(device=device, recurse=True)

    if checkpoint_path is not None:
        logger.info("Loading Falcon Motubrain weights from %s", checkpoint_path)
        start = time.time()
        load_model_weights(model, checkpoint_path)
        logger.info("Loaded Falcon Motubrain in %.1fs", time.time() - start)

    return model.requires_grad_(False)
