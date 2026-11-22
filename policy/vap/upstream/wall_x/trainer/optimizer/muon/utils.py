import logging

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ..utils import register_optimizer
from .muon import build_muon_optimizer
from .shape_aware_muon import build_shape_aware_muon_optimizer

_logger = logging.getLogger(__name__)


def _model_uses_fsdp(model: torch.nn.Module) -> bool:
    """True iff any submodule is an FSDP wrapper.

    Checked via ``isinstance`` rather than parameter-name sniffing so a rename
    of FSDP's flat-param naming convention can't silently flip the dispatch.
    """
    return any(isinstance(m, FSDP) for m in model.modules())


def _model_has_dtensor_params(model: torch.nn.Module) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return any(isinstance(p, DTensor) for p in model.parameters())


def get_muon_optimizer(model: torch.nn.Module, *, opt_cfg, param_groups=None, **kwargs):
    """Dispatch to shape-aware (FSDP) or vanilla muon based on model wrapping.

    Shape-aware muon needs parameter shapes captured before FSDP flattening.
    We read them off model._orig_param_shapes (attached by the trainer
    before _create_optimizer) rather than taking them as a kwarg, so the
    trainer stays optimizer-agnostic.

    opt_cfg must be a MuonConfig. Optional **kwargs are
    forwarded to the underlying builders (e.g. filter_fn).
    """
    from .config import MuonConfig

    if not isinstance(opt_cfg, MuonConfig):
        raise TypeError(
            f"get_muon_optimizer expects MuonConfig, got {type(opt_cfg).__name__}"
        )

    assert isinstance(
        model, torch.nn.Module
    ), f"model must be a torch.nn.Module but got {type(model)}"

    common = dict(
        lr=opt_cfg.learning_rate,
        weight_decay=opt_cfg.weight_decay,
        eps=opt_cfg.eps,
        beta_1=opt_cfg.beta_1,
        beta_2=opt_cfg.beta_2,
        param_groups=param_groups,
        **kwargs,
    )

    if _model_has_dtensor_params(model):
        raise ValueError(
            "optimizer_type='muon' is not compatible with FSDP2/DTensor "
            "parameters. Use optimizer_type='dmuon' for FSDP2 full_shard runs."
        )

    if _model_uses_fsdp(model):
        orig_shapes = getattr(model, "_orig_param_shapes", None)
        if orig_shapes is None:
            raise ValueError(
                "shape-aware muon requires `model._orig_param_shapes` "
                "(captured before FSDP flattening)"
            )
        _logger.info("using shape-aware muon optimizer")
        return build_shape_aware_muon_optimizer(model, orig_shapes, **common)

    _logger.info("using default muon optimizer")
    return build_muon_optimizer(model, **common)


register_optimizer("muon", get_muon_optimizer)
