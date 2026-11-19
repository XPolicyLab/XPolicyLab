"""Compose and validate adapter settings without Ray, weights or a simulator."""

import hydra
from omegaconf import OmegaConf

from XPolicyLab.infra.RLinf.core.configuration import register_config_paths
from XPolicyLab.infra.RLinf.core.validation import (
    checked_rlinf_revision,
    validate_adapter_config,
)


@hydra.main(
    version_base="1.1",
    config_path="../config",
    config_name=None,
)
def main(cfg):
    from rlinf.config import validate_fsdp_cfg

    checked_rlinf_revision()
    validate_adapter_config(cfg)
    if not cfg.runner.only_eval:
        validate_fsdp_cfg(cfg.actor)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(
        "Adapter configuration OK. Cluster placement, weights and simulator were not loaded."
    )


if __name__ == "__main__":
    register_config_paths()
    main()
