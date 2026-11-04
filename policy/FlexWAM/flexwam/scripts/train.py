from flexwam.config import resolve_config
import hydra
from omegaconf import DictConfig

from flexwam.utils.tmpdir import configure_tmpdir

configure_tmpdir()

from flexwam.runtime import run_training
from flexwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    cfg = resolve_config(cfg)
    run_training(cfg)


if __name__ == "__main__":
    main()
