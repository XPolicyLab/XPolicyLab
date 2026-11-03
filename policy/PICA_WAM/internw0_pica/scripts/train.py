import hydra
import sys
from omegaconf import DictConfig

from wam.cache import CacheManager, CacheMode
from wam.runtime import run_training
from wam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def run_from_config(cfg: DictConfig) -> None:
    cache_manager = CacheManager.from_config(cfg)
    if cache_manager.mode is CacheMode.GENERATE:
        cache_manager.generate()
        return
    run_training(cfg, cache_manager=cache_manager)


def _strip_launcher_rank_args(argv: list[str]) -> list[str]:
    """DeepSpeed appends local-rank CLI args that Hydra does not accept."""
    cleaned = [argv[0]]
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--local_rank", "--local-rank"}:
            skip_next = True
            continue
        if arg.startswith("--local_rank=") or arg.startswith("--local-rank="):
            continue
        cleaned.append(arg)
    return cleaned


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_from_config(cfg)


if __name__ == "__main__":
    sys.argv = _strip_launcher_rank_args(sys.argv)
    main()
