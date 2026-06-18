import sys

import hydra
from omegaconf import DictConfig

from ahawam.runtime import run_training
from ahawam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def _strip_launcher_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--local_rank", "--local-rank"}:
            if i + 1 < len(argv):
                skip_next = True
            continue
        if arg.startswith("--local_rank=") or arg.startswith("--local-rank="):
            continue
        cleaned.append(arg)
    return cleaned


sys.argv = _strip_launcher_args(sys.argv)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
