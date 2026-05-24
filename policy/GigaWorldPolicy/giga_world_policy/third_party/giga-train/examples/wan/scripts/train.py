from typing import List

import torch
import tyro

from giga_train import launch_from_config, load_config, setup_environment


def train(config: str, launch: bool = True) -> None:
    """Train WAN models from a config.

    Args:
        config (str): Path to a Python config file.
        launch (bool): If True, use the project's launcher for distributed
            execution. If False, run single-GPU training directly.
    """
    setup_environment()
    if launch:
        launch_from_config(config)
    else:
        from wan import WanTrainer

        config = load_config(config)
        gpu_ids: List[int] = config.launch.gpu_ids
        assert len(gpu_ids) == 1
        torch.cuda.set_device(gpu_ids[0])
        runner = WanTrainer.load(config)
        runner.print(config)
        runner.save_config(config)
        if config.train.get('resume', False):
            runner.resume()
        runner.train()


if __name__ == '__main__':
    tyro.cli(train)
