import tyro
import argparse
from giga_train import launch_from_config


def train(config: str):
    launch_from_config(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='world_action_model.configs.fold_the_shirt.config',
                        help="Path to config, e.g. configs.xxx.config")
    tyro.cli(train)
