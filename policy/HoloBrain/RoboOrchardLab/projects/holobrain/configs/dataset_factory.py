# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from typing import Callable

REGISTERED = False
TRAIN_DATASET_BUILD_FUNCS = set()
VALIDATION_DATASET_BUILD_FUNCS = set()
PROCESSOR_BUILD_FUNCS = set()


def train_dataset_register():
    def decorator(func: Callable):
        TRAIN_DATASET_BUILD_FUNCS.add(func)
        return func

    return decorator


def validation_dataset_register():
    def decorator(func: Callable):
        VALIDATION_DATASET_BUILD_FUNCS.add(func)
        return func

    return decorator


def processor_register():
    def decorator(func: Callable):
        PROCESSOR_BUILD_FUNCS.add(func)
        return func

    return decorator


def apply_dataset_register():
    global REGISTERED
    if REGISTERED:
        return
    import config_agibot_geniesim_dataset  # noqa: F401
    import config_agilex_ro_dataset  # noqa: F401
    import config_robotwin_dataset  # noqa: F401

    REGISTERED = True


def build_training_dataset(config, lazy_init=False):
    from robo_orchard_lab.dataset.dataset_wrapper import ConcatDatasetWithFlag

    apply_dataset_register()
    datasets = []
    for build_func in TRAIN_DATASET_BUILD_FUNCS:
        datasets.extend(
            build_func(
                config,
                config["training_datasets"],
                mode="training",
                lazy_init=lazy_init,
            )
        )
    dataset = ConcatDatasetWithFlag(datasets=datasets)
    return dataset


def build_validation_dataset(config, lazy_init=False):
    from robo_orchard_lab.dataset.dataset_wrapper import ConcatDatasetWithFlag

    apply_dataset_register()
    datasets = []
    for build_func in VALIDATION_DATASET_BUILD_FUNCS:
        datasets.extend(
            build_func(
                config,
                config.get("validation_datasets", []),
                mode="validation",
                lazy_init=lazy_init,
            )
        )
    if len(datasets) == 0:
        return None
    else:
        dataset = ConcatDatasetWithFlag(datasets=datasets)
        return dataset


def build_processors(config):
    apply_dataset_register()
    processors = {}
    for build_func in PROCESSOR_BUILD_FUNCS:
        processors.update(build_func(config, config["deploy_datasets"]))
    return processors
