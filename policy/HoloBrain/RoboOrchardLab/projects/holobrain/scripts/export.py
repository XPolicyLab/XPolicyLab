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

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projects.holobrain.utils import load_checkpoint, load_config  # noqa: E402
from robo_orchard_lab.models.holobrain import HoloBrainProcessor  # noqa: E402
from robo_orchard_lab.models.holobrain.pipeline import (  # noqa: E402
    HoloBrainInferencePipeline,
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402
from robo_orchard_lab.utils import log_basic_config  # noqa: E402

logger = logging.getLogger(__file__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def main(args):
    os.makedirs(args.workspace, exist_ok=True)
    shutil.copytree(
        CONFIGS_DIR,
        os.path.join(args.workspace, "configs"),
        dirs_exist_ok=True,
    )

    config = load_config(args.config)
    build_model = config.build_model
    build_processors = config.build_processors
    config = config.config

    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            kwargs = json.load(open(args.kwargs, "r"))
        else:
            kwargs = json.loads(args.kwargs)
        config.update(kwargs)
    logger.info("\n" + json.dumps(config, indent=4))

    # export data processors and reload test
    processors = build_processors(config)
    for dataset_name, processor in processors.items():
        processor_name = f"{dataset_name}_processor.json"
        processor.save(args.workspace, processor_name)
        logger.info(f"Export {processor_name} successfully.")
        _processor = HoloBrainProcessor.load(args.workspace, processor_name)
        logger.info(f"Reload {processor_name} successfully.")

    # export model and reload test
    model = build_model(config)
    load_checkpoint(model, config.get("checkpoint"))
    model_path = os.path.join(args.workspace, "model")
    model.save_model(model_path, required_empty=False)
    logger.info("Export model successfully.")
    _model = ModelMixin.load_model(model_path, load_impl="native")
    logger.info("Reload model successfully.")

    # copy urdf
    urdf_src = os.path.join(args.workspace, "urdf")
    if os.path.isdir(urdf_src):
        shutil.copytree(
            urdf_src,
            os.path.join(model_path, "urdf"),
            dirs_exist_ok=True,
        )

    # export inference.config.json for each dataset's pipeline
    for dataset_name, processor in processors.items():
        inference_cfg = HoloBrainInferencePipelineCfg(
            class_type=HoloBrainInferencePipeline,
            model_cfg=None,
            processor=processor.cfg,
        )
        inference_config_name = f"{dataset_name}.config.json"
        inference_config_path = os.path.join(model_path, inference_config_name)
        with open(inference_config_path, "w") as fh:
            fh.write(inference_cfg.model_dump_json(indent=4))
        logger.info(f"Export {inference_config_name} successfully.")
        _pipeline = HoloBrainInferencePipeline.load_pipeline(
            directory=model_path,
            inference_prefix=dataset_name,
            device="cuda",
            load_weights=True,
            load_impl="native",
        )
        logger.info(f"Reload pipeline for {dataset_name} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )

    logger.info(f"Export to workspace dir {args.workspace}")
    main(args)
