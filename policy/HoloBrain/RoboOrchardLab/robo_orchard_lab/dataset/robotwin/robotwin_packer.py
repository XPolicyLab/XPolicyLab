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

import json
import logging
import os

import cv2
import h5py
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def images_decoding(padded_data):
    decoded_imgs = []
    for jpeg_data in padded_data:
        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        decoded_imgs.append(img)
    return decoded_imgs


class RobotwinDataPacker(BaseLmdbManipulationDataPacker):
    def __init__(
        self,
        input_path,
        output_path,
        task_names=None,
        embodiment=None,
        robotwin_aug=None,
        camera_name=None,
        config_name=None,
        simulation=True,
        **kwargs,
    ):
        super().__init__(input_path, output_path, **kwargs)
        self.task_names = task_names
        self.embodiment = embodiment
        self.robotwin_aug = robotwin_aug
        self.camera_name = camera_name
        if (
            (self.embodiment is not None)
            or (self.robotwin_aug is not None)
            or (self.camera_name is not None)
        ):
            logger.warning(
                "The embodiment/robotwin_aug/camera_name supports the "
                "robotwin_cvpr_round2 branch but is deprecated and will be "
                "removed in a future version. "
                "Please use config_name instead."
            )
        self.config_name = config_name
        self.simulation = simulation
        self.episodes = self.input_path_handler(self.input_path)

    def _check_valid(self, input_path, task_dir, config_dir):
        if self.task_names is not None and task_dir not in self.task_names:
            return None

        if (
            (self.embodiment is not None)
            or (self.robotwin_aug is not None)
            or (self.camera_name is not None)
        ):
            config_name = (
                f"{self.embodiment}-{self.robotwin_aug}_{self.camera_name}"
            )
        else:
            config_name = self.config_name

        if config_name is None:
            config_name = config_dir
        elif config_dir != config_name:
            return None

        task_name = task_dir

        valid_seed_file = os.path.join(
            input_path, task_name, config_name, "seed.txt"
        )
        if not os.path.isfile(valid_seed_file):
            return None

        return task_name, config_name, valid_seed_file

    def input_path_handler(self, input_path):
        episodes = []
        for task_dir in os.listdir(input_path):
            for config_dir in os.listdir(os.path.join(input_path, task_dir)):
                valid = self._check_valid(input_path, task_dir, config_dir)
                if not valid:
                    logger.warning(
                        f"invalid task/config dir: {task_dir}/{config_dir}"
                    )
                    continue
                task_name, config_name, valid_seed_file = valid

                seeds = open(valid_seed_file, "r").read().strip().split(" ")
                current_path = os.path.join(input_path, task_dir, config_name)

                if os.path.isdir(os.path.join(current_path, "data")):
                    current_path = os.path.join(current_path, "data")

                for ep in os.listdir(current_path):
                    if not ep.endswith(".hdf5"):
                        continue
                    ep_path = os.path.join(current_path, ep)
                    ep_id = int(ep.replace("episode", "").replace(".hdf5", ""))
                    seed = seeds[ep_id]
                    episodes.append([task_name, config_name, ep_path, seed])
        episodes.sort(key=lambda x: (x[0], x[1], int(x[3])))  # sort by seed
        logger.info(f"number of valid episodes: {len(episodes)}")
        return episodes

    def _pack(self):
        num_valid_ep = 0
        for ep_id, ep in enumerate(self.episodes):
            task_name, config_name, ep_path, seed = ep
            uuid = f"{task_name}_{config_name}_seed{seed}"
            logger.info(
                f"start process [{ep_id + 1}/{len(self.episodes)}] {uuid}"
            )
            num_steps = 0
            joint_positions = []
            cartesian_positions = []
            extrinsics = {}
            intrinsics = {}
            rgbs = {}
            depths = {}

            with h5py.File(ep_path, "r") as ep_file:
                # read cartesian positions
                cartesian_positions = ep_file["endpose"][:]

                # read joint positions
                joint_positions = ep_file["joint_action"]["vector"][:]

                # read camera data
                camera_names = []
                for camera_name, camera_data in ep_file["observation"].items():
                    camera_names.append(camera_name)
                    intrinsics[camera_name] = camera_data["intrinsic_cv"][0]
                    extrinsics[camera_name] = camera_data["extrinsic_cv"][:]

                    rgbs_encode = camera_data["rgb"][:]

                    depths_raw = camera_data["depth"][:]
                    depths_encode = []
                    for depth in depths_raw:
                        assert len(depth.shape) == 2
                        depth = depth.astype(np.uint16)
                        ret, depth = cv2.imencode(".PNG", depth)
                        assert ret
                        depths_encode.append(depth)

                    rgbs[camera_name] = rgbs_encode
                    depths[camera_name] = depths_encode

            num_steps = len(cartesian_positions)

            for camera in camera_names:
                assert len(rgbs[camera]) == num_steps
                assert len(depths[camera]) == num_steps
                for i, (rgb, depth) in enumerate(
                    zip(rgbs[camera], depths[camera], strict=False)
                ):
                    self.image_pack_file.write(f"{uuid}/{camera}/{i}", rgb)
                    self.depth_pack_file.write(f"{uuid}/{camera}/{i}", depth)

            assert len(joint_positions) == num_steps
            assert len(cartesian_positions) == num_steps
            joint_positions = np.stack(joint_positions)
            cartesian_positions = np.stack(cartesian_positions)

            self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsics)
            self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsics)
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_positions",
                joint_positions,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/cartesian_position",
                cartesian_positions,
            )
            self.meta_pack_file.write(f"{uuid}/camera_names", camera_names)

            instruction_file = os.path.join(
                os.path.dirname(os.path.dirname(ep_path)),
                "instructions",
                os.path.basename(ep_path).replace("hdf5", "json"),
            )
            if os.path.exists(instruction_file):
                instructions = json.load(open(instruction_file))
                self.meta_pack_file.write(
                    f"{uuid}/instructions",
                    instructions,
                )

            index_data = dict(
                uuid=uuid,
                task_name=task_name,
                config_name=config_name,
                num_steps=num_steps,
                seeed=seed,
                simulation=True,
            )
            if self.camera_name is not None:
                index_data["camera_type"] = self.camera_name
            if self.embodiment is not None:
                index_data["embodiment"] = self.embodiment
            self.meta_pack_file.write(f"{uuid}/meta_data", index_data)
            self.write_index(ep_id, index_data)
            num_valid_ep += 1
            logger.info(
                f"finish process [{ep_id + 1}/{len(self.episodes)}] {uuid}, "
                f"num_steps:{num_steps} \n"
            )
        self.index_pack_file.write("__len__", num_valid_ep, commit=True)
        self.close()


if __name__ == "__main__":
    import argparse

    from robo_orchard_lab.utils import log_basic_config

    log_basic_config(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--embodiment", type=str, default=None)
    parser.add_argument("--robotwin_aug", type=str, default=None)
    parser.add_argument("--camera_name", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    args = parser.parse_args()

    if args.task_names is None:
        task_names = None
    else:
        task_names = args.task_names.split(",")

    packer = RobotwinDataPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        task_names=task_names,
        embodiment=args.embodiment,
        robotwin_aug=args.robotwin_aug,
        camera_name=args.camera_name,
        config_name=args.config_name,
    )
    packer()
