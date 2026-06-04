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

import copy
import importlib
import os

import cv2
import numpy as np
import torch
import yaml


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except AttributeError:
        raise SystemExit(f"No Task, {task_name}")
    return env_instance


def deploy_data_process(dataset, obs, task_name, joint_state=None):
    images = []
    depths = []
    T_world2cam = []  # noqa: N806
    intrinsic = []
    for camera_data in obs["observation"].values():
        if dataset.load_image:
            images.append(camera_data["rgb"])
        if dataset.load_depth:
            depths.append(camera_data["depth"] / 1000)

        _tmp = np.eye(4)
        _tmp[:3] = camera_data["extrinsic_cv"]
        T_world2cam.append(_tmp)

        _tmp = np.eye(4)
        _tmp[:3, :3] = camera_data["intrinsic_cv"]
        intrinsic.append(_tmp)

    joint_state.append(obs["joint_action"]["vector"])
    data = dict(
        intrinsic=np.stack(intrinsic),
        T_world2cam=np.stack(T_world2cam),
        T_base2world=copy.deepcopy(dataset.T_base2world),
        step_index=len(joint_state) - 1,
        joint_state=np.stack(joint_state),
    )
    if dataset.T_base2ego is not None:
        data["T_base2ego"] = copy.deepcopy(dataset.T_base2ego)
    if dataset.load_image:
        data["imgs"] = np.stack(images)
    if dataset.load_depth:
        data["depths"] = np.stack(depths)

    instructions = dataset.instructions.get(
        task_name,
        dataset.DEFAULT_INSTRUCTIONS["others"],
    )
    if isinstance(instructions, str):
        text = instructions
    elif len(instructions) == 0:
        text = ""
    else:
        text = instructions[0]
    data["text"] = text

    for transform in dataset.transforms:
        if transform is None:
            continue
        data = transform(data)
    for k, v in data.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            data[k] = v[None]
        else:
            data[k] = [v]
    return data, joint_state


def update_task_config(task_args, CONFIGS_PATH):  # noqa: N803
    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(
        CONFIGS_PATH, "_embodiment_config.yml"
    )

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    task_args["dual_arm_embodied"] = True

    task_args["left_embodiment_config"] = get_embodiment_config(
        task_args["left_robot_file"]
    )
    task_args["right_embodiment_config"] = get_embodiment_config(
        task_args["right_robot_file"]
    )
    return task_args


def visualize(data, video_writer, task_env, save_path):
    if save_path is None:
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    imgs1 = np.concatenate(
        [
            data["observation"]["head_camera"]["rgb"],
            data["observation"]["front_camera"]["rgb"],
        ],
        axis=1,
    )
    imgs2 = np.concatenate(
        [
            data["observation"]["left_camera"]["rgb"],
            data["observation"]["right_camera"]["rgb"],
        ],
        axis=1,
    )
    imgs = np.concatenate([imgs1, imgs2], axis=0)

    if video_writer is None:
        file = os.path.join(save_path, f"episodes_test{task_env.test_num}.mp4")
        print(f"visualization result: {file}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            file,
            fourcc,
            20,
            imgs.shape[:2][::-1],
        )
    video_writer.write(np.uint8(imgs)[..., ::-1])
    return video_writer


def evaluation(task_env, model, dataset, save_path):
    succ = False
    joint_state = []
    task_name = task_env.task_name
    video_writer = None
    while task_env.take_action_cnt < task_env.step_lim:
        # get model input
        observation = task_env.get_obs()
        data, joint_state = deploy_data_process(
            dataset, observation, task_name, joint_state
        )

        # get action chunk
        actions = model(data)[0]["pred_actions"][0]
        valid_action_step = 32
        actions = actions[:valid_action_step, :, 0].cpu().numpy()

        # take action
        for action in actions:
            observation = task_env.get_obs()
            video_writer = visualize(
                observation, video_writer, task_env, save_path
            )
            task_env.take_action(action)

        if task_env.eval_success:
            succ = True
            break
    if video_writer is not None:
        video_writer.release()
    return succ
