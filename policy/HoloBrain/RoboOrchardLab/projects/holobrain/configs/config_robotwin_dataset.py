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

import os

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

# XPolicyLab integration: allow overriding the LMDB path via env var so
# train.sh / eval.sh can plug in the converted dataset without editing
# this file. Multiple paths can be passed comma-separated.
_XP_LMDB = os.environ.get("XPOLICY_HOLOBRAIN_LMDB")
if _XP_LMDB:
    _ROBOTWIN2_0_PATHS = [p.strip() for p in _XP_LMDB.split(",") if p.strip()]
else:
    _ROBOTWIN2_0_PATHS = [
        "./data/robotwin2.0/aloha_agilex_demo_clean",
        "./data/robotwin2.0/aloha_agilex_demo_randomized",
    ]

# Resolve URDF paths relative to this config file so they work regardless
# of cwd (accelerate worker subprocesses, export.py with absolute workspace,
# etc).
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POLICY_DIR = os.path.abspath(os.path.join(_PROJECT_DIR, "..", "..", ".."))
_DEFAULT_URDF = os.path.join(
    _POLICY_DIR, "embodiments", "arx_x5", "dual_x5_exact_from_x5a.urdf"
)
_ARX5_URDF = os.environ.get("XPOLICY_HOLOBRAIN_URDF", _DEFAULT_URDF)

dataset_config = dict(
    robotwin2_0=dict(
        kinematics_config=dict(
            urdf=_ARX5_URDF,
            # left/right_arm_joint_id and link keys all use upstream defaults
            # (DualArmKinematics: [10-15]/[18-23], fl/fr_link1-6 + fl/fr_link7-8),
            # which match RoboTwin2.0's official arx5_description_isaac.urdf.
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        paths=_ROBOTWIN2_0_PATHS,
        scale_shift=[
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.5, 0.5],
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.5, 0.5],
        ],
        num_joint=14,
        cam_names=[
            "front_camera",
            "left_camera",
            "right_camera",
            "head_camera",
        ],
    ),
    robotwin2_0_ur5_wsg=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin2_dual_arm_ur5_wsg.urdf",
            left_arm_link_keys=[
                "left_shoulder_link",
                "left_upper_arm_link",
                "left_forearm_link",
                "left_wrist_1_link",
                "left_wrist_2_link",
                "left_wrist_3_link",
            ],
            left_finger_keys=["left_finger_left"],
            right_arm_link_keys=[
                "right_shoulder_link",
                "right_upper_arm_link",
                "right_forearm_link",
                "right_wrist_1_link",
                "right_wrist_2_link",
                "right_wrist_3_link",
            ],
            right_finger_keys=["right_finger_left"],
            left_arm_joint_id=list(range(6)),
            right_arm_joint_id=list(range(8, 14)),
        ),
        T_base2world=[
            [1, 0, 0, 0],
            [0, 1, 0, -0.65],
            [0, 0, 1, 0.65],
            [0, 0, 0, 1],
        ],
        paths=[
            "./data/robotwin2.0/ur5_wsg_demo_clean",
        ],
        scale_shift=[
            [2.400281548500061, -0.1310516595840454],
            [1.445511817932129, -1.445511817932129],
            [2.16847026348114, -0.23492777347564697],
            [1.7424615025520325, -0.007538259029388428],
            [2.8101450204849243, 0.15472495555877686],
            [2.9653799533843994, 0.02583003044128418],
            [0.5, 0.5],
            [2.400281548500061, -0.1310516595840454],
            [1.445511817932129, -1.445511817932129],
            [2.16847026348114, -0.23492777347564697],
            [1.7424615025520325, -0.007538259029388428],
            [2.8101450204849243, 0.15472495555877686],
            [2.9653799533843994, 0.02583003044128418],
            [0.5, 0.5],
        ],
        num_joint=14,
        cam_names=["left_camera", "right_camera", "head_camera"],
    ),
)


def build_transforms(
    config, mode, kinematics_config, t_base2world, scale_shift, num_joint
):
    import numpy as np

    from robo_orchard_lab.dataset.robotwin.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        DualArmKinematics,
        GetProjectionMat,
        ImageChannelFlip,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    num_joint_per_arm = num_joint // 2 - 1
    joint_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    ee_state_loss_weights = [1, 2, 2, 2, 0.2, 0.2, 0.2, 0.2]
    loss_weights = np.array(
        [
            [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
        ]
    ).tolist()
    joint_mask = ([True] * num_joint_per_arm + [False]) * 2

    if mode == "training":
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
            joint_mask=joint_mask,
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])
    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="ego")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=dict(
            imgs="float32",
            depths="float32",
            image_wh="float32",
            projection_mat="float32",
            embodiedment_mat="float32",
        ),
    )

    kinematics = dict(type=DualArmKinematics, **kinematics_config)

    scale_shift = dict(type=AddScaleShift, scale_shift=scale_shift)

    if mode == "training":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "kinematics",
                "fk_loss_weight",
                "state_loss_weights",
                "text",
                "uuid",
                "joint_mask",
            ],
        )
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=([[-0.02, 0.02]] * num_joint_per_arm + [[0.0, 0.0]])
            * 2,
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
            joint_state_noise,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "validation":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "kinematics",
                "text",
                "uuid",
                "joint_mask",
            ],
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "joint_scale_shift",
                "kinematics",
                "text",
                "joint_mask",
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return transforms


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    datasets = []
    for dataset_name, data_config in dataset_config.items():
        if (
            "robotwin" not in dataset_names
            and dataset_name not in dataset_names
        ):
            continue
        transforms = build_transforms(
            config,
            mode,
            data_config["kinematics_config"],
            data_config["T_base2world"],
            data_config["scale_shift"],
            data_config["num_joint"],
        )
        dataset = RoboTwinLmdbDataset(
            paths=dataset_config[dataset_name]["paths"],
            task_names=config.get("task_names"),
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name=dataset_name,
            cam_names=data_config["cam_names"],
            reset_step=1000,
        )
        datasets.append(dataset)
    return datasets


@processor_register()
def build_processors(config, dataset_names):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    processors = {}
    for dataset_name, data_config in dataset_config.items():
        if dataset_name not in dataset_names:
            continue

        transforms = build_transforms(
            config,
            "deploy",
            data_config["kinematics_config"],
            data_config["T_base2world"],
            data_config["scale_shift"],
            data_config["num_joint"],
        )
        processor = HoloBrainProcessor(
            HoloBrainProcessorCfg(
                load_image=True,
                load_depth=config["with_depth"],
                valid_action_step=None,
                transforms=transforms,
                cam_names=data_config["cam_names"],
            )
        )
        processors[dataset_name] = processor
    return processors
