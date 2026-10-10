# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field
import os

@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})

    label_path: str = field(default=None, metadata={"help": "Path to the label data."})
    stats_path: str = field(default=None, metadata={"help": "Path to the stats data."})
    image_mapping_path: str = field(default=None, metadata={"help": "Path to the image mapping."})
    data_skip: int =field(default=1, metadata={"help": "for get a subset of "})


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})



def register_datasets_mixtures():
    DATA_ROOT=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "data",
    )

    print(DATA_ROOT)
    holoassist_train_abs_skip2v3 = Dataset(
        dataset_name="holoassist_train_abs_skip2v3",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2v3_train",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=1
    )
    add_dataset(holoassist_train_abs_skip2v3)

    holoassist_eval_abs_skip2v3_sub20 = Dataset(
        dataset_name="holoassist_eval_abs_skip2v3_sub20",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2v3_val",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=20
    )
    add_dataset(holoassist_eval_abs_skip2v3_sub20)

    holoassist_train_abs_skip2top10 = Dataset(
        dataset_name="holoassist_train_abs_skip2top10",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2top10_train",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=1
    )
    add_dataset(holoassist_train_abs_skip2top10)

    holoassist_eval_abs_skip2top10_sub20 = Dataset(
        dataset_name="holoassist_eval_abs_skip2top10_sub20",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2top10_val",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=20
    )
    add_dataset(holoassist_eval_abs_skip2top10_sub20)

    holoassist_train_abs_skip2_rot = Dataset(
        dataset_name="holoassist_train_abs_skip2_rot",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2_rot_train",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=1
    )
    add_dataset(holoassist_train_abs_skip2_rot)

    holoassist_train_abs_skip2_mano_optim = Dataset(
        dataset_name="holoassist_train_abs_skip2_mano_optim",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2_mano_optim_train",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=1
    )
    add_dataset(holoassist_train_abs_skip2_mano_optim)

    holoassist_train_abs_skip2_mano_optim_sub5 = Dataset(
        dataset_name="holoassist_train_abs_skip2_mano_optim_sub5",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2_mano_optim_train",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=5
    )
    add_dataset(holoassist_train_abs_skip2_mano_optim_sub5)

    holoassist_eval_abs_skip2_rot_sub20 = Dataset(
        dataset_name="holoassist_eval_abs_skip2_rot_sub20",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2_rot_val",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=20
    )
    add_dataset(holoassist_eval_abs_skip2_rot_sub20)

    holoassist_eval_abs_skip2_mano_optim_sub20 = Dataset(
        dataset_name="holoassist_eval_abs_skip2_mano_optim_sub20",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2_mano_optim_val",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=20
    )
    add_dataset(holoassist_eval_abs_skip2_mano_optim_sub20)

    holoassist_eval_abs_skip2_mano_optim_sub20 = Dataset(
        dataset_name="holoassist_eval_abs_skip2_mano_optim_sub20",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2_mano_optim_val",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=20
    )
    add_dataset(holoassist_eval_abs_skip2_mano_optim_sub20)

    holoassist_train_abs_frameskip2sampleskip1_full = Dataset(
        dataset_name="holoassist_train_abs_ss1",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_frameskip2sampleskip1_full_train_incremental",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=1
    )
    add_dataset(holoassist_train_abs_frameskip2sampleskip1_full)

    holoassist_eval_abs_ss1_sub20 = Dataset(
        dataset_name="holoassist_eval_abs_ss1_sub20",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered_skip2_rot_val",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=100
    )
    add_dataset(holoassist_eval_abs_ss1_sub20)

    hot3d_train_abs_v3 = Dataset(
        dataset_name="hot3d_v3_train_abs",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_V3_train",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hot3d_train_abs_v3)

    hot3d_eval_abs_v3_sub20 = Dataset(
        dataset_name="hot3d_v3_eval_abs_sub20",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_V3_val",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=20
    )
    add_dataset(hot3d_eval_abs_v3_sub20)

    hot3d_train_abs_v4 = Dataset(
        dataset_name="hot3d_v4_train_abs",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_V4_train",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hot3d_train_abs_v4)

    hot3d_eval_abs_v4_sub20 = Dataset(
        dataset_name="hot3d_v4_eval_abs_sub20",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_V4_val",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=20
    )
    add_dataset(hot3d_eval_abs_v4_sub20)

    hot3d_train_abs_v5 = Dataset(
        dataset_name="hot3d_v5_train_abs",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_V5_train",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hot3d_train_abs_v5)

    hot3d_train_abs_v5_sub5 = Dataset(
        dataset_name="hot3d_v5_train_abs_sub5",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_V5_train",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=5
    )
    add_dataset(hot3d_train_abs_v5_sub5)

    hot3d_eval_abs_v5_sub20 = Dataset(
        dataset_name="hot3d_v5_eval_abs_sub20",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_V5_val",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=20
    )
    add_dataset(hot3d_eval_abs_v5_sub20)

    hot3d_train_abs_ss1 = Dataset(
        dataset_name="hot3d_ss1_train_abs",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_sampleskip1_train",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hot3d_train_abs_ss1)

    hot3d_eval_abs_ss1_sub20 = Dataset(
        dataset_name="hot3d_ss1_eval_abs_sub20",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_sampleskip1_val",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=100
    )
    add_dataset(hot3d_eval_abs_ss1_sub20)

    hoi4d_train_abs_v1 = Dataset(
        dataset_name="hoi4d_v1_train_abs",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_V1_train",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images",
        description="hoi4d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hoi4d_train_abs_v1)

    hoi4d_eval_abs_v1_sub20 = Dataset(
        dataset_name="hoi4d_v1_eval_abs_sub20",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_V1_val",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images",
        description="hoi4d Full Dataset in HF",
        data_skip=20
    )
    add_dataset(hoi4d_eval_abs_v1_sub20)


    hoi4d_train_abs_v2 = Dataset(
        dataset_name="hoi4d_v2_train_abs",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_V2_train",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images",
        description="hoi4d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hoi4d_train_abs_v2)

    hoi4d_train_abs_v2_sub5 = Dataset(
        dataset_name="hoi4d_v2_train_abs_sub5",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_V2_train",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images",
        description="hoi4d Full Dataset in HF",
        data_skip=5
    )
    add_dataset(hoi4d_train_abs_v2_sub5)

    hoi4d_eval_abs_v2_sub20 = Dataset(
        dataset_name="hoi4d_v2_eval_abs_sub20",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_V2_val",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images",
        description="hoi4d Full Dataset in HF",
        data_skip=20
    )
    add_dataset(hoi4d_eval_abs_v2_sub20)

    hoi4d_train_abs_v2_ss1 = Dataset(
        dataset_name="hoi4d_v2_ss1_train_abs",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_V2_sampleskip1_train",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images",
        description="hoi4d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hoi4d_train_abs_v2_ss1)

    hoi4d_eval_abs_v2_ss1_sub20 = Dataset(
        dataset_name="hoi4d_v2_ss1_eval_abs_sub20",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_V2_sampleskip1_val",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images",
        description="hoi4d Full Dataset in HF",
        data_skip=100
    )
    add_dataset(hoi4d_eval_abs_v2_ss1_sub20)

    # TACO
    taco_train_abs_v1 = Dataset(
        dataset_name="taco_v1_train_abs",
        dataset_type="taco_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/TACO_HF/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/TACO_HF/stats",
        data_path=f"{DATA_ROOT}/TACO_HF/HF_hand_V1_train",
        image_path=f"{DATA_ROOT}/TACO_HF/HF_images",
        description="TACO Full Dataset in HF",
        data_skip=1
    )
    add_dataset(taco_train_abs_v1)

    taco_eval_abs_v1_sub20 = Dataset(
        dataset_name="taco_v1_eval_abs_sub20",
        dataset_type="taco_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/TACO_HF/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/TACO_HF/stats",
        data_path=f"{DATA_ROOT}/TACO_HF/HF_hand_V1_val",
        image_path=f"{DATA_ROOT}/TACO_HF/HF_images",
        description="TACO Full Dataset in HF",
        data_skip=20
    )
    add_dataset(taco_eval_abs_v1_sub20)

    # AUG_SHIFT
    # for data_version in ["FIXED_SET", "V1"]:
    for dataset_name, root_path, data_version, scale in [
        # ("otv_sim_fixed_set", "otv_isaaclab_hf_fixed_set", "FIXED_SET", 50),
        ("otv_sim_fixed_set", "EgoVLA_SIM_Processed", "FIXED_SET_MIX", 50),
    ]:
        for set_name in ["train"]:
            print(f"{dataset_name}_{data_version}_{set_name}")
            otv_abs_fixed_set = Dataset(
                dataset_name=f"{dataset_name}_{data_version}_{set_name}",
                dataset_type="otv_sim",
                image_mapping_path=f"{DATA_ROOT}/{root_path}/hf_images_mapping.pkl",
                stats_path="",
                data_path=f"{DATA_ROOT}/{root_path}/HF_hand_{data_version}_{set_name}",
                image_path=f"{DATA_ROOT}/{root_path}/HF_images",
                description="otv simulation Full Dataset in HF",
                data_skip=1
            )
            add_dataset(otv_abs_fixed_set)

            otv_abs_fixed_set_sub50 = Dataset(
                dataset_name=f"{dataset_name}_{data_version}_{set_name}_sub{str(scale)}",
                dataset_type="otv_sim",
                image_mapping_path=f"{DATA_ROOT}/{root_path}/hf_images_mapping.pkl",
                stats_path="",
                data_path=f"{DATA_ROOT}/{root_path}/HF_hand_{data_version}_{set_name}",
                image_path=f"{DATA_ROOT}/{root_path}/HF_images",
                description="otv simulation Full Dataset in HF",
                data_skip=scale
            )
            add_dataset(otv_abs_fixed_set_sub50)

    # Single TASK Dataset
    for name in [
        "Insert-Cans", "Sort-Cans", "Push-Box",
        "Flip-Mug", "Insert-And-Unload-Cans"
    ]:
        for data_version in ["V1", "V1_CLIPSTARTEND", "V1_3dim", "V1_2dim", "V1_1dim"]:
            for set_name in ["train", "val", "all"]:
                otv_abs_fixed_set = Dataset(
                    dataset_name=f"otv_sim_fixed_set_{name}_{data_version}_{set_name}",
                    dataset_type="otv_sim",
                    image_mapping_path=f"{DATA_ROOT}/otv_isaaclab_hf_fixed_set/hf_images_mapping_{name}.pkl",
                    stats_path="",
                    data_path=f"{DATA_ROOT}/otv_isaaclab_hf_fixed_set/HF_hand_{name}_{data_version}_{set_name}",
                    image_path=f"{DATA_ROOT}/otv_isaaclab_hf_fixed_set/HF_images_{name}",
                    description="otv simulation Full Dataset in HF",
                    data_skip=1
                )
                add_dataset(otv_abs_fixed_set)

                otv_abs_fixed_set_sub20 = Dataset(
                    dataset_name=f"otv_sim_fixed_set_{name}_{data_version}_{set_name}_sub20",
                    dataset_type="otv_sim",
                    image_mapping_path=f"{DATA_ROOT}/otv_isaaclab_hf_fixed_set/hf_images_mapping_{name}.pkl",
                    stats_path="",
                    data_path=f"{DATA_ROOT}/otv_isaaclab_hf_fixed_set/HF_hand_{name}_{data_version}_{set_name}",
                    image_path=f"{DATA_ROOT}/otv_isaaclab_hf_fixed_set/HF_images_{name}",
                    description="otv simulation Full Dataset in HF",
                    data_skip=20
                )
                add_dataset(otv_abs_fixed_set_sub20)


    ### 30Hz Data
    holoassist_train_3ohz_mano_optim_sub5 = Dataset(
        dataset_name="holoassist_train_30hz_sub5",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered30Hz_train",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=5
    )
    add_dataset(holoassist_train_3ohz_mano_optim_sub5)

    holoassist_train_3ohz_mano_optim_sub100 = Dataset(
        dataset_name="holoassist_eval_30hz_sub100",
        dataset_type="holoassist_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/ha_dataset/hf_images_v2_mapping.pkl",
        stats_path=f"{DATA_ROOT}/ha_dataset/stats",
        data_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_hands_filtered30Hz_val",
        image_path=f"{DATA_ROOT}/ha_dataset/HoloAssist_HF_images_v2",
        description="HoloAssist Full Dataset in HF",
        data_skip=100
    )
    add_dataset(holoassist_train_3ohz_mano_optim_sub100)

    hot3d_train_abs_30hz = Dataset(
        dataset_name="hot3d_30hz_train",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_30Hz_train",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hot3d_train_abs_30hz)

    hot3d_eval_abs_30hz_sub20 = Dataset(
        dataset_name="hot3d_30hz_eval_sub20",
        dataset_type="hot3d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hot3d_hf/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/hot3d_hf/stats",
        data_path=f"{DATA_ROOT}/hot3d_hf/HF_hand_30Hz_val",
        image_path=f"{DATA_ROOT}/hot3d_hf/HF_images",
        description="hot3d Full Dataset in HF",
        data_skip=20
    )
    add_dataset(hot3d_eval_abs_30hz_sub20)


    hoi4d_train_abs_30hz = Dataset(
        dataset_name="hoi4d_30hz_train",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping_30hz.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_30Hz_train",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images_30hz",
        description="hoi4d Full Dataset in HF",
        data_skip=1
    )
    add_dataset(hoi4d_train_abs_30hz)

    hoi4d_eval_abs_30hz_sub20 = Dataset(
        dataset_name="hoi4d_30hz_eval_sub20",
        dataset_type="hoi4d_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/hoi4d_hf/hf_images_mapping_30hz.pkl",
        stats_path=f"{DATA_ROOT}/hoi4d_hf/stats",
        data_path=f"{DATA_ROOT}/hoi4d_hf/HF_hand_30Hz_val",
        image_path=f"{DATA_ROOT}/hoi4d_hf/HF_images_30hz",
        description="hoi4d Full Dataset in HF",
        data_skip=20
    )
    add_dataset(hoi4d_eval_abs_30hz_sub20)


    taco_train_abs_30hz = Dataset(
        dataset_name="taco_30hz_train",
        dataset_type="taco_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/TACO_HF/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/TACO_HF/stats",
        data_path=f"{DATA_ROOT}/TACO_HF/HF_hand_30Hz_train",
        image_path=f"{DATA_ROOT}/TACO_HF/HF_images",
        description="TACO Full Dataset in HF",
        data_skip=1
    )
    add_dataset(taco_train_abs_30hz)

    taco_eval_abs_30hz_sub20 = Dataset(
        dataset_name="taco_30hz_eval_sub20",
        dataset_type="taco_hf_abs_hand",
        image_mapping_path=f"{DATA_ROOT}/TACO_HF/hf_images_mapping.pkl",
        stats_path=f"{DATA_ROOT}/TACO_HF/stats",
        data_path=f"{DATA_ROOT}/TACO_HF/HF_hand_30Hz_val",
        image_path=f"{DATA_ROOT}/TACO_HF/HF_images",
        description="TACO Full Dataset in HF",
        data_skip=20
    )
    add_dataset(taco_eval_abs_30hz_sub20)
