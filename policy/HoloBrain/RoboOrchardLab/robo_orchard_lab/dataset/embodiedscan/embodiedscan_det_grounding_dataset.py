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
import json
import logging
import os
import pickle
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import tqdm
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.embodiedscan.utils import sample
from robo_orchard_lab.utils.build import build
from robo_orchard_lab.utils.misc import as_sequence

logger = logging.getLogger(__name__)


# fmt: off
DEFAULT_CLASSES = (
    "adhesive tape", "air conditioner", "alarm", "album", "arch", "backpack",
    "bag", "balcony", "ball", "banister", "bar", "barricade", "baseboard",
    "basin", "basket", "bathtub", "beam", "beanbag", "bed", "bench", "bicycle",
    "bidet", "bin", "blackboard", "blanket", "blinds", "board", "body loofah",
    "book", "boots", "bottle", "bowl", "box", "bread", "broom", "brush",
    "bucket", "cabinet", "calendar", "camera", "can", "candle", "candlestick",
    "cap", "car", "carpet", "cart", "case", "chair", "chandelier", "cleanser",
    "clock", "clothes", "clothes dryer", "coat hanger", "coffee maker", "coil",
    "column", "commode", "computer", "conducting wire", "container", "control",
    "copier", "cosmetics", "couch", "counter", "countertop", "crate", "crib",
    "cube", "cup", "curtain", "cushion", "decoration", "desk", "detergent",
    "device", "dish rack", "dishwasher", "dispenser", "divider", "door",
    "door knob", "doorframe", "doorway", "drawer", "dress", "dresser", "drum",
    "duct", "dumbbell", "dustpan", "dvd", "eraser", "excercise equipment",
    "fan", "faucet", "fence", "file", "fire extinguisher", "fireplace",
    "flowerpot", "flush", "folder", "food", "footstool", "frame", "fruit",
    "furniture", "garage door", "garbage", "glass", "globe", "glove",
    "grab bar", "grass", "guitar", "hair dryer", "hamper", "handle", "hanger",
    "hat", "headboard", "headphones", "heater", "helmets", "holder", "hook",
    "humidifier", "ironware", "jacket", "jalousie", "jar", "kettle",
    "keyboard", "kitchen island", "kitchenware", "knife", "label", "ladder",
    "lamp", "laptop", "ledge", "letter", "light", "luggage", "machine",
    "magazine", "mailbox", "map", "mask", "mat", "mattress", "menu",
    "microwave", "mirror", "molding", "monitor", "mop", "mouse", "napkins",
    "notebook", "ottoman", "oven", "pack", "package", "pad", "pan", "panel",
    "paper", "paper cutter", "partition", "pedestal", "pen", "person", "piano",
    "picture", "pillar", "pillow", "pipe", "pitcher", "plant", "plate",
    "player", "plug", "plunger", "pool", "pool table", "poster", "pot",
    "price tag", "printer", "projector", "purse", "rack", "radiator", "radio",
    "rail", "range hood", "refrigerator", "remote control", "ridge", "rod",
    "roll", "roof", "rope", "sack", "salt", "scale", "scissors", "screen",
    "seasoning", "shampoo", "sheet", "shelf", "shirt", "shoe", "shovel",
    "shower", "sign", "sink", "soap", "soap dish", "soap dispenser", "socket",
    "speaker", "sponge", "spoon", "stairs", "stall", "stand", "stapler",
    "statue", "steps", "stick", "stool", "stopcock", "stove", "structure",
    "sunglasses", "support", "switch", "table", "tablet", "teapot",
    "telephone", "thermostat", "tissue", "tissue box", "toaster", "toilet",
    "toilet paper", "toiletry", "tool", "toothbrush", "toothpaste", "towel",
    "toy", "tray", "treadmill", "trophy", "tube", "tv", "umbrella", "urn",
    "utensil", "vacuum cleaner", "vanity", "vase", "vent", "ventilation",
    "wardrobe", "washbasin", "washing machine", "water cooler", "water heater",
    "window", "window frame", "windowsill", "wine", "wire", "wood", "wrap"
)

HEAD_LABELS = (
    48, 177, 82, 179, 37, 243, 28, 277, 32, 84, 215, 145, 182, 170, 22, 72, 30,
    141, 65, 257, 221, 225, 52, 75, 231, 158, 236, 156, 47, 74, 6, 18, 71, 242,
    217, 251, 66, 263, 5, 45, 14, 73, 278, 198, 24, 23, 196, 252, 19, 135, 26,
    229, 183, 200, 107, 272, 246, 269, 125, 59, 279, 15, 163, 258, 57, 195, 51,
    88, 97, 58, 102, 36, 137, 31, 80, 160, 155, 61, 238, 96, 190, 25, 219, 152,
    142, 201, 274, 249, 178, 192
)

COMMON_LABELS = (
    189, 164, 101, 205, 273, 233, 131, 180, 86, 220, 67, 268, 224, 270, 53,
    203, 237, 226, 10, 133, 248, 41, 55, 16, 199, 134, 99, 185, 2, 20, 234,
    194, 253, 35, 174, 8, 223, 13, 91, 262, 230, 121, 49, 63, 119, 162, 79,
    168, 245, 267, 122, 104, 100, 1, 176, 280, 140, 209, 259, 143, 165, 147,
    117, 85, 105, 95, 109, 207, 68, 175, 106, 60, 4, 46, 171, 204, 111, 211,
    108, 120, 157, 222, 17, 264, 151, 98, 38, 261, 123, 78, 118, 127, 240, 124
)

TAIL_LABELS = (
    76, 149, 173, 250, 275, 255, 34, 77, 266, 283, 112, 115, 186, 136, 256, 40,
    254, 172, 9, 212, 213, 181, 154, 94, 191, 193, 3, 130, 146, 70, 128, 167,
    126, 81, 7, 11, 148, 228, 239, 247, 21, 42, 89, 153, 161, 244, 110, 0, 29,
    114, 132, 159, 218, 232, 260, 56, 92, 116, 282, 33, 113, 138, 12, 188, 44,
    150, 197, 271, 169, 206, 90, 235, 103, 281, 184, 208, 216, 202, 214, 241,
    129, 210, 276, 64, 27, 87, 139, 227, 187, 62, 43, 50, 69, 93, 144, 166,
    265, 54, 83, 39
)
# fmt: on


class EmbodiedScanDetGroundingDataset(Dataset):
    """EmbodiedScanDetGroundingDataset.

    A dataset class for detection and grounding for EmbodiedScan.
    Following EmbodiedScan, the data directory structure is as follows::

        self.data_root
        ├──embodiedscan
        │   ├──embodiedscan_infos_train.pkl
        │   ├──embodiedscan_infos_val.pkl
        │   ├──embodiedscan_train_vg_all.json
        │   ...
        ├──3rscan
        │   ├──00d42bed-778d-2ac6-86a7-0e0e5f5f5660
        │   ...
        ├──scannet
        │   └──posed_images
        ├──matterport3d
        │   ├──17DRP5sb8fy
        │   ...
        └──arkitscenes
            ├──Training
            └──Validation

    Args:
        ann_file (str): Path to the annotation file, e.g.,
            ``"embodiedscan/embodiedscan_infos_train.pkl"``.
        data_root (str): Root directory of the dataset.
            Defaults to ``""``.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to ``False``.
        load_anns (bool): Whether to load annotations.
            Defaults to ``True``.
        remove_dontcare (bool): Whether to remove "don't care"
            regions. Defaults to ``False``.
        dataset_length (Optional[int]): Maximum length of the
            dataset. Defaults to ``None``.
        mode (Literal["detection", "grounding"]): Mode of the dataset.
            Defaults to ``"detection"``.
        max_refetch (int,): Maximum number of refetch attempts during
            data loading. Defaults to ``100``.
        part (Optional[str]): Specific part of the dataset to load.
            For example, set it to ``"scannet"`` to load ScanNet data only.
            Defaults to ``None``.
        transforms (List[Union[dict, Callable]]): List of transforms
            to apply. Defaults to ``[]``.
        vg_file (Optional[str]): Path to the visual grounding file,
            e.g., ``"embodiedscan/embodiedscan_train_vg_all.json"``.
            Defaults to ``None``.
        num_text (int): Max number of text descriptions per sample
            from the grounding task. Defaults to ``1``.
        tokens_positive_rebuild (bool, optional): Whether to rebuild positive
            tokens. Defaults to ``True``.
        sep_token (str): Separator token for text processing.
            Defaults to ``"[SEP]"``.
        lazy_init (bool): Whether to initialize the dataset lazily.
            Defaults to ``False``.
        classes (Optional[List[str]]): List of class names.
            If ``None``, ``DEFAULT_CLASSES`` will be used.
    """

    def __init__(
        self,
        ann_file: str,
        data_root: str = "",
        test_mode: bool = False,
        load_anns: bool = True,
        remove_dontcare: bool = False,
        dataset_length: Optional[int] = None,
        mode: Literal["detection", "grounding"] = "detection",
        max_refetch: int = 100,
        part: Optional[str] = None,
        transforms: List[Union[dict, Callable]] | None = None,
        vg_file: Optional[str] = None,
        num_text: int = 1,
        tokens_positive_rebuild: bool = True,
        sep_token: str = "[SEP]",
        lazy_init: bool = False,
        classes: Optional[List[str]] = None,
    ):
        super().__init__()
        if transforms is None:
            transforms = []

        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.load_anns = load_anns
        self.remove_dontcare = remove_dontcare
        self.dataset_length = dataset_length
        self.mode = mode
        assert self.mode in ["detection", "grounding"]
        self.max_refetch = max_refetch
        self.part = part
        self.transforms = [build(x) for x in as_sequence(transforms)]

        # grounding kwargs
        self.vg_file = vg_file
        self.num_text = num_text
        self.tokens_positive_rebuild = tokens_positive_rebuild
        self.sep_token = sep_token

        if classes is None:
            self.classes = DEFAULT_CLASSES
        else:
            self.classes = classes

        self.initialized = False
        if not lazy_init:
            self._init()

    def _init(self):
        self.data_list = self.load_data_list()
        if self.mode == "grounding":
            self.load_grounding_data()
        self.initialized = True
        logger.info(f"dataset length : {self.__len__()}")

    def __len__(self):
        if not self.initialized:
            self._init()
        return len(self.data_list)

    def load_data_list(self):
        annotations = pickle.load(
            open(os.path.join(self.data_root, self.ann_file), "rb")
        )
        metainfo = annotations["metainfo"]
        raw_data_list = annotations["data_list"]

        self.label_mapping = np.full(
            max(metainfo["categories"].values()) + 1, -1, dtype=int
        )
        for key, value in metainfo["categories"].items():
            if key in self.classes:
                self.label_mapping[value] = self.classes.index(key)

        data_list = []
        for raw_data_info in tqdm.tqdm(
            raw_data_list,
            mininterval=10,
            desc=f"Loading {'Test' if self.test_mode else 'Train'} dataset",
        ):
            if self.part is not None:
                valid = any(
                    [x in raw_data_info["sample_idx"] for x in self.part]
                )
                if not valid:
                    continue

            data_info = self.parse_data_info(raw_data_info)
            if data_info is None:
                continue
            data_list.append(data_info)
            if (
                self.dataset_length is not None
                and len(data_list) >= self.dataset_length
            ):
                break
        return data_list

    def parse_data_info(self, info: dict):
        output = {
            "scan_id": info["sample_idx"],
            "sample_idx": info["sample_idx"],
            "img_path": [],
            "depth_img_path": [],
        }

        if "matterport3d" in info["sample_idx"]:
            output["depth_shift"] = 4000.0
        else:
            output["depth_shift"] = 1000.0

        img_path = []
        depth_path = []
        if "cam2img" not in info:
            intrinsic = []
        else:
            intrinsic = info["cam2img"]
        extrinsic = []

        axis_align_matrix = info.get("axis_align_matrix", np.eye(4))
        for _, img_info in enumerate(info["images"]):
            img_path.append(os.path.join(self.data_root, img_info["img_path"]))
            depth_path.append(
                os.path.join(self.data_root, img_info["depth_path"])
            )
            extrinsic.append(
                np.linalg.inv(
                    axis_align_matrix @ img_info["cam2global"]
                ).astype(np.float32)
            )
            if "cam2img" not in info:
                intrinsic.append(img_info["cam2img"].astype(np.float32))

        output["img_path"] = img_path
        output["depth_path"] = depth_path
        output["intrinsic"] = intrinsic
        output["extrinsic"] = extrinsic
        if "depth_cam2img" not in info:
            output["depth_intrinsic"] = intrinsic
        else:
            output["depth_intrinsic"] = info["depth_cam2img"]

        if self.load_anns:
            output["ann_info"] = self.parse_ann_info(info)
        return output

    def parse_ann_info(self, info: dict):
        instances = info.get("instances", [])
        num_instances = len(instances)
        ann_info = dict(
            gt_bboxes_3d=np.zeros((num_instances, 9), dtype=np.float32),
            gt_labels_3d=np.zeros((num_instances,), dtype=np.int64),
            gt_names=[],
            bbox_id=np.zeros((num_instances,), dtype=np.int64) - 1,
        )
        for idx, instance in enumerate(info["instances"]):
            ann_info["gt_bboxes_3d"][idx] = instance["bbox_3d"]
            label = self.label_mapping[instance["bbox_label_3d"]]
            ann_info["gt_labels_3d"][idx] = label
            if label >= 0:
                name = self.classes[label]
            else:
                name = "others"
            ann_info["gt_names"].append(name)  # type: ignore
            ann_info["bbox_id"][idx] = instance["bbox_id"]

        if "visible_instance_ids" in info["images"][0]:
            masks = []
            for i in range(len(info["images"])):
                mask = np.zeros((num_instances), dtype=bool)
                visible_id = info["images"][i]["visible_instance_ids"]
                mask[visible_id] = True
                masks.append(mask)
            ann_info["visible_instance_masks"] = np.stack(masks, axis=1)

        if self.remove_dontcare:
            filter_mask = ann_info["gt_labels_3d"] >= 0  # type: ignore
            for key in [
                "gt_bboxes_3d",
                "gt_labels_3d",
                "bbox_id",
                "visible_instance_masks",
            ]:
                ann_info[key] = ann_info[key][filter_mask]  # type: ignore
            ann_info["gt_names"] = [
                x for i, x in enumerate(ann_info["gt_names"]) if filter_mask[i]
            ]
        return ann_info

    def load_grounding_data(self):
        self.scans = dict()
        for data in self.data_list:
            scan_id = data["scan_id"]
            self.scans[scan_id] = data
        self.scan_ids = list(self.scans.keys())

        self.data_list = []
        for vg_file in as_sequence(self.vg_file):
            self.data_list.extend(
                json.load(open(os.path.join(self.data_root, vg_file), "r"))
            )

        if self.dataset_length is not None:
            interval = len(self.data_list) / self.dataset_length
            output = []
            for i in range(self.dataset_length):
                output.append(self.data_list[int(interval * i)])
            self.data_list = output

        for data in self.data_list:
            if "distractor_ids" in data:
                data["is_unique"] = len(data["distractor_ids"]) == 0
                data["is_hard"] = len(data["distractor_ids"]) > 3
            if "text" in data:
                data["is_view_dep"] = self._is_view_dep(data["text"])

        self.scan_id_to_data_idx = {}
        for scan_id in self.scan_ids:
            self.scan_id_to_data_idx[scan_id] = []
        for i, d in enumerate(self.data_list):
            self.scan_id_to_data_idx[d["scan_id"]].append(i)

    @staticmethod
    def _is_view_dep(text):
        rels = [
            "front",
            "behind",
            "back",
            "left",
            "right",
            "facing",
            "leftmost",
            "rightmost",
            "looking",
            "across",
        ]
        words = set(text.split())
        return any(rel in words for rel in rels)

    def merge_grounding_data(self, data_infos):
        output = dict(
            scan_id=data_infos[0].pop("scan_id"),
            text="",
        )
        if "target_id" in data_infos[0]:
            for key in [
                "target_id",
                "distractor_ids",
                "target",
                "anchors",
                "anchor_ids",
                "tokens_positive",
            ]:
                if key in data_infos[0]:
                    output[key] = []  # type: ignore

        for data_info in data_infos:
            if (
                "target_id" in data_info
                and data_info["target_id"] in output["target_id"]
            ):
                continue

            if self.tokens_positive_rebuild and "target" in data_info:
                start_idx = data_info["text"].find(data_info["target"])
                end_idx = start_idx + len(data_info["target"])
                tokens_positive = [[start_idx, end_idx]]
            elif "tokens_positive" in data_info:
                tokens_positive = data_info["tokens_positive"]
            else:
                tokens_positive = None

            if len(output["text"]) == 0:
                output["text"] = data_info["text"]
            else:
                if tokens_positive is not None:
                    tokens_positive = np.array(tokens_positive)
                    tokens_positive += len(output["text"]) + len(
                        self.sep_token
                    )
                    tokens_positive = tokens_positive.tolist()
                output["text"] += self.sep_token + data_info["text"]
            if tokens_positive is not None:
                output["tokens_positive"].append(tokens_positive)  # type: ignore
            for k in [
                "target_id",
                "distractor_ids",
                "target",
                "anchors",
                "anchor_ids",
            ]:
                if k not in data_info:
                    continue
                output[k].append(data_info[k])  # type: ignore
        return output

    def get_data_grounding(self, data_info):
        scan_id = data_info["scan_id"]
        scan_data = copy.deepcopy(self.scans[scan_id])
        data_info = [data_info]
        if self.num_text > 1:
            data_idx = self.scan_id_to_data_idx[scan_id]
            num_sample = int(np.random.rand() * self.num_text)
            num_sample = max(min(num_sample, len(data_idx)) - 1, 1)
            sample_idx = sample(len(data_idx), num_sample, fix_interval=False)
            for i in sample_idx:
                data_info.append(copy.deepcopy(self.data_list[data_idx[i]]))

        data_info = self.merge_grounding_data(data_info)
        scan_data["text"] = data_info["text"]

        if "ann_info" in scan_data and "target_id" in data_info:
            tokens_positive = []
            obj_idx = []
            for i, (target_name, id) in enumerate(
                zip(data_info["target"], data_info["target_id"], strict=False)
            ):
                mask = np.logical_and(
                    scan_data["ann_info"]["bbox_id"] == id,
                    np.array(scan_data["ann_info"]["gt_names"]) == target_name,
                )
                if np.sum(mask) != 1:
                    continue
                obj_idx.append(np.where(mask)[0][0])
                tokens_positive.append(data_info["tokens_positive"][i])
            obj_idx = np.array(obj_idx, dtype=np.int32)
            scan_data["ann_info"]["gt_bboxes_3d"] = scan_data["ann_info"][
                "gt_bboxes_3d"
            ][obj_idx]
            scan_data["ann_info"]["gt_labels_3d"] = scan_data["ann_info"][
                "gt_labels_3d"
            ][obj_idx]
            scan_data["ann_info"]["gt_names"] = [
                scan_data["ann_info"]["gt_names"][i] for i in obj_idx
            ]
            if "visible_instance_masks" in scan_data["ann_info"]:
                scan_data["ann_info"]["visible_instance_masks"] = scan_data[
                    "ann_info"
                ]["visible_instance_masks"][obj_idx]
            scan_data["tokens_positive"] = tokens_positive
        elif "tokens_positive" in data_info:
            scan_data["tokens_positive"] = data_info.get("tokens_positive")
        return scan_data

    def get_data(self, idx):
        data = copy.deepcopy(self.data_list[idx])
        if self.mode == "grounding":
            data = self.get_data_grounding(data)

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

    def __getitem__(self, idx):
        if not self.initialized:
            self._init()

        if self.test_mode:
            data = self.get_data(idx)
            if data is None:
                raise Exception(
                    "Test time pipline should not get `None` data_sample"
                )
            return data

        for _ in range(self.max_refetch + 1):
            data = self.get_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = np.random.randint(0, len(self))
                continue
            return data

        raise FileNotFoundError(
            f"Cannot find valid image after {self.max_refetch}! "
            "Please check your image path and pipeline"
        )
