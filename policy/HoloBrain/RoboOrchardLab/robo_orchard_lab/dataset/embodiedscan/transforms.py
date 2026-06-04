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
import random
from typing import List

import cv2
import numpy as np
import torch

from robo_orchard_lab.dataset.embodiedscan.embodiedscan_det_grounding_dataset import (  # noqa: E501
    DEFAULT_CLASSES,
)
from robo_orchard_lab.dataset.embodiedscan.utils import sample


class LoadMultiViewImageDepthFromFile:
    def __init__(
        self,
        num_views=50,
        max_num_views=None,
        sample_mode="fix",
        rotate_3rscan=False,
        load_img=True,
        load_depth=True,
        dst_intrinsic=None,
        dst_wh=None,
        random_crop_range=None,
    ):
        self.num_views = num_views
        if max_num_views is None:
            max_num_views = num_views
        self.max_num_views = max_num_views
        assert sample_mode in ["fix", "random"]
        self.sample_mode = sample_mode
        self.rotate_3rscan = rotate_3rscan
        self.load_img = load_img
        self.load_depth = load_depth
        self.random_crop_range = random_crop_range

        if isinstance(dst_intrinsic, List):
            dst_intrinsic = np.array(dst_intrinsic)

        self.dst_wh = dst_wh
        if dst_intrinsic is not None:
            assert dst_wh is not None, (
                "dst_wh should be set when dst_intrinsic is set"
            )
            _tmp = np.eye(4)
            _tmp[:3, :3] = dst_intrinsic[:3, :3]
            self.dst_intrinsic = _tmp
            u, v = np.arange(dst_wh[0]), np.arange(dst_wh[1])
            u = np.repeat(u[None], dst_wh[1], 0)
            v = np.repeat(v[:, None], dst_wh[0], 1)
            uv = np.stack([u, v, np.ones_like(u)], axis=-1)
            self.dst_pts = uv @ np.linalg.inv(self.dst_intrinsic[:3, :3]).T
        else:
            self.dst_intrinsic = None

    def __call__(self, data):
        num_views = len(data["img_path"])
        num_sample = min(max(self.num_views, num_views), self.max_num_views)
        sample_idx = sample(num_views, num_sample, self.sample_mode == "fix")
        imgs = []
        depths = []
        intrinsics = []
        extrinsics = []
        for idx in sample_idx:
            if isinstance(data["intrinsic"], List):
                intrinsic = copy.deepcopy(data["intrinsic"][idx])
            else:
                intrinsic = copy.deepcopy(data["intrinsic"])
            extrinsic = data["extrinsic"][idx]

            if self.load_img:
                img = cv2.imread(data["img_path"][idx])
            else:
                img = None

            if self.load_depth:
                depth = (
                    cv2.imread(data["depth_path"][idx], cv2.IMREAD_UNCHANGED)
                    / data["depth_shift"]
                )
            else:
                depth = None

            if (
                img is not None
                and depth is not None
                and depth.shape[:2] != img.shape[:2]
            ):
                depth = cv2.resize(
                    depth, img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR
                )

            if img is not None or depth is not None:
                img, depth, intrinsic = self.resize(img, depth, intrinsic)

            if self.random_crop_range is not None:
                img, depth, intrinsic = self.random_crop(img, depth, intrinsic)

            if self.rotate_3rscan and "3rscan" in data["scan_id"]:
                if img is not None:
                    img = np.transpose(img, (1, 0, 2))
                if depth is not None:
                    depth = np.transpose(depth, (1, 0))
                rot_mat = np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
                rot_mat = np.linalg.inv(intrinsic) @ rot_mat @ intrinsic
                extrinsic = rot_mat @ extrinsic

            if img is not None:
                imgs.append(img)
            if depth is not None:
                depths.append(depth)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

        if len(imgs) != 0:
            data["imgs"] = np.stack(imgs)
        if len(depths) != 0:
            data["depths"] = np.stack(depths)[..., None]
        data["intrinsic"] = np.stack(intrinsics)
        data["extrinsic"] = np.stack(extrinsics)

        if "ann_info" in data and "visible_instance_masks" in data["ann_info"]:
            data["ann_info"]["visible_instance_masks"] = data["ann_info"][
                "visible_instance_masks"
            ][:, sample_idx]
        return data

    def random_crop(self, img=None, depth=None, intrinsic=None):
        if (img is None) and (depth is None):
            return img, depth, intrinsic
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = depth.shape[:2]
        x = int(np.random.uniform(*self.random_crop_range[0]) * w)
        y = int(np.random.uniform(*self.random_crop_range[1]) * h)
        transform_matrix = np.eye(4)
        transform_matrix[:2, 2] -= np.array([x, y])
        intrinsic = transform_matrix @ intrinsic

        if img is not None:
            _img = copy.deepcopy(img)
            img = cv2.warpAffine(
                img, transform_matrix[:2, :3], img.shape[:2][::-1]
            )

        if depth is not None:
            depth = cv2.warpAffine(
                depth, transform_matrix[:2, :3], depth.shape[:2][::-1]
            )
        return img, depth, intrinsic

    def resize(self, img=None, depth=None, intrinsic=None):
        if self.dst_intrinsic is not None:
            assert intrinsic is not None, "intrinsic should not be None"
            src_intrinsic = intrinsic[:3, :3]
            src_uv = self.dst_pts @ src_intrinsic.T
            src_uv = src_uv.astype(np.float32)
            if img is not None:
                img = cv2.remap(
                    img,
                    src_uv[..., 0],
                    src_uv[..., 1],
                    cv2.INTER_NEAREST,
                )
            if depth is not None:
                depth = cv2.remap(
                    depth,
                    src_uv[..., 0],
                    src_uv[..., 1],
                    cv2.INTER_NEAREST,
                )
            intrinsic = self.dst_intrinsic
        elif self.dst_wh is not None:
            origin_wh = img.shape[:2][::-1]
            trans_mat = np.eye(4)
            trans_mat[0, 0] = self.dst_wh[0] / origin_wh[0]
            trans_mat[1, 1] = self.dst_wh[1] / origin_wh[1]
            intrinsic = trans_mat @ intrinsic
            if img is not None:
                img = cv2.resize(img, self.dst_wh)
            if depth is not None:
                depth = cv2.resize(depth, self.dst_wh)
        return img, depth, intrinsic


class CategoryGroundingDataPrepare:
    def __init__(
        self,
        training,
        classes=None,
        max_class=None,
        sep_token="[SEP]",
        filter_others=True,
        z_range=None,
        filter_invisible=True,
        instance_mask_key="visible_instance_masks",
    ):
        if classes is None:
            classes = copy.deepcopy(DEFAULT_CLASSES)
        self.classes = list(classes)
        self.training = training
        self.max_class = max_class
        self.sep_token = sep_token
        self.filter_others = filter_others
        self.z_range = z_range
        self.filter_invisible = filter_invisible
        self.instance_mask_key = instance_mask_key

    def __call__(self, data):
        if "ann_info" in data and "visible_instance_masks" in data["ann_info"]:
            visible_instance_masks = data["ann_info"]["visible_instance_masks"]
            if visible_instance_masks.ndim == 2:
                visible_instance_masks = visible_instance_masks.any(axis=1)

        flag = "ann_info" in data and "gt_labels_3d" in data["ann_info"]

        if flag:
            ann_info = data["ann_info"]
            gt_names = ann_info["gt_names"]
            if self.filter_others:
                mask = ann_info["gt_labels_3d"] >= 0
                ann_info["gt_labels_3d"] = ann_info["gt_labels_3d"][mask]
                ann_info["gt_bboxes_3d"] = ann_info["gt_bboxes_3d"][mask]
                visible_instance_masks = visible_instance_masks[mask]
                gt_names = [x for i, x in enumerate(gt_names) if mask[i]]

        if self.z_range is not None and self.training and flag:
            mask = np.logical_and(
                ann_info["gt_bboxes_3d"][..., 2] >= self.z_range[0],
                ann_info["gt_bboxes_3d"][..., 2] <= self.z_range[1],
            )
            ann_info["gt_labels_3d"] = ann_info["gt_labels_3d"][mask]
            ann_info["gt_bboxes_3d"] = ann_info["gt_bboxes_3d"][mask]
            visible_instance_masks = visible_instance_masks[mask]
            gt_names = [x for i, x in enumerate(gt_names) if mask[i]]

        if self.filter_invisible and self.training and flag:
            ann_info["gt_labels_3d"] = ann_info["gt_labels_3d"][
                visible_instance_masks
            ]
            ann_info["gt_bboxes_3d"] = ann_info["gt_bboxes_3d"][
                visible_instance_masks
            ]
            gt_names = [
                x for i, x in enumerate(gt_names) if visible_instance_masks[i]
            ]

        if self.training and flag:
            ann_info["gt_names"] = gt_names
            if (
                self.max_class is not None
                and len(self.classes) > self.max_class
            ):
                classes = copy.deepcopy(gt_names)
                random.shuffle(self.classes)
                for c in self.classes:
                    if c in classes:
                        continue
                    classes.append(c)
                    if len(classes) >= self.max_class:
                        break
                random.shuffle(classes)
            else:
                classes = copy.deepcopy(self.classes)
        else:
            classes = copy.deepcopy(self.classes)
            gt_names = classes

        data["text"] = self.sep_token.join(classes)
        tokens_positive = []
        for name in gt_names:
            start = data["text"].find(self.sep_token + name + self.sep_token)
            if start == -1:
                if data["text"].startswith(name + self.sep_token):
                    start = 0
                else:
                    start = data["text"].find(self.sep_token + name) + len(
                        self.sep_token
                    )
            else:
                start += len(self.sep_token)
            end = start + len(name)
            tokens_positive.append([[start, end]])
        data["tokens_positive"] = tokens_positive
        return data


class DepthProbLabelGenerator:
    def __init__(
        self,
        max_depth=10,
        min_depth=0.25,
        num_depth=64,
        stride=(8, 16, 32, 64),
        origin_stride=1,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.stride = [x // origin_stride for x in stride]
        self.origin_stride = origin_stride

    def __call__(self, data):
        depth = data["depths"]
        if self.origin_stride != 1:
            H, W = depth.shape[1:3]  # noqa: N806
            depth = [
                cv2.resize(
                    x,
                    (W // self.origin_stride, H // self.origin_stride),
                    interpolation=cv2.INTER_LINEAR,
                )
                for x in depth
            ]
            depth = np.stack(depth)[..., None]
        depth = np.clip(
            depth,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )
        depth_anchor = np.linspace(
            self.min_depth, self.max_depth, self.num_depth
        )
        distance = np.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = np.where(mask, depth_anchor, 0)
        y = depth_gt.sum(axis=-1, keepdims=True) - depth_gt
        depth_valid_mask = depth > 0

        depth_prob_gt = np.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        views, H, W, _ = depth.shape  # noqa: N806
        gt = []
        for s in self.stride:
            gt_tmp = np.reshape(
                depth_prob_gt, (views, H // s, s, W // s, s, self.num_depth)
            )
            gt_tmp = gt_tmp.sum(axis=-2).sum(axis=2)
            mask_tmp = depth_valid_mask.reshape(views, H // s, s, W // s, s, 1)
            mask_tmp = mask_tmp.sum(axis=-2).sum(axis=2)
            gt_tmp /= np.clip(mask_tmp, a_min=1, a_max=None)
            gt_tmp = gt_tmp.reshape(views, -1, self.num_depth)
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        data["depth_prob_gt"] = gt
        return data


class Format:
    def __call__(self, data):
        for key in [
            "imgs",
            "depths",
            "extrinsic",
            "intrinsic",
            "depth_prob_gt",
        ]:
            if key in data:
                data[key] = torch.from_numpy(data[key].astype(np.float32))

        if "imgs" in data:
            data["image_wh"] = torch.tensor(data["imgs"].shape[1:3][::-1])
        elif "depths" in data:
            data["image_wh"] = torch.tensor(data["depths"].shape[1:3][::-1])

        data["projection_mat"] = data["intrinsic"] @ data["extrinsic"]
        if "ann_info" in data:
            ann_info = data.pop("ann_info")
            data["gt_bboxes_3d"] = torch.from_numpy(
                ann_info["gt_bboxes_3d"].astype(np.float32)
            )
            data["gt_labels_3d"] = torch.from_numpy(
                ann_info["gt_labels_3d"].astype(np.int64)
            )
            data["gt_names"] = ann_info["gt_names"]
        return data
