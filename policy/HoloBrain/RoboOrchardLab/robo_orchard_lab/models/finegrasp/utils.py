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

import numpy as np
import open3d as o3d
import torch

from robo_orchard_lab.ops.knn.knn_modules import knn


class ModelFreeCollisionDetector:
    """Model-free collision detection in unlabeled scenes.

    Reference:
        https://github.com/iSEE-Laboratory/EconomicGrasp/blob/main/utils/collision_detector.py

    The detector assumes fixed finger width and length for grasp simulation.

    Args:
        scene_points: [numpy.ndarray, (N,3), numpy.float32]
            the scene points to detect
        voxel_size: [float]
            used for downsample

    Examples:
        >>> mfcdetector = ModelFreeCollisionDetector(
        ...     scene_points, voxel_size=0.005
        ... )

        >>> collision_mask = mfcdetector.detect(
        ...     grasp_group, approach_dist=0.03
        ... )

        >>> collision_mask, iou_list = mfcdetector.detect(
        ...     grasp_group,
        ...     approach_dist=0.03,
        ...     collision_thresh=0.05,
        ...     return_ious=True,
        ... )

        >>> collision_mask, empty_mask = mfcdetector.detect(
        ...     grasp_group,
        ...     approach_dist=0.03,
        ...     collision_thresh=0.05,
        ...     return_empty_grasp=True,
        ...     empty_thresh=0.01,
        ... )

        >>> collision_mask, empty_mask, iou_list = mfcdetector.detect(
        ...     grasp_group,
        ...     approach_dist=0.03,
        ...     collision_thresh=0.05,
        ...     return_empty_grasp=True,
        ...     empty_thresh=0.01,
        ...     return_ious=True,
        ... )


    """

    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(
        self,
        grasp_group,
        approach_dist=0.03,
        collision_thresh=0.05,
        return_empty_grasp=False,
        empty_thresh=0.01,
        return_ious=False,
    ):
        """Detect collision of grasps.

        Args:
            grasp_group: [GraspGroup, M grasps]
                the grasps to check
            approach_dist: [float]
                the distance for a gripper to move along approaching direction
                before grasping this shifting space requires no point either
            collision_thresh: [float]
                if global collision iou is greater than this threshold,
                a collision is detected
            return_empty_grasp: [bool]
                if True, return a mask to imply whether there are
                objects in a grasp
            empty_thresh: [float]
                if inner space iou is smaller than this threshold,
                a collision is detected
                only set when [return_empty_grasp] is True
            return_ious: [bool]
                if True, return global collision iou and part collision ious

        Returns:
            collision_mask: [numpy.ndarray, (M,), numpy.bool]
                True implies collision
            [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                True implies empty grasp
                only returned when [return_empty_grasp] is True
            [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                global and part collision ious, containing
                [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations  # noqa: N806
        R = grasp_group.rotation_matrices  # noqa: N806
        heights = grasp_group.heights[:, np.newaxis]
        depths = grasp_group.depths[:, np.newaxis]
        widths = grasp_group.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        # collision detection
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (
            targets[:, :, 2] < heights / 2
        )
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - self.finger_length) & (
            targets[:, :, 0] < depths
        )
        mask3 = targets[:, :, 1] > -(widths / 2 + self.finger_width)
        mask4 = targets[:, :, 1] < -widths / 2
        # right finger mask
        mask5 = targets[:, :, 1] < (widths / 2 + self.finger_width)
        mask6 = targets[:, :, 1] > widths / 2
        # bottom mask
        mask7 = (targets[:, :, 0] <= depths - self.finger_length) & (
            targets[:, :, 0] > depths - self.finger_length - self.finger_width
        )
        # shifting mask
        mask8 = (
            targets[:, :, 0] <= depths - self.finger_length - self.finger_width
        ) & (
            targets[:, :, 0]
            > depths - self.finger_length - self.finger_width - approach_dist
        )

        # get collision mask of each point
        left_mask = mask1 & mask2 & mask3 & mask4
        right_mask = mask1 & mask2 & mask5 & mask6
        bottom_mask = mask1 & mask3 & mask5 & mask7
        shifting_mask = mask1 & mask3 & mask5 & mask8
        global_mask = left_mask | right_mask | bottom_mask | shifting_mask

        # calculate equivalant volume of each part
        left_right_volume = (
            heights
            * self.finger_length
            * self.finger_width
            / (self.voxel_size**3)
        ).reshape(-1)
        bottom_volume = (
            heights
            * (widths + 2 * self.finger_width)
            * self.finger_width
            / (self.voxel_size**3)
        ).reshape(-1)
        shifting_volume = (
            heights
            * (widths + 2 * self.finger_width)
            * approach_dist
            / (self.voxel_size**3)
        ).reshape(-1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)

        # get collison mask
        collision_mask = global_iou > collision_thresh

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [
            collision_mask,
        ]
        if return_empty_grasp:
            inner_mask = mask1 & mask2 & (~mask4) & (~mask6)
            inner_volume = (
                heights * self.finger_length * widths / (self.voxel_size**3)
            ).reshape(-1)
            empty_mask = inner_mask.sum(axis=-1) / inner_volume < empty_thresh
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume + 1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume + 1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume + 1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume + 1e-6)
            ret_value.append(
                [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
            )
        return ret_value


def transform_point_cloud(cloud, transform, format="4x4"):
    """Transform points to new coordinates with transformation matrix.

    Args:
        cloud: [torch.FloatTensor, (N,3)]
            points in original coordinates
        transform: [torch.FloatTensor, (3,3)/(3,4)/(4,4)]
            transformation matrix, could be rotation only or
            rotation+translation
        format: [string, '3x3'/'3x4'/'4x4']
            the shape of transformation matrix
            '3x3' --> rotation matrix
            '3x4'/'4x4' --> rotation matrix + translation matrix

    Returns:
        cloud_transformed: [torch.FloatTensor, (N,3)]
            points in new coordinates
    """
    if not (format == "3x3" or format == "4x4" or format == "3x4"):
        raise ValueError(
            "Unknown transformation format, "
            "only support '3x3' or '4x4' or '3x4'."
        )
    if format == "3x3":
        cloud_transformed = torch.matmul(transform, cloud.T).T
    elif format == "4x4" or format == "3x4":
        ones = cloud.new_ones(cloud.size(0), device=cloud.device).unsqueeze(-1)
        cloud_ = torch.cat([cloud, ones], dim=1)
        cloud_transformed = torch.matmul(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def generate_grasp_views(N=300, phi=None, center=None, r=1):  # noqa: N803
    """View sampling on a unit sphere using Fibonacci lattices.

    Refernce:
        https://arxiv.org/abs/0912.4540

    Args:
        N: [int]
            number of sampled views
        phi: [float]
            constant for view coordinate calculation, different phi's bring
            different distributions, default: (sqrt(5)-1)/2
        center: [np.ndarray, (3,), np.float32]
            sphere center
        r: [float]
            sphere radius

    Returns:
        views: [torch.FloatTensor, (N,3)]
            sampled view coordinates
    """
    if phi is None:
        phi = (np.sqrt(5) - 1) / 2
    if center is None:
        center = np.zeros(3)

    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return torch.from_numpy(views.astype(np.float32))


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """Convert approach vectors and in-plane angles to rotation matrices.

    This function computes rotation matrices based on approach directions
    and in-plane rotation angles, commonly used in grasp representation.

    Args:
        batch_towards: [torch.FloatTensor, (N,3)]
            approach vectors in batch
        batch_angle: [torch.floatTensor, (N,)]
            in-plane rotation angles in batch

    Returns:
        batch_matrix: [torch.floatTensor, (N,3,3)]
            rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(
        axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device
    )
    zeros = torch.zeros(
        axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device
    )
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = torch.norm(axis_y, dim=-1) == 0
    axis_y[mask_y, 1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.linalg.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack(  # noqa: N806
        [ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1
    )
    R1 = R1.reshape([-1, 3, 3])  # noqa: N806
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)  # noqa: N806
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix


def compute_pointwise_dists(A, B):  # noqa: N803
    """Compute pair-wise point distances in two matrices.

    Args:
        A: [torch.tensor, (N,3), np.float32]
            point cloud A
        B: [torch.tensor, (N,3), np.float32]
            point cloud B

    Returns:
        dists: [np.ndarray, (N,), np.float32]
            distance matrix
    """
    dists = torch.norm(A - B, dim=-1)
    return dists


def process_grasp_labels(end_points, model_cfgs):
    """Process labels according to scene points and object poses.

    Reference:
        https://github.com/iSEE-Laboratory/EconomicGrasp/blob/main/utils/label_generation.py

    """
    seed_xyzs = end_points[
        "xyz_graspable"
    ]  # [B (batch size), 1024 (scene points after sample), 3]
    pred_top_view_inds = end_points[
        "grasp_top_view_inds"
    ]  # [B (batch size), 1024 (scene points after sample)]
    batch_size, num_samples, _ = seed_xyzs.size()

    valid_points_count = 0
    valid_views_count = 0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_valid_mask = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # [1024 (scene points after sample), 3]
        pred_top_view = pred_top_view_inds[
            i
        ]  # [1024 (scene points after sample)]
        poses = end_points["object_poses_list"][
            i
        ]  # a list with length of object amount, each has size [3, 4]

        # get merged grasp points for label computation
        # transform the view from object coordinate system
        # to scene coordinate system
        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        view_graspness_merged = []
        top_view_index_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points["grasp_points_list"][i][
                obj_idx
            ]  # [objects points, 3]
            grasp_rotations = end_points["grasp_rotations_list"][i][
                obj_idx
            ]  # [objects points, num_of_view]
            grasp_depth = end_points["grasp_depth_list"][i][
                obj_idx
            ]  # [objects points, num_of_view]
            grasp_scores = end_points["grasp_scores_list"][i][
                obj_idx
            ]  # [objects points, num_of_view]
            grasp_widths = end_points["grasp_widths_list"][i][
                obj_idx
            ]  # [objects points, num_of_view]
            view_graspness = end_points["view_graspness_list"][i][
                obj_idx
            ]  # [objects points, 300]
            top_view_index = end_points["top_view_index_list"][i][
                obj_idx
            ]  # [objects points, num_of_view]
            num_grasp_points = grasp_points.shape[0]

            # generate and transform template grasp views
            grasp_views = generate_grasp_views(model_cfgs.num_view).to(
                pose.device
            )  # [300 (views), 3 (coordinate)]
            grasp_points_trans = transform_point_cloud(
                grasp_points, pose, "3x4"
            )
            grasp_views_trans = transform_point_cloud(
                grasp_views, pose[:3, :3], "3x3"
            )
            # [300 (views), 3 (coordinate)],
            # after translation to scene coordinate system

            # generate and transform template grasp view rotation
            angles = torch.zeros(
                grasp_views.size(0),
                dtype=grasp_views.dtype,
                device=grasp_views.device,
            )
            grasp_views_rot = batch_viewpoint_params_to_matrix(
                -grasp_views, angles
            )
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)
            # [300 (views), 3, 3 (the rotation matrix)]

            # assign views after transform (the view will not exactly match)
            grasp_views_ = (
                grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            )
            grasp_views_trans_ = (
                grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            )
            view_inds = (
                knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1
            )  # [300]
            view_graspness_trans = torch.index_select(
                view_graspness, 1, view_inds
            )  # [object points, 300]
            grasp_views_rot_trans = torch.index_select(
                grasp_views_rot_trans, 0, view_inds
            )
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(
                num_grasp_points, -1, -1, -1
            )
            # [object points, 300, 3, 3]

            # -1 means that when we transform the top 60 views into the
            # scene coordinate,
            # some views will have no matching
            # It means that two views in the object coordinate match to one
            # view in the scene coordinate
            top_view_index_trans = -1 * torch.ones(
                (num_grasp_points, grasp_rotations.shape[1]), dtype=torch.long
            ).to(seed_xyz.device)
            tpid, tvip, tids = torch.where(
                view_inds == top_view_index.unsqueeze(-1)
            )
            top_view_index_trans[tpid, tvip] = (
                tids  # [objects points, num_of_view]
            )

            # add to list
            grasp_points_merged.append(grasp_points_trans)
            view_graspness_merged.append(view_graspness_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_rotations_merged.append(grasp_rotations)
            grasp_depth_merged.append(grasp_depth)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_views_rot_merged.append(grasp_views_rot_trans)

        grasp_points_merged = torch.cat(
            grasp_points_merged, dim=0
        )  # [all object points, 3]
        view_graspness_merged = torch.cat(
            view_graspness_merged, dim=0
        )  # [all object points, 300]
        top_view_index_merged = torch.cat(
            top_view_index_merged, dim=0
        )  # [all object points, num_of_view]
        grasp_rotations_merged = torch.cat(
            grasp_rotations_merged, dim=0
        )  # [all object points, num_of_view]
        grasp_depth_merged = torch.cat(
            grasp_depth_merged, dim=0
        )  # [all object points, num_of_view]
        grasp_scores_merged = torch.cat(
            grasp_scores_merged, dim=0
        )  # [all object points, num_of_view]
        grasp_widths_merged = torch.cat(
            grasp_widths_merged, dim=0
        )  # [all object points, num_of_view]
        grasp_views_rot_merged = torch.cat(
            grasp_views_rot_merged, dim=0
        )  # [all object points, 300, 3, 3]

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
        grasp_points_merged_ = (
            grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
        )
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(
            grasp_points_merged, 0, nn_inds
        )
        # [1024 (scene points after sample), 3]
        grasp_views_rot_merged = torch.index_select(
            grasp_views_rot_merged, 0, nn_inds
        )
        # [1024 (scene points after sample), 300, 3, 3]
        view_graspness_merged = torch.index_select(
            view_graspness_merged, 0, nn_inds
        )
        # [1024 (scene points after sample), 300]
        top_view_index_merged = torch.index_select(
            top_view_index_merged, 0, nn_inds
        )
        # [1024 (scene points after sample), num_of_view]
        grasp_rotations_merged = torch.index_select(
            grasp_rotations_merged, 0, nn_inds
        )
        # [1024 (scene points after sample), num_of_view]
        grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_scores_merged = torch.index_select(
            grasp_scores_merged, 0, nn_inds
        )
        # [1024 (scene points after sample), num_of_view]
        grasp_widths_merged = torch.index_select(
            grasp_widths_merged, 0, nn_inds
        )
        # [1024 (scene points after sample), num_of_view]

        # select top view's rot, score and width
        # we only assign labels when the pred view is in the pre-defined
        # 60 top view, others are zero
        pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(
            -1, -1, 3, 3
        )
        # [1024 (points after sample), 1, 3, 3]
        top_grasp_views_rot = torch.gather(
            grasp_views_rot_merged, 1, pred_top_view_
        ).squeeze(1)
        # [1024 (points after sample), 3, 3]
        pid, vid = torch.where(
            pred_top_view.unsqueeze(-1) == top_view_index_merged
        )
        # both pid and vid are [true numbers], where(condition) equals to
        # nonzero(condition)
        top_grasp_rotations = 12 * torch.ones(
            num_samples, dtype=torch.int32
        ).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_depth = 4 * torch.ones(num_samples, dtype=torch.int32).to(
            seed_xyz.device
        )
        # [1024 (points after sample)]
        top_grasp_scores = torch.zeros(num_samples, dtype=torch.float32).to(
            seed_xyz.device
        )
        # [1024 (points after sample)]
        top_grasp_widths = 0.1 * torch.ones(
            num_samples, dtype=torch.float32
        ).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_rotations[pid] = torch.gather(
            grasp_rotations_merged[pid], 1, vid.view(-1, 1)
        ).squeeze(1)
        top_grasp_depth[pid] = torch.gather(
            grasp_depth_merged[pid], 1, vid.view(-1, 1)
        ).squeeze(1)
        top_grasp_scores[pid] = torch.gather(
            grasp_scores_merged[pid], 1, vid.view(-1, 1)
        ).squeeze(1)
        top_grasp_widths[pid] = torch.gather(
            grasp_widths_merged[pid], 1, vid.view(-1, 1)
        ).squeeze(1)

        # only compute loss in the points with correct matching
        # (so compute the mask first)
        dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
        valid_point_mask = dist < 0.005
        valid_view_mask = torch.zeros(num_samples, dtype=torch.bool).to(
            seed_xyz.device
        )
        valid_view_mask[pid] = True
        valid_points_count = valid_points_count + torch.sum(valid_point_mask)
        valid_views_count = valid_views_count + torch.sum(valid_view_mask)
        valid_mask = valid_point_mask & valid_view_mask

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)
        batch_grasp_rotations.append(top_grasp_rotations)
        batch_grasp_depth.append(top_grasp_depth)
        batch_grasp_scores.append(top_grasp_scores)
        batch_grasp_widths.append(top_grasp_widths)
        batch_valid_mask.append(valid_mask)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    # [B (batch size), 1024 (scene points after sample), 3]
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    # [B (batch size), 1024 (scene points after sample), 3, 3]
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    # [B (batch size), 1024 (scene points after sample), 300]
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_valid_mask = torch.stack(batch_valid_mask, 0)
    # [B (batch size), 1024 (scene points after sample)]

    end_points["batch_grasp_point"] = batch_grasp_points
    end_points["batch_grasp_rotations"] = batch_grasp_rotations
    end_points["batch_grasp_depth"] = batch_grasp_depth
    end_points["batch_grasp_score"] = batch_grasp_scores
    end_points["batch_grasp_width"] = batch_grasp_widths
    end_points["batch_grasp_view_graspness"] = batch_view_graspness
    end_points["batch_valid_mask"] = batch_valid_mask
    end_points["C: Valid Points"] = valid_points_count / batch_size
    return batch_grasp_views_rot, end_points


def pred_decode(model_outputs, grasp_max_width, num_seed_points):
    batch_size = len(model_outputs["point_clouds"])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = model_outputs["xyz_graspable"][i].float()

        # composite score estimation
        grasp_score_prob = model_outputs["grasp_score_pred"][i].float()
        score = (
            torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1])
            .view(-1, 1)
            .expand(-1, grasp_score_prob.shape[1])
            .to(grasp_score_prob)
        )
        score = torch.sum(score * grasp_score_prob, dim=0)
        grasp_score = score.view(-1, 1)

        grasp_angle_pred = model_outputs["grasp_angle_pred"][i].float()
        grasp_angle, grasp_angle_indxs = torch.max(
            grasp_angle_pred.squeeze(0), 0
        )
        grasp_angle = grasp_angle_indxs * np.pi / 12

        grasp_depth_pred = model_outputs["grasp_depth_pred"][i].float()
        grasp_depth, grasp_depth_indxs = torch.max(
            grasp_depth_pred.squeeze(0), 0
        )
        grasp_depth = (grasp_depth_indxs + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)

        grasp_width = 1.2 * model_outputs["grasp_width_pred"][i] / 10.0
        grasp_width = torch.clamp(grasp_width, min=0.0, max=grasp_max_width)
        grasp_width = grasp_width.view(-1, 1)

        approaching = -model_outputs["grasp_top_view_xyz"][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(num_seed_points, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat(
                [
                    grasp_score,
                    grasp_width,
                    grasp_height,
                    grasp_depth,
                    grasp_rot,
                    grasp_center,
                    obj_ids,
                ],
                axis=-1,
            )
        )
    return grasp_preds
