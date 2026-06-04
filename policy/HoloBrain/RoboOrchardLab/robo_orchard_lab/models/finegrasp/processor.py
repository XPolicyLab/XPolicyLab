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
from dataclasses import dataclass, field
from typing import Dict, List

import fsspec
import numpy as np
import open3d
import scipy.io as scio
import torch
from graspnetAPI.graspnet_eval import GraspGroup
from PIL import Image

from robo_orchard_lab.models.finegrasp.utils import (
    ModelFreeCollisionDetector,
    pred_decode,
)
from robo_orchard_lab.processing.io_processor.base import (
    ClassType_co,
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.utils import depth_to_range_image, seed_everything

__all__ = [
    "GraspInput",
    "GraspOutput",
    "FineGraspProcessor",
    "FineGraspProcessorCfg",
]

import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@dataclass
class GraspInput:
    """Input data for FineGraspProcessor.

    Args:
        rgb_image (str | np.ndarray): RGB image of shape (H, W, 3),
        depth_image (str | np.ndarray): Depth image of shape (H, W),
        intrinsic_matrix (str | np.ndarray): Camera intrinsic matrix of shape (3, 3),
        points (np.ndarray | None): Point cloud of shape (N, 3). If provided,
            depth_image will be ignored.
        depth_scale (float): Scale factor to convert raw depth values to
            metric units. For example, if the depth map values are in
            millimeters and you want meters, set `depth_scale=1000.0`.
        grasp_workspace (List[float]): Workspace limits [xmin, xmax, ymin,
            ymax, zmin, zmax].
        num_sample_points (int): Number of points to sample from the point
            cloud.

    Example:
        >>> from robo_orchard_lab.models.finegrasp.processor import GraspInput
        >>> import numpy as np
        >>> rgb_image = np.random.rand(480, 640, 3).astype(np.float32)
        >>> depth_image = (
        ...     np.random.rand(480, 640).astype(np.float32) * 1000
        ... )  # in mm
        >>> intrinsic_matrix = np.array(
        ...     [[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32
        ... )
        >>> depth_scale = 1000.0
        >>> grasp_workspace = [-1, 1, -1, 1, 0.0, 2.0]
        >>> input_data = GraspInput(
        ...     rgb_image=rgb_image,
        ...     depth_image=depth_image,
        ...     depth_scale=depth_scale,
        ...     intrinsic_matrix=intrinsic_matrix,
        ...     grasp_workspace=grasp_workspace,
        ... )
        >>> print(input_data.grasp_workspace)
        [-1, 1, -1, 1, 0.0, 2.0]
        >>> print(input_data.rgb_image.shape)
        (480, 640, 3)
        >>> print(input_data.depth_image.shape)
        (480, 640)
        >>> print(input_data.intrinsic_matrix.shape)
        (3, 3)

    """  # noqa: E501

    rgb_image: str | np.ndarray

    depth_image: str | np.ndarray

    intrinsic_matrix: str | np.ndarray

    points: np.ndarray | None = None

    depth_scale: float = 1.0

    grasp_workspace: List[float] = field(
        default_factory=list
    )  # xmin, xmax, ymin, ymax, zmin, zmax

    num_sample_points: int = 20000

    def __post_init__(self):
        if isinstance(self.depth_image, str):
            with fsspec.open(self.rgb_image, "rb") as f:
                self.rgb_image = np.array(Image.open(f), dtype=np.float32)

        if isinstance(self.depth_image, str):
            with fsspec.open(self.depth_image, "rb") as f:
                self.depth_image = np.array(Image.open(f), dtype=np.float32)

        if isinstance(self.intrinsic_matrix, str):
            with fsspec.open(self.intrinsic_matrix, "rb") as f:
                self.intrinsic_matrix = scio.loadmat(f)["intrinsic_matrix"]


@dataclass
class GraspOutput:
    """Output data for FineGraspProcessor.

    Args:
        grasp_poses (GraspGroup): Detected grasp poses.

    Example:
        >>> from robo_orchard_lab.models.finegrasp.processor import GraspOutput
        >>> import numpy as np
        >>> from graspnetAPI.grasp import GraspGroup, GRASP_ARRAY_LEN
        >>> np.random.seed(0)
        >>> grasp_poses_array = np.random.rand(10, GRASP_ARRAY_LEN).astype(
        ...     np.float32
        ... )
        >>> grasp_poses = GraspGroup(grasp_poses_array)
        >>> output_data = GraspOutput(grasp_poses=grasp_poses)
        >>> print(output_data.grasp_poses[0])
        Grasp: score:0.54881352186203, width:0.7151893377304077,
        ... height:0.6027633547782898, depth:0.5448831915855408,
        ... translation:[0.92559665 0.07103606 0.0871293 ]
        rotation:
        [[0.4236548  0.6458941  0.4375872 ]
        [0.891773   0.96366274 0.3834415 ]
        [0.79172504 0.5288949  0.56804454]]
        object id:0
    """

    grasp_poses: GraspGroup


class FineGraspProcessor(ModelIOProcessor):
    cfg: "FineGraspProcessorCfg"  # for type hint

    def __init__(self, cfg: "FineGraspProcessorCfg"):
        seed_everything(2025)
        super().__init__(cfg)

    def get_normal(self, point_cloud):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(point_cloud)

        # compute normals
        pcd.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        return np.asarray(pcd.normals)

    def pre_process(
        self, input_data: GraspInput, device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(input_data, GraspInput)
        assert (
            input_data.depth_image is not None or input_data.points is not None
        ), "Either depth image or point cloud must be provided."

        if input_data.points is None:
            points = depth_to_range_image(
                depth=input_data.depth_image,
                camera_intrinsic=input_data.intrinsic_matrix,
                depth_scale=input_data.depth_scale,
            )
        else:
            points = input_data.points

        # Normalize RGB image to [0, 1]
        colors = input_data.rgb_image / 255.0

        # Set grasp_workspace to filter input points and colors
        if (
            input_data.grasp_workspace is None
            or len(input_data.grasp_workspace) != 6
        ):
            raise ValueError(
                "grasp_workspace must be provided and have length 6."
            )

        xmin, xmax, ymin, ymax, zmin, zmax = (
            input_data.grasp_workspace[0],
            input_data.grasp_workspace[1],
            input_data.grasp_workspace[2],
            input_data.grasp_workspace[3],
            input_data.grasp_workspace[4],
            input_data.grasp_workspace[5],
        )
        x_mask = (points[..., 0] > xmin) & (points[..., 0] < xmax)
        y_mask = (points[..., 1] > ymin) & (points[..., 1] < ymax)
        z_mask = (points[..., 2] > zmin) & (points[..., 2] < zmax)
        mask = x_mask & y_mask & z_mask
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        # Downsample to fixed number of points
        num_sample_points = input_data.num_sample_points
        if len(points) >= num_sample_points:
            idxs = np.random.choice(
                len(points), num_sample_points, replace=False
            )
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(
                len(points),
                num_sample_points - len(points),
                replace=True,
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        points = points[idxs].astype(np.float32)
        colors = colors[idxs].astype(np.float32)
        coordinates_for_voxel = points.astype(np.float32) / self.cfg.voxel_size
        cloud_normal = self.get_normal(points)

        # Convert to torch tensors and move to device
        point_clouds = torch.from_numpy(points).unsqueeze(0).to(device)
        cloud_colors = torch.from_numpy(colors).unsqueeze(0).to(device)
        cloud_normal = (
            torch.from_numpy(cloud_normal).unsqueeze(0).float().to(device)
        )
        coordinates_for_voxel = (
            torch.from_numpy(coordinates_for_voxel)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        data_dict = {
            "point_clouds": point_clouds,
            "cloud_colors": cloud_colors,
            "cloud_normal": cloud_normal,
            "coordinates_for_voxel": coordinates_for_voxel,
        }

        return data_dict

    def post_process(self, model_outputs, input_data) -> GraspOutput:
        # Decode the model outputs to grasp predictions
        grasp_preds = pred_decode(
            model_outputs,
            self.cfg.grasp_max_width,
            self.cfg.num_seed_points,
        )

        preds = grasp_preds[0].detach().cpu().numpy()

        # Filter the predictions which width is larger than max_gripper_width
        if self.cfg.max_gripper_width is not None:
            width_mask = preds[:, 1] < self.cfg.max_gripper_width
            preds = preds[width_mask]

        gg = GraspGroup(preds)

        # Collision detection
        if self.cfg.collision_thresh > 0:
            logger.info("Start collision detection")

            cloud = input_data["point_clouds"]
            mfcdetector = ModelFreeCollisionDetector(
                cloud[0].cpu().numpy(), voxel_size=self.cfg.voxel_size_cd
            )
            collision_mask = mfcdetector.detect(
                gg,
                approach_dist=0.05,
                collision_thresh=self.cfg.collision_thresh,
            )
            gg = gg[~collision_mask]
        else:
            logger.info("Skip collision detection")

        if len(gg) == 0:
            logger.info("No Grasp detected after collision detection!")

        # Grasp NMS and sort by score
        gg = gg.nms().sort_by_score()
        return GraspOutput(grasp_poses=gg)


class FineGraspProcessorCfg(ModelIOProcessorCfg[FineGraspProcessor]):
    """Configuration for FineGraspProcessor.

    Args:
        voxel_size (float): Voxel size for point cloud downsampling.
        grasp_max_width (float): Maximum width of the gripper for grasp
            prediction.
        num_seed_points (int): Number of seed points for grasp prediction.
        max_gripper_width (float): Maximum gripper width to filter grasps.
        collision_thresh (float): Distance threshold for collision detection.
        voxel_size_cd (float): Voxel size for collision detection.

    Example:
        >>> from robo_orchard_lab.models.finegrasp.processor import (
        ...     FineGraspProcessorCfg,
        ... )
        >>> cfg = FineGraspProcessorCfg()
        >>> print(cfg.voxel_size)
        0.005
        >>> print(cfg.grasp_max_width)
        0.1
    """

    class_type: ClassType_co[FineGraspProcessor] = FineGraspProcessor
    voxel_size: float = 0.005
    grasp_max_width: float = 0.1
    num_seed_points: int = 1024
    max_gripper_width: float = 0.10

    collision_thresh: float = 0.01
    voxel_size_cd: float = 0.01
