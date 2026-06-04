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


import logging
import os
from typing import List

import numpy as np
import open3d as o3d
import torch
from graspnetAPI import GraspGroup, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    GraspQualityConfigFactory,
    collision_detection,
    compute_closest_points,
    create_table_points,
    get_grasp_score,
    get_scene_name,
    transform_points,
    voxel_sample_points,
)
from graspnetAPI.utils.utils import generate_scene_model

from robo_orchard_lab.models.finegrasp.utils import (
    ModelFreeCollisionDetector,
    pred_decode,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class GraspNetEvalScale(GraspNetEval):
    """GraspNet evaluation for scale-balanced grasping.

    Reference:
        https://github.com/mahaoxiang822/Scale-Balanced-Grasp/blob/main/evaluate_scale.py
    """

    def eval(self, dump_folder, scene_ids, scale, note, proc=2):
        res = np.array(
            self.parallel_eval_scenes(
                scene_ids=scene_ids,
                dump_folder=dump_folder,
                scale=scale,
                proc=proc,
            )
        )
        ap = np.mean(res)
        logger.info(
            "\nEvaluation Result:\n----------\n{}, SCALE={}, AP={}, AP {note}={}".format(  # noqa F524
                self.camera, scale, ap, ap
            )
        )
        return res, ap

    def eval_all(self, dump_folder, scale, proc=2):
        res = np.array(
            self.parallel_eval_scenes(
                scene_ids=list(range(100, 190)),
                dump_folder=dump_folder,
                scale=scale,
                proc=proc,
            )
        )
        ap = [
            np.mean(res),
            np.mean(res[0:30]),
            np.mean(res[30:60]),
            np.mean(res[60:90]),
        ]
        logger.info(
            "\nEvaluation Result:\n----------\n{}, SCALE={}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}".format(  # noqa F524
                self.camera, scale, ap[0], ap[1], ap[2], ap[3]
            )
        )
        return res, ap

    def eval_grasp(
        self,
        grasp_group,
        models,
        dexnet_models,
        poses,
        config,
        table=None,
        voxel_size=0.008,
        TOP_K=50,  # noqa: N803
    ):
        """Evaluate grasps.

        Args:
            grasp_group: GraspGroup instance for evaluation.
            models: in model coordinate
            dexnet_models: models in dexnet format
            poses: from model to camera coordinate
            config: dexnet config.
            table: in camera coordinate
            voxel_size: float of the voxel size.
            TOP_K: int of the number of top grasps to evaluate.
        """
        num_models = len(models)
        # grasp nms
        grasp_group = grasp_group.nms(0.03, 30.0 / 180 * np.pi)

        # assign grasps to object
        # merge and sample scene
        model_trans_list = list()
        seg_mask = list()
        for i, model in enumerate(models):
            model_trans = transform_points(model, poses[i])
            seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
            model_trans_list.append(model_trans)
            seg_mask.append(seg)
        seg_mask = np.concatenate(seg_mask, axis=0)
        scene = np.concatenate(model_trans_list, axis=0)

        # assign grasps
        indices = compute_closest_points(grasp_group.translations, scene)
        model_to_grasp = seg_mask[indices]
        pre_grasp_list = list()
        for i in range(num_models):
            grasp_i = grasp_group[model_to_grasp == i]
            grasp_i.sort_by_score()
            pre_grasp_list.append(grasp_i[:5].grasp_group_array)
        all_grasp_list = np.vstack(pre_grasp_list)
        remain_mask = np.argsort(all_grasp_list[:, 0])[::-1]
        if len(remain_mask) == 0:
            grasp_list = []
            score_list = []
            collision_mask_list = []
            for _i in range(num_models):
                grasp_list.append([])
                score_list.append([])
                collision_mask_list.append([])
            return grasp_list, score_list, collision_mask_list

        min_score = all_grasp_list[
            remain_mask[min(49, len(remain_mask) - 1)], 0
        ]

        grasp_list = []
        for i in range(num_models):
            remain_mask_i = pre_grasp_list[i][:, 0] >= min_score
            grasp_list.append(pre_grasp_list[i][remain_mask_i])
        # grasp_list = pre_grasp_list

        # collision detection
        if table is not None:
            scene = np.concatenate([scene, table])

        collision_mask_list, empty_list, dexgrasp_list = collision_detection(
            grasp_list,
            model_trans_list,
            dexnet_models,
            poses,
            scene,
            outlier=0.05,
            return_dexgrasps=True,
        )

        # evaluate grasps
        # score configurations
        force_closure_quality_config = dict()
        fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
        for value_fc in fc_list:
            value_fc = round(value_fc, 2)
            config["metrics"]["force_closure"]["friction_coef"] = value_fc
            force_closure_quality_config[value_fc] = (
                GraspQualityConfigFactory.create_config(
                    config["metrics"]["force_closure"]
                )
            )
        # get grasp scores
        score_list = list()

        for i in range(num_models):
            dexnet_model = dexnet_models[i]
            collision_mask = collision_mask_list[i]
            dexgrasps = dexgrasp_list[i]
            scores = list()
            num_grasps = len(dexgrasps)
            for grasp_id in range(num_grasps):
                if collision_mask[grasp_id]:
                    scores.append(-1.0)
                    continue
                if dexgrasps[grasp_id] is None:
                    scores.append(-1.0)
                    continue
                grasp = dexgrasps[grasp_id]
                score = get_grasp_score(
                    grasp, dexnet_model, fc_list, force_closure_quality_config
                )
                scores.append(score)
            score_list.append(np.array(scores))

        return grasp_list, score_list, collision_mask_list

    def eval_scene(
        self,
        scene_id,
        dump_folder,
        scale,
        TOP_K=20,  # noqa: N803
        return_list=False,
        vis=False,
        max_width=0.1,
    ):
        """Evaluate a single scene.

        Args:
            scene_id: int of the scene index.
            dump_folder: string of the folder that saves the dumped npy files.
            TOP_K: int of the top number of grasp to evaluate
            return_list: bool of whether to return the result list.
            vis: bool of whether to show the result
            max_width: float of the maximum gripper width in evaluation

        Returns:
            scene_accuracy: np.array[256, 50, 6] of the accuracy tensor.
        """
        config = get_config()
        table = create_table_points(
            1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008
        )

        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

        model_list, dexmodel_list, _ = self.get_scene_models(
            scene_id, ann_id=0
        )

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        for ann_id in range(256):
            grasp_group = GraspGroup().from_npy(
                os.path.join(
                    dump_folder,
                    get_scene_name(scene_id),
                    self.camera,
                    "%04d.npy" % (ann_id,),
                )
            )
            _, pose_list, camera_pose, align_mat = self.get_model_poses(
                scene_id, ann_id
            )
            table_trans = transform_points(
                table, np.linalg.inv(np.matmul(align_mat, camera_pose))
            )

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = gg_array[:, 1] < 0
            max_width_mask = gg_array[:, 1] > max_width
            gg_array[min_width_mask, 1] = 0
            gg_array[max_width_mask, 1] = max_width
            if scale == "small":
                width_mask = gg_array[:, 1] < 0.04
            elif scale == "medium":
                width_mask = (gg_array[:, 1] >= 0.04) * (gg_array[:, 1] < 0.07)
            elif scale == "large":
                width_mask = gg_array[:, 1] >= 0.07
            else:
                print("unknown scale")
                exit(0)
            gg_array = gg_array[width_mask]
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = self.eval_grasp(
                grasp_group,
                model_sampled_list,
                dexmodel_list,
                pose_list,
                config,
                table=table_trans,
                voxel_size=0.008,
                TOP_K=TOP_K,
            )

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [
                x for x in collision_mask_list if len(x) != 0
            ]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = (
                np.concatenate(grasp_list),
                np.concatenate(score_list),
                np.concatenate(collision_mask_list),
            )

            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(
                    self.root,
                    "scene_%04d" % scene_id,
                    ann_id,
                    return_poses=False,
                    align=False,
                    camera=self.camera,
                )
                import copy

                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5  # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype=np.float32)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                o3d.visualization.draw_geometries(
                    [pcd, *grasps_geometry, *model_list]
                )
                o3d.visualization.draw_geometries(
                    [*grasps_geometry, *model_list, t]
                )
            # sort in scene level
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = (
                grasp_list[indices],
                score_list[indices],
                collision_mask_list[indices],
            )

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            # calculate AP
            grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0, TOP_K):
                    if k + 1 > len(score_list):
                        grasp_accuracy[k, fric_idx] = np.sum(
                            ((score_list <= fric) & (score_list > 0)).astype(
                                int
                            )
                        ) / (k + 1)
                    else:
                        grasp_accuracy[k, fric_idx] = np.sum(
                            (
                                (score_list[0 : k + 1] <= fric)
                                & (score_list[0 : k + 1] > 0)
                            ).astype(int)
                        ) / (k + 1)

            scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return (
                scene_accuracy,
                grasp_list_list,
                score_list_list,
                collision_list_list,
            )

    def parallel_eval_scenes(self, scene_ids, dump_folder, scale, proc=2):
        """Parallel evaluation of multiple scenes.

        Args:
            scene_ids: list of int of scene index.
            dump_folder: string of the folder that saves the npy files.
            proc: int of the number of processes to use to evaluate.

        Returns:
            scene_acc_list: list of the scene accuracy.
        """
        from multiprocessing import Pool

        p = Pool(processes=proc)
        res_list = []
        for scene_id in scene_ids:
            res_list.append(
                p.apply_async(self.eval_scene, (scene_id, dump_folder, scale))
            )
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list


class GraspNetMetric:
    def __init__(
        self,
        voxel_size_cd: float = 0.005,
        collision_thresh: float = 0.01,
        eval_save_dir: str = "./eval_results",
        camera: str = "realsense",
        test_mode: List[str] = ["test", "test_scale"],  # noqa: B006
        num_test_proc: int = 16,
        grasp_max_width: float = 0.1,
        num_seed_points: int = 4096,
        data_root: str = "./data",
    ):
        self.voxel_size_cd = voxel_size_cd
        self.collision_thresh = collision_thresh
        self.eval_save_dir = eval_save_dir
        self.camera = camera
        self.test_mode = test_mode
        self.num_test_proc = num_test_proc
        self.grasp_max_width = grasp_max_width
        self.num_seed_points = num_seed_points
        self.data_root = data_root
        self.results = dict()
        self.reset()

    def update(self, batch, model_outputs):
        grasp_preds = pred_decode(
            model_outputs, self.grasp_max_width, self.num_seed_points
        )
        data_len = len(grasp_preds)
        for i in range(data_len):
            data_idx = batch["data_idx"][i].item()
            preds = grasp_preds[i].detach().cpu().numpy()

            gg = GraspGroup(preds)
            # collision detection
            if self.collision_thresh > 0:
                cloud = batch["cloud_masked"][i][0]
                mfcdetector = ModelFreeCollisionDetector(
                    cloud.cpu().numpy(), voxel_size=self.voxel_size_cd
                )
                collision_mask = mfcdetector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=self.collision_thresh,
                )
                gg = gg[~collision_mask]

            # save grasps
            scene_name = batch["scene_name"][i]
            save_dir = os.path.join(
                self.eval_save_dir, scene_name, self.camera
            )
            save_path = os.path.join(
                save_dir, str(data_idx % 256).zfill(4) + ".npy"
            )
            os.makedirs(save_dir, exist_ok=True)
            gg.save_npy(save_path)

    def compute(self, accelerator):
        return None

    def eval(self):
        torch.cuda.empty_cache()  # 清空 GPU 缓存
        test_mode_list = self.test_mode
        for test_mode in test_mode_list:
            if test_mode == "test":
                logger.info("=" * 50 + f"BEGIN EVAL {test_mode}" + "=" * 50)
                ge = GraspNetEval(
                    root=self.data_root, camera=self.camera, split=test_mode
                )
                res, ap = ge.eval_all(
                    dump_folder=self.eval_save_dir, proc=self.num_test_proc
                )
                self.results[test_mode] = res
                np.save(
                    os.path.join(
                        self.eval_save_dir,
                        f"ap_{self.camera}_{test_mode}_{ap}.npy",
                    ),
                    res,
                )

            elif test_mode == "test_scale":
                scales = ["small", "medium", "large"]
                ge_scale = GraspNetEvalScale(
                    root=self.data_root, camera=self.camera, split="test"
                )
                for scale in scales:
                    logger.info(
                        "=" * 50
                        + f"BEGIN EVAL {test_mode} SCALE {scale}"
                        + "=" * 50
                    )
                    res_scale, ap_scale = ge_scale.eval_all(
                        dump_folder=self.eval_save_dir,
                        SCALE=scale,
                        proc=self.num_test_proc,
                    )
                    self.results[f"{test_mode}_{scale}"] = res_scale
                    np.save(
                        os.path.join(
                            self.eval_save_dir,
                            f"ap_{self.camera}_{test_mode}_{scale}_{ap_scale}.npy",
                        ),
                        res_scale,
                    )
            elif test_mode == "test_scale_mini":
                scales = ["small", "medium", "large"]
                ge_scale = GraspNetEvalScale(
                    root=self.data_root, camera=self.camera, split="test"
                )
                for scale in scales:
                    logger.info(
                        "=" * 50
                        + f"BEGIN EVAL {test_mode} SCALE {scale}"
                        + "=" * 50
                    )
                    res_scale, ap_scale = ge_scale.eval(
                        dump_folder=self.eval_save_dir,
                        SCALE=scale,
                        scene_ids=[0],
                        proc=self.num_test_proc,
                    )
                    self.results[f"{test_mode}_{scale}"] = res_scale
                    np.save(
                        os.path.join(
                            self.eval_save_dir,
                            f"ap_{self.camera}_{test_mode}_{scale}_{ap_scale}.npy",
                        ),
                        res_scale,
                    )

        return self.results

    def reset(self):
        self.results = dict()
