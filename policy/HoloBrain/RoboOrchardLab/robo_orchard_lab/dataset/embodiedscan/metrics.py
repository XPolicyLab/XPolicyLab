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
import pickle
import shutil
from typing import Optional

import numpy as np
import torch
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms import euler_angles_to_matrix
from terminaltables import AsciiTable

from robo_orchard_lab.dataset.embodiedscan.embodiedscan_det_grounding_dataset import (  # noqa: E501
    COMMON_LABELS,
    DEFAULT_CLASSES,
    HEAD_LABELS,
    TAIL_LABELS,
)
from robo_orchard_lab.utils import as_sequence

logger = logging.getLogger(__name__)


def average_precision(recalls, precisions, mode="area"):
    assert recalls.shape == precisions.shape
    ap = 0
    if mode == "area":
        mrec = np.hstack((0, recalls, 1))
        mpre = np.hstack((0, precisions, 0))
        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        ind = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[ind + 1] - mrec[ind]) * mpre[ind + 1])
    elif mode == "11points":
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[recalls >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported'
        )
    return ap


def bbox_to_corners(bbox):
    assert len(bbox.shape) == 2, (
        "bbox must be 2D tensor of shape (N, 6) or (N, 7) or (N, 9)"
    )
    if bbox.shape[-1] == 6:
        rot_mat = (
            torch.eye(3, device=bbox.device)
            .unsqueeze(0)
            .repeat(bbox.shape[0], 1, 1)
        )
    elif bbox.shape[-1] == 7:
        angles = bbox[:, 6:]
        fake_angles = torch.zeros_like(angles).repeat(1, 2)
        angles = torch.cat((angles, fake_angles), dim=1)
        rot_mat = euler_angles_to_matrix(angles, "ZXY")
    elif bbox.shape[-1] == 9:
        rot_mat = euler_angles_to_matrix(bbox[:, 6:], "ZXY")
    else:
        raise NotImplementedError
    centers = bbox[:, :3].unsqueeze(1).repeat(1, 8, 1)  # shape (N, 8, 3)
    half_sizes = bbox[:, 3:6].unsqueeze(1).repeat(1, 8, 1) / 2

    eight_corners = torch.tensor(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [1, 1, -1],
        ],
    ).to(bbox)
    eight_corners = eight_corners * half_sizes  # shape (N, 8, 3)
    rotated_corners = torch.matmul(eight_corners, rot_mat.transpose(1, 2))
    corners = rotated_corners + centers
    return corners


def compute_iou3d(bbox_1, bbox_2, eps=1e-4):
    corners_1 = bbox_to_corners(bbox_1)
    corners_2 = bbox_to_corners(bbox_2)
    return box3d_overlap(corners_1, corners_2, eps=eps)[1]


def compute_precision_recall_AP(pred, gt, iou_thr, area_range=None):  # noqa: N802
    gt_bbox_code_size = 9
    pred_bbox_code_size = 9
    for img_id in gt.keys():
        if len(gt[img_id]) != 0:
            gt_bbox_code_size = gt[img_id][0].shape[0]
            break
    for img_id in pred.keys():
        if len(pred[img_id][0]) != 0:
            pred_bbox_code_size = pred[img_id][0][0].shape[0]
            break
    assert gt_bbox_code_size == pred_bbox_code_size

    class_recs = {}
    num_gt = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            bbox = torch.stack(gt[img_id])
        else:
            bbox = torch.zeros([0, gt_bbox_code_size], dtype=torch.float32)
        det = [[False] * cur_gt_num for i in iou_thr]
        num_gt += cur_gt_num
        class_recs[img_id] = {"bbox": bbox, "det": det}
        if area_range is not None and len(bbox) > 0:
            area = bbox[:, 3:6].cumprod(dim=-1)[..., -1]
            ignore = torch.logical_or(
                area < area_range[0], area > area_range[1]
            )
            num_gt -= ignore.sum().item()
            class_recs[img_id]["ignore"] = ignore

    if num_gt == 0:
        return [(float("nan"),) * 3 for _ in iou_thr]

    image_ids = []
    confidence = []
    ious = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue

        pred_cur, _confidence = list(zip(*pred[img_id], strict=False))
        pred_cur = torch.stack(pred_cur)
        mask = torch.any(
            torch.stack(
                [
                    pred_cur[:, 3] * pred_cur[:, 4],
                    pred_cur[:, 3] * pred_cur[:, 5],
                    pred_cur[:, 4] * pred_cur[:, 5],
                ],
                dim=-1,
            )
            < 2e-4,
            dim=-1,
            keepdims=True,
        )  # type: ignore
        pred_cur[:, 3:6] = torch.where(
            mask, torch.clamp(pred_cur[:, 3:6], min=2e-2), pred_cur[:, 3:6]
        )

        confidence.extend(_confidence)
        image_ids.extend([img_id] * cur_num)

        gt_cur = class_recs[img_id]["bbox"]
        if len(gt_cur) > 0:
            iou_cur = compute_iou3d(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for _ in range(cur_num):
                ious.append(torch.zeros(0))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    num_preds = len(image_ids)
    tp_thr = [np.zeros(num_preds) for i in iou_thr]
    fp_thr = [np.zeros(num_preds) for i in iou_thr]
    for d in range(num_preds):
        R = class_recs[image_ids[d]]  # noqa: N806
        gt_bbox = R["bbox"]
        cur_iou = ious[d]

        if len(gt_bbox) != 0:
            iou_max, jmax = cur_iou.max(dim=0)
        else:
            iou_max = float("-inf")

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if "ignore" in R and R["ignore"][jmax]:
                    continue
                if not R["det"][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.0
                    R["det"][iou_idx][jmax] = 1
                else:
                    fp_thr[iou_idx][d] = 1.0
            else:
                fp_thr[iou_idx][d] = 1.0

    ret = []
    for iou_idx, _ in enumerate(iou_thr):
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(num_gt)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))
    return ret


def compute_precision_recall_AP_multi_cls(  # noqa: N802
    pred, gt, iou_thresholds, area_range=None
):
    recall = [{} for _ in iou_thresholds]
    precision = [{} for _ in iou_thresholds]
    ap = [{} for _ in iou_thresholds]

    for label in gt.keys():
        if label in pred:
            eval_ret = compute_precision_recall_AP(
                pred[label], gt[label], iou_thresholds, area_range
            )
            if np.isnan(eval_ret[0][-1]):
                continue
            for iou_idx, _ in enumerate(iou_thresholds):
                (
                    recall[iou_idx][label],
                    precision[iou_idx][label],
                    ap[iou_idx][label],
                ) = eval_ret[iou_idx]

    return recall, precision, ap


def eval_function(
    pred,
    gt,
    iou_thresholds,
    label2cat,
    classes_split=None,
    area_split=None,
    verbose=True,
    part="Overall",
):
    rec, prec, ap = compute_precision_recall_AP_multi_cls(
        pred, gt, iou_thresholds
    )

    ret_dict = dict()
    header = ["classes"]
    table_columns = [[label2cat[label] for label in ap[0].keys()] + [part]]
    num_border_row = 1

    for i, iou_thresh in enumerate(iou_thresholds):
        header.append(f"AP_{iou_thresh:.2f}")
        header.append(f"AR_{iou_thresh:.2f}")
        ap_list = []
        for label in ap[i].keys():
            ret_dict[f"{label2cat[label]}_AP_{iou_thresh:.2f}"] = ap[i][label]
            ap_list.append(ap[i][label])
        mean_ap = np.mean(ap_list)
        ret_dict[f"mAP_{iou_thresh:.2f}"] = mean_ap
        table_columns.append([f"{x:.4f}" for x in ap_list + [mean_ap]])

        rec_list = []
        for label in rec[i].keys():
            ret_dict[f"{label2cat[label]}_rec_{iou_thresh:.2f}"] = rec[i][
                label
            ][-1]
            rec_list.append(rec[i][label][-1])
        mean_rec = np.mean(rec_list)
        ret_dict[f"mAR_{iou_thresh:.2f}"] = mean_rec
        table_columns.append([f"{x:.4f}" for x in rec_list + [mean_rec]])

    if classes_split is not None:
        num_border_row += len(classes_split)
        for split, classes in classes_split.items():
            table_columns[0].append(split)
            for i, iou_thresh in enumerate(iou_thresholds):
                ap_list = [ap[i][label] for label in classes if label in ap[i]]
                mean_ap = np.mean(ap_list)
                ret_dict[f"{split}_mAP_{iou_thresh:.2f}"] = mean_ap
                table_columns[2 * i + 1].append(f"{mean_ap:.4f}")

                rec_list = [
                    rec[i][label][-1] for label in classes if label in rec[i]
                ]
                mean_rec = np.mean(rec_list)
                table_columns[2 * i + 2].append(f"{mean_rec:.4f}")
                ret_dict[f"{split}_mAR_{iou_thresh:.2f}"] = mean_rec

    if area_split is not None:
        num_border_row += len(area_split)
        table_rows = []
        for _, (area, area_range) in enumerate(area_split.items()):
            rec, prec, ap = compute_precision_recall_AP_multi_cls(
                pred, gt, iou_thresholds, area_range
            )
            table_columns[0].append(area)
            for i, iou_thresh in enumerate(iou_thresholds):
                mean_ap = np.mean(list(ap[i].values()))
                ret_dict[f"{area}_mAP_{iou_thresh:.2f}"] = mean_ap
                table_columns[2 * i + 1].append(f"{mean_ap:.4f}")
                mean_rec = np.mean([x[-1] for x in rec[i].values()])
                ret_dict[f"{area}_mAR_{iou_thresh:.2f}"] = mean_rec
                table_columns[2 * i + 2].append(f"{mean_rec:.4f}")

    if verbose:
        table_data = [header]
        table_rows = list(zip(*table_columns, strict=False))
        table_data += table_rows
        table = AsciiTable(table_data)
        border_row = ["-" * len(cell) for cell in header]
        table.table_data.insert(-num_border_row, border_row)
        logger.info(f"eval results of {part}:\n" + table.table)
    return ret_dict


def format_results(results):
    pred, gt = {}, {}
    for sample_id, result in enumerate(results):
        num_pred = len(result["labels_3d"])
        for i in range(num_pred):
            label = int(result["labels_3d"][i])
            if label not in pred:
                pred[label] = {}
            if sample_id not in pred[label]:
                pred[label][sample_id] = []

            if label not in gt:
                gt[label] = {}
            if sample_id not in gt[label]:
                gt[label][sample_id] = []

            pred[label][sample_id].append(
                (result["bboxes_3d"][i], result["scores_3d"][i])
            )

        num_gt = len(result["gt_labels_3d"])
        for i in range(num_gt):
            label = int(result["gt_labels_3d"][i])
            if label not in gt:
                gt[label] = {}
            if sample_id not in gt[label]:
                gt[label][sample_id] = []
            gt[label][sample_id].append(result["gt_bboxes_3d"][i])
    return pred, gt


class DetMetric:
    def __init__(
        self,
        iou_thresholds: tuple[float, float] = (0.25, 0.5),
        save_result_path=None,
        eval_part=("scannet", "3rscan", "matterport3d", "arkit"),
        classes=None,
        classes_split=None,
        area_split=None,
        gather_device: str = "cpu",
        share_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.iou_thresholds = as_sequence(iou_thresholds)
        self.save_result_path = save_result_path
        self.eval_part = eval_part
        if classes is None:
            classes = DEFAULT_CLASSES
        self.classes = classes
        if classes_split is None:
            classes_split = dict(
                head=HEAD_LABELS,
                common=COMMON_LABELS,
                tail=TAIL_LABELS,
            )
        self.classes_split = classes_split
        if area_split is None:
            area_split = dict(
                small=[0, 0.2**3],
                medium=[0.2**3, 1.0**3],
                large=[1.0**3, float("inf")],
            )
        self.area_split = area_split
        assert gather_device in ["cpu", "gpu"]
        self.gather_device = gather_device
        self.share_dir = share_dir
        self.reset()

    def reset(self):
        self.results = []

    def update(self, batch, model_outputs):
        for i, output in enumerate(model_outputs):
            result = dict(
                scan_id=batch["scan_id"][i],
                gt_bboxes_3d=batch["gt_bboxes_3d"][i].cpu(),
                gt_labels_3d=batch["gt_labels_3d"][i].cpu().numpy(),
            )
            for k, v in output.items():
                if hasattr(v, "to"):
                    result[k] = v.to("cpu")
                else:
                    result[k] = v
            self.results.append(result)

    def compute(self, accelerator):
        results = self.gather_all_results(accelerator)
        if not accelerator.is_main_process:
            return None

        logger.info(f"number of results: {len(results)}")
        metric = {}
        for part in ("overall",) + self.eval_part:
            if part is None:
                continue
            if part == "overall":
                pred, gt = format_results(results)
            else:
                ret = [x for x in results if part in x["scan_id"]]
                logger.info(f"number of {part} results: {len(ret)}")
                if len(ret) == 0:
                    continue
                pred, gt = format_results(ret)
            metric = eval_function(
                pred,
                gt,
                self.iou_thresholds,
                self.classes,
                classes_split=self.classes_split,
                area_split=self.area_split,
                part=part,
            )
        return metric

    def gather_all_results(self, accelerator):
        if accelerator.num_processes == 1:
            return self.results

        if self.gather_device == "gpu":
            return accelerator.gather_for_metrics(
                self.results, use_gather_object=True
            )

        if self.share_dir is None:
            share_dir = "/tmp/.dist_metric_gather"
        else:
            share_dir = self.share_dir

        if accelerator.is_main_process:
            os.makedirs(share_dir, exist_ok=True)

        accelerator.wait_for_everyone()
        rank = accelerator.state.process_index
        with open(os.path.join(share_dir, f"part_{rank}.pkl"), "wb") as f:
            pickle.dump(self.results, f, protocol=2)
        accelerator.wait_for_everyone()

        if not accelerator.is_main_process:
            return None

        results = []
        for i in range(accelerator.state.num_processes):
            path = os.path.join(share_dir, f"part_{i}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{share_dir} is not an shared directory for "
                    f"rank {i}, please make sure {share_dir} is a shared "
                    "directory for all ranks!"
                )
            with open(path, "rb") as f:
                results.extend(pickle.load(f))
        shutil.rmtree(share_dir)
        return results
