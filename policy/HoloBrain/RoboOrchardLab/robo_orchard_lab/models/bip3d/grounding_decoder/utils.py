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

from typing import List

import torch
from pytorch3d.transforms import euler_angles_to_matrix, quaternion_to_matrix
from torch import nn

from robo_orchard_lab.utils.build import build


def decode_box(box, min_size=None, max_size=None):
    size = box[..., 3:6].exp()
    if min_size is not None or max_size is not None:
        size = size.clamp(min=min_size, max=max_size)
    box = torch.cat(
        [box[..., :3], size, box[..., 6:]],
        dim=-1,
    )
    return box


def convert_grounding_to_cls_scores(
    logits: torch.Tensor, positive_maps: List[dict]
):
    """Convert logits to class scores."""
    assert len(positive_maps) == logits.shape[0]  # batch size

    scores = torch.zeros(
        logits.shape[0], logits.shape[1], len(positive_maps[0])
    ).to(logits.device)
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            for label_j in positive_map:
                scores[:, :, label_j - 1] = logits[
                    :, :, torch.LongTensor(positive_map[label_j])
                ].mean(-1)
        else:
            for i, positive_map in enumerate(positive_maps):
                for label_j in positive_map:
                    scores[i, :, label_j - 1] = logits[
                        i, :, torch.LongTensor(positive_map[label_j])
                    ].mean(-1)
    return scores


def create_positive_map_label_to_token(
    positive_map: torch.Tensor, plus: int = 0
):
    """Create a dictionary mapping the label to the token.

    Args:
        positive_map (Tensor): The positive map tensor.
        plus (int, optional): Value added to the label for indexing.
            Defaults to 0.

    Returns:
        dict: The dictionary mapping the label to the token.
    """
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(
            positive_map[i], as_tuple=True
        )[0].tolist()
    return positive_map_label_to_token


def rotation_3d_in_euler(points, angles, return_mat=False):
    """Rotate points by angles according to axis.

    This function was originally copied from the [mmdetection3d] repository:
    https://github.com/open-mmlab/mmdetection3d

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple):
            Vector of angles in shape (N, 3)
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if len(angles.shape) == 1:
        angles = angles.expand(points.shape[:1] + (3,))
        # angles = torch.full(points.shape[:1], angles)

    assert (
        len(points.shape) == 3
        and len(angles.shape) == 2
        and points.shape[0] == angles.shape[0]
    ), f"Incorrect shape of points angles: {points.shape}, {angles.shape}"

    assert points.shape[-1] in [
        2,
        3,
    ], f"Points size should be 2 or 3 instead of {points.shape[-1]}"

    if angles.shape[1] == 3:
        rot_mat_T = euler_angles_to_matrix(  # noqa: N806
            angles, "ZXY"
        )  # N, 3,3
    else:
        rot_mat_T = quaternion_to_matrix(angles)  # N, 3,3  # noqa: N806
    rot_mat_T = rot_mat_T.transpose(-2, -1)  # noqa: N806

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.bmm(points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)  # noqa: N806
        return points_new, rot_mat_T
    else:
        return points_new


def wasserstein_distance(source, target):
    rot_mat_src = euler_angles_to_matrix(source[..., 6:9], "ZXY")
    sqrt_sigma_src = rot_mat_src @ (
        source[..., 3:6, None] * rot_mat_src.transpose(-2, -1)
    )

    rot_mat_tgt = euler_angles_to_matrix(target[..., 6:9], "ZXY")
    sqrt_sigma_tgt = rot_mat_tgt @ (
        target[..., 3:6, None] * rot_mat_tgt.transpose(-2, -1)
    )

    sigma_distance = sqrt_sigma_src - sqrt_sigma_tgt
    sigma_distance = sigma_distance.pow(2).sum(dim=-1).sum(dim=-1)
    center_distance = ((source[..., :3] - target[..., :3]) ** 2).sum(dim=-1)
    distance = sigma_distance + center_distance
    distance = distance.clamp(1e-7).sqrt()
    return distance


def center_distance(source, target):
    return torch.norm(source[..., :3] - target[..., :3], p=2, dim=-1)


def get_positive_map(char_positive, text_dict):
    bs, text_length = text_dict["embedded"].shape[:2]
    tokenized = text_dict["tokenized"]
    positive_maps = []
    for i in range(bs):
        num_target = len(char_positive[i])
        positive_map = torch.zeros(
            (num_target, text_length), dtype=torch.float
        )
        for j, tok_list in enumerate(char_positive[i]):
            for beg, end in tok_list:
                try:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                except Exception as e:
                    print("beg:", beg, "end:", end)
                    print("token_positive:", char_positive[i])
                    raise e
                if beg_pos is None:
                    try:
                        beg_pos = tokenized.char_to_token(i, beg + 1)
                        if beg_pos is None:
                            beg_pos = tokenized.char_to_token(i, beg + 2)
                    except Exception:
                        beg_pos = None
                if end_pos is None:
                    try:
                        end_pos = tokenized.char_to_token(i, end - 2)
                        if end_pos is None:
                            end_pos = tokenized.char_to_token(i, end - 3)
                    except Exception:
                        end_pos = None
                if beg_pos is None or end_pos is None:
                    continue

                assert beg_pos is not None and end_pos is not None
                positive_map[j, beg_pos : end_pos + 1].fill_(1)
        positive_map /= positive_map.sum(-1)[:, None] + 1e-6
        positive_maps.append(positive_map)
    return positive_maps


def get_entities(text, char_positive, sep_token="[SEP]"):
    batch_entities = []
    for bs_idx in range(len(char_positive)):
        entities = []
        for obj_idx in range(len(char_positive[bs_idx])):
            entity = ""
            for beg, end in char_positive[bs_idx][obj_idx]:
                if len(entity) == 0:
                    entity = text[bs_idx][beg:end]
                else:
                    entity += sep_token + text[bs_idx][beg:end]
            entities.append(entity)
        batch_entities.append(entities)
    return batch_entities


def linear_act_ln(
    embed_dims,
    in_loops,
    out_loops,
    input_dims=None,
    act_cfg=None,
    norm_cfg=None,
):
    if act_cfg is None:
        act_cfg = dict(type=nn.ReLU, inplace=True)

    if norm_cfg is None:
        norm_cfg = dict(type=nn.LayerNorm, normalized_shape=embed_dims)

    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(build(act_cfg))
            input_dims = embed_dims
        layers.append(build(norm_cfg))
    return layers
