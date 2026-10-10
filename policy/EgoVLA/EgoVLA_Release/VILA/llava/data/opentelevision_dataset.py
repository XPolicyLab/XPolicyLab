from llava.data.dataset import LazyVLAHFAbsDataset
import torch
import numpy as np
from typing import Dict, Sequence


OTV_SIM_IMAGE_WIDTH = 1280
OTV_SIM_IMAGE_HEIGHT = 720


# mano_per_dim_min = [
#   -1,
#   1.5,
#   -2,
#   -3,
#   -1.5,
#   -1
# ]
# mano_per_dim_min = torch.concat([
#   torch.Tensor(mano_per_dim_min),
#   -4 * torch.ones(9),
# ])
# mano_per_dim_max = [
#   2.2,
#   3.5,
#   1,
#   0.5,
#   4,
#   5
# ]
# mano_per_dim_max = torch.concat([
#   torch.Tensor(mano_per_dim_max),
#   4 * torch.ones(9),
# ])
# mano_range = mano_per_dim_max - mano_per_dim_min

# def norm_hand_dof(hand_dof):
#   return (hand_dof - mano_per_dim_min) / mano_range

from llava.data.utils import norm_hand_dof


class LazyVLAOTVSimHFAbsDataset(LazyVLAHFAbsDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)


  def init_dataset_specific_info(self):
    self.frame_count_scaler_up = 1
    self.frame_count_scaler = 1
    self.reverse_channel_order = False

    self.raw_image_width = OTV_SIM_IMAGE_WIDTH
    self.raw_image_height = OTV_SIM_IMAGE_HEIGHT


  def get_current_hand_data(self, sample, hand) -> Sequence[torch.Tensor]:
    valid_mask = torch.ones((1, 1))
    # single_ee_3d = torch.zeros((1, 3))
    # single_ee_2d = torch.zeros((1, 2))
    # single_ee_rot = torch.zeros((1, 4))
    # single_handkp_3d = torch.zeros((1, 21, 3))
    # return valid_mask, \
    #   single_ee_3d, \
    #   single_ee_2d, \
    #   single_ee_rot, \ 
    #   single_handkp_3d

    single_ee_3d = torch.tensor(
        sample["current_" + hand + "_mano_trans"]
    ).reshape(-1, 3)

    single_ee_2d = torch.tensor(
        sample["current_" + hand + "_mano_ee_2d"]
    ).reshape(-1, 2)

    single_hand_rot = sample["current_" + hand + "_mano_rot"].reshape(-1, 3)
    # r = R.from_matrix(current_rot)
    # current_rot = r.as_rotvec()
    single_hand_rot = torch.tensor(
      single_hand_rot
    ).reshape(-1, 3)

    single_handkp_3d = torch.tensor(
      sample["current_" + hand + "_mano_kps3d"]
    ).reshape(-1, 21, 3)

    hand_dof = torch.tensor(
      sample["current_" + hand + "_mano_parameters"]
    ).reshape(-1, 15)
    # print("before norm", hand_dof)
    # hand_dof = norm_hand_dof(hand_dof)
    # hand_dof[..., self.training_args.hand_loss_dim:] = 0
    # print("After norm", hand_dof)

    hand_finger_tip = torch.tensor(
      sample["current_" + hand + "_finger_tip_cam_pos"]
      # current_left_finger_tip_cam_pos
    ).reshape(-1, 5, 3)

    return valid_mask, \
      single_ee_3d, \
      single_ee_2d, \
      single_hand_rot, \
      single_handkp_3d, \
      hand_finger_tip, \
      hand_dof

    # return valid_mask, single_hand_trans, single_hand_trans_2d, \
    #   single_hand_pose, single_hand_rot

  def get_future_hand_data(
    self, sample, hand, future_step, future_idx
  ) -> Sequence[torch.Tensor]:

    valid_mask = torch.zeros((1, 1))

    max_len = sample["future_" + hand + "_mano_trans"].reshape(-1, 3).shape[0]

    target_idx = min(
        future_idx * (future_step + 1), max_len - 1
    )

    if self.data_args.ee_relative_transformation:
      single_ee_3d_label = torch.tensor(
          sample[
              "future_" + hand + "_mano_ee_relative_trans"
          ].reshape(-1, 3)[target_idx]
      ).reshape(1, 3)
    else:
      single_ee_3d_label = torch.tensor(
          sample[
              "future_" + hand + "_mano_trans"
          ].reshape(-1, 3)[target_idx]
      ).reshape(1, 3)

    single_ee_2d_label = torch.tensor(
        sample[
            "future_" + hand + "_mano_ee_2d"
        ].reshape(-1, 2)[target_idx]
    ).reshape(1, 2)

    single_handkp_3d_label = torch.tensor(
        sample[
            "future_" + hand + "_mano_kps3d"
        ].reshape(-1, 21, 3)[target_idx]
    ).reshape(1, 21, 3)

    assert self.data_args.use_mano
    hand_dof = torch.tensor(
      sample[
        "future_" + hand + "_mano_parameters"
      ].reshape(-1, 15)[target_idx]
    )
    hand_dof = torch.tensor(hand_dof)
    # hand_dof = norm_hand_dof(hand_dof)
    # hand_dof[..., self.training_args.hand_loss_dim:] = 0

    assert self.data_args.no_norm_ee_label

    valid_mask = torch.tensor(
      sample["future_" + hand + "_flag"].reshape(-1, 1)[target_idx]
    ).unsqueeze(-1)
  
    valid_mask = valid_mask.reshape(1, 1)

    if self.data_args.ee_relative_transformation:
      future_rot = sample[
          "future_" + hand + "_mano_ee_relative_rot"
      ].reshape(-1, 3)[target_idx]
    else:
      future_rot = sample[
          "future_" + hand + "_mano_rot"
      ].reshape(-1, 3)[target_idx]

    # r = R.from_matrix(future_rot)
    # future_rot = r.as_rotvec()
    single_ee_rot_label = torch.tensor(
        future_rot
    ).reshape(1, 3)

    # single_handkp_3d_label = torch.tensor(
    #     sample["future_"+ hand + "_mano_kps3d"].reshape(-1, 21, 3)[target_idx]
    # ).reshape(1, 21, 3)

    return valid_mask, \
      single_ee_3d_label, \
      single_ee_2d_label, \
      hand_dof, \
      single_ee_rot_label, \
      single_handkp_3d_label

  def get_current_language_label(self, sample):
    # current_language_label = f"{sample['language_label_verb']} {sample['language_label_noun']}"
    current_language_label = sample["language_label"]
    return current_language_label