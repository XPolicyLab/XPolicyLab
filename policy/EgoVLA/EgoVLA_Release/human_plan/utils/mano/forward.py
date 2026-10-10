import torch
import numpy as np
from human_plan.utils.mano.model import (
  mano_left, 
  mano_right
)
from human_plan.utils.mano.constants import (
  EMPTY_ROT,
  MANO_FINGERTIP_VERT_INDICES,
  LEFT_PELVIS,
  RIGHT_PELVIS,
  mano_joint_mapping,
  RIGHT_AXIS_TRANSFORMATION_RETARGET,
  LEFT_AXIS_TRANSFORMATION_RETARGET,
  RIGHT_RETARGET_MANO_TRANSFORMATION,
  LEFT_RETARGET_MANO_TRANSFORMATION,
  # ISAACLAB Frame
  RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB,
  LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB
)


# Function to compute joint positions from MANO parameters
def mano_forward(
  mano_model, 
  hand_joint_angles,

  global_rot,
  global_trans,
):
  batch_size = np.prod(hand_joint_angles.shape) // 15
  beta = torch.zeros(
    (batch_size, 10), dtype=torch.float32
  ).to(hand_joint_angles.device).to(hand_joint_angles.dtype)
  # print(beta, global_rot, hand_joint_angles, global_trans)
  output = mano_model(
      betas=beta.reshape(batch_size, 10),
      global_orient=global_rot.reshape(-1, 3),
      hand_pose=hand_joint_angles.reshape(-1,15),
      transl=global_trans.reshape(-1, 3),
      return_verts=True,  # MANO doesn't return landmarks as well if this is false
  )
  extra_joints = torch.index_select(
      output.vertices, 1,
      torch.tensor(
          list(MANO_FINGERTIP_VERT_INDICES.values()),
          dtype=torch.long,
      ).to(hand_joint_angles.device),
  )
  joints = torch.cat([output.joints, extra_joints], dim=1)
  joints = joints[:, mano_joint_mapping]
  return joints.squeeze()  # We only need the 21 main joints


def mano_forward_retarget(
  pca_components,
  is_right
):
  mano_model = mano_right if is_right else mano_left
  # mano_model.to(hand_label_component.device)
  mano_model.to(pca_components.device).to(pca_components.dtype)

  batch_size = np.prod(pca_components.shape) // 15

  pca_components = pca_components.reshape(-1, 15).to(mano_model.dtype)

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS

  rot = torch.repeat_interleave(
    EMPTY_ROT.reshape(1, 3),
    batch_size, dim=0
  ).to(pca_components.device).to(mano_model.dtype)
  # print("ROT shape:", rot.shape)
  # print(pca_components.shape)

  beta = torch.zeros(
    (batch_size, 10),
    dtype=mano_model.dtype
  ).to(pca_components.device)
  output = mano_model(
      betas=beta,
      global_orient=rot.reshape(-1, 3),
      # hand_pose=torch.zeros(1,45),
      hand_pose=pca_components.reshape(-1,15),
      transl=None,
      return_verts=True,  # MANO doesn't return landmarks as well if this is false
  )

  extra_joints = torch.index_select(
      output.vertices, 1,
      torch.tensor(
          list(MANO_FINGERTIP_VERT_INDICES.values()),
          dtype=torch.long,
      ).to(pca_components.device),
  )
  joints = torch.cat([output.joints, extra_joints], dim=1)

  # # Move to Wrist
  joints -= pelvis.to(joints.device)
  # estimate_frame_from_hand_points
  joints = joints[:, mano_joint_mapping]
  joints = joints - joints[:, 0:1, :]

  mano_retarget_rot = RIGHT_RETARGET_MANO_TRANSFORMATION if is_right else \
    LEFT_RETARGET_MANO_TRANSFORMATION
  mano_retarget_rot = mano_retarget_rot.to(pca_components.device)

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET
  retarget_axis_transformation = retarget_axis_transformation.to(pca_components.device)

  joints = joints @ mano_retarget_rot @ retarget_axis_transformation

  return joints.squeeze()  # We only need the 21 main joints


def mano_forward_retarget_isaaclab(
  pca_components,
  is_right
):
  mano_model = mano_right if is_right else mano_left

  mano_model.to(pca_components.device)

  batch_size = np.prod(pca_components.shape) // 15

  pca_components = pca_components.reshape(-1, 15)

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS

  rot = torch.repeat_interleave(
    EMPTY_ROT.reshape(1, 3),
    batch_size, dim=0
  ).to(pca_components.device)
  print("ROT shape:", rot.shape)
  print(pca_components.shape)

  beta = torch.zeros((batch_size, 10), dtype=torch.float32).to(pca_components.device)
  output = mano_model(
      betas=beta,
      global_orient=rot.reshape(-1, 3),
      # hand_pose=torch.zeros(1,45),
      hand_pose=pca_components.reshape(-1,15),
      transl=None,
      return_verts=True,  # MANO doesn't return landmarks as well if this is false
  )

  extra_joints = torch.index_select(
      output.vertices, 1,
      torch.tensor(
          list(MANO_FINGERTIP_VERT_INDICES.values()),
          dtype=torch.long,
      ).to(pca_components.device),
  )
  joints = torch.cat([output.joints, extra_joints], dim=1)

  # # Move to Wrist
  joints -= pelvis.to(joints.device)
  # estimate_frame_from_hand_points
  joints = joints[:, mano_joint_mapping]
  joints = joints - joints[:, 0:1, :]

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB
  retarget_axis_transformation = retarget_axis_transformation.to(pca_components.device)


  joints = joints @ retarget_axis_transformation

  return joints.squeeze()  # We only need the 21 main joints
