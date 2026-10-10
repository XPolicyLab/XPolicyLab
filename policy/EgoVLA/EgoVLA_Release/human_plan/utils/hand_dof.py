
import torch

def compute_hand_dof_5dim(
  palm_center: torch.Tensor,
  wrist: torch.Tensor,
  finger_tip: torch.Tensor
):
  # Palm Center Shape: [Batch Size, 1, 3]
  # Wrist Shape: [Batch, 1, 3]
  # Finger Tip Shape: [Batch Size, 5, 3]
  finger_tip_delta = torch.linalg.norm(
    finger_tip - palm_center,
    dim=-1
  )

  wrist_to_palm = torch.linalg.norm(
    (wrist - palm_center), dim=-1
  )
  finger_tip_range = finger_tip_delta / (wrist_to_palm + 1e-6)
  # Finger tip range: [Batch Size, 5]
  return finger_tip_range


from human_plan.dataset_preprocessing.utils.mano_utils import (
  mano_left,
  mano_right,
  MANO_FINGERTIP_VERT_INDICES,
  device,
  mano_joint_mapping,
  RIGHT_AXIS_TRANSFORMATION,
  LEFT_AXIS_TRANSFORMATION,
)

import numpy as np

def convert_full_mano_to_pca_dof(
  rotations, hand_mean, hand_components
):
  hand_mean = hand_mean
  rotations = rotations - hand_mean #.detach().cpu().numpy()
  hand_pose_pca = np.dot(rotations, hand_components.T)
  return hand_pose_pca

def convert_full_mano_to_pca(
  rotations, is_right
):
  mano_model = mano_right if is_right else mano_left
  # hand_mean = mano_model.flat_hand_mean
  hand_mean = mano_model.hand_mean
  rotations = rotations - hand_mean.detach().cpu().numpy()

  hand_components = mano_model.np_hand_components

  # hand_pose_aa = np.dot(hand_pose_pca, hand_components)
  hand_pose_pca = np.dot(rotations, hand_components.T)

  return hand_pose_pca

import torch.nn as nn
# Function to compute joint positions from MANO parameters
def mano_forward(
  pca_components, hand_info, is_right
):
  pca_components = torch.FloatTensor(
    pca_components
  ).to(device).reshape(-1, 15)
  mano_model = mano_right if is_right else mano_left
  mano_model.to(device)

  beta = nn.Parameter(torch.FloatTensor(hand_info['beta']).unsqueeze(0)).to(device)
  trans = nn.Parameter(torch.FloatTensor(hand_info['trans']).unsqueeze(0)).to(device)

  # theta_np = np.array(hand_info['poseCoeff'])
  wrist_rot = torch.FloatTensor(hand_info['poseCoeff'][:3]).unsqueeze(0).to(device)

  output = mano_model(
      betas=beta,
      global_orient=wrist_rot,
      # hand_pose=torch.zeros(1,45),
      hand_pose=pca_components.reshape(-1,15),
      transl=trans,
      return_verts=True,  # MANO doesn't return landmarks as well if this is false
  )

  extra_joints = torch.index_select(
      output.vertices, 1,
      torch.tensor(
          list(MANO_FINGERTIP_VERT_INDICES.values()),
          dtype=torch.long,
      ).to(device),
  )
  joints = torch.cat([output.joints, extra_joints], dim=1)

  # # Move to Wrist
  # joints -= pelvis
  
  joints = joints[:, mano_joint_mapping]
  # if is_right:
  #   joints = torch.einsum(
  #     "ij, nkj -> nki",
  #     RIGHT_AXIS_TRANSFORMATION,
  #     joints
  #   )
  # else:
  #   joints = torch.einsum(
  #     "ij, nkj -> nki",
  #     LEFT_AXIS_TRANSFORMATION,
  #     joints
  #   )
  return joints.squeeze()  # We only need the 21 main joints


import torch.nn as nn
# Function to compute joint positions from MANO parameters

from scipy.spatial.transform import Rotation as R

from smplx.lbs import blend_shapes, vertices2joints



LEFT_AXIS_TRANSFORMATION_RETARGET = torch.tensor([
  [0, 0, -1],
  [1, 0, 0],
  [0, -1, 0],
]).float().to(device)

RIGHT_AXIS_TRANSFORMATION_RETARGET = torch.tensor([
  [0, 0, -1],
  [-1, 0, 0],
  [0, 1, 0],
]).float().to(device)


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    # print(points.shape)
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame

def mano_forward_empty(
  pca_components, 
  # hand_info, 
  is_right
):
  mano_model = mano_right if is_right else mano_left

  mano_model.to("cpu")
  batch_size = np.prod(pca_components.shape) // 15
  betas = torch.zeros(10, dtype=torch.float32).unsqueeze(0).to("cpu")

  v_shaped = mano_model.v_template + blend_shapes(betas, mano_model.shapedirs)  # type: ignore
  pelvis = vertices2joints(
    mano_model.J_regressor[0:1], v_shaped
  ).squeeze(dim=1)

  shape = torch.zeros(batch_size, 10, dtype=torch.float32).to("cpu") # Neutral shape

  pca_components = torch.FloatTensor(
    pca_components
  ).to("cpu").reshape(-1, 15)

  # beta = nn.Parameter(torch.FloatTensor(hand_info['beta']).unsqueeze(0)).to(device)
  beta = torch.zeros((1, 10), dtype=torch.float32).to("cpu")
  # trans = nn.Parameter(torch.FloatTensor(hand_info['trans']).unsqueeze(0)).to(device)

  # theta_np = np.array(hand_info['poseCoeff'])
  # wrist_rot = torch.FloatTensor(hand_info['poseCoeff'][:3]).unsqueeze(0).to("cpu")


  # trans = raw_trans - pelvis.numpy()
  trans = torch.zeros((1, 3), dtype=torch.float32).to("cpu")
  rot = R.from_matrix(
    np.eye(3)
  ).as_rotvec()
  rot = torch.tensor(rot, dtype=torch.float32)
  # trans = torch.tensor(trans, dtype=torch.float32)

  output = mano_model(
      betas=beta,
      global_orient=rot.reshape(1,3),
      # hand_pose=torch.zeros(1,45),
      hand_pose=pca_components.reshape(-1,15),
      transl=trans,
      return_verts=True,  # MANO doesn't return landmarks as well if this is false
  )

  extra_joints = torch.index_select(
      output.vertices, 1,
      torch.tensor(
          list(MANO_FINGERTIP_VERT_INDICES.values()),
          dtype=torch.long,
      ).to("cpu"),
  )
  joints = torch.cat([output.joints, extra_joints], dim=1)

  # # Move to Wrist
  joints -= pelvis
  joints = joints.squeeze().detach().cpu().numpy()

  # estimate_frame_from_hand_points
  joints = joints[mano_joint_mapping]


  joints = joints - joints[0:1, :]

  mediapipe_wrist_rot = estimate_frame_from_hand_points(joints)

  joints = joints @ mediapipe_wrist_rot @ (RIGHT_AXIS_TRANSFORMATION_RETARGET.detach().cpu().numpy())
  # if is_right:
  #   joints = torch.einsum(
  #     "ij, nkj -> nki",
  #     RIGHT_AXIS_TRANSFORMATION_RETARGET.to("cpu"),
  #     joints
  #   )
  # else:
  #   joints = torch.einsum(
  #     "ij, nkj -> nki",
  #     LEFT_AXIS_TRANSFORMATION_RETARGET.to("cpu"),
  #     joints
  #   )
  return joints.squeeze()  # We only need the 21 main joints
