
import torch
from scipy.spatial.transform import Rotation as R
import torch.nn as nn
import numpy as np
import smplx
from smplx.lbs import blend_shapes, vertices2joints

import tqdm

device="cuda"


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



def obtain_raw_mano_rotation(is_right):
  mano_model = mano_right if is_right else mano_left

  mano_model.to("cpu")

  # batch_size = np.prod(pca_components.shape) // 15
  betas = torch.zeros(10, dtype=torch.float32).unsqueeze(0).to("cpu")
  v_shaped = mano_model.v_template + blend_shapes(betas, mano_model.shapedirs)  # type: ignore
  pelvis = vertices2joints(
    mano_model.J_regressor[0:1], v_shaped
  ).squeeze(dim=1)

  pca_components = pca_components.reshape(-1, 15)

  # beta = torch.zeros((1, 10), dtype=torch.float32).to("cpu")

  # trans = raw_trans - pelvis.numpy()
  # trans = torch.zeros((1, 3), dtype=torch.float32).to("cpu")
  rot = R.from_matrix(
    np.eye(3)
  ).as_rotvec()
  rot = torch.tensor(rot, dtype=torch.float32)
  rot = torch.repeat_interleave(
    rot.reshape(1, 3),
    batch_size, dim=0
  ).to(pca_components.device)
  # trans = torch.tensor(trans, dtype=torch.float32)

  output = mano_model(
      betas=None,
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
  joints -= pelvis
  # print(pelvis.shape)
  # joints = joints.squeeze().detach().cpu().numpy()
  # print(joints.shape)
  joints = joints.squeeze()

  # estimate_frame_from_hand_points
  joints = joints[mano_joint_mapping]

  joints = joints - joints[0:1, :]

  # print(joints.shape)
  mediapipe_wrist_rot = estimate_frame_from_hand_points(joints)

RETARGET_MANO_TRANSFORMATION = obtain_raw_mano_rotation()