import numpy as np


# CAM_AXIS_TRANSFORM = np.linalg.inv(np.array([
#     [0, 0, 1, 0],
#     [-1, 0, 0, 0],
#     [0, -1, 0, 0],
#     [0, 0, 0, 1]
# ]))

# Rotate by Yaw for 90 degree then transform the coordinate system

# CAM_AXIS_TRANSFORM = np.array([
#     [0, -1, 0, 0],
#     [0, 0, -1, 0],
#     [1, 0, 0, 0],
#     [0, 0, 0, 1]
# ]) @ np.linalg.inv(np.array([
#     [0, -1, 0, 0],
#     [1, 0, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])) @ np.linalg.inv(np.array([
#     [1, 0, 0, 0],
#     [0, 0, -1, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]
# ]))


CAM_AXIS_TRANSFORM = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])
# CAM_AXIS_TRANSFORM = np.eye(4)


# ISAAC_LAB_CAMERA_FRAME_CHANGE = np.linalg.inv(np.array([
#     [1, 0, 0, 0],
#     [0, 0, -1, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]
# ])) @ np.linalg.inv(np.array([
#     [0, -1, 0, 0],
#     [1, 0, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ]))

from scipy.spatial.transform import Rotation as R

# Use manually calibrated pair to compute the relative transformation
def get_isaac_lab_cam_frame_change_mat():
  # Isaac Lab Convention WXYZ
  cam_quat = (0.66446, 0.24184, -0.24184, -0.664464)
  cam_quat_xyzw = (0.24184, -0.24184, -0.664464, 0.66446)
  cam_rotmat = R.from_quat(cam_quat_xyzw)
  cam_rotmat = cam_rotmat.as_matrix()

  gt_cam_quat = (0.9063077870366499, 0.0, 0.42261826174069944, 0.0)
  gt_cam_quat_xyzw = (0.0, 0.42261826174069944, 0.0, 0.9063077870366499, )
  gt_rotmat = R.from_quat(gt_cam_quat_xyzw)
  gt_rotmat = gt_rotmat.as_matrix()

  # relative_rotmat = np.eye(4)
  relative_rotmat = gt_rotmat @ np.linalg.inv(cam_rotmat)
  
  return relative_rotmat

ISAAC_LAB_CAMERA_FRAME_CHANGE = get_isaac_lab_cam_frame_change_mat()


def homogeneous_coord(points):
  points_shape = list(points.shape)
  points_shape[-1] = 1
  # print(np.ones(tuple(points_shape)).shape)
  points_3d_homogeneous = np.concatenate([
    points, 
    np.ones(tuple(points_shape))
  ], axis=-1)
  return points_3d_homogeneous


def point_from_frame_to_frame(
    pos, frameA, frameB
):
  R_A, t_A = frameA[:3, :3], frameA[:3, -1]
  R_B, t_B = frameB[:3, :3], frameB[:3, -1]
  # Step 1: Transform point from Frame A to World frame
  P_W = R_A @ pos + t_A

  # Step 2: Transform point from World frame to Frame B
  P_B = R_B.T @ (P_W - t_B)

  return P_B
# Output the result


def ee_from_frame_to_frame(
    pos, frameA, frameB
):
  # print(pos.shape)
  left_A, right_A = pos[:3], pos[3:]
  # print(left_A.shape, right_A.shape)

  left_B = point_from_frame_to_frame(
      left_A, frameA, frameB
  )
  right_B = point_from_frame_to_frame(
      right_A, frameA, frameB
  )

  P_B = np.concatenate([
      left_B, right_B,
  ], axis=-1)

  return P_B


def coord_from_frame_to_frame_homo(
    coord, homoA, homoB
):
  coord_left, coord_right = coord[0][0], coord[1][0]
  # print(pos.shape)
  coord_left = np.append(coord_left, 1)
  coord_right = np.append(coord_right, 1)

  coord_left = homoB @ np.linalg.inv(homoA) @ coord_left
  coord_right = homoB @ np.linalg.inv(homoA) @ coord_right

  coord_left = coord_left[:2] / coord_left[-1]
  coord_right = coord_right[:2] / coord_right[-1]

  return coord_left, coord_right
