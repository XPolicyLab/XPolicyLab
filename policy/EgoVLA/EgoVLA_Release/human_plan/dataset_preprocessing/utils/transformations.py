import numpy as np
import cv2

def matrix_to_trans_rot(matrix):
  return np.squeeze(
    matrix[..., :3, -1]
  ), np.squeeze(
    matrix[..., :3, :3]
  )

def get_homo(point):
  if len(point.shape) == 2:
    point = np.concatenate([point, np.ones((point.shape[0], 1))], axis=-1)
    return point[np.newaxis, ...]
  return np.concatenate([point, np.ones((point.shape[0], point.shape[1], 1))], axis=-1)

def transform_to_world_frame(
  cam_pose, cam_frame_pos
):
  points = np.einsum(
    "nij, nkj-> nki",
    cam_pose,
    get_homo(cam_frame_pos)
  ).astype(dtype=np.float64)
  points = points[..., :3].astype(dtype=np.float64)
  return points

def transform_to_world_frame_rot(
  cam_pose, cam_frame_rot
):
  rots = np.einsum(
    # "nij, nkj-> nki",
    "nij, nkjl-> nkil",
    cam_pose[:, :3, :3],
    cam_frame_rot,
  ).astype(dtype=np.float64)
  return rots

def transform_to_world_frame_pose(
  cam_pose, cam_frame_pose
):
  # print(get_homo(cam_frame_pos).shape)
  poses = np.einsum(
    "nij, nkjl-> nkil",
    cam_pose,
    cam_frame_pose
  ).astype(dtype=np.float64)
  # points = points[..., :3].astype(dtype=np.float64)
  return poses

def transform_to_current_frame(
  inv_cam_pose, world_frame_pos
):
  points = np.einsum(
    "ij, nkj-> nki",
    inv_cam_pose,
    get_homo(world_frame_pos)
  ).astype(dtype=np.float64)
  points = points[..., :3].astype(dtype=np.float64)
  return points

def transform_to_current_frame_rot(
  inv_cam_pose, world_frame_rot
):
  rots = np.einsum(
    "ij, nkjl-> nkil",
    inv_cam_pose[:3, :3],
    world_frame_rot
    # get_homo(world_frame_pos)
  ).astype(dtype=np.float64)
  # points = points[..., :3].astype(dtype=np.float64)
  return rots

def transform_to_current_frame_pose(
  inv_cam_pose, world_frame_pose
):
  poses = np.einsum(
    "ij, nkjl-> nkil",
    inv_cam_pose,
    world_frame_pose
  ).astype(dtype=np.float64)
  return poses


def rotate_clockwise_batch(points, width, height):
  """
  Rotate a batch of points by 90 degrees clockwise using NumPy arrays.

  :param points: NumPy array of shape (N, 2), where each row is (x, y) coordinates of a point.
  :param width: The width of the image.
  :param height: The height of the image.
  :return: NumPy array of shape (N, 2) representing the new (x', y') coordinates after rotation.
  """
  # Extract x and y columns
  x = points[:, 0]
  y = points[:, 1]

  # Apply the 90-degree clockwise rotation formula
  new_x = height - y - 1
  new_y = x

  # Combine the results into a new array
  rotated_points = np.column_stack((new_x, new_y))

  return rotated_points

def project_set(
  points, 
  camera_intrinsics, 
  width, height
):
  rvec = np.array([[0.0, 0.0, 0.0]])
  tvec = np.array([0.0, 0.0, 0.0])
  # print(points.shape, points.dtype)
  points = points.reshape(-1, 3)
  points, _ = cv2.projectPoints(
    # points,
    points.astype(dtype=np.float64),
    rvec, tvec, camera_intrinsics, np.array([])
  )
  points = np.array(points).reshape(-1, 2)
  # points = rotate_clockwise_batch(points, width, height)

  mask_x = np.bitwise_and(
    points[..., 0] < width, points[..., 0] >= 0
  )
  mask_y = np.bitwise_and(
    points[..., 1] < height, points[..., 1] >= 0
  )
  mask = np.bitwise_and(
    mask_x,
    mask_y
  )
  return points, mask