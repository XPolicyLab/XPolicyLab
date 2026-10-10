import cv2
import numpy as np

def project_points(
  points, img_intrinsics
):
  points = points.reshape(-1, 3)
  # Put an empty camera pose for image.
  rvec = np.array([[0.0, 0.0, 0.0]])
  tvec = np.array([0.0, 0.0, 0.0])
  # print(points.shape, points.dtype)
  points = points.astype(np.float32)
  points, _ = cv2.projectPoints(
    points, rvec, tvec, img_intrinsics, np.array([])
  )

  return np.array(points).reshape(-1, 2)


def plot_points(points, img, color):
  points = points.reshape(-1, 2)
  # img_list = []
  for idx, point in enumerate(points):
    
    img = cv2.circle(
      img, (int(round(point[0])), int(round(point[1]))),
      radius=10, color=color,
      thickness=-1
    )
  return img


def plot_hand(points, img, color, get_handpose_connectivity_func):
  points = points.reshape(-1, 2)

  connectivity = get_handpose_connectivity_func()
  thickness = 4

  if not (np.isnan(points).any()):
    for limb in connectivity:
      cv2.line(
        img, (int(points[limb[0]][0]), int(points[limb[0]][1])),
               (int(points[limb[1]][0]), int(points[limb[1]][1])), color, thickness)

  return img

def get_handpose_connectivity_sim():
  return [
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
      [0, 5],
  ]

def plot_hand_sim(points, img, color):
  return plot_hand(points, img, color, get_handpose_connectivity_sim)

def get_handpose_connectivity_mano():
  return [
      # Thumb
      [0, 1], [1, 2], [2, 3], [3, 4],
      # Index
      [0, 5], [5, 6], [6, 7], [7, 8],
      # Middle
      [0, 9], [9, 10], [10, 11], [11, 12],
      # Ring
      [0, 13], [13, 14], [14, 15], [15, 16],
      # Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
  ]

def plot_hand_mano(points, img, color):
  return plot_hand(points, img, color, get_handpose_connectivity_mano)
