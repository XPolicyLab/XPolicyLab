import pickle
from pathlib import Path

import cv2
import tqdm
# import tyro
from typing import List

import torch
import numpy as np

LEFT_AXIS_TRANSFORMATION_RETARGET = np.array([
  [0, 0, -1],
  [1, 0, 0],
  [0, -1, 0],
])#.float()

RIGHT_AXIS_TRANSFORMATION_RETARGET = np.array([
  [0, 0, -1],
  [-1, 0, 0],
  [0, 1, 0],
])


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


def mano_joint_pos_to_retargetting(joint_pos):
  # Change Origin
  joint_pos = joint_pos - joint_pos[0:1, :]
  # Reorient & Mano have some weird rotation 
  mediapipe_wrist_rot = estimate_frame_from_hand_points(joint_pos)
  joint_pos = joint_pos @ mediapipe_wrist_rot @ RIGHT_AXIS_TRANSFORMATION_RETARGET
  return joint_pos

def retarget_single_step(
  # retargeting: SeqRetargeting,
  retargeting,
  joint_pos,
):
  # num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)
  # Joint Pos from Hand
  retargeting_type = retargeting.optimizer.retargeting_type
  indices = retargeting.optimizer.target_link_human_indices
  if retargeting_type == "POSITION":
      indices = indices
      ref_value = joint_pos[indices, :]
  else:
      origin_indices = indices[0, :]
      task_indices = indices[1, :]
      ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
  qpos = retargeting.retarget(ref_value)
  
  return qpos

def retarget_sequence(
  # retargeting: SeqRetargeting,
  retargeting,
  joint_pos_sequence: List,
):
  
  data = []
  for reference_data in joint_pos_sequence:
    qpos = retarget_sequence
    data.append(qpos)
  

  meta_data = dict(
      dof=len(retargeting.optimizer.robot.dof_joint_names),
      joint_names=retargeting.optimizer.robot.dof_joint_names,
  )

  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("wb") as f:
      pickle.dump(dict(data=data, meta_data=meta_data), f)

  retargeting.verbose()


# def main(
#     robot_name: RobotName, video_path: str, output_path: str, retargeting_type: RetargetingType, hand_type: HandType
# ):
#     """
#     Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

#     Args:
#         robot_name: The identifier for the robot. This should match one of the default supported robots.
#         video_path: The file path for the input video in .mp4 format.
#         output_path: The file path for the output data in .pickle format.
#         retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
#         hand_type: Specifies which hand is being tracked, either left or right.
#             Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
#             to another left robot hand, and the same applies for the right hand.
#     """

#     config_path = get_default_config_path(
#       robot_name, 
#       retargeting_type, 
#       hand_type
#     )
#     root_path = "external/dex-urdf"
#     robot_dir = Path(root_path).absolute() / "robots" / "hands"
#     RetargetingConfig.set_default_urdf_dir(str(robot_dir))
#     retargeting = RetargetingConfig.load_from_file(config_path).build()
#     # retarget_video(retargeting, video_path, output_path, str(config_path))


if __name__ == "__main__":
  pass
    # tyro.cli(main)
    
    # cd example/vector_retargeting
    # python3 detect_from_video.py \
    #   --robot-name allegro \
    #   --video-path data/human_hand_video.mp4 \
    #   --retargeting-type dexpilot \
    #   --hand-type right \
    #   --output-path data/allegro_joints.pkl