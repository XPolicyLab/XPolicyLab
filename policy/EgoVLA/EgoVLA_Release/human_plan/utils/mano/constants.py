import torch
import numpy as np
from smplx.lbs import blend_shapes, vertices2joints

holoassist_to_mano_joint_mapping = np.array([
  1, # wrist 
  2, 3, 4, 5, # Thumb
  7, 8, 9, 10, # Index
  12, 13, 14, 15, # middle
  17, 18, 19, 20, # Ring
  22, 23, 24, 25 # Index
])

mano_joint_mapping = np.array([
  0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
])


MANO_FINGERTIP_VERT_INDICES = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}

LEFT_AXIS_TRANSFORMATION_RETARGET = torch.tensor([
  [0, 0, -1],
  [1, 0, 0],
  [0, -1, 0],
]).float().to("cpu")

RIGHT_AXIS_TRANSFORMATION_RETARGET = torch.tensor([
  [0, 0, -1],
  [-1, 0, 0],
  [0, 1, 0],
]).float().to("cpu")


LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB = torch.tensor([
  [0, 1, 0],
  [-1, 0, 0],
  [0, 0, 1],
]).float().to("cpu")

# LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB = torch.tensor([
#   [0, -1, 0],
#   [1, 0, 0],
#   [0, 0, 1],
# ]).float().to("cpu")

RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB = torch.tensor([
  [0, 1, 0],
  [1, 0, 0],
  [0, 0,  -1],
]).float().to("cpu")


LEFT_AXIS_TRANSFORMATION = torch.tensor([
  [1, 0, 0],
  [0,  0, -1],
  [0,  1, 0]
]).float().to("cpu")

RIGHT_AXIS_TRANSFORMATION = torch.tensor([
  [-1, 0, 0],
  [0,  0, 1],
  [0,  1, 0]
]).float().to("cpu")


LEFT_AXIS_TRANSFORMATION_44 = torch.tensor([
  [1, 0, 0, 0],
  [0,  0, -1, 0],
  [0,  1, 0, 0],
  [0,  0, 0, 1]
]).float().to("cpu")

RIGHT_AXIS_TRANSFORMATION_44 = torch.tensor([
  [-1, 0, 0, 0],
  [0,  0, 1, 0],
  [0,  1, 0, 0],
  [0,  0, 0, 1]
]).float().to("cpu")

def obtain_mano_pelvis(mano_model):
  betas = torch.zeros(
    10, dtype=torch.float32
  ).unsqueeze(0).to("cpu")
  v_shaped = mano_model.v_template + \
    blend_shapes(betas, mano_model.shapedirs)  # type: ignore
  pelvis = vertices2joints(
    mano_model.J_regressor[0:1], v_shaped
  ).squeeze(dim=1)
  return pelvis.reshape(1, 3)

from human_plan.utils.mano.model import (
  mano_left,
  mano_right
)

LEFT_PELVIS = obtain_mano_pelvis(mano_left)
RIGHT_PELVIS = obtain_mano_pelvis(mano_right)

from human_plan.utils.mano.functions import (
  estimate_frame_from_hand_points
)

from scipy.spatial.transform import Rotation as R

def obtain_raw_mano_rotation(is_right):
  mano_model = mano_right if is_right else mano_left

  mano_model.to("cpu")

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS
  pelvis = pelvis.to("cpu")

  pca_components = torch.zeros((1, 15), dtype=mano_model.dtype)

  beta = torch.zeros((1, 10), dtype=torch.float32).to("cpu")

  # trans = raw_trans - pelvis.numpy()
  trans = torch.zeros((1, 3), dtype=torch.float32).to("cpu")
  rot = R.from_matrix(
    np.eye(3)
  ).as_rotvec()
  rot = torch.tensor(rot, dtype=torch.float32).reshape(1, 3)

  output = mano_model(
      betas=beta,
      global_orient=rot.reshape(-1, 3),
      hand_pose=pca_components.reshape(-1,15),
      transl=trans,
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
  # Move to Wrist
  joints -= pelvis
  joints = joints.squeeze()

  # estimate_frame_from_hand_points
  joints = joints[mano_joint_mapping]

  joints = joints - joints[0:1, :]

  mediapipe_wrist_rot = estimate_frame_from_hand_points(joints.detach().cpu().numpy())
  return mediapipe_wrist_rot

RIGHT_RETARGET_MANO_TRANSFORMATION = torch.tensor(obtain_raw_mano_rotation(is_right=True)).cpu()
LEFT_RETARGET_MANO_TRANSFORMATION = torch.tensor(obtain_raw_mano_rotation(is_right=False)).cpu()

EMPTY_ROT = R.from_matrix(
  np.eye(3)
).as_rotvec()
EMPTY_ROT = torch.tensor(EMPTY_ROT, dtype=torch.float32)


if __name__ == "__main__":
  print(LEFT_RETARGET_MANO_TRANSFORMATION)
  print(RIGHT_RETARGET_MANO_TRANSFORMATION)