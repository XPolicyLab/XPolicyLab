import torch
import numpy as np
import smplx
from smplx.lbs import blend_shapes, vertices2joints

from human_plan.utils.mano.constants import (
  holoassist_to_mano_joint_mapping,
  mano_joint_mapping,
  MANO_FINGERTIP_VERT_INDICES,
)


mano_left = smplx.create(
  'mano_v1_2/models/MANO_LEFT.pkl',
  "mano",
  use_pca=True,
  is_rhand=False,
  num_pca_comps=15,
)
mano_left.to("cpu")

mano_right = smplx.create(
  'mano_v1_2/models/MANO_RIGHT.pkl',
  "mano",
  use_pca=True,
  is_rhand=True,
  num_pca_comps=15,
)
mano_right.to("cpu")

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

def mano_handpose_connectivity():
  return [
      [0, 1],

      # Thumb
      [1, 2], [2, 3], [3, 4],
      # Index
      [0, 5], [5, 6], [6, 7], [7, 8],

      # Middle
      [0, 9], [9, 10], [10, 11], [11, 12],

      # Ring
      [0, 13], [13, 14], [14, 15], [15, 16],
      
      # Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
  ]

def obtain_mano_pelvis(mano_model):
  betas = torch.zeros(10, dtype=torch.float32).unsqueeze(0).to(mano_model.v_template.device)
  # print("shapedirs",mano_model.shapedirs.device)
  # print("betas",betas.device)
  v_shaped = mano_model.v_template + blend_shapes(betas, mano_model.shapedirs)  # type: ignore

  pelvis = vertices2joints(
    mano_model.J_regressor[0:1], v_shaped
  ).squeeze(dim=1)
  return pelvis.detach().cpu().numpy()

# from human_plan.utils.mano.functions import (
#   obtain_mano_pelvis
# )

# Function to compute joint positions from MANO parameters
def mano_forward(
  mano_model, 
  hand_joint_angles,
  # rot, trans,
  global_rot,
  global_trans,
  # target_transformation,
  # pelvis,
  # shape,
  # is_right
):
  batch_size = np.prod(hand_joint_angles.shape) // 15
  beta = torch.zeros((batch_size, 10), dtype=torch.float32).to(hand_joint_angles.device).to(hand_joint_angles.dtype)
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
  # print("joints",joints.shape)
  return joints.squeeze()  # We only need the 21 main joints


# from scipy.spatial.transform import Rotation as R
# from human_plan.utils.mano.constants import (
#   RIGHT_AXIS_TRANSFORMATION_RETARGET,
#   LEFT_AXIS_TRANSFORMATION_RETARGET,
#   estimate_frame_from_hand_points
# )

# def mano_forward_retarget(
#   pca_components, 
#   # hand_info, 
#   is_right
# ):
#   mano_model = mano_right if is_right else mano_left

#   mano_model.to(pca_components.device)

#   batch_size = np.prod(pca_components.shape) // 15
#   betas = torch.zeros(10, dtype=torch.float32).unsqueeze(0).to(pca_components.device)

#   v_shaped = mano_model.v_template + blend_shapes(betas, mano_model.shapedirs)  # type: ignore
#   pelvis = vertices2joints(
#     mano_model.J_regressor[0:1], v_shaped
#   ).squeeze(dim=1)

#   pca_components = pca_components.reshape(-1, 15)

#   # beta = torch.zeros((1, 10), dtype=torch.float32).to("cpu")

#   # trans = raw_trans - pelvis.numpy()
#   # trans = torch.zeros((1, 3), dtype=torch.float32).to("cpu")
#   rot = R.from_matrix(
#     np.eye(3)
#   ).as_rotvec()
#   rot = torch.tensor(rot, dtype=torch.float32)
#   rot = torch.repeat_interleave(
#     rot.reshape(1, 3),
#     batch_size, dim=0
#   ).to(pca_components.device)
#   # trans = torch.tensor(trans, dtype=torch.float32)

#   output = mano_model(
#       betas=None,
#       global_orient=rot.reshape(-1, 3),
#       # hand_pose=torch.zeros(1,45),
#       hand_pose=pca_components.reshape(-1,15),
#       transl=None,
#       return_verts=True,  # MANO doesn't return landmarks as well if this is false
#   )

#   extra_joints = torch.index_select(
#       output.vertices, 1,
#       torch.tensor(
#           list(MANO_FINGERTIP_VERT_INDICES.values()),
#           dtype=torch.long,
#       ).to(pca_components.device),
#   )
#   joints = torch.cat([output.joints, extra_joints], dim=1)

#   # # Move to Wrist
#   joints -= pelvis
#   # print(pelvis.shape)
#   # joints = joints.squeeze().detach().cpu().numpy()
#   # print(joints.shape)
#   joints = joints.squeeze()

#   # estimate_frame_from_hand_points
#   joints = joints[mano_joint_mapping]

#   joints = joints - joints[0:1, :]

#   print(joints.shape)
#   mediapipe_wrist_rot = estimate_frame_from_hand_points(joints)

#   if is_right:
#     RETARGET_AXIS_TRANSFORMATION = RIGHT_AXIS_TRANSFORMATION_RETARGET
#   else:
#     RETARGET_AXIS_TRANSFORMATION = LEFT_AXIS_TRANSFORMATION_RETARGET

#   joints = joints @ mediapipe_wrist_rot @ RETARGET_AXIS_TRANSFORMATION

#   return joints.squeeze()  # We only need the 21 main joints
