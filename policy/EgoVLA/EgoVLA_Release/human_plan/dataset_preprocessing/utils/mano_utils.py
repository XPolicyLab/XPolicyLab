import torch
from scipy.spatial.transform import Rotation as R
import torch.nn as nn
import numpy as np
import smplx
from smplx.lbs import blend_shapes, vertices2joints

import tqdm

device = "cpu"
if torch.cuda.is_available():
  device="cuda"

mano_joint_mapping = np.array([
  0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
])

holoassist_to_mano_joint_mapping = np.array([
  1, # wrist 
  2, 3, 4, 5, # Thumb
  7, 8, 9, 10, # Index
  12, 13, 14, 15, # middle
  17, 18, 19, 20, # Ring
  22, 23, 24, 25 # Index
])

MANO_FINGERTIP_VERT_INDICES = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}

mano_left = smplx.create(
  'mano_v1_2/models/MANO_LEFT.pkl',
  "mano",
  use_pca=True,
  is_rhand=False,
  num_pca_comps=15,
)
mano_left.to(device)

mano_right = smplx.create(
  'mano_v1_2/models/MANO_RIGHT.pkl',
  "mano",
  use_pca=True,
  is_rhand=True,
  num_pca_comps=15,
)
mano_right.to(device)


LEFT_AXIS_TRANSFORMATION = torch.tensor([
  [1, 0, 0],
  [0,  0, -1],
  [0,  1, 0]
]).float().to(device)

RIGHT_AXIS_TRANSFORMATION = torch.tensor([
  [-1, 0, 0],
  [0,  0, 1],
  [0,  1, 0]
]).float().to(device)

# Function to compute joint positions from MANO parameters
def mano_forward_reverse(mano_model, hand_joint_angles, is_right):
  batch_size = np.prod(hand_joint_angles.shape) // 15
  betas = torch.zeros(10, dtype=torch.float32).unsqueeze(0).to(device)
  mano_model.to(device)
  # print(betas.device)
  # print(mano_model.device)
  # print(hand_joint_angles.device)
  v_shaped = mano_model.v_template + blend_shapes(betas, mano_model.shapedirs)  # type: ignore
  pelvis = vertices2joints(
    mano_model.J_regressor[0:1], v_shaped
  ).squeeze(dim=1)

  r = R.from_matrix(np.eye(3))
  temp_global_rot = r.as_rotvec()
  temp_global_rot = torch.tensor(temp_global_rot, dtype=torch.float32).reshape(1, 3).to(device)
  temp_global_rot = torch.repeat_interleave(
    temp_global_rot, batch_size, dim=0
  )
  
  tmp_trans = torch.zeros(batch_size, 3, dtype=torch.float32).to(device)
  shape = torch.zeros(batch_size, 10, dtype=torch.float32).to(device) # Neutral shape
  # pose = torch.tensor(params[:45], dtype=torch.float32)  # 15 joints * 3 axis-angle
  output = mano_model(
      betas=shape,
      global_orient=temp_global_rot,
      # hand_pose=torch.zeros(1,45),
      hand_pose=hand_joint_angles.reshape(-1,15),
      transl=tmp_trans,
      return_verts=True,  # MANO doesn't return landmarks as well if this is false
  )
  # print(output)
  extra_joints = torch.index_select(
      output.vertices, 1,
      torch.tensor(
          list(MANO_FINGERTIP_VERT_INDICES.values()),
          dtype=torch.long,
      ).to(device),
  )
  joints = torch.cat([output.joints, extra_joints], dim=1)

  # Move to Wrist
  joints -= pelvis
  
  joints = joints[:, mano_joint_mapping]
  joints = joints.float()
  if is_right:
    joints = torch.einsum(
      "ij, nkj -> nki",
      RIGHT_AXIS_TRANSFORMATION,
      joints
    )
  else:
    joints = torch.einsum(
      "ij, nkj -> nki",
      LEFT_AXIS_TRANSFORMATION,
      joints
    )

  return joints.squeeze()  # We only need the 21 main joints

def obtain_mano_parameters_holoassist(mano_model, hand_info, is_right):
  # N, 21, 4, 4
  observed_joints = hand_info[
    "hand_pose"
  ][:, holoassist_to_mano_joint_mapping]

  # N, 1, 4, 4
  global_pose = observed_joints[:, 0:1]


  observed_joints = np.linalg.inv(global_pose) @ observed_joints
  observed_joints = observed_joints[:, :, :3, -1]
  observed_joints = observed_joints - observed_joints[:, 0:1]
  observed_joints = torch.tensor(observed_joints).float().to(device)
  # N, 21, 3

  r = R.from_matrix(np.eye(3))
  global_rot = r.as_rotvec()
  global_rot = torch.tensor(global_rot, dtype=torch.float32).reshape(1, 3)

  # Initialize pose parameters (45 axis-angle parameters: 15 joints * 3)
  batch_size = observed_joints.shape[0]
  initial_pose = nn.Parameter(torch.rand(batch_size, 15, requires_grad=True, device=device) * 0.1)

  # Set up the optimizer
  optimizer = torch.optim.Adam([initial_pose], lr=0.05)
  # Optimization loop
  # for epoch in tqdm.tqdm(range(500)):  # Adjust the number of iterations as needed
  for epoch in range(500):  # Adjust the number of iterations as needed
      optimizer.zero_grad()
      # Forward pass: Predict joints using the current pose parameters
      predicted_joints = mano_forward_reverse(mano_model, initial_pose, is_right)
      # Compute the loss (L2 distance between observed and predicted joints)

      loss = torch.nn.functional.mse_loss(predicted_joints, observed_joints)
      # Backward pass: Compute gradients
      loss.backward()
      optimizer.step()

  optimized_pose = initial_pose.detach()

  kps3d = mano_forward_reverse(mano_model, optimized_pose, is_right)
  kps3d = kps3d.detach().cpu().numpy()
  kps3d = np.einsum(
    "nij, nkj -> nki",
    global_pose[:, 0],
    np.concatenate([
        kps3d,
       np.ones((kps3d.shape[0], kps3d.shape[1], 1))
    ], axis=-1)
  )
  kps3d = kps3d[:, :,:3]

  return {
    "mano_kp_predicted": kps3d,
    "optimized_mano_parameters": optimized_pose.cpu().numpy(),
  }

from human_plan.utils.transformation import (
   homogeneous_coord
)


mano_to_inspire_mapping = np.array([
  0, 4, 8, 12, 16, 20
])

inspire_hand_scaling = 1.5

from human_plan.utils.mano_to_robot import (
  estimate_frame_from_hand_points
)

from human_plan.utils.mano.forward import (
  mano_forward,
  mano_forward_retarget,
  mano_forward_retarget_isaaclab
)

# from human_plan.utils.mano.mano_model import (
#   mano_forward
# )

from human_plan.utils.mano.model import (
  mano_left,
  mano_right
)

#   return joints.squeeze()  # We only need the 21 main joints

def obtain_mano_parameters_otv_inspire_hand(
    mano_model, finger_tip, ee_pose, is_right
):
  # N, 5, 3
  observed_joints = np.concatenate([
    ee_pose[:, :3, -1].reshape(-1, 1, 3),
    finger_tip
  ], axis=1)

  # N, 4, 4
  global_pose = ee_pose.copy()
  print("Global Pose", np.linalg.inv(global_pose)[:, np.newaxis, :, :].shape)
  print("Homo Coord", homogeneous_coord(observed_joints)[:, :, :, np.newaxis].shape)
  observed_joints = np.linalg.inv(global_pose)[:, np.newaxis, :, :] @ \
    homogeneous_coord(observed_joints)[:, :, :, np.newaxis]
  observed_joints = observed_joints[:, :, :3, 0]
  # print(observed_joints.shape)
  observed_joints = observed_joints - observed_joints[:, 0:1]
  observed_joints = observed_joints / inspire_hand_scaling
  # print(observed_joints)
  observed_joints = torch.tensor(observed_joints).float().to(device)
  # N, 6, 3

  # Initialize pose parameters (45 axis-angle parameters: 15 joints * 3)
  batch_size = observed_joints.shape[0]
  initial_pose = nn.Parameter(torch.rand(batch_size, 15, requires_grad=True, device=device) * 0.1)

  # Set up the optimizer
  optimizer = torch.optim.Adam([initial_pose], lr=0.1)
  # Optimization loop
  # for epoch in tqdm.tqdm(range(500)):  # Adjust the number of iterations as needed
  for epoch in range(500):  # Adjust the number of iterations as needed
    optimizer.zero_grad()
    # Forward pass: Predict joints using the current pose parameters
    # predicted_joints = mano_forward_empty(
    #   # mano_model, 
    #   initial_pose, is_right)
    print(initial_pose.shape)
    predicted_joints = mano_forward_retarget_isaaclab(initial_pose, is_right)
    # Compute the loss (L2 distance between observed and predicted joints)
    # print(predicted_joints.shape)
    # fix_rotattion = estimate_frame_from_hand_points(predicted_joints.detach().cpu().numpy())

    predicted_joints_inspire = predicted_joints[:, mano_to_inspire_mapping]
    # print(predicted_joints_inspire.shape, observed_joints.shape)
    loss = torch.nn.functional.smooth_l1_loss(predicted_joints_inspire, observed_joints)
    # loss = torch.nn.functional.mse_loss(predicted_joints_inspire, observed_joints)
    # Backward pass: Compute gradients
    loss.backward()
    optimizer.step()

    # print(loss)
  print(loss)
  optimized_pose = initial_pose.detach()
  # optimized_pose = torch.zeros_like(initial_pose)

  kps3d = mano_forward_retarget_isaaclab(
    # mano_model, 
    optimized_pose, is_right)
  kps3d = kps3d.detach().cpu().numpy()
  kps3d = kps3d * inspire_hand_scaling
  # print(kps3d)
  # print(kps3d.shape, global_pose.shape)
  kps3d = np.einsum(
    "nij, nkj -> nki",
    global_pose,
    np.concatenate([
        kps3d,
       np.ones((kps3d.shape[0], kps3d.shape[1], 1))
    ], axis=-1)
    # homogeneous_coord(kps3d)
  )
  # print(kps3d.shape)
  kps3d = kps3d[:, :,:3]

  return {
    "mano_kp_predicted": kps3d,
    "optimized_mano_parameters": optimized_pose.cpu().numpy(),
  }


from human_plan.utils.mano.constants import (
  RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB,
  LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB,
  LEFT_PELVIS,
  RIGHT_PELVIS
)

def obtain_mano_parameters_otv_inspire_hand_mano_rot(
    mano_model, finger_tip, ee_pose, is_right
):
  mano_model = mano_right if is_right else mano_left
  mano_model.to(device)
  # N, 5, 3
  observed_joints = np.concatenate([
    ee_pose[:, :3, -1].reshape(-1, 1, 3),
    finger_tip
  ], axis=1)

  # N, 4, 4
  global_pose = ee_pose.copy()
  # print("Global Pose", np.linalg.inv(global_pose)[:, np.newaxis, :, :].shape)
  # print("Homo Coord", homogeneous_coord(observed_joints)[:, :, :, np.newaxis].shape)

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB

  global_rot = global_pose[:, :3, :3] @ retarget_axis_transformation.cpu().numpy()
  global_transformed_pose = np.eye(4)

  global_transformed_pose = np.repeat(
    global_transformed_pose[None, :, :],
    global_rot.shape[0],
    axis=0
  )
  global_transformed_pose[:, :3, :3] = global_rot.copy()


  global_rot = R.from_matrix(global_rot)
  global_rot = global_rot.as_rotvec()
  global_rot = torch.tensor(global_rot).to(device).float()

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS
  global_trans = torch.tensor(global_pose[:, :3, -1] - pelvis.cpu().numpy()).float().to(device)

  global_transformed_pose[:, :3, -1] = global_trans.cpu().numpy()

  observed_joints = torch.tensor(observed_joints).float().to(device)

  observed_joints = (
    observed_joints - observed_joints[:, 0:1]
  ) / inspire_hand_scaling + observed_joints[:, 0:1]

  # observed_joints = observed_joints - observed_joints[:, 0:1]
  # observed_joints = observed_joints / inspire_hand_scaling
  # observed_joints = observed_joints 
  # N, 6, 3

  # Initialize pose parameters (45 axis-angle parameters: 15 joints * 3)
  hand_dim = 6
  batch_size = observed_joints.shape[0]
  # initial_pose = nn.Parameter(torch.rand(batch_size, 15, requires_grad=True, device=device) * 0.1)
  # initial_pose = nn.Parameter(torch.rand(batch_size, 6, requires_grad=True, device=device) * 0.1)
  initial_pose = nn.Parameter(torch.rand(batch_size, hand_dim, requires_grad=True, device=device) * 0.1)

  # Set up the optimizer
  optimizer = torch.optim.Adam([initial_pose], lr=0.1)
  # Optimization loop
  # for epoch in tqdm.tqdm(range(500)):  # Adjust the number of iterations as needed
  for epoch in range(500):  # Adjust the number of iterations as needed
    optimizer.zero_grad()
    # Forward pass: Predict joints using the current pose parameters
    # predicted_joints = mano_forward_empty(
    #   # mano_model, 
    #   initial_pose, is_right)
    # print(initial_pose.shape)
    # predicted_joints = mano_forward(
    #   initial_pose, is_right)
    predicted_joints = mano_forward(
      mano_model,
      # initial_pose,
      torch.concat([
        initial_pose,
        torch.zeros((batch_size, 15 - hand_dim)).to(initial_pose.device).to(initial_pose.dtype)
      ], dim=-1),
      global_rot,
      global_trans,
    )
    # predicted_joints = mano_forward_retarget_isaaclab(
    #   initial_pose, is_right
    # )
    # Compute the loss (L2 distance between observed and predicted joints)
    # print(predicted_joints.shape)
    # fix_rotattion = estimate_frame_from_hand_points(predicted_joints.detach().cpu().numpy())

    predicted_joints_inspire = predicted_joints[:, mano_to_inspire_mapping]

    # predic

    # Inspire Hand need to scale
    # predicted_joints_inspire = (
    #   (predicted_joints - predicted_joints[:, 0:1]) * inspire_hand_scaling + \
    #   predicted_joints[:, 0:1]
    # )[:, mano_to_inspire_mapping]

    # print(predicted_joints_inspire.shape, observed_joints.shape)
    loss = torch.nn.functional.smooth_l1_loss(predicted_joints_inspire, observed_joints)
    # loss = torch.nn.functional.mse_loss(predicted_joints_inspire, observed_joints)
    # Backward pass: Compute gradients
    loss.backward()
    optimizer.step()

    # print(loss)
  print(loss)
  optimized_pose = initial_pose.detach()
  optimized_pose = torch.concat([
    optimized_pose,
    torch.zeros((batch_size, 15 - hand_dim)).to(initial_pose.device).to(initial_pose.dtype)
  ], dim=-1)
  # optimized_pose = torch.zeros_like(initial_pose)

  # kps3d = mano_forward_retarget_isaaclab(
  #   # mano_model, 
  #   optimized_pose, is_right)
  kps3d = mano_forward(
      mano_model,
      optimized_pose,
      global_rot,
      global_trans,
  )
  kps3d = kps3d.detach().cpu().numpy()
  # kps3d = kps3d * inspire_hand_scaling
  # # print(kps3d)
  # # print(kps3d.shape, global_pose.shape)
  # kps3d = np.einsum(
  #   "nij, nkj -> nki",
  #   global_pose,
  #   np.concatenate([
  #       kps3d,
  #      np.ones((kps3d.shape[0], kps3d.shape[1], 1))
  #   ], axis=-1)
  #   # homogeneous_coord(kps3d)
  # )
  # print(kps3d.shape)
  kps3d = kps3d[:, :,:3]
  kps3d = (kps3d - kps3d[:, 0:1]) * inspire_hand_scaling + kps3d[:, 0:1]

  return {
    "mano_kp_predicted": kps3d,
    "optimized_mano_parameters": optimized_pose.cpu().numpy(),
    "optimized_mano_rot": global_rot.cpu().numpy(),
    "optimized_mano_trans": global_trans.cpu().numpy(),
    "optimized_mano_t_pose": global_transformed_pose
  }

# DATA PROCESSING INDICES
RETARGETTING_INDICES = [0, 4, 9, 14, 19, 24]
# 0 indices are set at the origin of palm, which is always zero
VALID_RETARGETTING_INDICES = [4, 9, 14, 19, 24]

def obtain_mano_parameters_otv_inspire_hand_mano_no_rot(
    mano_model, finger_tip, is_right
):
  mano_model = mano_right if is_right else mano_left
  mano_model.to(device)

  ee_pose = np.eye(4)
  ee_pose = np.repeat( 
    ee_pose[None, :, :],
    finger_tip.shape[0],
    axis=0
  )
  print("EE Pose", ee_pose.shape)
  # N, 5, 3
  observed_joints = np.concatenate([
    ee_pose[:, :3, -1].reshape(-1, 1, 3),
    finger_tip[:, VALID_RETARGETTING_INDICES]
  ], axis=1)

  print("Observed Joints", observed_joints.shape)


  # N, 4, 4
  global_pose = ee_pose.copy()

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB

  global_rot = global_pose[:, :3, :3] @ retarget_axis_transformation.cpu().numpy()
  global_transformed_pose = np.eye(4)

  global_transformed_pose = np.repeat(
    global_transformed_pose[None, :, :],
    global_rot.shape[0],
    axis=0
  )
  global_transformed_pose[:, :3, :3] = global_rot.copy()


  global_rot = R.from_matrix(global_rot)
  global_rot = global_rot.as_rotvec()
  global_rot = torch.tensor(global_rot).to(device).float()

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS
  global_trans = torch.tensor(global_pose[:, :3, -1] - pelvis.cpu().numpy()).float().to(device)

  global_transformed_pose[:, :3, -1] = global_trans.cpu().numpy()

  observed_joints = torch.tensor(observed_joints).float().to(device)

  observed_joints = (
    observed_joints - observed_joints[:, 0:1]
  ) / inspire_hand_scaling + observed_joints[:, 0:1]

  # observed_joints = observed_joints - observed_joints[:, 0:1]
  # observed_joints = observed_joints / inspire_hand_scaling
  # observed_joints = observed_joints 
  # N, 6, 3

  # Initialize pose parameters (45 axis-angle parameters: 15 joints * 3)
  hand_dim = 6
  batch_size = observed_joints.shape[0]
  # initial_pose = nn.Parameter(torch.rand(batch_size, 15, requires_grad=True, device=device) * 0.1)
  # initial_pose = nn.Parameter(torch.rand(batch_size, 6, requires_grad=True, device=device) * 0.1)
  initial_pose = nn.Parameter(torch.rand(batch_size, hand_dim, requires_grad=True, device=device) * 0.1)

  # Set up the optimizer
  optimizer = torch.optim.Adam([initial_pose], lr=0.1)
  # Optimization loop
  # for epoch in tqdm.tqdm(range(500)):  # Adjust the number of iterations as needed
  for epoch in range(500):  # Adjust the number of iterations as needed
    optimizer.zero_grad()
    # Forward pass: Predict joints using the current pose parameters
    # predicted_joints = mano_forward_empty(
    #   # mano_model, 
    #   initial_pose, is_right)
    # print(initial_pose.shape)
    # predicted_joints = mano_forward(
    #   initial_pose, is_right)
    predicted_joints = mano_forward(
      mano_model,
      # initial_pose,
      torch.concat([
        initial_pose,
        torch.zeros((batch_size, 15 - hand_dim)).to(initial_pose.device).to(initial_pose.dtype)
      ], dim=-1),
      global_rot,
      global_trans,
    )
    # predicted_joints = mano_forward_retarget_isaaclab(
    #   initial_pose, is_right
    # )
    # Compute the loss (L2 distance between observed and predicted joints)
    # print(predicted_joints.shape)
    # fix_rotattion = estimate_frame_from_hand_points(predicted_joints.detach().cpu().numpy())

    predicted_joints_inspire = predicted_joints[:, mano_to_inspire_mapping]

    # predic

    # Inspire Hand need to scale
    # predicted_joints_inspire = (
    #   (predicted_joints - predicted_joints[:, 0:1]) * inspire_hand_scaling + \
    #   predicted_joints[:, 0:1]
    # )[:, mano_to_inspire_mapping]

    # print(predicted_joints_inspire.shape, observed_joints.shape)
    loss = torch.nn.functional.smooth_l1_loss(predicted_joints_inspire, observed_joints)
    # loss = torch.nn.functional.mse_loss(predicted_joints_inspire, observed_joints)
    # Backward pass: Compute gradients
    loss.backward()
    optimizer.step()

    # print(loss)
  print(loss)
  optimized_pose = initial_pose.detach()
  optimized_pose = torch.concat([
    optimized_pose,
    torch.zeros((batch_size, 15 - hand_dim)).to(initial_pose.device).to(initial_pose.dtype)
  ], dim=-1)
  # optimized_pose = torch.zeros_like(initial_pose)

  # kps3d = mano_forward_retarget_isaaclab(
  #   # mano_model, 
  #   optimized_pose, is_right)
  kps3d = mano_forward(
      mano_model,
      optimized_pose,
      global_rot,
      global_trans,
  )
  kps3d = kps3d.detach().cpu().numpy()
  # kps3d = kps3d * inspire_hand_scaling
  # # print(kps3d)
  # # print(kps3d.shape, global_pose.shape)
  # kps3d = np.einsum(
  #   "nij, nkj -> nki",
  #   global_pose,
  #   np.concatenate([
  #       kps3d,
  #      np.ones((kps3d.shape[0], kps3d.shape[1], 1))
  #   ], axis=-1)
  #   # homogeneous_coord(kps3d)
  # )
  # print(kps3d.shape)
  kps3d = kps3d[:, :,:3]
  kps3d = (kps3d - kps3d[:, 0:1]) * inspire_hand_scaling + kps3d[:, 0:1]

  return {
    "mano_kp_predicted": kps3d,
    "optimized_mano_parameters": optimized_pose.cpu().numpy(),
    "optimized_mano_rot": global_rot.cpu().numpy(),
    "optimized_mano_trans": global_trans.cpu().numpy(),
    "optimized_mano_t_pose": global_transformed_pose
  }

def obtain_mano_parameters_otv_inspire_hand_full_optimization(
    mano_model, finger_tip, ee_pose, is_right
):
  mano_model = mano_right if is_right else mano_left
  mano_model.to(device)
  # print(ee_pose[:, :3, -1].reshape(-1, 1, 3).shape)
  # print(finger_tip.shape)
  # N, 5, 3
  observed_joints = np.concatenate([
    ee_pose[:, :3, -1].reshape(-1, 1, 3),
    finger_tip
  ], axis=1)

  observed_joints = torch.tensor(observed_joints).float().to(device)
  # N, 6, 3

  # Initialize pose parameters (45 axis-angle parameters: 15 joints * 3)
  batch_size = observed_joints.shape[0]
  initial_pose = nn.Parameter(torch.rand(batch_size, 15 + 3 + 3, requires_grad=True, device=device) * 0.1)

  # Set up the optimizer
  optimizer = torch.optim.Adam([initial_pose], lr=0.1)
  # Optimization loop
  # for epoch in tqdm.tqdm(range(500)):  # Adjust the number of iterations as needed
  for epoch in range(500):  # Adjust the number of iterations as needed
    optimizer.zero_grad()
    # Forward pass: Predict joints using the current pose parameters
    predicted_joints = mano_forward(
      mano_model,
      initial_pose[:, :15],
      initial_pose[:, 15:18],
      initial_pose[:, 18:21],
    )
    # Compute the loss (L2 distance between observed and predicted joints)
    # print(predicted_joints.shape)
    # fix_rotattion = estimate_frame_from_hand_points(predicted_joints.detach().cpu().numpy())

    predicted_joints_inspire = predicted_joints[:, mano_to_inspire_mapping]
    
    # print(predicted_joints_inspire.shape, observed_joints.shape)
    loss = torch.nn.functional.smooth_l1_loss(predicted_joints_inspire, observed_joints)
    # loss = torch.nn.functional.mse_loss(predicted_joints_inspire, observed_joints)
    # Backward pass: Compute gradients
    loss.backward()
    optimizer.step()

    # print(loss)
  print(loss)
  optimized_pose = initial_pose.detach()
  # optimized_pose = torch.zeros_like(initial_pose)

  # kps3d = mano_forward_retarget_isaaclab(
  #   # mano_model, 
  #   optimized_pose, is_right)

  kps3d = mano_forward(
    mano_model,
    optimized_pose[:, :15], 
    optimized_pose[:, 15:18],
    optimized_pose[:, 18:21],
    # is_right
  )
  kps3d = kps3d.detach().cpu().numpy()
  # kps3d = kps3d * inspire_hand_scaling
  # print(kps3d)
  # print(kps3d.shape, global_pose.shape)
  # kps3d = np.einsum(
  #   "nij, nkj -> nki",
  #   global_pose,
  #   np.concatenate([
  #       kps3d,
  #      np.ones((kps3d.shape[0], kps3d.shape[1], 1))
  #   ], axis=-1)
  #   # homogeneous_coord(kps3d)
  # )
  # print(kps3d.shape)
  kps3d = kps3d[:, :, :3]
  print("kps3d", kps3d.shape)

  return {
    "mano_kp_predicted": kps3d,
    "optimized_mano_parameters": optimized_pose[:, :15].cpu().numpy(),
    "optimized_mano_rot": optimized_pose[:, 15:18].cpu().numpy(),
    "optimized_mano_trans": optimized_pose[:, 18:].cpu().numpy(),
  }



def obtain_mano_parameters_otv_inspire_hand_mano_rot_single_step(
    finger_tip, ee_pose, is_right
):
  mano_model = mano_right if is_right else mano_left
  mano_model.to(device)
  # N, 5, 3
  observed_joints = np.concatenate([
    ee_pose[:, :3, -1].reshape(-1, 1, 3),
    finger_tip
  ], axis=1)

  # 1, 4, 4
  global_pose = ee_pose.copy()

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB

  # Isaac Lab using WXYZ
  # Make it XYZW
  # global_rot = R.from_quat(np.concatenate([
  #   global_pose[:, 4:], global_pose[:, 3:4]
  # ], axis=-1)).as_matrix()

  global_rot = global_pose[:, :3, :3] @ retarget_axis_transformation.cpu().numpy()
  global_rot = R.from_matrix(global_rot)
  global_rot = global_rot.as_rotvec()
  global_rot = torch.tensor(global_rot).to(device).float()

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS
  global_trans = torch.tensor(global_pose[:, :3, -1] - pelvis.cpu().numpy()).float().to(device)

  observed_joints = torch.tensor(observed_joints).float().to(device)

  observed_joints = (
    observed_joints - observed_joints[:, 0:1]
  ) / inspire_hand_scaling + observed_joints[:, 0:1]

  # N, 6, 3
  # Initialize pose parameters (45 axis-angle parameters: 15 joints * 3)
  batch_size = observed_joints.shape[0]
  initial_pose = nn.Parameter(torch.rand(batch_size, 6, requires_grad=True, device=device) * 0.1)
  # initial_pose = nn.Parameter(torch.rand(batch_size, 15, requires_grad=True, device=device) * 0.1)

  # Set up the optimizer
  optimizer = torch.optim.Adam([initial_pose], lr=0.1)
  # Optimization loop
  for epoch in range(500):  # Adjust the number of iterations as needed
    optimizer.zero_grad()
    # print(initial_pose)
    # print(torch.__version__)
    # print(torch.concat([initial_pose, torch.zeros((batch_size, 9)).to(initial_pose.device).to(initial_pose.dtype)], dim=-1))
    predicted_joints = mano_forward(
      mano_model,
      torch.concat([
        initial_pose,
        torch.zeros((batch_size, 9)).to(initial_pose.device).to(initial_pose.dtype)
      ], dim=-1),
      global_rot,
      global_trans,
    )
    # Compute the loss (L2 distance between observed and predicted joints)

    # print(predicted_joints)
    predicted_joints_inspire = predicted_joints[mano_to_inspire_mapping].unsqueeze(0)
    # print(predicted_joints_inspire)
    loss = torch.nn.functional.smooth_l1_loss(predicted_joints_inspire, observed_joints)
    # Backward pass: Compute gradients
    loss.backward()
    optimizer.step()

  optimized_pose = initial_pose.detach()
  optimized_pose = torch.concat([
    optimized_pose,
    torch.zeros((batch_size, 9)).to(initial_pose.device).to(initial_pose.dtype)
  ], dim=-1)

  kps3d = mano_forward(
      mano_model,
      optimized_pose,
      global_rot,
      global_trans,
  )
  kps3d = kps3d.detach().cpu().numpy()
  # print(kps3d.shape)
  kps3d = kps3d[:,:3]
  kps3d = (kps3d - kps3d[0:1]) * inspire_hand_scaling + kps3d[0:1]

  return {
    "mano_kp_predicted": kps3d,
    "optimized_mano_parameters": optimized_pose.cpu().numpy(),
    "optimized_mano_rot": global_rot.cpu().numpy(),
    "optimized_mano_trans": global_trans.cpu().numpy()
  }


def obtain_mano_pose_otv_inspire_single_step(
    ee_pose, is_right
):
  mano_model = mano_right if is_right else mano_left
  mano_model.to(device)
  # 1, 4, 4
  global_pose = ee_pose.copy()

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB

  # Isaac Lab using WXYZ
  # Make it XYZW
  # global_rot = R.from_quat(np.concatenate([
  #   global_pose[:, 4:], global_pose[:, 3:4]
  # ], axis=-1)).as_matrix()

  global_rot = global_pose[:, :3, :3] @ retarget_axis_transformation.cpu().numpy()
  global_rot = R.from_matrix(global_rot)
  global_rot = global_rot.as_rotvec()
  global_rot = torch.tensor(global_rot).to(device).float()

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS
  global_trans = torch.tensor(global_pose[:, :3, -1] - pelvis.cpu().numpy()).float().to(device)

  return {
    "optimized_mano_rot": global_rot.cpu().numpy(),
    "optimized_mano_trans": global_trans.cpu().numpy()
  }
