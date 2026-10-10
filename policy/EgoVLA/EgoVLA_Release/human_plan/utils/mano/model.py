import torch
import numpy as np
import smplx
from smplx.lbs import blend_shapes, vertices2joints

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