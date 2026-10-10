import numpy as np
import os
import torch


def get_quantile(quantile_path):
  return np.load(quantile_path
  ).reshape(1, -1)


# NUM_CLASS = np.prod(CLASS_QUANTILE.shape)


def get_class_label(
  data: torch.Tensor, 
  thresholds: torch.Tensor
):
  # Use broadcasting and vectorized operations to quantize the data
  quantized_data = torch.sum(
      data.unsqueeze(1) >= thresholds,
      dim=1
  )
  return quantized_data
