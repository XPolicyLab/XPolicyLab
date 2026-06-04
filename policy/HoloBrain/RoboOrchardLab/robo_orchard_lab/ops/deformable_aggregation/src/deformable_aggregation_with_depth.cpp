/*
 * Project RoboOrchard
 *
 * Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

void deformable_aggregation_with_depth(
    float* output, const float* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const float* sample_location,
    const float* weights, int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups, int num_depths);

void deformable_aggregation_with_depth_grad(
    const float* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const float* sample_location,
    const float* weights, const float* grad_output, float* grad_mc_ms_feat,
    float* grad_sampling_location, float* grad_weights, int batch_size,
    int num_cams, int num_feat, int num_embeds, int num_scale, int num_pts,
    int num_groups, int num_depths);

/* _mc_ms_feat: b, cam, feat, C+D */
/* _spatial_shape: scale, 2 */
/* _scale_start_index: scale */
/* _sampling_location: bs, pts, */

at::Tensor deformable_aggregation_with_depth_forward(
    const at::Tensor& _mc_ms_feat, const at::Tensor& _spatial_shape,
    const at::Tensor& _scale_start_index, const at::Tensor& _sampling_location,
    const at::Tensor& _weights, const int num_depths) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_cams = _mc_ms_feat.size(1);
  int num_feat = _mc_ms_feat.size(2);
  int num_embeds = _mc_ms_feat.size(3) - num_depths;
  int num_scale = _spatial_shape.size(0);
  int num_pts = _sampling_location.size(1);
  int num_groups = _weights.size(4);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  auto output =
      at::zeros({batch_size, num_pts, num_embeds}, _mc_ms_feat.options());
  deformable_aggregation_with_depth(
      output.data_ptr<float>(), mc_ms_feat, spatial_shape, scale_start_index,
      sampling_location, weights, batch_size, num_cams, num_feat, num_embeds,
      num_scale, num_pts, num_groups, num_depths);
  return output;
}

void deformable_aggregation_with_depth_backward(
    const at::Tensor& _mc_ms_feat, const at::Tensor& _spatial_shape,
    const at::Tensor& _scale_start_index, const at::Tensor& _sampling_location,
    const at::Tensor& _weights, const int num_depths,
    const at::Tensor& _grad_output, at::Tensor& _grad_mc_ms_feat,
    at::Tensor& _grad_sampling_location, at::Tensor& _grad_weights) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_cams = _mc_ms_feat.size(1);
  int num_feat = _mc_ms_feat.size(2);
  int num_embeds = _mc_ms_feat.size(3) - num_depths;
  int num_scale = _spatial_shape.size(0);
  int num_pts = _sampling_location.size(1);
  int num_groups = _weights.size(4);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();
  const float* grad_output = _grad_output.data_ptr<float>();

  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights = _grad_weights.data_ptr<float>();

  deformable_aggregation_with_depth_grad(
      mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
      grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
      batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts,
      num_groups, num_depths);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deformable_aggregation_with_depth_forward",
        &deformable_aggregation_with_depth_forward,
        "deformable_aggregation_with_depth_forward");
  m.def("deformable_aggregation_with_depth_backward",
        &deformable_aggregation_with_depth_backward,
        "deformable_aggregation_with_depth_backward");
}
