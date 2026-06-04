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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>

__device__ float bilinear_sampling(const float *&bottom_data, const int &height,
                                   const int &width,
                                   const int &num_embeds_with_depth,
                                   const int &num_depths, const float &h_im,
                                   const float &w_im, const float &loc_d,
                                   const int &base_ptr, const int &depth_ptr) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const int d_low = floorf(loc_d);
  const int d_high = d_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const float ld = loc_d - d_low;
  const float hd = 1 - ld;

  const int w_stride = num_embeds_with_depth;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  float dp_low1 = 0;
  float dp_high1 = 0;
  const bool flag_d_low = d_low >= 0 && d_low < num_depths;
  const bool flag_d_high = d_high >= 0 && d_high < num_depths;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset;
    v1 = bottom_data[ptr1 + base_ptr];
    const int ptr_d1 = ptr1 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low1 = bottom_data[ptr_d1];
    }
    if (flag_d_high) {
      dp_high1 = bottom_data[ptr_d1 + 1];
    }
  }

  float v2 = 0;
  float dp_low2 = 0;
  float dp_high2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset;
    v2 = bottom_data[ptr2 + base_ptr];
    const int ptr_d2 = ptr2 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low2 = bottom_data[ptr_d2];
    }
    if (flag_d_high) {
      dp_high2 = bottom_data[ptr_d2 + 1];
    }
  }

  float v3 = 0;
  float dp_low3 = 0;
  float dp_high3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset;
    v3 = bottom_data[ptr3 + base_ptr];
    const int ptr_d3 = ptr3 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low3 = bottom_data[ptr_d3];
    }
    if (flag_d_high) {
      dp_high3 = bottom_data[ptr_d3 + 1];
    }
  }

  float v4 = 0;
  float dp_low4 = 0;
  float dp_high4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset;
    v4 = bottom_data[ptr4 + base_ptr];
    const int ptr_d4 = ptr4 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low4 = bottom_data[ptr_d4];
    }
    if (flag_d_high) {
      dp_high4 = bottom_data[ptr_d4 + 1];
    }
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float val = hd * (w1 * v1 * dp_low1 + w2 * v2 * dp_low2 +
                          w3 * v3 * dp_low3 + w4 * v4 * dp_low4) +
                    ld * (w1 * v1 * dp_high1 + w2 * v2 * dp_high2 +
                          w3 * v3 * dp_high3 + w4 * v4 * dp_high4);
  return val;
}

__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight, const int &height,
    const int &width, const int &num_embeds_with_depth, const int &num_depths,
    const float &h_im, const float &w_im, const float &loc_d,
    const int &base_ptr, const int &depth_ptr, const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location,
    float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const int d_low = floorf(loc_d);
  const int d_high = d_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;
  const float ld = loc_d - d_low;
  const float hd = 1 - ld;

  const int w_stride = num_embeds_with_depth;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;

  const bool flag_d_low = d_low >= 0 && d_low < num_depths;
  const bool flag_d_high = d_high >= 0 && d_high < num_depths;

  float v1 = 0;
  float dp_low1 = 0;
  float dp_high1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset;
    v1 = bottom_data[ptr1 + base_ptr];
    const int ptr_d1 = ptr1 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low1 = bottom_data[ptr_d1];
      atomicAdd(grad_mc_ms_feat + ptr_d1, w1 * top_grad_mc_ms_feat * v1 * hd);
    }
    if (flag_d_high) {
      dp_high1 = bottom_data[ptr_d1 + 1];
      atomicAdd(grad_mc_ms_feat + ptr_d1 + 1,
                w1 * top_grad_mc_ms_feat * v1 * ld);
    }
    atomicAdd(grad_mc_ms_feat + ptr1 + base_ptr,
              w1 * top_grad_mc_ms_feat * (dp_low1 * hd + dp_high1 * ld));
  }
  float v2 = 0;
  float dp_low2 = 0;
  float dp_high2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset;
    v2 = bottom_data[ptr2 + base_ptr];
    const int ptr_d2 = ptr2 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low2 = bottom_data[ptr_d2];
      atomicAdd(grad_mc_ms_feat + ptr_d2, w2 * top_grad_mc_ms_feat * v2 * hd);
    }
    if (flag_d_high) {
      dp_high2 = bottom_data[ptr_d2 + 1];
      atomicAdd(grad_mc_ms_feat + ptr_d2 + 1,
                w2 * top_grad_mc_ms_feat * v2 * ld);
    }

    atomicAdd(grad_mc_ms_feat + ptr2 + base_ptr,
              w2 * top_grad_mc_ms_feat * (dp_low2 * hd + dp_high2 * ld));
  }
  float v3 = 0;
  float dp_low3 = 0;
  float dp_high3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset;
    v3 = bottom_data[ptr3 + base_ptr];
    const int ptr_d3 = ptr3 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low3 = bottom_data[ptr_d3];
      atomicAdd(grad_mc_ms_feat + ptr_d3, w3 * top_grad_mc_ms_feat * v3 * hd);
    }
    if (flag_d_high) {
      dp_high3 = bottom_data[ptr_d3 + 1];
      atomicAdd(grad_mc_ms_feat + ptr_d3 + 1,
                w3 * top_grad_mc_ms_feat * v3 * ld);
    }

    atomicAdd(grad_mc_ms_feat + ptr3 + base_ptr,
              w3 * top_grad_mc_ms_feat * (dp_low3 * hd + dp_high3 * ld));
  }
  float v4 = 0;
  float dp_low4 = 0;
  float dp_high4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset;
    v4 = bottom_data[ptr4 + base_ptr];
    const int ptr_d4 = ptr4 + depth_ptr + d_low;
    if (flag_d_low) {
      dp_low4 = bottom_data[ptr_d4];
      atomicAdd(grad_mc_ms_feat + ptr_d4, w4 * top_grad_mc_ms_feat * v4 * hd);
    }
    if (flag_d_high) {
      dp_high4 = bottom_data[ptr_d4 + 1];
      atomicAdd(grad_mc_ms_feat + ptr_d4 + 1,
                w4 * top_grad_mc_ms_feat * v4 * ld);
    }
    atomicAdd(grad_mc_ms_feat + ptr4 + base_ptr,
              w4 * top_grad_mc_ms_feat * (dp_low4 * hd + dp_high4 * ld));
  }

  const float val1 = w1 * v1 * dp_low1 + w2 * v2 * dp_low2 + w3 * v3 * dp_low3 +
                     w4 * v4 * dp_low4;
  const float val2 = w1 * v1 * dp_high1 + w2 * v2 * dp_high2 +
                     w3 * v3 * dp_high3 + w4 * v4 * dp_high4;

  const float val = hd * val1 + ld * val2;

  atomicAdd(grad_weights, grad_output * val);

  const float grad_w_weight = hd * (-hh * v1 * dp_low1 + hh * v2 * dp_low2 -
                                    lh * v3 * dp_low3 + lh * v4 * dp_low4) +
                              ld * (-hh * v1 * dp_high1 + hh * v2 * dp_high2 -
                                    lh * v3 * dp_high3 + lh * v4 * dp_high4);
  const float grad_h_weight = hd * (-hw * v1 * dp_low1 - lw * v2 * dp_low2 +
                                    hw * v3 * dp_low3 + lw * v4 * dp_low4) +
                              ld * (-hw * v1 * dp_high1 - lw * v2 * dp_high2 +
                                    hw * v3 * dp_high3 + lw * v4 * dp_high4);

  /* const float grad_d_weight = -1 * (w1 * v1 * dp_low1 + w2 * v2 * dp_low2 +
   * w3 * v3 * dp_low3 + w4 * v4 * dp_low4) */
  /*     + (w1 * v1 * dp_high1 + w2 * v2 * dp_high2 + w3 * v3 * dp_high3 + w4 *
   * v4 * dp_high4); */
  const float grad_d_weight = val2 - val1;
  /* const float grad_d_weight = w1 * v1 * (dp_high1 - dp_low1) */
  /*     + w2 * v2 * (dp_high2 - dp_low2) */
  /*     + w3 * v3 * (dp_high3 - dp_low3) */
  /*     + w4 * v4 * (dp_high4 - dp_low4); */

  atomicAdd(grad_sampling_location,
            width * top_grad_mc_ms_feat * grad_w_weight);
  atomicAdd(grad_sampling_location + 1,
            height * top_grad_mc_ms_feat * grad_h_weight);
  atomicAdd(grad_sampling_location + 2, top_grad_mc_ms_feat * grad_d_weight);
}

__global__ void deformable_aggregation_with_depth_kernel(
    const int num_kernels, float *output, const float *mc_ms_feat,
    const int *spatial_shape, const int *scale_start_index,
    const float *sample_location, const float *weights, int batch_size,
    int num_cams, int num_feat, int num_embeds, int num_scale, int num_pts,
    int num_groups, int num_depths) {
  long int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kernels) return;

  float *output_ptr = output + idx;
  const int channel_index = idx % num_embeds;
  const int groups_index = channel_index / (num_embeds / num_groups);
  idx /= num_embeds;
  const int pts_index = idx % num_pts;
  idx /= num_pts;
  const int batch_index = idx;

  const int num_embeds_with_depth = num_embeds + num_depths;
  const int value_cam_stride = num_feat * num_embeds_with_depth;
  const int weight_cam_stride = num_scale * num_groups;
  int loc_offset = (batch_index * num_pts + pts_index) * num_cams * 3;
  int value_offset = batch_index * num_cams * value_cam_stride;
  int depth_offset = value_offset + num_embeds;
  value_offset = value_offset + channel_index;
  int weight_offset =
      ((batch_index * num_pts + pts_index) * num_cams * weight_cam_stride +
       groups_index);

  float result = 0;
  for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 3) {
    const float loc_w = sample_location[loc_offset];
    const float loc_h = sample_location[loc_offset + 1];
    const float loc_d = sample_location[loc_offset + 2];

    if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1 && loc_d > -1 &&
        loc_d < num_depths) {
      for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
        const int scale_offset =
            scale_start_index[scale_index] * num_embeds_with_depth;

        const int spatial_shape_ptr = scale_index << 1;
        const int h = spatial_shape[spatial_shape_ptr];
        const int w = spatial_shape[spatial_shape_ptr + 1];

        const float h_im = loc_h * h - 0.5;
        const float w_im = loc_w * w - 0.5;

        const int value_ptr =
            value_offset + scale_offset + value_cam_stride * cam_index;
        const int depth_ptr =
            depth_offset + scale_offset + value_cam_stride * cam_index;

        const float *weights_ptr =
            (weights + weight_offset + scale_index * num_groups +
             weight_cam_stride * cam_index);
        result += bilinear_sampling(mc_ms_feat, h, w, num_embeds_with_depth,
                                    num_depths, h_im, w_im, loc_d, value_ptr,
                                    depth_ptr) *
                  *weights_ptr;
      }
    }
  }
  *output_ptr = result;
}

__global__ void deformable_aggregation_with_depth_grad_kernel(
    const int num_kernels, const float *mc_ms_feat, const int *spatial_shape,
    const int *scale_start_index, const float *sample_location,
    const float *weights, const float *grad_output, float *grad_mc_ms_feat,
    float *grad_sampling_location, float *grad_weights, int batch_size,
    int num_cams, int num_feat, int num_embeds, int num_scale, int num_pts,
    int num_groups, int num_depths) {
  long int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kernels) return;
  const float grad = grad_output[idx];

  const int channel_index = idx % num_embeds;
  const int groups_index = channel_index / (num_embeds / num_groups);
  idx /= num_embeds;
  const int pts_index = idx % num_pts;
  idx /= num_pts;
  const int batch_index = idx;

  const int num_embeds_with_depth = num_embeds + num_depths;
  const int value_cam_stride = num_feat * num_embeds_with_depth;
  const int weight_cam_stride = num_scale * num_groups;
  int loc_offset = (batch_index * num_pts + pts_index) * num_cams * 3;
  int value_offset = batch_index * num_cams * value_cam_stride;
  int depth_offset = value_offset + num_embeds;
  value_offset = value_offset + channel_index;
  int weight_offset =
      ((batch_index * num_pts + pts_index) * num_cams * weight_cam_stride +
       groups_index);

  for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 3) {
    const float loc_w = sample_location[loc_offset];
    const float loc_h = sample_location[loc_offset + 1];
    const float loc_d = sample_location[loc_offset + 2];

    if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1 && loc_d > -1 &&
        loc_d < num_depths) {
      for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
        const int scale_offset =
            scale_start_index[scale_index] * num_embeds_with_depth;

        const int spatial_shape_ptr = scale_index << 1;
        const int h = spatial_shape[spatial_shape_ptr];
        const int w = spatial_shape[spatial_shape_ptr + 1];

        const float h_im = loc_h * h - 0.5;
        const float w_im = loc_w * w - 0.5;

        const int value_ptr =
            value_offset + scale_offset + value_cam_stride * cam_index;
        const int depth_ptr =
            depth_offset + scale_offset + value_cam_stride * cam_index;
        const int weights_ptr = weight_offset + scale_index * num_groups +
                                weight_cam_stride * cam_index;
        const float weight = weights[weights_ptr];

        float *grad_location_ptr = grad_sampling_location + loc_offset;
        float *grad_weights_ptr = grad_weights + weights_ptr;
        bilinear_sampling_grad(mc_ms_feat, weight, h, w, num_embeds_with_depth,
                               num_depths, h_im, w_im, loc_d, value_ptr,
                               depth_ptr, grad, grad_mc_ms_feat,
                               grad_location_ptr, grad_weights_ptr);
      }
    }
  }
}

void deformable_aggregation_with_depth(
    float *output, const float *mc_ms_feat, const int *spatial_shape,
    const int *scale_start_index, const float *sample_location,
    const float *weights, int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups,
    int num_depths) {
  const long int num_kernels = batch_size * num_pts * num_embeds;
  deformable_aggregation_with_depth_kernel<<<
      (int)ceil(((double)num_kernels / 512)), 512>>>(
      num_kernels, output, mc_ms_feat, spatial_shape, scale_start_index,
      sample_location, weights, batch_size, num_cams, num_feat, num_embeds,
      num_scale, num_pts, num_groups, num_depths);
}

void deformable_aggregation_with_depth_grad(
    const float *mc_ms_feat, const int *spatial_shape,
    const int *scale_start_index, const float *sample_location,
    const float *weights, const float *grad_output, float *grad_mc_ms_feat,
    float *grad_sampling_location, float *grad_weights, int batch_size,
    int num_cams, int num_feat, int num_embeds, int num_scale, int num_pts,
    int num_groups, int num_depths) {
  const long int num_kernels = batch_size * num_pts * num_embeds;
  deformable_aggregation_with_depth_grad_kernel<<<
      (int)ceil(((double)num_kernels / 512)), 512>>>(
      num_kernels, mc_ms_feat, spatial_shape, scale_start_index,
      sample_location, weights, grad_output, grad_mc_ms_feat,
      grad_sampling_location, grad_weights, batch_size, num_cams, num_feat,
      num_embeds, num_scale, num_pts, num_groups, num_depths);
}
