// Copyright 2022 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <config.hpp>
#include <preprocess_kernel.hpp>
#include <utils.hpp>

namespace
{
const std::size_t WARPS_PER_BLOCK = 4;
const std::size_t FEATURE_SIZE = 10;  // same as `encoder_in_features` in config.hpp
const std::size_t THREADS_FOR_VOXEL = 256;    // threads number for a block
const std::size_t POINTS_PER_VOXEL = 32;      // depands on "params.h"
}  // namespace

namespace centerpoint
{

__global__ void generateVoxels_random_kernel(float 
*points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        int *mask, float *voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(point_idx >= points_size) return;

  float4 point = ((float4*)points)[point_idx];

  if(point.x<min_x_range||point.x>=max_x_range
    || point.y<min_y_range||point.y>=max_y_range
    || point.z<min_z_range||point.z>=max_z_range) return;

  int voxel_idx = floorf((point.x - min_x_range)/pillar_x_size);
  int voxel_idy = floorf((point.y - min_y_range)/pillar_y_size);
  int voxel_index = voxel_idy * grid_x_size
                            + voxel_idx;

  int point_id = atomicAdd(&(mask[voxel_index]), 1);

  if(point_id >= POINTS_PER_VOXEL) return;
  float *address = voxels + (voxel_index*POINTS_PER_VOXEL + point_id)*4;
  atomicExch(address+0, point.x);
  atomicExch(address+1, point.y);
  atomicExch(address+2, point.z);
  atomicExch(address+3, point.w);
}

cudaError_t generateVoxels_random_launch(float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        int *mask, float *voxels,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((points_size+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generateVoxels_random_kernel<<<blocks, threads, 0, stream>>>
    (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        pillar_x_size, pillar_y_size, pillar_z_size,
        grid_y_size, grid_x_size,
        mask, voxels);
  cudaError_t err = cudaGetLastError();
  return err;
}
__global__ void generateBaseFeatures_kernel(int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        int *pillar_num,
        float *voxel_features,
        float *voxel_num_points,
        int *voxel_idxs)
{
  int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int voxel_idy = blockIdx.y * blockDim.y + threadIdx.y;

  if(voxel_idx >= grid_x_size ||voxel_idy >= grid_y_size) return;

  int voxel_index = voxel_idy * grid_x_size
                           + voxel_idx;
  int count = mask[voxel_index];
  if( !(count>0) ) return;
  count = count<POINTS_PER_VOXEL?count:POINTS_PER_VOXEL;

  int current_pillarId = 0;
  current_pillarId = atomicAdd(pillar_num, 1);

  voxel_num_points[current_pillarId] = count;

  int3 idx = {0, voxel_idy, voxel_idx};
  ((int3*)voxel_idxs)[current_pillarId] = idx;

  for (int i=0; i<count; i++){
    int inIndex = voxel_index*POINTS_PER_VOXEL + i;
    int outIndex = current_pillarId*POINTS_PER_VOXEL + i;
    ((float4*)voxel_features)[outIndex] = ((float4*)voxels)[inIndex];
  }

//   // clear buffer for next infer
  atomicExch(mask + voxel_index, 0);
}
cudaError_t generateBaseFeatures_launch(int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        int *pillar_num,
        float *voxel_features,
        float*voxel_num_points,
        int *voxel_idxs,
        cudaStream_t stream)
{
  dim3 threads = {32,32};
  dim3 blocks = {(grid_x_size + threads.x -1)/threads.x,
                 (grid_y_size + threads.y -1)/threads.y};

  generateBaseFeatures_kernel<<<blocks, threads, 0, stream>>>
      (mask, voxels, grid_y_size, grid_x_size,
       pillar_num,
       voxel_features,
       voxel_num_points,
       voxel_idxs);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void generateFeatures_kernel(
  const float * voxel_features, const float * voxel_num_points, const int * coords,
  const int num_voxels, const float voxel_x, const float voxel_y, const float voxel_z,
  const float range_min_x, const float range_min_y, const float range_min_z, float * features)
{
  // voxel_features (float): (max_num_voxels, max_num_points_per_voxel, point_feature_size)
  // voxel_num_points (int): (max_num_voxels)
  // coords (int): (max_num_voxels, point_dim_size)
  int pillar_idx = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / POINTS_PER_VOXEL;
  int point_idx = threadIdx.x % POINTS_PER_VOXEL;
  int pillar_idx_inBlock = threadIdx.x / POINTS_PER_VOXEL;

  if (pillar_idx >= num_voxels) return;

  // load src
  __shared__ float4 pillarSM[WARPS_PER_BLOCK][POINTS_PER_VOXEL];
  __shared__ float3 pillarSumSM[WARPS_PER_BLOCK];
  __shared__ int3 cordsSM[WARPS_PER_BLOCK];
  __shared__ int pointsNumSM[WARPS_PER_BLOCK];
  __shared__ float pillarOutSM[WARPS_PER_BLOCK][POINTS_PER_VOXEL][FEATURE_SIZE];

  if (threadIdx.x < WARPS_PER_BLOCK) {
    pointsNumSM[threadIdx.x] = voxel_num_points[blockIdx.x * WARPS_PER_BLOCK + threadIdx.x];
    cordsSM[threadIdx.x] = ((int3 *)coords)[blockIdx.x * WARPS_PER_BLOCK + threadIdx.x];
    pillarSumSM[threadIdx.x] = {0, 0, 0};
  }

  pillarSM[pillar_idx_inBlock][point_idx] =
    ((float4 *)voxel_features)[pillar_idx * POINTS_PER_VOXEL + point_idx]; 
  __syncthreads();

  // calculate sm in a pillar
  if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
    atomicAdd(&(pillarSumSM[pillar_idx_inBlock].x), pillarSM[pillar_idx_inBlock][point_idx].x);
    atomicAdd(&(pillarSumSM[pillar_idx_inBlock].y), pillarSM[pillar_idx_inBlock][point_idx].y);
    atomicAdd(&(pillarSumSM[pillar_idx_inBlock].z), pillarSM[pillar_idx_inBlock][point_idx].z);
  }
  __syncthreads();

  // feature-mean
  float3 mean;
  float validPoints = pointsNumSM[pillar_idx_inBlock];
  mean.x = pillarSumSM[pillar_idx_inBlock].x / validPoints;
  mean.y = pillarSumSM[pillar_idx_inBlock].y / validPoints;
  mean.z = pillarSumSM[pillar_idx_inBlock].z / validPoints;

  mean.x = pillarSM[pillar_idx_inBlock][point_idx].x - mean.x;
  mean.y = pillarSM[pillar_idx_inBlock][point_idx].y - mean.y;
  mean.z = pillarSM[pillar_idx_inBlock][point_idx].z - mean.z;

  // calculate offset
  float x_offset = voxel_x / 2 + cordsSM[pillar_idx_inBlock].z * voxel_x + range_min_x;
  float y_offset = voxel_y / 2 + cordsSM[pillar_idx_inBlock].y * voxel_y + range_min_y;
  float z_offset = voxel_z / 2 + cordsSM[pillar_idx_inBlock].x * voxel_z + range_min_z;

  // feature-offset
  float3 center;
  center.x = pillarSM[pillar_idx_inBlock][point_idx].x - x_offset;
  center.y = pillarSM[pillar_idx_inBlock][point_idx].y - y_offset;
  center.z = pillarSM[pillar_idx_inBlock][point_idx].z - z_offset;

  // store output
  if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
    pillarOutSM[pillar_idx_inBlock][point_idx][0] = pillarSM[pillar_idx_inBlock][point_idx].x;
    pillarOutSM[pillar_idx_inBlock][point_idx][1] = pillarSM[pillar_idx_inBlock][point_idx].y;
    pillarOutSM[pillar_idx_inBlock][point_idx][2] = pillarSM[pillar_idx_inBlock][point_idx].z;
    pillarOutSM[pillar_idx_inBlock][point_idx][3] = pillarSM[pillar_idx_inBlock][point_idx].w;
    
    pillarOutSM[pillar_idx_inBlock][point_idx][4] = mean.x;
    pillarOutSM[pillar_idx_inBlock][point_idx][5] = mean.y;
    pillarOutSM[pillar_idx_inBlock][point_idx][6] = mean.z;

    pillarOutSM[pillar_idx_inBlock][point_idx][7] = center.x;
    pillarOutSM[pillar_idx_inBlock][point_idx][8] = center.y;
    pillarOutSM[pillar_idx_inBlock][point_idx][9] = center.z;//modify

  } else {
    pillarOutSM[pillar_idx_inBlock][point_idx][0] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][1] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][2] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][3] = 0;

    pillarOutSM[pillar_idx_inBlock][point_idx][4] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][5] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][6] = 0;

    pillarOutSM[pillar_idx_inBlock][point_idx][7] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][8] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][9] = 0;//modify
  }

  __syncthreads();

  for (int i = 0; i < FEATURE_SIZE; i++) {
    int outputSMId = pillar_idx_inBlock * POINTS_PER_VOXEL * FEATURE_SIZE + i * POINTS_PER_VOXEL + point_idx;
    int outputId = pillar_idx * POINTS_PER_VOXEL * FEATURE_SIZE + i * POINTS_PER_VOXEL + point_idx;
    features[outputId] = ((float *)pillarOutSM)[outputSMId];
  }
}

cudaError_t generateFeatures_launch(
  const float * voxel_features, const float * voxel_num_points, const int * coords,
  const int num_voxels, float * features, cudaStream_t stream)
{
  dim3 blocks(divup(Config::max_num_voxels, WARPS_PER_BLOCK));
  dim3 threads(WARPS_PER_BLOCK * POINTS_PER_VOXEL);
  generateFeatures_kernel<<<blocks, threads, 0, stream>>>(
    voxel_features, voxel_num_points, coords, num_voxels, Config::voxel_size_x,
    Config::voxel_size_y, Config::voxel_size_z, Config::range_min_x, Config::range_min_y,
    Config::range_min_z, features);

  return cudaGetLastError();
}



}  // namespace centerpoint
