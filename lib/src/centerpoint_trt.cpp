// Copyright 2021 TIER IV, Inc.
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

#include <centerpoint_trt.hpp>
#include <preprocess_kernel.hpp>
#include <scatter_kernel.hpp>
#include <tier4_autoware_utils/math/constants.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace centerpoint
{
//计算每层的耗时
struct SimpleProfiler:public nvinfer1::IProfiler
{
    struct Record{
        float time{0};
        int count{0};
    };

    virtual void reportLayerTime(const char* layerName, float ms) noexcept
    {
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
        if(std::find(mLayerNames.begin(),mLayerNames.end(),layerName) == mLayerNames.end()){
            mLayerNames.push_back(layerName);
        }
    }

    SimpleProfiler(const char* name):mName(name){}

    friend std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value)
    {
        out << "========== " << value.mName << " profile ==========" << std::endl;
        float totalTime = 0;
        std::string layerNameStr = "TensorRT layer name";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
        for (const auto& elem : value.mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
        }

        auto old_settings = out.flags();
        auto old_precision = out.precision();
        // Output header
        {
            out << std::setw(maxLayerNameLength) << layerNameStr << " ";
            out << std::setw(12) << "Runtime, "
                << "%"
                << " ";
            out << std::setw(12) << "Invocations"
                << " ";
            out << std::setw(12) << "Runtime, ms" << std::endl;
        }
        for (size_t i = 0; i < value.mLayerNames.size(); i++)
        {
            const std::string layerName = value.mLayerNames[i];
            auto elem = value.mProfile.at(layerName);
            out << std::setw(maxLayerNameLength) << layerName << " ";
            out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.time * 100.0F / totalTime) << "%"
                << " ";
            out << std::setw(12) << elem.count << " ";
            out << std::setw(12) << std::fixed << std::setprecision(2) << elem.time << std::endl;
        }
        out.flags(old_settings);
        out.precision(old_precision);
        out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========" << std::endl;

        return out;
    }

    private:
    std::string mName;
    std::vector<std::string> mLayerNames;
    std::map<std::string, Record> mProfile;
};

CenterPointTRT::CenterPointTRT(
  const std::size_t num_class, const float score_threshold, const NetworkParam & encoder_param,
  const NetworkParam & head_param, const DensificationParam & densification_param)
: num_class_(num_class)
{
  vg_ptr_ = std::make_unique<VoxelGenerator>(densification_param);
  post_proc_ptr_ = std::make_unique<PostProcessCUDA>(num_class, score_threshold);
  // encoder
  encoder_trt_ptr_ = std::make_unique<VoxelEncoderTRT>(verbose_);
  encoder_trt_ptr_->init(
    encoder_param.onnx_path(), encoder_param.engine_path(), encoder_param.trt_precision());
  encoder_trt_ptr_->context_->setBindingDimensions(
    0,
    nvinfer1::Dims3(
      Config::max_num_voxels, Config::max_num_points_per_voxel, Config::encoder_in_feature_size));
  // head
  head_trt_ptr_ = std::make_unique<HeadTRT>(num_class, verbose_);
  head_trt_ptr_->init(head_param.onnx_path(), head_param.engine_path(), head_param.trt_precision());
  head_trt_ptr_->context_->setBindingDimensions(
    0, nvinfer1::Dims4(
         Config::batch_size, Config::encoder_out_feature_size, Config::grid_size_y,
         Config::grid_size_x));
  initPtr();
  do_profile = false;
  cudaStreamCreate(&stream_);
}

CenterPointTRT::~CenterPointTRT()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

void CenterPointTRT::initPtr()
{
   voxels_feature_size_ =
    Config::max_num_voxels * Config::max_num_points_per_voxel * Config::point_feature_size;
   coordinates_size_ = Config::max_num_voxels * Config::point_dim_size;
  encoder_in_feature_size_ =
    Config::max_num_voxels * Config::max_num_points_per_voxel * Config::encoder_in_feature_size;
  pillar_features_size_ = Config::max_num_voxels * Config::encoder_out_feature_size;
  spatial_features_size_ =
    Config::grid_size_x * Config::grid_size_y * Config::encoder_out_feature_size;
  grid_xy_ = Config::down_grid_size_x * Config::down_grid_size_y;
  mask_size_ = Config::grid_size_z * Config::grid_size_y * Config::grid_size_x;
   voxels_size_ = Config::grid_size_z * Config::grid_size_y * Config::grid_size_x * 
    Config::point_feature_size * Config::max_num_points_per_voxel;
  // device
  pillar_num_d_ = cuda::make_unique<int>();
  voxels_d_ = cuda::make_unique<float[]>(voxels_size_);
  voxels_feature_d_ = cuda::make_unique<float[]>(voxels_feature_size_);
  coordinates_d_ = cuda::make_unique<int[]>(coordinates_size_);
  num_points_per_voxel_d_ = cuda::make_unique<float[]>(Config::max_num_points_per_voxel);
  mask_d_ = cuda::make_unique<int[]>(mask_size_);
  encoder_in_features_d_ = cuda::make_unique<float[]>(encoder_in_feature_size_);
  pillar_features_d_ = cuda::make_unique<float[]>(pillar_features_size_);
  spatial_features_d_ = cuda::make_unique<float[]>(spatial_features_size_);
  head_out_heatmap_d_ = cuda::make_unique<float[]>(grid_xy_ * num_class_);
  head_out_offset_d_ = cuda::make_unique<float[]>(grid_xy_ * Config::head_out_offset_size);
  head_out_z_d_ = cuda::make_unique<float[]>(grid_xy_ * Config::head_out_z_size);
  head_out_dim_d_ = cuda::make_unique<float[]>(grid_xy_ * Config::head_out_dim_size);
  head_out_rot_d_ = cuda::make_unique<float[]>(grid_xy_ * Config::head_out_rot_size);
  
}

bool CenterPointTRT::detect(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & input, const tf2_ros::Buffer & tf_buffer,
  std::vector<Box3D> & det_boxes3d)
{
//   uint64_t begin_1 = GetCurTime();
//   std::fill(voxels_.begin(), voxels_.end(), 0);
//   std::fill(coordinates_.begin(), coordinates_.end(), -1);
//   std::fill(num_points_per_voxel_.begin(), num_points_per_voxel_.end(), 0);
//    uint64_t end_1 = GetCurTime();
//   std::cout << "========== " <<"The cudaMemset module runtime is  " << (end_1 - begin_1)/1000 <<"  ms ==========" << std::endl;
  CHECK_CUDA_ERROR(cudaMemsetAsync(pillar_num_d_.get(), 0, sizeof(int), stream_));
  CHECK_CUDA_ERROR(cudaMemsetAsync(voxels_feature_d_.get(), 0, voxels_feature_size_* sizeof(float), stream_));
  CHECK_CUDA_ERROR(cudaMemsetAsync(voxels_d_.get(), 0, voxels_size_ * sizeof(float), stream_)); 
  CHECK_CUDA_ERROR(cudaMemsetAsync(coordinates_d_.get(), 0, coordinates_size_ * sizeof(int), stream_)); 
  CHECK_CUDA_ERROR(cudaMemsetAsync(num_points_per_voxel_d_.get(), 0,Config::max_num_points_per_voxel* sizeof(int), stream_));
  CHECK_CUDA_ERROR(cudaMemsetAsync(mask_d_.get(), 0,  mask_size_* sizeof(int), stream_));
  CHECK_CUDA_ERROR(cudaMemsetAsync(encoder_in_features_d_.get(), 0, encoder_in_feature_size_ * sizeof(float), stream_));
   CHECK_CUDA_ERROR(cudaMemsetAsync(spatial_features_d_.get(), 0, spatial_features_size_ * sizeof(float), stream_));
  //计算前处理用时
  uint64_t begin_2 = GetCurTime();
  if (!preprocess(input, tf_buffer)) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_centerpoint"), "Fail to preprocess and skip to detect.");
    return false;
  }
  uint64_t end_2 = GetCurTime();
  std::cout << "========== " <<"The preprocess module runtime is  " << (end_2 - begin_2)/1000 <<"  ms ==========" << std::endl;

  inference();
  //计算后处理用时
  uint64_t begin_3 = GetCurTime();
  postProcess(det_boxes3d);
  uint64_t end_3 = GetCurTime();
  std::cout << "========== " <<"The postprocess module runtime is  " << (end_3 - begin_3)/1000 <<"  ms ==========" << std::endl;
  return true;
}

bool CenterPointTRT::preprocess(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & input, const tf2_ros::Buffer & tf_buffer)
{
  //TODO cuda
  std::size_t points_size = input->points.size();
  int points_data_size = points_size * Config::point_feature_size * sizeof(float);
  std::vector<float>pc_array;
  pc_array.resize(points_size * Config::point_feature_size);
  for(std::size_t i = 0 ; i < points_size; i++){
      pc_array[i * Config::point_feature_size + 0] = input->points[i].x;
      pc_array[i * Config::point_feature_size + 1] = input->points[i].y;
      pc_array[i * Config::point_feature_size + 2] = input->points[i].z;
      pc_array[i * Config::point_feature_size + 3] = 1.0;
  }
  cuda::unique_ptr<float[]> points_data_d{nullptr};
  points_data_d = cuda::make_unique<float[]>(points_data_size);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    points_data_d.get(), pc_array.data(), points_data_size,cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(generateVoxels_random_launch(points_data_d.get(), points_size,
        Config::range_min_x, Config::range_max_x,
        Config::range_min_y, Config::range_max_y,
        Config::range_min_z, Config::range_max_z,
        Config::voxel_size_x, Config::voxel_size_y, Config::voxel_size_z,
        Config::grid_size_y,  Config::grid_size_x,
        mask_d_.get(), voxels_d_.get(),stream_));
  CHECK_CUDA_ERROR(generateBaseFeatures_launch(mask_d_.get(), voxels_d_.get(),
      Config::grid_size_y, Config::grid_size_x,
      pillar_num_d_.get(),
      voxels_feature_d_.get(),
      num_points_per_voxel_d_.get(),
      coordinates_d_.get(), stream_));
  
  CHECK_CUDA_ERROR(cudaMemcpyAsync(&num_voxels_, pillar_num_d_.get(), sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(generateFeatures_launch(
    voxels_feature_d_.get(), num_points_per_voxel_d_.get(), coordinates_d_.get(), num_voxels_,
    encoder_in_features_d_.get(), stream_));
   
  return true;
}

void CenterPointTRT::inference()
{
  if (!encoder_trt_ptr_->context_ || !head_trt_ptr_->context_) {
    throw std::runtime_error("Failed to create tensorrt context.");
  }
    // pillar encoder network
    std::vector<void *> encoder_buffers{encoder_in_features_d_.get(), pillar_features_d_.get()};
    SimpleProfiler profiler_pillar("pillarEncoder");
    if(do_profile){
        encoder_trt_ptr_->context_->setProfiler(&profiler_pillar);
    }
    encoder_trt_ptr_->context_->enqueueV2(encoder_buffers.data(), stream_, nullptr);
    if(do_profile){
        std::cout << profiler_pillar;
    }
  // scatter
    CHECK_CUDA_ERROR(scatterFeatures_launch(
        pillar_features_d_.get(), coordinates_d_.get(), num_voxels_, spatial_features_d_.get(),
        stream_));
  // head network
  std::vector<void *> head_buffers = {spatial_features_d_.get(),head_out_offset_d_.get(),
                                                                              head_out_z_d_.get(),  head_out_dim_d_.get(),
                                                                               head_out_rot_d_.get(),head_out_heatmap_d_.get()};        
  SimpleProfiler profiler_head("headNetwork");
  if(do_profile){
        head_trt_ptr_->context_->setProfiler(&profiler_head);
    }          
    head_trt_ptr_->context_->enqueueV2(head_buffers.data(), stream_, nullptr);
    if(do_profile){
        std::cout << profiler_head;
    }
}

void CenterPointTRT::postProcess(std::vector<Box3D> & det_boxes3d)
{
    CHECK_CUDA_ERROR(post_proc_ptr_->generateDetectedBoxes3D_launch(
    head_out_heatmap_d_.get(), head_out_offset_d_.get(), head_out_z_d_.get(), head_out_dim_d_.get(),
    head_out_rot_d_.get(), det_boxes3d, stream_));
  if (det_boxes3d.size() == 0) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("lidar_centerpoint"), "No detected boxes.");
  }
}

}  // namespace centerpoint
