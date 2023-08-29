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

#include "lidar_centerpoint/node.hpp"

#include <config.hpp>
#include <pcl_ros/transforms.hpp>
#include <pointcloud_densification.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>
#include <tier4_autoware_utils/math/constants.hpp>
#include <utils.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <memory>
#include <string>
#include <vector>

namespace centerpoint
{
LidarCenterPointNode::LidarCenterPointNode(const rclcpp::NodeOptions & node_options)
: Node("lidar_center_point", node_options), tf_buffer_(this->get_clock())
{
    score_threshold_ = this->declare_parameter("score_threshold", 0.5);
    std::string densification_world_frame_id =
        this->declare_parameter("densification_world_frame_id", "map");
    int densification_num_past_frames = this->declare_parameter("densification_num_past_frames", 1);
    std::string trt_precision = this->declare_parameter("trt_precision", "fp16");
    std::string encoder_onnx_path = this->declare_parameter("encoder_onnx_path", "");
    std::string encoder_engine_path = this->declare_parameter("encoder_engine_path", "");
    std::string head_onnx_path = this->declare_parameter("head_onnx_path", "");
    std::string head_engine_path = this->declare_parameter("head_engine_path", "");
    class_names_ = this->declare_parameter<std::vector<std::string>>("class_names");//modify config/default.param.yaml
    rename_car_to_truck_and_bus_ = this->declare_parameter("rename_car_to_truck_and_bus", false);
    
    NetworkParam encoder_param(encoder_onnx_path, encoder_engine_path, trt_precision);
    NetworkParam head_param(head_onnx_path, head_engine_path, trt_precision);
    DensificationParam densification_param(
    densification_world_frame_id, densification_num_past_frames);
    //groundSegment
    float voxel_size_x = this->declare_parameter("voxel_size_x", 0.1);
    float voxel_size_y = this->declare_parameter("voxel_size_y", 0.1);
    float voxel_size_z = this->declare_parameter("voxel_size_z", 0.1);
    float outlier_threshold = this->declare_parameter("outlier_threshold", 0.01);
    float max_iterations = this->declare_parameter("max_iterations", 1000);
    float plane_slope_threshold = this->declare_parameter("plane_slope_threshold", 10.0);
    gd_seg_ = std::make_unique<GroundSegment>(voxel_size_x, voxel_size_y,voxel_size_z,outlier_threshold,max_iterations,plane_slope_threshold);
    isThreadRun_ = true;
    //detector
    top2ground_ = this->declare_parameter<std::vector<double>>("top2Ground");
    detector_ptr_ = std::make_unique<CenterPointTRT>(
    class_names_.size(), score_threshold_, encoder_param, head_param, densification_param);
    //
    mgroundSegThread = std::thread(&LidarCenterPointNode::dogroundSegThread, this);
    //pub sub
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "~/input/pointcloud", rclcpp::SensorDataQoS{}.keep_last(1),
        std::bind(&LidarCenterPointNode::pointCloudCallback, this, std::placeholders::_1));
    objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>(
        "~/output/objects", rclcpp::QoS{1});
    gd_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("~/ground/pointcloud",
        rclcpp::QoS{1});
    cluster_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("~/cluster/pointcloud",
        rclcpp::QoS{1});
    hull_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/hull_markers", rclcpp::QoS{1});
}

void LidarCenterPointNode::pointCloudCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg)
{
   const auto objects_sub_count =
   objects_pub_->get_subscription_count() + objects_pub_->get_intra_process_subscription_count();
   if (objects_sub_count < 1) {
       return;
    }
  // TransformPointCloud to ground
   uint64_t begin = GetCurTime();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*input_pointcloud_msg, *pcl_pc_ptr);

  Eigen::Matrix4d Top2Ground = Eigen::Map<Eigen::Matrix4d>(top2ground_.data()).transpose();
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*pcl_pc_ptr, *transformed_cloud, Top2Ground);
  //inference
  pcl::toROSMsg(*transformed_cloud, trans_pointcloud_msg);
  std::vector<Box3D> det_boxes3d;
  bool is_success = detector_ptr_->detect(transformed_cloud, tf_buffer_, det_boxes3d);
  if (!is_success) {
    return;
  }
  autoware_auto_perception_msgs::msg::DetectedObjects output_msg;
  output_msg.header = input_pointcloud_msg->header;
  for (const auto & box3d : det_boxes3d) {
    if (box3d.score < score_threshold_) {
      continue;
    }
    autoware_auto_perception_msgs::msg::DetectedObject obj;
    box3DToDetectedObject(box3d, obj, Top2Ground);
    output_msg.objects.emplace_back(obj);
  }
  if (objects_sub_count > 0) {
    objects_pub_->publish(output_msg);
  }
  uint64_t end = GetCurTime();
  std::cout << "========== " <<"The detect module total runtime is  " << (end - begin)/1000 <<"  ms ==========" << std::endl;
}

void LidarCenterPointNode::dogroundSegThread()
{
    while(isThreadRun_){
    if(trans_pointcloud_msg.data.size()>0){
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(trans_pointcloud_msg, *transformed_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pc(new pcl::PointCloud<pcl::PointXYZ>);
        gd_seg_->filter(transformed_cloud,ground_pc);
        sensor_msgs::msg::PointCloud2  ground_pointcloud_msg;
        pcl::toROSMsg(*ground_pc, ground_pointcloud_msg);
        ground_pointcloud_msg.header.frame_id = "top_lidar";
        ground_pointcloud_msg.header.stamp = trans_pointcloud_msg.header.stamp;
        gd_pc_pub_->publish(ground_pointcloud_msg);
        float min_y = 0;
        float max_y = 0;
        for(size_t i = 0; i < ground_pc->points.size(); i++){
            float y  =  ground_pc->points[i].y;
            if(y < min_y){
                min_y = y;
            }
            if(y > max_y){
                max_y  =  y;
            }
        }
        min_y_ = min_y;
        max_y_ = max_y;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void LidarCenterPointNode::box3DToDetectedObject(
  const Box3D & box3d, autoware_auto_perception_msgs::msg::DetectedObject & obj, Eigen::Matrix4d &rotation_matrix)
{
  
  obj.existence_probability = box3d.score;
  
  // classification
  autoware_auto_perception_msgs::msg::ObjectClassification classification;
  classification.probability = 1.0f;
  classification.label = Label::CAR;
  float l = box3d.length;
  float w = box3d.width;
  obj.classification.emplace_back(classification);

  // transform pose to top_lidar
  float yaw = box3d.yaw;
  Eigen::Vector4d pose_ground(box3d.x , box3d.y, box3d.z , 1.0);
  Eigen::Vector4d pose_top =  rotation_matrix.inverse() * pose_ground;
  //filter
  if(pose_top(1) >(min_y_ + (w/2 + 0.5)) && pose_top(1) < (max_y_ - (w/2 + 0.5)) && yaw < M_PI/2){
        obj.kinematics.pose_with_covariance.pose.position =
        tier4_autoware_utils::createPoint(pose_top(0), pose_top(1), pose_top(2));
        obj.kinematics.pose_with_covariance.pose.orientation =
        tier4_autoware_utils::createQuaternionFromYaw(yaw);
        obj.shape.type = autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX;
        obj.shape.dimensions =
            tier4_autoware_utils::createTranslation(box3d.length, box3d.width, box3d.height);
  }
}

}  // namespace centerpoint

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(centerpoint::LidarCenterPointNode)
