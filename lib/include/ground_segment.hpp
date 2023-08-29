/**
 * @file ground_segmen.hpp
 * @author zhaobinzhen@tongxin.cn
 * @brief 
 * @version 0.1
 * @date 2022-10-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef GROUND_SEGMENT_HPP_
#define GROUND_SEGMENT_HPP_

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include<pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/transform_datatypes.h>
#include<Eigen/Core>

namespace centerpoint{

class GroundSegment
{
public:
    GroundSegment(float voxel_size_x, float voxel_size_y, float voxel_size_z, float outlier_threshold, int max_iterations, float plane_slope_threshold_);
    void filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &output);
    void applyRANSAC(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, 
        pcl::PointIndices::Ptr & output_inliers, pcl::ModelCoefficients::Ptr & output_coefficients);
    void extractPointsIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr &input,  const pcl::PointIndices & in_indices,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &ground, pcl::PointCloud<pcl::PointXYZ>::Ptr &no_ground);

private:
    float voxel_size_x_;
    float voxel_size_y_;
    float voxel_size_z_;
    float outlier_threshold_;
    int max_iterations_;
    float plane_slope_threshold_;
    Eigen::Vector3d unit_vec_ = Eigen::Vector3d::UnitZ();

};
#endif //GROUND_SEGMENT_HPP_











}//centerpoint
