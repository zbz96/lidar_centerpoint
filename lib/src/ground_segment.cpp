/**
 * @file ground_segment.cpp
 * @author zhaobingzhen@tongxin.cn
 * @brief 
 * @version 0.1
 * @date 2022-10-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include"ground_segment.hpp"
namespace centerpoint{

GroundSegment::GroundSegment
    (float voxel_size_x, float voxel_size_y, float voxel_size_z, float outlier_threshold, int max_iterations, float plane_slope_threshold):
    voxel_size_x_(voxel_size_x),
    voxel_size_y_(voxel_size_y),
    voxel_size_z_(voxel_size_z),
    outlier_threshold_(outlier_threshold),
    max_iterations_(max_iterations),
    plane_slope_threshold_(plane_slope_threshold)
{

}
void  GroundSegment::applyRANSAC(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, 
        pcl::PointIndices::Ptr & output_inliers, pcl::ModelCoefficients::Ptr & output_coefficients)
{
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setRadiusLimits(0.3, std::numeric_limits<double>::max());
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(outlier_threshold_);
    seg.setInputCloud(input);
    seg.setMaxIterations(max_iterations_);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.segment(*output_inliers, *output_coefficients);
}

void GroundSegment::extractPointsIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr &input,  const pcl::PointIndices & in_indices,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &ground, pcl::PointCloud<pcl::PointXYZ>::Ptr &no_ground
)
{
    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;
    extract_ground.setInputCloud(input);
    extract_ground.setIndices(pcl::make_shared<pcl::PointIndices>(in_indices));
    extract_ground.setNegative(false); 
    extract_ground.filter(*ground);
    extract_ground.setNegative(true); 
    extract_ground.filter(*no_ground);
}

void GroundSegment::filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &output)
{
    //passThroygh filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter_y(new pcl::PointCloud<pcl::PointXYZ>);
     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-15,15);
    pass.setFilterLimitsNegative(false);
    pass.filter(*cloud_filter_y);
    //
    pass.setInputCloud(cloud_filter_y);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.5,0.5);
    pass.setFilterLimitsNegative(false);
    pass.filter(*cloud_filter_z);
    // std::cout<<"cloud_filter_z poins is :  "<<cloud_filter_z->points.size()<<std::endl;

   // downsample pointcloud to reduce ransac calculation cost
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new  pcl::PointCloud<pcl::PointXYZ>);
    downsampled_cloud->points.reserve(cloud_filter_z->points.size());
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud_filter_z);
    filter.setLeafSize(voxel_size_x_, voxel_size_y_, voxel_size_z_);
    filter.filter(*downsampled_cloud);
    //apply RANSAC
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    applyRANSAC(downsampled_cloud,inliers,coefficients);
    if (coefficients->values.empty()) {
        std::cout<<"empty coefficients"<<std::endl;
        output = cloud_filter_z;
    return;
    }
    //过滤坡度较大的平面防止过拟合(比如墙面)
    Eigen::Vector3d plane_normal(
    coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    const auto plane_slope = std::abs(
      std::acos(plane_normal.dot(unit_vec_) / (plane_normal.norm() * unit_vec_.norm())) * 180 /
      M_PI);
    if (plane_slope > plane_slope_threshold_) {
      output = input;
      return;
    }
    //extract ground
    pcl::PointCloud<pcl::PointXYZ>::Ptr segment_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr segment_no_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    extractPointsIndices(
        downsampled_cloud, *inliers, segment_ground_cloud_ptr, segment_no_ground_cloud_ptr);
    output = segment_ground_cloud_ptr;
}
}//centerpoint
