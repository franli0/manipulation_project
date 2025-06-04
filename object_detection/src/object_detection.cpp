#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

// Include custom messages
#include "object_detection/msg/detected_surfaces.hpp"
#include "object_detection/msg/detected_objects.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

struct Point3D {
    float x, y, z;
    Point3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

struct Plane {
    float a, b, c, d;  // ax + by + cz + d = 0
    std::vector<int> inliers;
};

class ObjectDetectionNode : public rclcpp::Node
{
public:
    ObjectDetectionNode() : Node("object_detection_node")
    {
        // Initialize TF2
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Subscribers
        pc_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/wrist_rgbd_depth_sensor/points", 10,
            std::bind(&ObjectDetectionNode::pointCloudCallback, this, std::placeholders::_1));
        
        // Publishers using custom messages
        surface_pub_ = this->create_publisher<object_detection::msg::DetectedSurfaces>("/surface_detected", 10);
        object_pub_ = this->create_publisher<object_detection::msg::DetectedObjects>("/object_detected", 10);
        surface_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/surface_markers", 10);
        object_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/object_markers", 10);
        
        RCLCPP_INFO(this->get_logger(), "Object Detection Node initialized (C++ with REAL perception)");
        
        // Perception parameters
        plane_distance_threshold_ = 0.02f;  // 2cm tolerance for plane fitting
        min_cluster_size_ = 50;
        max_cluster_size_ = 2000;
        cluster_tolerance_ = 0.02f;
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        try
        {
            RCLCPP_INFO(this->get_logger(), "Processing point cloud with %d points", 
                       msg->width * msg->height);
            
            // Convert PointCloud2 to our internal format and transform to base_link
            std::vector<Point3D> points = convertAndTransformPointCloud(msg);
            
            if (points.empty()) {
                RCLCPP_WARN(this->get_logger(), "No valid points after conversion/transform");
                return;
            }
            
            // Filter points for workspace
            std::vector<Point3D> filtered_points = filterPointCloud(points);
            
            if (filtered_points.size() < 100) {
                RCLCPP_WARN(this->get_logger(), "Not enough points after filtering: %zu", filtered_points.size());
                return;
            }
            
            RCLCPP_INFO(this->get_logger(), "After filtering: %zu points", filtered_points.size());
            
            // Detect table surface using RANSAC plane fitting
            Plane detected_plane = detectPlaneRANSAC(filtered_points);
            
            if (detected_plane.inliers.size() < 200) {
                RCLCPP_WARN(this->get_logger(), "No sufficient plane detected");
                return;
            }
            
            RCLCPP_INFO(this->get_logger(), "Detected plane with %zu inliers", detected_plane.inliers.size());
            
            // Extract plane points and non-plane points
            std::vector<Point3D> plane_points, non_plane_points;
            extractPlanePoints(filtered_points, detected_plane, plane_points, non_plane_points);
            
            // Process surface detection
            processSurfaceDetection(plane_points, msg->header);
            
            // Process object detection on points above the plane
            if (!non_plane_points.empty()) {
                std::vector<Point3D> above_plane_points = filterPointsAbovePlane(non_plane_points, detected_plane);
                if (!above_plane_points.empty()) {
                    processObjectDetection(above_plane_points, msg->header);
                }
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in point cloud processing: %s", e.what());
        }
    }
    
    std::vector<Point3D> convertAndTransformPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg)
    {
        std::vector<Point3D> points;
        
        try {
            // Get transform from camera frame to base_link
            geometry_msgs::msg::TransformStamped transform;
            transform = tf_buffer_->lookupTransform("base_link", msg->header.frame_id, tf2::TimePointZero);
            
            // Parse point cloud data
            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
            
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z)) {
                    continue;
                }
                
                // Transform point to base_link frame
                geometry_msgs::msg::PointStamped point_in, point_out;
                point_in.header = msg->header;
                point_in.point.x = *iter_x;
                point_in.point.y = *iter_y;
                point_in.point.z = *iter_z;
                
                tf2::doTransform(point_in, point_out, transform);
                
                points.emplace_back(point_out.point.x, point_out.point.y, point_out.point.z);
            }
        }
        catch (const tf2::TransformException& e) {
            RCLCPP_WARN(this->get_logger(), "Transform failed: %s", e.what());
            // Fallback: use points without transform
            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
            
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                if (!std::isnan(*iter_x) && !std::isnan(*iter_y) && !std::isnan(*iter_z)) {
                    points.emplace_back(*iter_x, *iter_y, *iter_z);
                }
            }
        }
        
        return points;
    }
    
    std::vector<Point3D> filterPointCloud(const std::vector<Point3D>& points)
    {
        std::vector<Point3D> filtered;
        
        for (const auto& point : points) {
            // Filter by workspace bounds (adjust these for your robot setup)
            if (point.x > -1.0f && point.x < 2.0f &&  // Forward/backward range
                point.y > -1.0f && point.y < 1.0f &&  // Left/right range  
                point.z > -0.1f && point.z < 1.5f) {  // Height range
                
                // Filter by distance from origin
                float distance = std::sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
                if (distance > 0.2f && distance < 2.0f) {
                    filtered.push_back(point);
                }
            }
        }
        
        return filtered;
    }
    
    Plane detectPlaneRANSAC(const std::vector<Point3D>& points)
    {
        Plane best_plane;
        int max_inliers = 0;
        const int max_iterations = 1000;
        
        std::srand(std::time(nullptr));
        
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            // Randomly sample 3 points
            if (points.size() < 3) break;
            
            std::vector<int> sample_indices;
            for (int i = 0; i < 3; ++i) {
                int idx;
                do {
                    idx = std::rand() % points.size();
                } while (std::find(sample_indices.begin(), sample_indices.end(), idx) != sample_indices.end());
                sample_indices.push_back(idx);
            }
            
            // Calculate plane equation from 3 points
            const Point3D& p1 = points[sample_indices[0]];
            const Point3D& p2 = points[sample_indices[1]];
            const Point3D& p3 = points[sample_indices[2]];
            
            // Calculate normal vector using cross product
            float v1x = p2.x - p1.x, v1y = p2.y - p1.y, v1z = p2.z - p1.z;
            float v2x = p3.x - p1.x, v2y = p3.y - p1.y, v2z = p3.z - p1.z;
            
            float nx = v1y * v2z - v1z * v2y;
            float ny = v1z * v2x - v1x * v2z;
            float nz = v1x * v2y - v1y * v2x;
            
            // Normalize normal vector
            float norm = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (norm < 1e-6) continue;
            
            nx /= norm; ny /= norm; nz /= norm;
            
            // Calculate d parameter
            float d = -(nx * p1.x + ny * p1.y + nz * p1.z);
            
            // Count inliers
            std::vector<int> inliers;
            for (size_t i = 0; i < points.size(); ++i) {
                float distance = std::abs(nx * points[i].x + ny * points[i].y + nz * points[i].z + d);
                if (distance < plane_distance_threshold_) {
                    inliers.push_back(i);
                }
            }
            
            // Update best plane if this one is better
            if (inliers.size() > max_inliers) {
                max_inliers = inliers.size();
                best_plane.a = nx;
                best_plane.b = ny;
                best_plane.c = nz;
                best_plane.d = d;
                best_plane.inliers = inliers;
            }
        }
        
        return best_plane;
    }
    
    void extractPlanePoints(const std::vector<Point3D>& points, const Plane& plane,
                           std::vector<Point3D>& plane_points, std::vector<Point3D>& non_plane_points)
    {
        std::vector<bool> is_inlier(points.size(), false);
        for (int idx : plane.inliers) {
            is_inlier[idx] = true;
        }
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (is_inlier[i]) {
                plane_points.push_back(points[i]);
            } else {
                non_plane_points.push_back(points[i]);
            }
        }
    }
    
    std::vector<Point3D> filterPointsAbovePlane(const std::vector<Point3D>& points, const Plane& plane)
    {
        std::vector<Point3D> above_plane;
        
        for (const auto& point : points) {
            float distance = plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d;
            // Points above the plane (positive distance) within reasonable height
            if (distance > 0.005f && distance < 0.3f) {  // 5mm to 30cm above plane
                above_plane.push_back(point);
            }
        }
        
        return above_plane;
    }
    
    void processSurfaceDetection(const std::vector<Point3D>& plane_points, const std_msgs::msg::Header& header)
    {
        if (plane_points.empty()) return;
        
        // Calculate surface properties
        Point3D min_pt(plane_points[0]), max_pt(plane_points[0]);
        Point3D centroid(0, 0, 0);
        
        for (const auto& point : plane_points) {
            min_pt.x = std::min(min_pt.x, point.x);
            min_pt.y = std::min(min_pt.y, point.y);
            min_pt.z = std::min(min_pt.z, point.z);
            max_pt.x = std::max(max_pt.x, point.x);
            max_pt.y = std::max(max_pt.y, point.y);
            max_pt.z = std::max(max_pt.z, point.z);
            
            centroid.x += point.x;
            centroid.y += point.y;
            centroid.z += point.z;
        }
        
        centroid.x /= plane_points.size();
        centroid.y /= plane_points.size();
        centroid.z /= plane_points.size();
        
        // Publish surface detection
        auto surface_msg = object_detection::msg::DetectedSurfaces();
        surface_msg.surface_id = 0;
        surface_msg.position.x = centroid.x;
        surface_msg.position.y = centroid.y;
        surface_msg.position.z = centroid.z;
        surface_msg.height = max_pt.y - min_pt.y;
        surface_msg.width = max_pt.x - min_pt.x;
        
        surface_pub_->publish(surface_msg);
        
        // Publish surface marker
        publishSurfaceMarker(surface_msg, header);
        
        RCLCPP_INFO(this->get_logger(), "Detected surface at [%.3f, %.3f, %.3f] with dimensions [%.3f x %.3f]",
                   centroid.x, centroid.y, centroid.z, surface_msg.width, surface_msg.height);
    }
    
    void processObjectDetection(const std::vector<Point3D>& points, const std_msgs::msg::Header& header)
    {
        if (points.size() < min_cluster_size_) return;
        
        // Simple clustering based on spatial proximity
        std::vector<std::vector<Point3D>> clusters = performClustering(points);
        
        auto marker_array = visualization_msgs::msg::MarkerArray();
        
        for (size_t i = 0; i < clusters.size(); ++i) {
            if (clusters[i].size() < min_cluster_size_) continue;
            
            // Calculate cluster properties
            Point3D min_pt(clusters[i][0]), max_pt(clusters[i][0]);
            Point3D centroid(0, 0, 0);
            
            for (const auto& point : clusters[i]) {
                min_pt.x = std::min(min_pt.x, point.x);
                min_pt.y = std::min(min_pt.y, point.y);
                min_pt.z = std::min(min_pt.z, point.z);
                max_pt.x = std::max(max_pt.x, point.x);
                max_pt.y = std::max(max_pt.y, point.y);
                max_pt.z = std::max(max_pt.z, point.z);
                
                centroid.x += point.x;
                centroid.y += point.y;
                centroid.z += point.z;
            }
            
            centroid.x /= clusters[i].size();
            centroid.y /= clusters[i].size();
            centroid.z /= clusters[i].size();
            
            // Publish object detection
            auto object_msg = object_detection::msg::DetectedObjects();
            object_msg.object_id = i;
            object_msg.position.x = centroid.x;
            object_msg.position.y = centroid.y;
            object_msg.position.z = centroid.z;
            object_msg.height = max_pt.y - min_pt.y;
            object_msg.width = max_pt.x - min_pt.x;
            object_msg.thickness = max_pt.z - min_pt.z;
            
            object_pub_->publish(object_msg);
            
            // Create object marker
            auto marker = createObjectMarker(object_msg, header, i);
            marker_array.markers.push_back(marker);
            
            RCLCPP_INFO(this->get_logger(), "Detected object %zu at [%.3f, %.3f, %.3f] with dimensions [%.3f x %.3f x %.3f]",
                       i, centroid.x, centroid.y, centroid.z, object_msg.width, object_msg.height, object_msg.thickness);
        }
        
        if (!marker_array.markers.empty()) {
            object_marker_pub_->publish(marker_array);
        }
    }
    
    std::vector<std::vector<Point3D>> performClustering(const std::vector<Point3D>& points)
    {
        std::vector<std::vector<Point3D>> clusters;
        std::vector<bool> processed(points.size(), false);
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (processed[i]) continue;
            
            std::vector<Point3D> cluster;
            std::vector<size_t> queue;
            queue.push_back(i);
            processed[i] = true;
            
            while (!queue.empty() && cluster.size() < max_cluster_size_) {
                size_t current_idx = queue.back();
                queue.pop_back();
                cluster.push_back(points[current_idx]);
                
                // Find nearby points
                for (size_t j = 0; j < points.size(); ++j) {
                    if (processed[j]) continue;
                    
                    float distance = std::sqrt(
                        std::pow(points[current_idx].x - points[j].x, 2) +
                        std::pow(points[current_idx].y - points[j].y, 2) +
                        std::pow(points[current_idx].z - points[j].z, 2)
                    );
                    
                    if (distance < cluster_tolerance_) {
                        processed[j] = true;
                        queue.push_back(j);
                    }
                }
            }
            
            if (cluster.size() >= min_cluster_size_) {
                clusters.push_back(cluster);
            }
        }
        
        return clusters;
    }
    
    void publishSurfaceMarker(const object_detection::msg::DetectedSurfaces& surface_msg, const std_msgs::msg::Header& header)
    {
        auto marker_array = visualization_msgs::msg::MarkerArray();
        auto marker = visualization_msgs::msg::Marker();
        
        marker.header = header;
        marker.header.frame_id = "base_link";
        marker.ns = "surfaces";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position = surface_msg.position;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = std::max(surface_msg.width, 0.1);
        marker.scale.y = std::max(surface_msg.height, 0.1);
        marker.scale.z = 0.02;  // Thin surface
        
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.7;
        
        marker_array.markers.push_back(marker);
        surface_marker_pub_->publish(marker_array);
    }
    
    visualization_msgs::msg::Marker createObjectMarker(const object_detection::msg::DetectedObjects& object_msg,
                                                       const std_msgs::msg::Header& header, int id)
    {
        auto marker = visualization_msgs::msg::Marker();
        
        marker.header = header;
        marker.header.frame_id = "base_link";
        marker.ns = "objects";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position = object_msg.position;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = std::max(object_msg.width, 0.02);
        marker.scale.y = std::max(object_msg.height, 0.02);
        marker.scale.z = std::max(object_msg.thickness, 0.02);
        
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        
        return marker;
    }
    
    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_;
    rclcpp::Publisher<object_detection::msg::DetectedSurfaces>::SharedPtr surface_pub_;
    rclcpp::Publisher<object_detection::msg::DetectedObjects>::SharedPtr object_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr surface_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr object_marker_pub_;
    
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Perception parameters
    float plane_distance_threshold_;
    size_t min_cluster_size_;
    size_t max_cluster_size_;
    float cluster_tolerance_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetectionNode>());
    rclcpp::shutdown();
    return 0;
}