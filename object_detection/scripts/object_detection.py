#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import pcl
import numpy as np
import tf2_ros
from tf2_ros import TransformException, ConnectivityException
from custom_msgs.msg import DetectedSurfaces, DetectedObjects
from typing import List, Tuple, Union

class ObjectDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__('object_detection_node')
        
        # Subscriber to point cloud data
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/wrist_rgbd_depth_sensor/points',
            self.callback,
            10)
        
        # Publishers for markers
        self.surface_marker_pub = self.create_publisher(
            MarkerArray,
            '/surface_markers',
            10)
        self.object_marker_pub = self.create_publisher(
            MarkerArray,
            '/object_markers',
            10)
        
        # Publishers for detection messages
        self.surface_detected_pub = self.create_publisher(
            DetectedSurfaces,
            '/surface_detected',
            10)
        self.object_detected_pub = self.create_publisher(
            DetectedObjects,
            '/object_detected',
            10)
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info('Object Detection Node initialized (Python with PCL)')

    def callback(self, msg: PointCloud2) -> None:
        try:
            # Convert PointCloud2 msg to pcl point cloud
            cloud = self.from_ros_msg(msg)
            
            if cloud is None:
                self.get_logger().warn('Failed to convert point cloud')
                return

            # Adjusted filtering for the specific detection area (around x=0.33, y=0, z=0)
            # Surface detection: focus on table level around z=-0.006
            filtered_cloud_plane = self.filter_cloud(cloud, 
                                                   min_x=0.1, max_x=0.6,
                                                   min_y=-0.3, max_y=0.3, 
                                                   min_height=-0.05, max_height=0.05)
            
            # Object detection: focus on area above table (around z=0.028)
            filtered_cloud_objects = self.filter_cloud(cloud,
                                                     min_x=0.1, max_x=0.6,
                                                     min_y=-0.3, max_y=0.3,
                                                     min_height=0.005, max_height=0.08)

            # Segmentation: Plane extraction for table surface
            if filtered_cloud_plane and filtered_cloud_plane.size > 100:
                plane_indices, plane_coefficients, plane_cloud = self.extract_plane(filtered_cloud_plane)
                
                # Clustering: Identify clusters corresponding to table surfaces
                surface_clusters, surface_centroids, surface_dimensions = self.extract_clusters(plane_cloud, "Surface")
                
                if surface_centroids:
                    # Publish the detected surface clusters as markers
                    self.pub_surface_marker(surface_centroids, surface_dimensions)
                    # Publish detected surface information
                    self.pub_surface_detected(surface_centroids, surface_dimensions)

            # Clustering: Identify clusters corresponding to objects placed on top of surfaces
            if filtered_cloud_objects and filtered_cloud_objects.size > 20:
                object_clusters, object_centroids, object_dimensions = self.extract_clusters(filtered_cloud_objects, "Object")
                
                if object_centroids:
                    # Publish the detected object clusters as markers
                    self.pub_object_marker(object_centroids, object_dimensions)
                    # Publish detected object information
                    self.pub_object_detected(object_centroids, object_dimensions)

        except (TransformException, ConnectivityException) as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def from_ros_msg(self, msg: PointCloud2) -> Union[pcl.PointCloud, None]:
        """Converts a ROS PointCloud2 message to a PCL point cloud"""
        try:
            # Try to get transform from camera frame to base_link
            try:
                transform = self.tf_buffer.lookup_transform('base_link',
                                                            msg.header.frame_id,
                                                            rclpy.time.Time(),
                                                            timeout=rclpy.duration.Duration(seconds=1.0))
                
                translation = np.array([transform.transform.translation.x,
                                        transform.transform.translation.y,
                                        transform.transform.translation.z])
                rotation_quaternion = np.array([transform.transform.rotation.x,
                                                transform.transform.rotation.y,
                                                transform.transform.rotation.z,
                                                transform.transform.rotation.w])

                # Convert quaternion to rotation matrix
                rotation_matrix = self.quaternion_to_rotation_matrix(rotation_quaternion)
                use_transform = True
                
            except (TransformException, ConnectivityException) as e:
                self.get_logger().warn(f"Transform lookup failed, using original coordinates: {e}")
                use_transform = False

            # Convert PointCloud2 msg to numpy array
            point_step = msg.point_step
            num_points = len(msg.data) // point_step
            points = []
            
            for i in range(num_points):
                start_index = i * point_step
                x_bytes = msg.data[start_index:start_index + 4]
                y_bytes = msg.data[start_index + 4:start_index + 8]
                z_bytes = msg.data[start_index + 8:start_index + 12]
                
                x = np.frombuffer(x_bytes, dtype=np.float32)[0]
                y = np.frombuffer(y_bytes, dtype=np.float32)[0]
                z = np.frombuffer(z_bytes, dtype=np.float32)[0]
                
                # Skip invalid points
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    continue
                    
                point = np.array([x, y, z])

                if use_transform:
                    # Apply the rotation to the point
                    rotated_point = np.dot(rotation_matrix, point)
                    # Apply the translation to get position relative to base_link
                    relative_point = rotated_point + translation
                    points.append(relative_point)
                else:
                    # Use original coordinates
                    points.append(point)

            if not points:
                self.get_logger().warn("No valid points found in point cloud")
                return None

            data = np.array(points, dtype=np.float32)
            cloud = pcl.PointCloud()
            cloud.from_array(data)
            
            self.get_logger().info(f"Converted point cloud with {cloud.size} points")
            return cloud

        except Exception as e:
            self.get_logger().error(f"Error in from_ros_msg: {e}")
            return None

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Converts a quaternion to a rotation matrix"""
        x, y, z, w = q
        rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                                    [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                                    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
        return rotation_matrix

    def filter_cloud(self, cloud: pcl.PointCloud, min_x: float, max_x: float, 
                     min_y: float, max_y: float, min_height: float, max_height: float) -> Union[pcl.PointCloud, None]:
        """Filter cloud based on specific spatial constraints to match target detection area"""
        try:
            if cloud.size == 0:
                return None
                
            indices = []

            for i in range(cloud.size):
                point = cloud[i]
                # Filter by specific coordinate ranges to target detection area
                if (min_x <= point[0] <= max_x and 
                    min_y <= point[1] <= max_y and 
                    min_height <= point[2] <= max_height):
                    indices.append(i)
            
            if not indices:
                self.get_logger().warn(f"No points found within filter constraints")
                return None
                
            filtered_cloud = cloud.extract(indices)
            self.get_logger().info(f"Filtered cloud: {len(indices)} points from {cloud.size}")
            
            return filtered_cloud

        except Exception as e:
            self.get_logger().error(f"Error in filter_cloud: {e}")
            return None

    def extract_plane(self, cloud: pcl.PointCloud) -> Tuple[np.ndarray, np.ndarray, pcl.PointCloud]:
        """Segmentation: Extracts a plane from the point cloud using RANSAC."""
        try:
            seg = cloud.make_segmenter()
            seg.set_model_type(pcl.SACMODEL_PLANE)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_distance_threshold(0.015)  # 1.5cm tolerance for table surface
            indices, coefficients = seg.segment()

            # Extract points belonging to the plane
            plane_cloud = cloud.extract(indices)

            self.get_logger().info(f"Plane extraction: {len(indices)} inliers from {cloud.size} points")
            self.get_logger().info(f"Plane coefficients: {coefficients}")
            
            return indices, coefficients, plane_cloud
            
        except Exception as e:
            self.get_logger().error(f"Error in extract_plane: {e}")
            return np.array([]), np.array([]), pcl.PointCloud()

    def extract_clusters(self, cloud: pcl.PointCloud, cluster_type: str) -> Tuple[List[pcl.PointCloud], List[List[float]], List[List[float]]]:
        """Extracts clusters using Euclidean clustering with adjusted parameters"""
        try:
            if cloud.size == 0:
                return [], [], []
                
            tree = cloud.make_kdtree()
            ec = cloud.make_EuclideanClusterExtraction()
            
            if cluster_type == "Surface":
                # Parameters for table surface detection
                ec.set_ClusterTolerance(0.03)    # 3cm tolerance for table surface
                ec.set_MinClusterSize(100)       # Minimum points for table
                ec.set_MaxClusterSize(50000)     # Large maximum for table
            else:  # Objects
                # Parameters for small object detection (to detect ~2cm objects)
                ec.set_ClusterTolerance(0.015)   # 1.5cm tolerance for small objects
                ec.set_MinClusterSize(20)        # Very small minimum for tiny objects
                ec.set_MaxClusterSize(2000)      # Small maximum for objects
                
            ec.set_SearchMethod(tree)

            # Extract clusters
            cluster_indices = ec.Extract()

            # Initialize lists to store clusters, centroids, and dimensions
            clusters = []
            cluster_centroids = []
            cluster_dimensions = []

            # Process each cluster
            for idx, indices in enumerate(cluster_indices):
                self.get_logger().info(f"Processing {cluster_type} cluster {idx + 1}...")

                # Extract points belonging to the cluster
                cluster = cloud.extract(indices)

                # Calculate centroid
                cluster_array = np.asarray(cluster)
                centroid = np.mean(cluster_array, axis=0)

                # Compute the min and max coordinates along each axis 
                min_coords = np.min(cluster_array, axis=0)
                max_coords = np.max(cluster_array, axis=0)
                dimensions = max_coords - min_coords

                # Append clusters, centroids and dimensions to lists
                clusters.append(cluster)
                cluster_centroids.append(centroid.tolist())
                cluster_dimensions.append(dimensions.tolist())

                # Log cluster information
                num_points = len(indices)
                self.get_logger().info(f"{cluster_type} cluster {idx + 1} has {num_points} points.")
                self.get_logger().info(f"Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
                self.get_logger().info(f"Dimensions: [{dimensions[0]:.3f}, {dimensions[1]:.3f}, {dimensions[2]:.3f}]")

            # Check if any clusters have been extracted
            if not clusters:
                self.get_logger().warning(f"No {cluster_type} clusters extracted...")

            return clusters, cluster_centroids, cluster_dimensions
            
        except Exception as e:
            self.get_logger().error(f"Error in extract_clusters: {e}")
            return [], [], []

    def pub_surface_marker(self, surface_centroids: List[List[float]], surface_dimensions: List[List[float]]) -> None:
        """Publishes the detected surface as cube markers"""
        marker_array = MarkerArray()
        surface_thickness = 0.02

        for idx, (centroid, dimensions) in enumerate(zip(surface_centroids, surface_dimensions)):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "surfaces"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = max(float(dimensions[0]), 0.1)
            marker.scale.y = max(float(dimensions[1]), 0.1)
            marker.scale.z = surface_thickness

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.get_logger().info(f"Published {len(marker_array.markers)} surface markers")
            self.surface_marker_pub.publish(marker_array)
        else:
            self.get_logger().warning("No surface markers to publish.")

    def pub_surface_detected(self, centroids: List[List[float]], dimensions: List[List[float]]) -> None:
        """Publishes the detected surface information"""
        for idx, (centroid, dimension) in enumerate(zip(centroids, dimensions)):
            surface_msg = DetectedSurfaces()
            surface_msg.surface_id = idx
            surface_msg.position.x = float(centroid[0])
            surface_msg.position.y = float(centroid[1])
            surface_msg.position.z = float(centroid[2])
            surface_msg.height = float(dimension[1])  # Y dimension
            surface_msg.width = float(dimension[0])   # X dimension
            
            self.surface_detected_pub.publish(surface_msg)
            self.get_logger().info(f"Published surface detection: ID={idx}")

    def pub_object_marker(self, object_centroids: List[List[float]], object_dimensions: List[List[float]]) -> None:
        """Publishes detected objects as markers"""
        marker_array = MarkerArray()

        for idx, (centroid, dimensions) in enumerate(zip(object_centroids, object_dimensions)):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "objects"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = max(float(dimensions[0]), 0.02)
            marker.scale.y = max(float(dimensions[1]), 0.02)
            marker.scale.z = max(float(dimensions[2]), 0.02)

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.get_logger().info(f"Published {len(marker_array.markers)} object markers")
            self.object_marker_pub.publish(marker_array)
        else:
            self.get_logger().warning("No object markers to publish.")

    def pub_object_detected(self, centroids: List[List[float]], dimensions: List[List[float]]) -> None:
        """Publishes the detected object information"""
        for idx, (centroid, dimension) in enumerate(zip(centroids, dimensions)):
            object_msg = DetectedObjects()
            object_msg.object_id = idx
            object_msg.position.x = float(centroid[0])
            object_msg.position.y = float(centroid[1])
            object_msg.position.z = float(centroid[2])
            object_msg.height = float(dimension[1])    # Y dimension
            object_msg.width = float(dimension[0])     # X dimension
            object_msg.thickness = float(dimension[2]) # Z dimension
            
            self.object_detected_pub.publish(object_msg)
            self.get_logger().info(f"Published object detection: ID={idx}")

def main(args=None) -> None:
    rclpy.init(args=args)
    object_detection = ObjectDetectionNode()
    rclpy.spin(object_detection)
    rclpy.shutdown()

if __name__ == '__main__':
    main()