#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations
import math

class StaticTransformPublisher(Node):
    def __init__(self):
        super().__init__('static_transform_publisher')
        
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.get_logger().info('Static transform publisher initialized (Python)')

    def publish_static_transform(self):
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'base_link'
        static_transform.child_frame_id = 'wrist_rgbd_camera_depth_optical_frame'
        
        # Use the ACTUAL transform values from tf2_echo
        static_transform.transform.translation.x = 0.338
        static_transform.transform.translation.y = 0.450
        static_transform.transform.translation.z = 0.100
        
        # Use the ACTUAL rotation quaternion from tf2_echo
        static_transform.transform.rotation.x = 0.000
        static_transform.transform.rotation.y = 0.866
        static_transform.transform.rotation.z = -0.500
        static_transform.transform.rotation.w = 0.000
        
        self.tf_static_broadcaster.sendTransform(static_transform)
        self.get_logger().info('Published static transform with ACTUAL camera position and orientation')
        self.get_logger().info('Translation: [0.338, 0.450, 0.100]')
        self.get_logger().info('RPY: [-120°, 0°, 180°]')

def main(args=None):
    rclpy.init(args=args)
    node = StaticTransformPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()