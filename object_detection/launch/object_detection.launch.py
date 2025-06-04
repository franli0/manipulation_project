import os
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    pkg_share = FindPackageShare('object_detection')
    rviz_config_file = PathJoinSubstitution([pkg_share, 'rviz', 'object_detection.rviz'])
    
    return LaunchDescription([
        
        Node(
            package='object_detection',
            executable='static_transform_publisher_cpp',
            name='static_transform_publisher',
            output='screen'
        ),
        
        Node(
            package='object_detection',
            executable='object_detection_cpp',
            name='object_detection_node',
            output='screen'
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
    ])