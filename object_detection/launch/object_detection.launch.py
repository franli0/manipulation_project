import os
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    pkg_share = FindPackageShare('object_detection')
    rviz_config_file = PathJoinSubstitution([pkg_share, 'rviz', 'object_detection.rviz'])
    
    return LaunchDescription([
        
        # Static transform publisher node
        Node(
            package='object_detection',
            executable='static_transform_publisher.py',
            name='static_transform_publisher',
            output='screen'
        ),
        
        # Object detection node  
        Node(
            package='object_detection',
            executable='object_detection.py',
            name='object_detection_node',
            output='screen'
        ),
        
        # RViz2 node
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
    ])