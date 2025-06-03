from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='moveit2_scripts',
            executable='pick_and_place',
            name='pick_and_place_node',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        )
    ])