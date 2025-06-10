from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.actions import EmitEvent
from launch.events import Shutdown

def generate_launch_description():
    
    # Static transform publisher
    static_transform_node = Node(
        package='object_detection',
        executable='static_transform_publisher.py',
        name='static_transform_publisher',
        output='screen'
    )
    
    # Object detection node
    object_detection_node = Node(
        package='object_detection',
        executable='object_detection.py',
        name='object_detection_node',
        output='screen'
    )
    
    # Pick and place with perception (main node)
    pick_place_node = Node(
        package='moveit2_scripts',
        executable='pick_and_place_perception_real',
        name='pick_and_place_perception_real_node',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )
    
    # Event handler to shutdown all nodes when pick_place_node exits
    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=pick_place_node,
            on_exit=[
                EmitEvent(event=Shutdown(reason='Pick and place completed'))
            ]
        )
    )
    
    return LaunchDescription([
        static_transform_node,
        object_detection_node,
        pick_place_node,
        shutdown_handler
    ])