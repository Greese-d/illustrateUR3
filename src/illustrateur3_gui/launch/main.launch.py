from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        # GUI node
        Node(
            package='illustrateur3_gui',
            executable='gui_main',
            name='gui_node',
            output='screen'
        ),

        # Calibration node
        Node(
            package='ur3_motion',
            executable='calibration_node',
            name='calibration_node',
            output='screen'
        ),

    ])