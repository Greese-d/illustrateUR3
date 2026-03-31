from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # -----------------------------
    # UR Robot Driver launch
    # -----------------------------
    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_robot_driver'),
                'launch',
                'ur_control.launch.py'
            )
        ),
        launch_arguments={
            'ur_type': 'ur3',
            'robot_ip': '192.168.0.195',
            'use_fake_hardware': 'true',
            'launch_rviz': 'false'
        }.items()
    )

    # -----------------------------
    # MoveIt launch
    # -----------------------------
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_moveit_config'),
                'launch',
                'ur_moveit.launch.py'
            )
        ),
        launch_arguments={
            'ur_type': 'ur3',
            'launch_rviz': 'true'
        }.items()
    )

    # -----------------------------
    # GUI node
    # -----------------------------
    gui_node = Node(
        package='illustrateur3_gui',
        executable='gui_main',
        name='gui_node',
        output='screen'
    )

    # -----------------------------
    # Calibration node
    # -----------------------------
    calibration_node = Node(
        package='ur3_motion',
        executable='calibration_node',
        name='calibration_node',
        output='screen'
    )

    # -----------------------------
    # Final LaunchDescription
    # -----------------------------
    return LaunchDescription([
        ur_driver,
        moveit,
        gui_node,
        calibration_node,
    ])