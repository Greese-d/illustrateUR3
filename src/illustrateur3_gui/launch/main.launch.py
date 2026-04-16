from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # -----------------------------
    # Launch arguments
    # -----------------------------
    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera',
        default_value='false',
        description=(
            'Set to true to launch the camera_publisher node. '
            'Leave false when replaying a rosbag instead of using a live camera.'
        )
    )

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
    # Camera publisher node
    # -----------------------------
    camera_publisher = Node(
        package='portrait_vectorisation',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_camera')),
        parameters=[{
            'device': 0,
            'fps': 30.0,
            'width': 1920,
            'height': 1080,
            'topic': '/camera/image_raw',
        }]
    )

    # -----------------------------
    # Image processing node
    # -----------------------------
    image_processing_node = Node(
        package='portrait_vectorisation',
        executable='image_processing_node',
        name='image_processing_node',
        output='screen',
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'snapshot_topic': '/camera/snapshot',
            'portrait_topic': '/portrait/preview',
            'strokes_topic': '/portrait/strokes',
            'markers_topic': '/portrait/markers',
            'stroke_publish_delay': 0.05,
        }]
    )

    # -----------------------------
    # Final LaunchDescription
    # -----------------------------
    return LaunchDescription([
        launch_camera_arg,
        ur_driver,
        moveit,
        gui_node,
        calibration_node,
        camera_publisher,
        image_processing_node,
    ])