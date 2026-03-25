
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import numpy as np
import json
import sys
import termios
import tty
import time
import os
class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.get_logger().info("Motion node ready")
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.timer = self.create_timer(1.0, self.print_pose)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
        )
        self.moveit2.max_velocity = 0.05
        self.moveit2.max_acceleration = 0.05
        self.fixed_orientation = [0.0, 1.0, 0.0, 0.0]   
        # self.timer = self.create_timer(2.0, self.start_drawing)
        self.start_drawing()
        # Move to home position
        # self.go_home()
        # self.go_to_pose([0.408, 0.113, 0.312])
        # self.go_to_pose([0.408, -0.184, 0.312])
        # Create subscriber to pen_path topic
        self.create_subscription(
            Path,
            "pen_path",
            self.pen_path_callback,
            10,
        )
        # Create publisher for drawing status
        self.status_pub = self.create_publisher(String, "/drawing/status", 10)
    # Extract calibration data from json file (rs2_ws/data/paper_calibration.json)
    def load_calibration(self):
        workspace = os.getcwd()  # assumes running from ~/rs2_ws
        json_path = os.path.join(workspace, "data", "paper_calibration.json")

        with open(json_path, "r") as f:
            data = json.load(f)

        P1 = np.array(data["P1"])
        P2 = np.array(data["P2"])
        P3 = np.array(data["P3"])

        width = data["width"]
        height = data["height"]

        x_axis= np.array(data["x_axis"])
        y_axis= np.array(data["y_axis"])
        z_axis= np.array(data["z_axis"])
        return P1, P2, P3, width, height, x_axis, y_axis, z_axis
    def generate_rectangle(self, P1, x_axis, y_axis, width, height, offset=0.01):

        # rectangle dimensions
        w = width - 2 * offset
        h = height - 2 * offset

        # shift origin inside
        origin = P1 + offset * x_axis + offset * y_axis

        # corners
        p1 = origin
        p2 = origin + w * x_axis
        p3 = origin + w * x_axis + h * y_axis
        p4 = origin + h * y_axis

        return [p1, p2, p3, p4, p1]  # closed loop
    def draw_rectangle(self):

        P1, P2, P3, width, height, x_axis, y_axis, z_axis = self.load_calibration()
        path = self.generate_rectangle(P1, x_axis, y_axis, width, height)
        self.get_logger().info("Drawing rectangle...")
        tcp_offset= 0.12
        for point in path:
            real_point = point - tcp_offset*z_axis
            self.moveit2.move_to_pose(
                position=real_point.tolist(),
                quat_xyzw=self.fixed_orientation,
                cartesian=True
            )
            self.moveit2.wait_until_executed()

        self.get_logger().info("Rectangle done ✅")

    def start_drawing(self):
        self.draw_rectangle()
        
    # Callback function for pen path subscriber
    def pen_path_callback(self, msg: Path):
        self.get_logger().info("Starting drawing...")
        # FIXED orientation (Z perpendicular to paper)
        for pose_stamped in msg.poses:
            pose = pose_stamped.pose
            position = [
                pose.position.x,
                pose.position.y,
                pose.position.z
            ]
            self.moveit2.move_to_pose(
                position=position,
                quat_xyzw=self.fixed_orientation,
                cartesian=True
            )
            self.moveit2.wait_until_executed()
        self.get_logger().info("Drawing completed")
        status_msg = String()
        status_msg.data = "completed"
        self.status_pub.publish(status_msg)

    # def get_current_pose(self):
    #     try:
    #         transform = self.tf_buffer.lookup_transform(
    #             "base_link",
    #             "tool0",
    #             rclpy.time.Time(),
    #             timeout=rclpy.duration.Duration(seconds=1.0)  # 🔥 KEY FIX
    #         )

    #         t = transform.transform.translation
    #         r = transform.transform.rotation

    #         position = [t.x, t.y, t.z]
    #         orientation = [r.x, r.y, r.z, r.w]

    #         return position, orientation

    #     except Exception as e:
    #         self.get_logger().warn(f"TF not ready yet: {e}")
    #         return None, None
        
    # def print_pose(self):
    #     pos, quat = self.get_current_pose()
    #     if pos is not None:
    #         self.get_logger().info(f"Position: {pos}")
    #         self.get_logger().info(f"Orientation: {quat}")
    def go_home(self):
    #     self.moveit2.move_to_pose(
    #     position=[0.298, 0.113, 0.312],
    #     quat_xyzw=self.fixed_orientation,
    #     cartesian=True
    # )
        self.moveit2.move_to_configuration([0.0, -1.57, 1.57, -1.57, -1.57, 1.57])
        self.moveit2.wait_until_executed()
    def go_to_pose(self, position):
        self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=self.fixed_orientation,
            cartesian=True
        )
        self.moveit2.wait_until_executed()


def main():

    rclpy.init()

    node = MotionNode()

    rclpy.spin(node)

    rclpy.shutdown()