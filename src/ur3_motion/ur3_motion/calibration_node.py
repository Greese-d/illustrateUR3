import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R

import numpy as np
import json
import sys
import termios
import tty
import time


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


class CalibrationNode(Node):

    def __init__(self):
        super().__init__("calibration_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(Marker, "paper_marker", 10)

        self.tcp_offset = 0.12

        # FIX: always fixed size (P1,P2,P3)
        self.points = [None, None, None]

        self.current_index = None
        self.preview_point = None

        self.last_key_time = 0
        self.debounce_time = 0.3

        self.show_menu()
        self.timer = self.create_timer(0.1, self.loop)

    def show_menu(self):
        self.get_logger().info("""
Calibration Started
-------------------
1 → Edit P1
2 → Edit P2
3 → Edit P3

Move robot → press ENTER to confirm

r → Reset
q → Quit
""")

    def transform_to_matrix(self, t, q):
        T = np.eye(4)
        T[0:3, 3] = [t.x, t.y, t.z]
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        T[0:3, 0:3] = rot.as_matrix()
        return T

    def apply_tcp_offset(self, T):
        T_offset = np.eye(4)
        T_offset[2, 3] = self.tcp_offset
        return T @ T_offset

    def get_pen_tip_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "tool0",
                rclpy.time.Time()
            )

            T = self.transform_to_matrix(
                transform.transform.translation,
                transform.transform.rotation
            )

            T_new = self.apply_tcp_offset(T)

            return T_new[0:3, 3]

        except Exception as e:
            self.get_logger().warn(f"TF not ready: {e}")
            return None

    def loop(self):
        key = get_key()

        now = time.time()
        if now - self.last_key_time < self.debounce_time:
            return

        if key:
            self.last_key_time = now
            self.handle_key(key)

    def handle_key(self, key):

        # select which point to edit
        if key in ["1", "2", "3"]:
            self.current_index = int(key) - 1
            self.get_logger().info(f"Editing P{key} → move robot, press ENTER")

        # confirm
        elif key == "\r" or key == "\n":
            if self.current_index is None:
                self.get_logger().warn("Select point first (1/2/3)")
                return

            pos = self.get_pen_tip_position()
            if pos is None:
                return

            # overwrite directly (NO append)
            self.points[self.current_index] = pos

            self.get_logger().info(
                f"P{self.current_index+1} CONFIRMED: {pos}"
            )

            self.current_index = None

            # auto visualize
            if all(p is not None for p in self.points):
                self.visualize_paper()

        elif key == "r":
            self.reset_calibration()

        elif key == "q":
            self.get_logger().info("Exit calibration")
            self.destroy_node()
            rclpy.shutdown()

        # preview continuously
        if self.current_index is not None:
            pos = self.get_pen_tip_position()
            if pos is not None:
                self.get_logger().info(
                    f"P{self.current_index+1} preview: {pos}"
                )

    def visualize_paper(self):

        P1, P2, P3 = self.points

        # X axis
        x_axis = P2 - P1
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            self.get_logger().error("P1 and P2 too close!")
            return
        x_axis /= x_norm

        # Y axis (orthogonalized)
        y_raw = P3 - P1
        y_proj = y_raw - np.dot(y_raw, x_axis) * x_axis
        y_norm = np.linalg.norm(y_proj)

        if y_norm < 1e-6:
            self.get_logger().error("Invalid P3 (collinear with P1→P2)")
            return

        y_axis = y_proj / y_norm

        # Z axis
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        width = np.linalg.norm(P2 - P1)
        height = np.linalg.norm(y_raw)

        center = P1 + 0.5 * width * x_axis + 0.5 * height * y_axis

        R_mat = np.column_stack((x_axis, y_axis, z_axis))
        quat = R.from_matrix(R_mat).as_quat()

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "paper"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker.scale.x = width
        marker.scale.y = height
        marker.scale.z = 0.001

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

        self.get_logger().info("🟦 Paper displayed correctly")

    def reset_calibration(self):
        self.points = [None, None, None]
        self.current_index = None

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.action = Marker.DELETE
        marker.id = 0
        self.marker_pub.publish(marker)

        self.get_logger().info("Calibration reset")
        self.show_menu()


def main():
    rclpy.init()
    node = CalibrationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()