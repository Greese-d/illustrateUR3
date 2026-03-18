import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

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

        self.tcp_offset = np.array([0.0, 0.0, -0.12])
        self.points = []

        self.last_key_time = 0
        self.debounce_time = 0.5

        self.show_menu()
        self.timer = self.create_timer(0.1, self.loop)

    def show_menu(self):
        self.get_logger().info("""
Calibration Started
-------------------
Move robot, THEN press key:

1 → Save P1 (origin)
2 → Save P2 (X direction)
3 → Save P3 (Y direction)

s → Compute & save
r → Reset calibration
q → Quit
""")

    def get_pen_tip_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "tool0",
                rclpy.time.Time()
            )

            t = transform.transform.translation
            tool0_pos = np.array([t.x, t.y, t.z])

            return tool0_pos + self.tcp_offset

        except Exception as e:
            self.get_logger().warn(f"TF not ready: {e}")
            return None

    def get_stable_position(self):
        prev = self.get_pen_tip_position()
        if prev is None:
            return None

        for _ in range(5):
            time.sleep(0.1)
            new = self.get_pen_tip_position()

            if new is None:
                continue

            if np.linalg.norm(new - prev) > 1e-5:
                return new

        return prev

    def loop(self):
        key = get_key()

        now = time.time()
        if now - self.last_key_time < self.debounce_time:
            return

        if key:
            self.last_key_time = now
            self.handle_key(key)

    def handle_key(self, key):

        if key in ["1", "2", "3"]:
            self.save_point()

        elif key == "s":
            self.compute_calibration()

        elif key == "r":
            self.reset_calibration()

        elif key == "q":
            self.get_logger().info("Exit calibration")
            self.destroy_node()
            rclpy.shutdown()

    def save_point(self):

        pos = self.get_stable_position()

        if pos is None:
            return

        # Prevent duplicates
        if len(self.points) > 0:
            if np.linalg.norm(pos - self.points[-1]) < 0.02:
                self.get_logger().warn("Point too close! Move further.")
                return

        if len(self.points) >= 3:
            self.get_logger().warn("Already have 3 points (press 'r')")
            return

        self.points.append(pos)

        label = f"P{len(self.points)}"
        self.get_logger().info(f"{label} saved: {pos}")

        # Show paper immediately after P3
        if len(self.points) == 3:
            self.visualize_paper()

    def visualize_paper(self):

        P1, P2, P3 = self.points

        x_axis = P2 - P1
        x_axis /= np.linalg.norm(x_axis)

        y_axis = P3 - P1
        y_axis /= np.linalg.norm(y_axis)

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        width = np.linalg.norm(P2 - P1)
        height = np.linalg.norm(P3 - P1)

        center = P1 + 0.5 * width * x_axis + 0.5 * height * y_axis

        R = np.column_stack((x_axis, y_axis, z_axis))
        q = self.rotation_matrix_to_quaternion(R)

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

        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.scale.x = width
        marker.scale.y = height
        marker.scale.z = 0.001

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.8

        self.marker_pub.publish(marker)

        self.get_logger().info("🟦 Paper displayed in RViz")

    def rotation_matrix_to_quaternion(self, R):
        q = np.zeros(4)
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2,1] - R[1,2]) * s
            q[1] = (R[0,2] - R[2,0]) * s
            q[2] = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                q[3] = (R[2,1] - R[1,2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0,1] + R[1,0]) / s
                q[2] = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                q[3] = (R[0,2] - R[2,0]) / s
                q[0] = (R[0,1] + R[1,0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                q[3] = (R[1,0] - R[0,1]) / s
                q[0] = (R[0,2] + R[2,0]) / s
                q[1] = (R[1,2] + R[2,1]) / s
                q[2] = 0.25 * s

        return q

    def reset_calibration(self):
        self.points.clear()
        self.last_key_time = 0

        # delete marker
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.action = Marker.DELETE
        marker.id = 0
        self.marker_pub.publish(marker)

        self.get_logger().info("Calibration reset")
        self.show_menu()

    def compute_calibration(self):

        if len(self.points) < 3:
            self.get_logger().error("Need 3 points first!")
            return

        P1, P2, P3 = self.points

        x_axis = (P2 - P1) / np.linalg.norm(P2 - P1)
        y_axis = (P3 - P1) / np.linalg.norm(P3 - P1)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        width = np.linalg.norm(P2 - P1)
        height = np.linalg.norm(P3 - P1)

        data = {
            "origin": P1.tolist(),
            "x_axis": x_axis.tolist(),
            "y_axis": y_axis.tolist(),
            "z_axis": z_axis.tolist(),
            "width": float(width),
            "height": float(height)
        }

        with open("paper_calibration.json", "w") as f:
            json.dump(data, f, indent=4)

        self.get_logger().info("Calibration saved!")


def main():
    rclpy.init()
    node = CalibrationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()