import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
# from ament_index_python.packages import get_package_share_directory
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur

import numpy as np
import json
import sys
import termios
import tty
import time
import os
from std_msgs.msg import String
# def get_key():
#     fd = sys.stdin.fileno()
#     old_settings = termios.tcgetattr(fd)
#     try:
#         tty.setraw(fd)
#         key = sys.stdin.read(1)
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#     return key


class CalibrationNode(Node):
    def __init__(self):
        super().__init__("calibration_node")

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
        )
        # self.moveit2.max_velocity = 0.1
        # self.moveit2.max_acceleration = 0.1

        # Get transformation data for new end effector (attached on pentip)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(String, "/calibration/command", self.command_callback, 10)
        self.status_pub = self.create_publisher(String, "/calibration/status", 10)
        self.tcp_offset = 0.17 # length of the pen ( from end-effector to pentip)

        # FIX: always fixed size (P1,P2,P3)
        self.points = [None, None, None]
        self.tool_points = [None, None, None]

        self.current_index = None
        self.preview_point = None
        self.load_existing_calibration()
        self.publish_tcp_offset_status("startup", recalculated=False)

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
            tool_position, pen_tip_position = self.get_tool_and_pen_tip_position()
            return pen_tip_position

        except Exception as e:
            self.get_logger().warn(f"TF not ready: {e}")
            return None

    def get_tool_and_pen_tip_position(self):
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

        return T[0:3, 3], T_new[0:3, 3]

    def command_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and data.get("command") == "set_tcp_offset":
            self.set_tcp_offset(data.get("tcp_offset"))
            return

        if msg.data in ("show_paper", "hide_paper", "show_axes", "hide_axes"):
            return

        elif msg.data == "set_p1":
            self.current_index = 0
            self.get_logger().info("Editing P1")

        elif msg.data == "set_p2":
            self.current_index = 1
            self.get_logger().info("Editing P2")

        elif msg.data == "set_p3":
            self.current_index = 2
            self.get_logger().info("Editing P3")

        elif msg.data == "toggle_freedrive":
            self.get_logger().info("Toggle freedrive (implement here)")

        elif msg.data == "confirm":
            if self.current_index is None:
                self.get_logger().warn("Select point first (1/2/3)")
                return

            try:
                tool_pos, pen_tip_pos = self.get_tool_and_pen_tip_position()
            except Exception as e:
                self.get_logger().warn(f"TF not ready: {e}")
                return

            # overwrite directly (NO append)
            self.tool_points[self.current_index] = tool_pos
            self.points[self.current_index] = pen_tip_pos

            self.get_logger().info(
                f"P{self.current_index+1} CONFIRMED: {pen_tip_pos}"
            )

            self.current_index = None

            # auto visualize
            if all(p is not None for p in self.points):
                self.get_paper_data()

               # preview continuously
        if self.current_index is not None:
            pos = self.get_pen_tip_position()
            if pos is not None:
                self.get_logger().info(
                    f"P{self.current_index+1} preview: {pos}"
                )

    # def reset_calibration(self):
    #     self.points = [None, None, None]
    #     self.current_index = None
    #     marker = Marker()
    #     marker.header.frame_id = "base_link"
    #     marker.action = Marker.DELETE
    #     marker.id = 0
    #     self.marker_pub.publish(marker)
    #     self.get_logger().info("Calibration reset okeeeeeeah ")
    #     self.show_menu()

    # After visualizing, also save the calibration to a JSON file for later use
    def get_workspace_data_path(self):
        # current file path (inside install or build)
        current_file = os.path.abspath(__file__)

        # go up until we find workspace root (has 'install' and 'src')
        path = current_file
        while path != "/":
            if os.path.exists(os.path.join(path, "src")) and \
            os.path.exists(os.path.join(path, "install")):
                return os.path.join(path, "data")
            path = os.path.dirname(path)

        # fallback (just in case)
        return os.path.expanduser("~/data")

    def get_calibration_json_path(self):
        data_dir = self.get_workspace_data_path()
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "paper_calibration.json")

    def load_calibration_file(self):
        json_path = self.get_calibration_json_path()
        with open(json_path, "r") as f:
            return json.load(f)

    def load_existing_calibration(self):
        try:
            data = self.load_calibration_file()
        except (FileNotFoundError, json.JSONDecodeError):
            return

        self.tcp_offset = float(data.get("tcp_offset", self.tcp_offset))

        if all(key in data for key in ("P1", "P2", "P3")):
            self.points = [
                np.array(data["P1"], dtype=float),
                np.array(data["P2"], dtype=float),
                np.array(data["P3"], dtype=float),
            ]

        tool_keys = ["tool_P1", "tool_P2", "tool_P3"]
        if all(key in data for key in tool_keys):
            self.tool_points = [np.array(data[key], dtype=float) for key in tool_keys]

    def publish_tcp_offset_status(self, source, recalculated=False, message=None):
        payload = {
            "command": "tcp_offset_status",
            "source": source,
            "tcp_offset": float(self.tcp_offset),
            "recalculated": bool(recalculated),
        }
        if message:
            payload["message"] = message
        self.status_pub.publish(String(data=json.dumps(payload)))

    def set_tcp_offset(self, tcp_offset):
        try:
            tcp_offset = float(tcp_offset)
        except (TypeError, ValueError):
            self.get_logger().warn(f"Invalid tcp_offset received: {tcp_offset}")
            self.publish_tcp_offset_status("gui", recalculated=False, message="Invalid TCP offset")
            return

        if tcp_offset <= 0.0:
            self.get_logger().warn("TCP offset must be positive")
            self.publish_tcp_offset_status("gui", recalculated=False, message="TCP offset must be positive")
            return

        self.tcp_offset = tcp_offset
        self.save_to_json(None, None, None, None, None, None, None, None, None)
        message = "TCP offset updated. Paper calibration unchanged."

        self.get_logger().info(message)
        self.publish_tcp_offset_status("gui", recalculated=False, message=message)

    def save_to_json(self, P1, P2, P3, width, height, center, quat, x_axis, y_axis, z_axis):
        
        json_path = self.get_calibration_json_path()

        try:
            data = self.load_calibration_file()
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        data["tcp_offset"] = float(self.tcp_offset)

        if all(point is not None for point in self.tool_points):
            data["tool_P1"] = self.tool_points[0].tolist()
            data["tool_P2"] = self.tool_points[1].tolist()
            data["tool_P3"] = self.tool_points[2].tolist()

        if all(value is not None for value in (P1, P2, P3, width, height, center, quat, x_axis, y_axis, z_axis)):
            data.update({
                "P1": P1.tolist(),
                "P2": P2.tolist(),
                "P3": P3.tolist(),
                "width": float(width),
                "height": float(height),
                "center": center.tolist(),
                "orientation": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
                "x_axis": x_axis.tolist(),
                "y_axis": y_axis.tolist(),
                "z_axis": z_axis.tolist()
            })

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        self.get_logger().info(f"Saved to {json_path}")
    
    # Visualize the paper as a marker in RViz based on the three points
    def get_paper_data(self):

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
        self.save_to_json(P1, P2, P3, width, height, center, quat, x_axis, y_axis, z_axis)


def main():
    rclpy.init()
    node = CalibrationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
