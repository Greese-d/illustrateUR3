import json
import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.node import Node


class GuiNode(Node):
    def __init__(self):
        super().__init__("illustrateur3_gui_node")

        self.current_state = "IDLE"
        self.freedrive_on = False

        self.state_callback_fn = None
        self.camera_callback_fn = None
        self.preview_callback_fn = None
        self.calibration_status_callback_fn = None

        self.calibration_pub = self.create_publisher(
            String, "/calibration/command", 10
        )

        self.calibration_status_sub = self.create_subscription(
            String,
            "/calibration/status",
            self.calibration_status_callback,
            10,
        )

        self.state_sub = self.create_subscription(
            String,
            "/state",
            self.state_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.camera_image_callback,
            10
        )

        self.preview_sub = self.create_subscription(
            Image,
            "/portrait/preview",
            self.preview_image_callback,
            10
        )

        self.get_logger().info("GUI node started")
        self.get_logger().info("Subscribed to /state")
        self.get_logger().info("Subscribed to /gesture/debug_image")
        self.get_logger().info("Subscribed to /preview/image")
        self.get_logger().info("Publishing to /calibration/command")
        self.get_logger().info("Subscribed to /calibration/status")

    def state_callback(self, msg):
        self.current_state = msg.data
        self.get_logger().info(f"Received state: {msg.data}")

        if self.state_callback_fn is not None:
            self.state_callback_fn(msg.data)

    def camera_image_callback(self, msg):
        try:
            frame_bgr = self.rosimg_to_bgr(msg)
            if self.camera_callback_fn is not None:
                self.camera_callback_fn(frame_bgr)
        except Exception as e:
            self.get_logger().warn(f"Failed to render camera image: {e}")

    def preview_image_callback(self, msg):
        try:
            frame_bgr = self.rosimg_to_bgr(msg)
            if self.preview_callback_fn is not None:
                self.preview_callback_fn(frame_bgr)
        except Exception as e:
            self.get_logger().warn(f"Failed to render preview image: {e}")

    def calibration_status_callback(self, msg):
        self.get_logger().info(f"Calibration status: {msg.data}")

        try:
            data = json.loads(msg.data)

            if "freedrive_on" in data:
                self.freedrive_on = data["freedrive_on"]

        except Exception as e:
            self.get_logger().warn(f"Failed to parse calibration status JSON: {e}")

        if self.calibration_status_callback_fn is not None:
            self.calibration_status_callback_fn(msg.data)

    def send_calibration_command(self, command: str):
        msg = String()
        msg.data = command
        self.calibration_pub.publish(msg)
        self.get_logger().info(f"Published calibration command: {command}")

    def set_point_1(self):
        self.send_calibration_command("set_p1")

    def set_point_2(self):
        self.send_calibration_command("set_p2")

    def set_point_3(self):
        self.send_calibration_command("set_p3")

    def on_confirm(self):
        self.ros_node.send_calibration_command("confirm")
        
    def toggle_freedrive(self):
        self.send_calibration_command("toggle_freedrive")

    def rosimg_to_bgr(self, msg: Image):
        enc = msg.encoding.lower()

        if enc not in ("bgr8", "rgb8"):
            raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

        if enc == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame