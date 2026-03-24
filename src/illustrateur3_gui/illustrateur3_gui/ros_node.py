import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.node import Node


class GuiNode(Node):
    def __init__(self):
        super().__init__("illustrateur3_gui_node")

        self.current_state = "IDLE"
        self.state_callback_fn = None
        self.camera_callback_fn = None
        self.preview_callback_fn = None

        self.state_sub = self.create_subscription(
            String,
            "/state",
            self.state_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            "/gesture/debug_image",
            self.camera_image_callback,
            10
        )

        self.preview_sub = self.create_subscription(
            Image,
            "/preview/image",
            self.preview_image_callback,
            10
        )

        self.get_logger().info("GUI node started")
        self.get_logger().info("Subscribed to /state")
        self.get_logger().info("Subscribed to /gesture/debug_image")
        self.get_logger().info("Subscribed to /preview/image")

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

    def rosimg_to_bgr(self, msg: Image):
        enc = msg.encoding.lower()

        if enc not in ("bgr8", "rgb8"):
            raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

        if enc == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame