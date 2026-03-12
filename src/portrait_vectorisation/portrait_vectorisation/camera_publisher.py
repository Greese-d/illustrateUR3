import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraPublisher(Node):
    def __init__(self):
        super().__init__("camera_publisher")

        self.declare_parameter("device", 1)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("topic", "/camera/image_raw")

        device = int(self.get_parameter("device").value)
        fps = float(self.get_parameter("fps").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        topic = str(self.get_parameter("topic").value)

        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        period = 1.0 / max(fps, 1.0)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"Publishing {width}x{height}@{fps} from device {device} on {topic}")

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Failed to read frame")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()