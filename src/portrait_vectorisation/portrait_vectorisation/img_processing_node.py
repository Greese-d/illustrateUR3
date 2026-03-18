import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_srvs.srv import Empty

import portrait_vectorisation.portrait_processor as pp
import cv2
from cv_bridge import CvBridge


class ImgProcessingNode(Node):

    def __init__(self):
        super().__init__('image_processing_node')

        # Store latest frame
        self.latest_image = None
        self.snapshot = None

        self.bridge = CvBridge()
        self.processor = pp.PortraitProcessor()

        # Subscriber to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for snapshot
        self.snapshot_pub = self.create_publisher(
            Image,
            '/camera/snapshot',
            10
        )

        self.portrait_pub = self.create_publisher(
            Image,
            '/portrait/preview',
            10
        )

        # Service to trigger snapshot
        self.service = self.create_service(
            Empty,
            '/capture_snapshot',
            self.capture_snapshot_callback
        )

        #Service to trigger portrait preview
        self.portrait_service = self.create_service(
            Empty,
            '/create_portrait',
            self.create_portrait_callback
        )

        self.get_logger().info('Image processing node ready.')

    def image_callback(self, msg):
        """Store the latest image from the camera."""
        self.latest_image = msg
        self.capture_snapshot_callback(None, None)  # Automatically update snapshot with latest image
        self.create_portrait_callback(None, None)  # Automatically update portrait preview with latest snapshot

    def capture_snapshot_callback(self, request, response):
        """Publish the latest image when service is called."""

        if self.latest_image is None:
            self.get_logger().warn('No image received yet.')
            return response

        self.snapshot = self.latest_image
        self.snapshot_pub.publish(self.snapshot)
        self.get_logger().info('Snapshot published.')

        return response
    
    def create_portrait_callback(self, request, response):
        """Publish the preview for portrait."""

        if self.snapshot is None:
            self.get_logger().warn('No snapshot captured yet. Creating snapshot first.')
            self.capture_snapshot_callback(request, response)

        snapshot_img = self.bridge.imgmsg_to_cv2(self.snapshot, desired_encoding='bgr8')
        portrait_msg = self.bridge.cv2_to_imgmsg(self.processor.process(snapshot_img), encoding='mono8')

        self.portrait_pub.publish(portrait_msg)
        self.get_logger().info('Portrait preview published.')

        return response



def main(args=None):
    rclpy.init(args=args)

    node = ImgProcessingNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()