import rclpy
from rclpy.node import Node


class GuiNode(Node):
    def __init__(self):
        super().__init__("illustrateur3_gui_node")
        self.get_logger().info("GUI node started")