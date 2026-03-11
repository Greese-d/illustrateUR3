from std_msgs.msg import String
from rclpy.node import Node


class GuiNode(Node):
    def __init__(self):
        super().__init__("illustrateur3_gui_node")

        self.current_state = "IDLE"
        self.state_callback_fn = None

        self.state_sub = self.create_subscription(
            String,
            "/state",
            self.state_callback,
            10
        )

        self.get_logger().info("GUI node started")
        self.get_logger().info("Subscribed to /state")

    def state_callback(self, msg):
        self.current_state = msg.data
        self.get_logger().info(f"Received state: {msg.data}")

        if self.state_callback_fn is not None:
            self.state_callback_fn(msg.data)