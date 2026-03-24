import tkinter as tk
import rclpy

from .ros_node import GuiNode
from .gui_app import GuiApp


def main(args=None):
    rclpy.init(args=args)

    ros_node = GuiNode()

    root = tk.Tk()
    app = GuiApp(root, ros_node)

    try:
        root.mainloop()
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()