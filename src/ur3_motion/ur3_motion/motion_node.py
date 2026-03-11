import rclpy
from rclpy.node import Node

from moveit_commander import MoveGroupCommander, roscpp_initialize


class MoveJointTest(Node):

    def __init__(self):

        super().__init__('move_joint_test')

        roscpp_initialize([])

        self.move_group = MoveGroupCommander("ur_manipulator")

        self.get_logger().info("Moving robot to test joint position")

        joint_goal = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

        self.move_group.go(joint_goal, wait=True)

        self.move_group.stop()

        self.get_logger().info("Movement finished")


def main():

    rclpy.init()

    node = MoveJointTest()

    rclpy.shutdown()