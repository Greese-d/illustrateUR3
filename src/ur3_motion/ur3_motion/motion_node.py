import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class MotionNode(Node):

    def __init__(self):

        super().__init__("motion_node")

        # Publisher to UR trajectory controller
        self.publisher = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10
        )

        # Wait a bit before sending motion
        self.timer = self.create_timer(2.0, self.move_robot)

        self.get_logger().info("Motion node started. Waiting before sending trajectory...")

    def move_robot(self):

        traj = JointTrajectory()

        traj.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        state = JointTrajectoryPoint()

        # Joint positions (radians)
        state.positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

        # Movement duration
        state.time_from_start.sec = 3

        traj.points.append(state)

        self.publisher.publish(traj)

        self.get_logger().info("Trajectory sent!")

        # Stop repeating
        self.timer.cancel()


def main(args=None):

    rclpy.init(args=args)

    node = MotionNode()

    rclpy.spin(node)

    rclpy.shutdown()