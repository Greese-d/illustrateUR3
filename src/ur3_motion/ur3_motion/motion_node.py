
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point

import numpy as np
import json
import sys
import termios
import tty
import time
import os
from visualization_msgs.msg import Marker
from visualization_msgs.msg import Marker
class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.get_logger().info("Motion node ready")
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.timer = self.create_timer(1.0, self.print_pose)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,   
        )
        self.moveit2.max_velocity = 0.05
        self.moveit2.max_acceleration = 0.05
        self.fixed_orientation = [0.0, 1.0, 0.0, 0.0]   
         # Create subscriber and publisher topic
        self.create_subscription(
            Path,
            "pen_path",
            self.pen_path_callback,
            10,
        )
        self.marker_pub = self.create_publisher(Marker, "paper_marker", 10)
        self.status_pub = self.create_publisher(String, "/drawing/status", 10)

        # Run the drawing function/other functions
        # self.go_home()
        # time.sleep(2)
        self.start_drawing()
       
    #------------Load Calibration Data and Draw rectangle frame on paper--------------------------------
    # Extract calibration data from json file (rs2_ws/data/paper_calibration.json)
    def load_calibration(self):
        workspace = os.getcwd()  # assumes running from ~/rs2_ws
        json_path = os.path.join(workspace, "data", "paper_calibration.json")

        with open(json_path, "r") as f:
            data = json.load(f)

        P1 = np.array(data["P1"])
        P2 = np.array(data["P2"])
        P3 = np.array(data["P3"])

        width = data["width"]
        height = data["height"]

        x_axis= np.array(data["x_axis"])
        y_axis= np.array(data["y_axis"])
        z_axis= np.array(data["z_axis"])

        orientation = data["orientation"]
        quat = [
            orientation["x"],
            orientation["y"],
            orientation["z"],
            orientation["w"]
            ]
    
        return P1, P2, P3, width, height, x_axis, y_axis, z_axis, quat
    def generate_rectangle(self, P1, x_axis, y_axis, width, height, offset=0.01):

        # rectangle dimensions
        w = width - 2 * offset
        h = height - 2 * offset

        # shift origin inside
        origin = P1 + offset * x_axis + offset * y_axis

        # corners
        p1 = origin
        p2 = origin + w * x_axis
        p3 = origin + w * x_axis + h * y_axis
        p4 = origin + h * y_axis

        return [p1, p2, p3, p4, p1]  # closed loop
    def draw_rectangle(self):

        P1, P2, P3, width, height, x_axis, y_axis, z_axis, quat = self.load_calibration()
        path = self.generate_rectangle(P1, x_axis, y_axis, width, height)
        self.get_logger().info("Drawing rectangle...")
        tcp_offset= 0.12
        drawn_path=[]
        for point in path:
            
            if(z_axis[2]<0):
                real_point = point - tcp_offset*z_axis
            else:
                real_point = point + tcp_offset*z_axis

            drawn_path.append(point)
            self.get_logger().info(f"Moving to point: {real_point}")
            self.visualize_rectangle(drawn_path)
            self.moveit2.move_to_pose(
                position=real_point.tolist(),
                quat_xyzw=quat,
                cartesian=True
            )
            self.moveit2.wait_until_executed()
            # if not success:
            #     self.get_logger().error(f"Failed to move to point: {real_point}")
            #     self.get_logger().error("Trying to use different orientation...")
            #     # Try with fixed orientation
            #     success = self.moveit2.move_to_pose(
            #         position=real_point.tolist(),
            #         quat_xyzw=self.fixed_orientation,
            #         cartesian=True
            #     )
            #     self.moveit2.wait_until_executed()
            #     if not success:
            #         self.get_logger().error(f"Failed to move to point: {real_point} with fixed orientation")
                # return
        self.get_logger().info("Rectangle done")

    def visualize_rectangle(self, path):

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "rectangle"
        marker.id = 10
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        from builtin_interfaces.msg import Duration
        marker.lifetime = Duration(sec=0)

        marker.scale.x = 0.005

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for p in path:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
            marker.points.append(pt)

        self.marker_pub.publish(marker)

    #----------------------------Fundamental Functions----------------------------------------------
    def start_drawing(self):
        self.draw_rectangle()
        
    
    def go_home(self):
    #     self.moveit2.move_to_pose(
    #     position=[0.298, 0.113, 0.312],
    #     quat_xyzw=self.fixed_orientation,
    #     cartesian=True
    # )
        self.moveit2.move_to_configuration([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.moveit2.wait_until_executed()

    def from_home_to_ready(self):
        self.moveit2.move_to_configuration([-4.71, -1.57, 1.57, -1.57, -1.57, 1.57])
    # Callback function for pen path subscriber
    def pen_path_callback(self, msg: Path):
        self.get_logger().info("Starting drawing...")
        # FIXED orientation (Z perpendicular to paper)
        for pose_stamped in msg.poses:
            pose = pose_stamped.pose
            position = [
                pose.position.x,
                pose.position.y,
                pose.position.z
            ]
            self.moveit2.move_to_pose(
                position=position,
                quat_xyzw=self.fixed_orientation,
                cartesian=True
            )
            self.moveit2.wait_until_executed()
        self.get_logger().info("Drawing completed")
        status_msg = String()
        status_msg.data = "completed"
        self.status_pub.publish(status_msg)


def main():

    rclpy.init()

    node = MotionNode()

    rclpy.spin(node)

    rclpy.shutdown()