
from asyncio.log import logger

from geometry_msgs import msg
import rclpy
from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
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
        self.marker_pub = self.create_publisher(Marker, "paper_marker", 10)
        self.status_pub = self.create_publisher(String, "/drawing/status", 10)
        self.stroke_queue = []
        self.create_subscription(Path,"/portrait/strokes", self.pen_path_callback,10)
    
        # Run the drawing function/other functions
        self.go_home()
        time.sleep(2)
        self.draw_rectangle()
        time.sleep(5)
        self.draw_portrait()

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
            # self.visualize_rectangle(drawn_path)
            self.moveit2.move_to_pose(
                position=real_point.tolist(),
                quat_xyzw=quat,
                cartesian=True
            )
            self.moveit2.wait_until_executed()
            self.visualize_rectangle(drawn_path)
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
    def go_home(self):
    #     self.moveit2.move_to_pose(
    #     position=[0.298, 0.113, 0.312],
    #     quat_xyzw=self.fixed_orientation,
    #     cartesian=True
    # )
        self.moveit2.move_to_configuration([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.moveit2.wait_until_executed()
    # Visualize the stroke points in RViz
    def visualize_stroke(self, msg, P1, width, height, x_axis, y_axis):

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "stroke_line"
        marker.id = 30
        marker.type = Marker.LINE_STRIP   # ✅ CHANGE HERE
        marker.action = Marker.ADD

        marker.scale.x = 0.003

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        image_width = 1920
        image_height = 1080

        for pose in msg.poses:
            u = pose.pose.position.x / image_width
            v = pose.pose.position.y / image_height

            point = P1 + u * width * x_axis + v * height * y_axis

            pt = Point()
            pt.x = float(point[0])
            pt.y = float(point[1])
            pt.z = float(point[2])

            marker.points.append(pt)

        self.marker_pub.publish(marker)
        
    def visualize_start_point(self, point):

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "start_point"
        marker.id = 21
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = float(point[2])

        self.marker_pub.publish(marker)
    # Callback function for pen path subscriber
    def pen_path_callback(self, msg: Path):
        # self.get_logger().info(f"Received stroke with {len(msg.poses)} points")
        self.stroke_queue.append(msg)
    def draw_portrait(self):

        self.get_logger().info(f"🔥 START DRAWING {len(self.stroke_queue)} strokes")

        while len(self.stroke_queue) > 0:
            msg = self.stroke_queue.pop(0)
            self.draw_stroke(msg)
        self.get_logger().info("✅ PORTRAIT DONE")

    def draw_stroke(self, msg: Path):
        P1, P2, P3, width, height, x_axis, y_axis, z_axis, quat = self.load_calibration()
        image_width = 1920
        image_height = 1080
        tcp_offset = 0.12
        lift_height = 0.02
        if len(msg.poses) == 0:
            return
        # -----------------------------
        # 1. MOVE ABOVE FIRST POINT
        # -----------------------------
        first = msg.poses[0].pose.position
        self.get_logger().info(f"Moving above first point: ({first.x}, {first.y})")
        u = first.x / image_width
        v = first.y / image_height
        self.get_logger().info(f"Moving above first point after scaling: ({u}, {v})")
        start_point = P1 + u * width * x_axis + v * height * y_axis

        if z_axis[2] < 0:
            start_up = start_point - (tcp_offset + lift_height) * z_axis
        else:
            start_up = start_point + (tcp_offset + lift_height) * z_axis
        self.get_logger().info(f"Moving above first point in real world: ({start_up[0]}, {start_up[1]}, {start_up[2]})")
        self.visualize_start_point(start_up)
        self.get_logger().info("Started point visualization")    
        self.moveit2.move_to_pose(
            position=start_up.tolist(),
            quat_xyzw=quat,
            cartesian=True
        )
        self.moveit2.wait_until_executed()
        self.get_logger().info("Reached above first point")

        # -----------------------------
        # 2. PEN DOWN
        # -----------------------------
        if z_axis[2] < 0:
            start_down = start_point - tcp_offset * z_axis
        else:
            start_down = start_point + tcp_offset * z_axis

        self.moveit2.move_to_pose(
            position=start_down.tolist(),
            quat_xyzw=quat,
            cartesian=True
        )
        self.moveit2.wait_until_executed()

        # -----------------------------
        # 3. DRAW CONTINUOUSLY
        # -----------------------------
        for pose in msg.poses:

            pixel_x = pose.pose.position.x
            pixel_y = pose.pose.position.y

            u = pixel_x / image_width
            v = pixel_y / image_height

            point = P1 + u * width * x_axis + v * height * y_axis

            if z_axis[2] < 0:
                real_point = point - tcp_offset * z_axis
            else:
                real_point = point + tcp_offset * z_axis

            self.moveit2.move_to_pose(
                position=real_point.tolist(),
                quat_xyzw=quat,
                cartesian=True
            )
            self.moveit2.wait_until_executed()
            self.visualize_stroke(msg, P1, width, height, x_axis, y_axis)

        # # -----------------------------
        # # 4. PEN UP AFTER STROKE
        # # -----------------------------
        last_point = point

        if z_axis[2] < 0:
            end_up = last_point - (tcp_offset + lift_height) * z_axis
        else:
            end_up = last_point + (tcp_offset + lift_height) * z_axis

        self.moveit2.move_to_pose(
            position=end_up.tolist(),
            quat_xyzw=quat,
            cartesian=True
        )
        self.moveit2.wait_until_executed()



def main():

    rclpy.init()

    node = MotionNode()

    rclpy.spin(node)

    rclpy.shutdown()