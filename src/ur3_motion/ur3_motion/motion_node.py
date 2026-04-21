
from asyncio.log import logger
import threading
from std_msgs.msg import String
from std_srvs.srv import Trigger

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
        self.stroke_id = 0
        self.create_subscription(Path,"/portrait/strokes", self.pen_path_callback,10)
        # Run the drawing function/other functions
        self.is_drawing = False
        self.drawing_requested = False
        self.inactivity_timer = None  # Timer to detect end of stroke messages
        self.strokes_reported = False  # Flag to report total strokes only once per batch
        self.start_drawing_srv = self.create_service(
            Trigger,
            "/start_drawing",
            self.handle_start_drawing
        )
        self.clear_strokes_srv = self.create_service(
            Trigger,
            "/clear_strokes",
            self.handle_clear_strokes
        )
        self.visualize_paper(show_axes=False)
        self.get_logger().info("Motion node waiting for GUI commands...")
        self.status_pub = self.create_publisher(String, "/drawing/status", 10)
        self.state_pub = self.create_publisher(String, "/state", 10)

    def wait_for_motion(self):
        return self.moveit2.wait_until_executed()
#-----------------------1. Load Calibration Data and Draw rectangle frame on paper--------------------------------
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
        center = np.array(data["center"])

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
    
        return P1, P2, P3, width, height, center, x_axis, y_axis, z_axis, quat
#----------------------- 2. Draw Rectange for checking calibration accuracy--------------------------------------
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
        P1, P2, P3, width, height, center, x_axis, y_axis, z_axis, quat = self.load_calibration()
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
            self.wait_for_motion()
            self.visualize_rectangle(drawn_path)
        self.get_logger().info("Rectangle done")
#--------------------------------------- 3. Visualization-----------------------------------------------
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

    def visualize_paper(self, show_axes= False):
        P1, P2, P3, width, height, center, x_axis, y_axis, z_axis, quat = self.load_calibration()
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "paper"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker.scale.x = width
        marker.scale.y = height
        marker.scale.z = 0.001

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)
        self.get_logger().info("Paper displayed correctly")
        
        # ===== AXES =====
        if (show_axes==True):
            axes = [
                (x_axis, (1, 0, 0), 1),
                (y_axis, (0, 1, 0), 2),
                (z_axis, (0, 0, 1), 3),
            ]
            for axis, color, mid in axes:
                m = Marker()
                m.header.frame_id = "base_link"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "axes"
                m.id = mid
                m.type = Marker.ARROW
                m.action = Marker.ADD
                # m.lifetime = Duration(sec=0)
            
                m.points.append(self.to_point(center))
                m.points.append(self.to_point(center + axis * 0.3))

                m.scale.x = 0.03
                m.scale.y = 0.06
                m.scale.z = 0.1

                m.color.r = float(color[0])
                m.color.g = float(color[1])
                m.color.b = float(color[2])
                m.color.a = 1.0
                self.marker_pub.publish(m)
            self.get_logger().info("✅ Paper + XYZ axes shown")

    def to_point(self, p):
        pt = Point()
        pt.x = float(p[0])
        pt.y = float(p[1])
        pt.z = float(p[2])
        return pt
    
    # Visualize the stroke points in RViz
    def visualize_stroke_path(self, path):

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "stroke_line"
        marker.id = self.stroke_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.003

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for p in path:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
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

#------------------------------4. Stroke Subcriber, Start Drawing Portrait---------------------------------------
    def handle_clear_strokes(self, request, response):
        if self.is_drawing:
            response.success = False
            response.message = "Cannot clear strokes while drawing"
            return response

        self.stroke_queue.clear()
        self.stroke_id = 0
        self.get_logger().info("Cleared portrait strokes")

        response.success = True
        response.message = "Portrait strokes cleared"
        return response
    def pen_path_callback(self, msg: Path):
        if len(self.stroke_queue) == 0:
            self.strokes_reported = False  # Reset flag for new batch
        self.get_logger().info(f"Received stroke with {len(msg.poses)} points")
        self.stroke_queue.append(msg)
        # Reset inactivity timer
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
        self.inactivity_timer = threading.Timer(0.1, self.report_stroke_count)
        self.inactivity_timer.start()
    def report_stroke_count(self):
        if not self.strokes_reported:
            self.get_logger().info(f"Total strokes received: {len(self.stroke_queue)}")
            self.strokes_reported = True
    def draw_portrait(self):

        self.get_logger().info(f"START DRAWING {len(self.stroke_queue)} strokes")

        while len(self.stroke_queue) > 0:
            msg = self.stroke_queue.pop(0)
            self.draw_stroke(msg)
        self.get_logger().info("PORTRAIT DONE")
    def draw_stroke(self, msg: Path):
        P1, P2, P3, width, height, center, x_axis, y_axis, z_axis, quat = self.load_calibration()
        image_width = 1920
        image_height = 1080
        tcp_offset = 0.12
        lift_height = 0.02
        scale = 1.5
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
        start_point = center + scale*(u-0.25) * width * x_axis + scale*(v-0.25) * height * y_axis 

        if z_axis[2] < 0:
            start_up = start_point - (tcp_offset + lift_height) * z_axis
        else:
            start_up = start_point + (tcp_offset + lift_height) * z_axis
        self.get_logger().info(f"Moving above first point in real world: ({start_up[0]}, {start_up[1]}, {start_up[2]})")
        # self.visualize_start_point(start_up)
        self.get_logger().info("Started point visualization")    
        self.moveit2.move_to_pose(
            position=start_up.tolist(),
            quat_xyzw=quat,
            cartesian=True
        )
        self.wait_for_motion()
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
        self.wait_for_motion()

        # -----------------------------
        # 3. DRAW CONTINUOUSLY
        # -----------------------------
        drawn_path = []
        for pose in msg.poses:

            pixel_x = pose.pose.position.x
            pixel_y = pose.pose.position.y

            u = pixel_x / image_width
            v = pixel_y / image_height

            point = center + scale*(u-0.25) * width * x_axis + scale*(v-0.25) * height * y_axis 

            if z_axis[2] < 0:
                real_point = point - tcp_offset * z_axis
            else:
                real_point = point + tcp_offset * z_axis

            self.moveit2.move_to_pose(
                position=real_point.tolist(),
                quat_xyzw=quat,
                cartesian=True
            )
            self.wait_for_motion()
            drawn_path.append(point)
            self.visualize_stroke_path(drawn_path)
            self.stroke_id += 1

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
        self.wait_for_motion()
#---------------------------------------------5. Fundamental Functions-------------------------------
    def go_home(self):
    #     self.moveit2.move_to_pose(
    #     position=[0.298, 0.113, 0.312],
    #     quat_xyzw=self.fixed_orientation,
    #     cartesian=True
    # )
        self.moveit2.move_to_configuration([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.wait_for_motion()

    def handle_start_drawing(self, request, response):
        if self.is_drawing:
            response.success = False
            response.message = "Robot is already drawing"
            self.get_logger().warn(response.message)
            return response

        if self.drawing_requested:
            response.success = False
            response.message = "Drawing sequence is already queued"
            self.get_logger().warn(response.message)
            return response

        if len(self.stroke_queue) == 0:
            response.success = False
            response.message = "No portrait strokes available. Capture portrait first."
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info("Start drawing service called")
        self.drawing_requested = True

        response.success = True
        response.message = "Drawing sequence started"
        return response


    def start_drawing_sequence(self):
        self.is_drawing = True
        try:
            self.state_pub.publish(String(data="DRAWING"))
            self.status_pub.publish(String(data="Drawing sequence started"))

            self.go_home()
            time.sleep(1)

            self.draw_rectangle()
            time.sleep(1)

            self.draw_portrait()
            time.sleep(1)

            self.go_home()
            time.sleep(1)

            self.state_pub.publish(String(data="IDLE"))
            self.status_pub.publish(String(data="Drawing sequence completed"))

        except Exception as e:
            self.get_logger().error(f"Drawing sequence failed: {e}")
            self.state_pub.publish(String(data="ERROR"))
            self.status_pub.publish(String(data=f"Drawing failed: {e}"))

        finally:
            self.is_drawing = False
def main():

    rclpy.init()

    node = MotionNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.drawing_requested and not node.is_drawing:
                node.drawing_requested = False
                node.start_drawing_sequence()
    finally:
        if node.inactivity_timer:
            node.inactivity_timer.cancel()
        node.destroy_node()
        rclpy.shutdown()
