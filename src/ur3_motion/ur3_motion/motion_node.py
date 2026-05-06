
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
from scipy.spatial.transform import Rotation as R

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
        # Initialize the motion node and set up MoveIt2, TF listener, publishers, subscribers, and service servers.
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
        self.tcp_offset = 0.17 # length of the pen ( from end-effector to pentip when pen is interacting with paper, i.e. pointing downwards)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
         # Create subscriber and publisher topic
        self.marker_pub = self.create_publisher(Marker, "paper_marker", 10)
        self.status_pub = self.create_publisher(String, "/drawing/status", 10)
        self.urscript_pub = self.create_publisher(String, "/urscript_interface/script_command", 10)
        self.stroke_queue = []
        self.stroke_id = 0
        self.create_subscription(Path,"/portrait/strokes", self.pen_path_callback,10)
        self.create_subscription(String, "/calibration/command", self.display_command_callback, 10)
        # State variables
        self.is_drawing = False
        self.drawing_requested = False
        self.stop_requested = False
        self.go_home_requested = False
        self.utility_motion_requested = None
        self.portrait_in_progress = False
        self.portrait_mapping = None
        self.show_paper = True
        self.show_axes = False
        # pencolor changing and pen docking parameters
        self.pen_calibration_data= "pen_storage_calibration.json"
        self.pen_ready_to_attach_pos = 0.20 # this is the fixed distance from the bottom of the pen storage to ready position for rotate  and attach (lock the pen)
        # Timer and flags for stroke reception
        self.inactivity_timer = None  # Timer to detect end of stroke messages
        self.strokes_reported = False  # Flag to report total strokes only once per batch
        # Service servers for drawing control
        self.start_drawing_srv = self.create_service(
            Trigger,
            "/start_drawing",
            self.handle_start_drawing
        )
        self.stop_drawing_srv = self.create_service(
            Trigger,
            "/stop_drawing",
            self.handle_stop_drawing
        )
        self.go_home_srv = self.create_service(
            Trigger,
            "/go_home",
            self.handle_go_home
        )
        self.clear_strokes_srv = self.create_service(
            Trigger,
            "/clear_strokes",
            self.handle_clear_strokes
        )
        self.update_paper_display()
        self.distance_timer = self.create_timer(0.2, self.update_distance_markers)
        self.get_logger().info("Motion node waiting for GUI commands...")
        self.status_pub = self.create_publisher(String, "/drawing/status", 10)
        self.state_pub = self.create_publisher(String, "/state", 10)
#----------------------------- 0. Service Handlers for Drawing Control --------------------------------
    def wait_for_motion(self):
        return self.moveit2.wait_until_executed()

    def stop_was_requested(self):
        return self.stop_requested

    def remaining_path_from(self, msg: Path, start_index: int):
        remaining = Path()
        remaining.header = msg.header
        remaining.poses = list(msg.poses[start_index:])
        return remaining

    def transform_to_matrix(self, t, q):
        T = np.eye(4)
        T[0:3, 3] = [t.x, t.y, t.z]
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        T[0:3, 0:3] = rot.as_matrix()
        return T

    def apply_tcp_offset(self, T):
        T_offset = np.eye(4)
        T_offset[2, 3] = self.tcp_offset
        return T @ T_offset

    def get_tool_and_pen_tip_positions(self):
        transform = self.tf_buffer.lookup_transform(
            "base_link",
            "tool0",
            rclpy.time.Time()
        )

        T_tool = self.transform_to_matrix(
            transform.transform.translation,
            transform.transform.rotation
        )
        T_tip = self.apply_tcp_offset(T_tool)
        return T_tool[0:3, 3], T_tip[0:3, 3]

    def point_to_plane_projection(self, point, plane_point, plane_normal):
        signed_distance = float(np.dot(point - plane_point, plane_normal))
        projected_point = point - signed_distance * plane_normal
        return signed_distance, projected_point

    def tool_distance_to_ground(self, tool_position):
        signed_distance = float(tool_position[2])
        projected_point = np.array([tool_position[0], tool_position[1], 0.0])
        return signed_distance, projected_point
#-----------------------1. Load Calibration Data and Draw rectangle frame on paper--------------------------------
    # Extract calibration data from json file (rs2_ws/data/paper_calibration.json)
    def load_calibration(self):
        workspace = os.getcwd()  # assumes running from ~/rs2_ws
        json_path = os.path.join(workspace, "data", "paper_calibration.json")

        with open(json_path, "r") as f:
            data = json.load(f)

        self.tcp_offset = float(data.get("tcp_offset", self.tcp_offset))

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

    def save_tcp_offset_to_calibration(self):
        workspace = os.getcwd()  # assumes running from ~/rs2_ws
        json_path = os.path.join(workspace, "data", "paper_calibration.json")

        with open(json_path, "r") as f:
            data = json.load(f)

        data["tcp_offset"] = float(self.tcp_offset)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
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
        drawn_path=[]
        for point in path:
            if self.stop_was_requested():
                self.get_logger().info("Rectangle drawing stopped")
                return False
            
            if(z_axis[2]<0):
                real_point = point - self.tcp_offset*z_axis
            else:
                real_point = point + self.tcp_offset*z_axis

            drawn_path.append(point)
            self.get_logger().info(f"Moving to point: {real_point}")
            # self.visualize_rectangle(drawn_path)
            self.moveit2.move_to_pose(
                position=real_point.tolist(),
                quat_xyzw=quat,
                cartesian=True
            )
            if not self.wait_for_motion() or self.stop_was_requested():
                self.get_logger().info("Rectangle drawing stopped")
                return False
            self.visualize_rectangle(drawn_path)
        self.get_logger().info("Rectangle done")
        return True
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

    def display_command_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and data.get("command") == "set_tcp_offset":
            tcp_offset = data.get("tcp_offset")
            try:
                self.tcp_offset = float(tcp_offset)
                self.save_tcp_offset_to_calibration()
                self.get_logger().info(f"Updated TCP offset from GUI: {self.tcp_offset:.4f} m")
            except (TypeError, ValueError):
                self.get_logger().warn(f"Ignoring invalid TCP offset from GUI: {tcp_offset}")
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                self.get_logger().warn(f"Updated TCP offset in memory, but could not save calibration file: {e}")
            return

        utility_commands = ("move_vertical", "rotate_end_effector", "attach_pen", "detach_pen")
        if isinstance(data, dict) and data.get("command") in utility_commands:
            if self.is_drawing or self.drawing_requested or self.go_home_requested or self.utility_motion_requested:
                self.get_logger().warn("Cannot start utility motion while another motion is active or queued")
                return

            self.stop_requested = False
            self.utility_motion_requested = data
            self.get_logger().info(f"Queued utility motion: {data}")
            return

        if msg.data == "show_paper":
            self.show_paper = True
            self.update_paper_display()
            self.get_logger().info("Paper display enabled")
        elif msg.data == "hide_paper":
            self.show_paper = False
            self.update_paper_display()
            self.get_logger().info("Paper display disabled")
        elif msg.data == "show_axes":
            self.show_axes = True
            self.update_paper_display()
            self.get_logger().info("Paper axes display enabled")
        elif msg.data == "hide_axes":
            self.show_axes = False
            self.update_paper_display()
            self.get_logger().info("Paper axes display disabled")

    def delete_marker(self, namespace, marker_id):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    def delete_paper_markers(self):
        self.delete_marker("paper", 0)

    def delete_axis_markers(self):
        for marker_id in (1, 2, 3):
            self.delete_marker("axes", marker_id)

    def delete_distance_markers(self):
        for marker_id in (100, 101, 102, 103):
            self.delete_marker("distance", marker_id)

    def update_paper_display(self):
        try:
            if self.show_paper or self.show_axes:
                self.visualize_paper(show_paper=self.show_paper, show_axes=self.show_axes)
            else:
                self.delete_paper_markers()
                self.delete_axis_markers()
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            self.delete_paper_markers()
            self.delete_axis_markers()
            self.get_logger().warn(f"Cannot display paper/axes: calibration data is not ready ({e})")

    def create_distance_line_marker(self, marker_id, start, end, color):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "distance"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [self.to_point(start), self.to_point(end)]
        marker.scale.x = 0.008
        marker.scale.y = 0.015
        marker.scale.z = 0.02
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0
        return marker

    def create_distance_text_marker(self, marker_id, position, text, color):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "distance"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = self.to_point(position)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.018
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0
        marker.text = text
        return marker

    def update_distance_markers(self):
        try:
            P1, _, _, _, _, _, _, _, z_axis, _ = self.load_calibration()
            tool_position, _ = self.get_tool_and_pen_tip_positions()
        except Exception:
            self.delete_distance_markers()
            return

        z_axis = z_axis / np.linalg.norm(z_axis)

        paper_distance, paper_projected = self.point_to_plane_projection(tool_position, P1, z_axis)
        ground_distance, ground_projected = self.tool_distance_to_ground(tool_position)

        text_anchor = 0.5 * (tool_position + ground_projected) + np.array([-0.28, 0.0, 0.02])
        text_gap = np.array([0.0, 0.0, 0.018])
        paper_text_position = text_anchor + text_gap
        ground_text_position = text_anchor

        markers = [
            self.create_distance_line_marker(100, tool_position, ground_projected, (1.0, 0.0, 0.0)),
            self.create_distance_text_marker(101, ground_text_position, f"Ground:{abs(ground_distance):.3f}m", (1.0, 0.0, 0.0)),
            self.create_distance_line_marker(102, tool_position, paper_projected, (0.0, 0.7, 1.0)),
            self.create_distance_text_marker(103, paper_text_position, f"Paper:{abs(paper_distance):.3f}m", (0.0, 0.7, 1.0)),
        ]

        for marker in markers:
            self.marker_pub.publish(marker)

    def visualize_paper(self, show_paper=True, show_axes=False):
        P1, P2, P3, width, height, center, x_axis, y_axis, z_axis, quat = self.load_calibration()

        if show_paper:
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
        else:
            self.delete_paper_markers()
        
        # ===== AXES =====
        if show_axes:
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
        else:
            self.delete_axis_markers()

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
        self.stop_requested = False
        self.portrait_in_progress = False
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
        self.portrait_mapping = self.compute_portrait_mapping(self.stroke_queue)
        self.portrait_in_progress = True

        while len(self.stroke_queue) > 0:
            if self.stop_was_requested():
                self.get_logger().info(
                    f"Portrait drawing stopped with {len(self.stroke_queue)} stroke(s) remaining"
                )
                return False
            msg = self.stroke_queue.pop(0)
            remaining = self.draw_stroke(msg)
            if remaining is not None:
                if len(remaining.poses) > 0:
                    self.stroke_queue.insert(0, remaining)
                self.get_logger().info(
                    f"Portrait drawing stopped with {len(self.stroke_queue)} stroke(s) remaining"
                )
                return False
        self.portrait_mapping = None
        self.portrait_in_progress = False
        self.get_logger().info("PORTRAIT DONE")
        return True

    def compute_portrait_mapping(self, strokes):
        P1, P2, P3, width, height, center, x_axis, y_axis, z_axis, quat = self.load_calibration()
        portrait_scale = 0.9
        rotation_degrees = 90.0
        points = [
            (pose.pose.position.x, pose.pose.position.y)
            for stroke in strokes
            for pose in stroke.poses
        ]

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        portrait_width = max(max_x - min_x, 1.0)
        portrait_height = max(max_y - min_y, 1.0)

        # Keep the portrait inside the rectangle drawn 1 cm in from the paper edge,
        # with a little extra padding so pen thickness/calibration error does not cross it.
        margin = min(0.015, width * 0.20, height * 0.20)
        drawable_width = max(width - 2.0 * margin, width * 0.50)
        drawable_height = max(height - 2.0 * margin, height * 0.50)
        metres_per_pixel = min(
            drawable_width / portrait_width,
            drawable_height / portrait_height,
        ) * portrait_scale
        rotation_radians = np.deg2rad(rotation_degrees)
        rotation = np.array([
            [np.cos(rotation_radians), -np.sin(rotation_radians)],
            [np.sin(rotation_radians), np.cos(rotation_radians)],
        ])

        bbox_center_x = 0.5 * (min_x + max_x)
        bbox_center_y = 0.5 * (min_y + max_y)

        self.get_logger().info(
            "Portrait fit: "
            f"pixel_bbox=({portrait_width:.1f} x {portrait_height:.1f}), "
            f"drawable=({drawable_width:.3f}m x {drawable_height:.3f}m), "
            f"scale={metres_per_pixel:.6f}m/px, "
            f"rotation={rotation_degrees:.1f}deg"
        )

        return {
            "center": center,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "pixel_center": np.array([bbox_center_x, bbox_center_y], dtype=float),
            "metres_per_pixel": metres_per_pixel,
            "rotation": rotation,
        }

    def pixel_to_paper_point(self, pixel_x, pixel_y, mapping):
        pixel = np.array([pixel_x, pixel_y], dtype=float)
        delta = mapping["rotation"] @ (pixel - mapping["pixel_center"])
        return (
            mapping["center"]
            + delta[0] * mapping["metres_per_pixel"] * mapping["x_axis"]
            + delta[1] * mapping["metres_per_pixel"] * mapping["y_axis"]
        )

    def draw_stroke(self, msg: Path):
        P1, P2, P3, width, height, center, x_axis, y_axis, z_axis, quat = self.load_calibration()
        lift_height = 0.02
        if len(msg.poses) == 0:
            return None
        if self.stop_was_requested():
            return msg
        mapping = self.portrait_mapping or self.compute_portrait_mapping([msg])
        # -----------------------------
        # 1. MOVE ABOVE FIRST POINT
        # -----------------------------
        first = msg.poses[0].pose.position
        self.get_logger().info(f"Moving above first point: ({first.x}, {first.y})")
        start_point = self.pixel_to_paper_point(first.x, first.y, mapping)

        if z_axis[2] < 0:
            start_up = start_point - (self.tcp_offset + lift_height) * z_axis
        else:
            start_up = start_point + (self.tcp_offset + lift_height) * z_axis
        self.get_logger().info(f"Moving above first point in real world: ({start_up[0]}, {start_up[1]}, {start_up[2]})")
        # self.visualize_start_point(start_up)
        self.get_logger().info("Started point visualization")    
        self.moveit2.move_to_pose(
            position=start_up.tolist(),
            quat_xyzw=quat,
            cartesian=True
        )
        if not self.wait_for_motion() or self.stop_was_requested():
            return msg
        self.get_logger().info("Reached above first point")

        # -----------------------------
        # 2. PEN DOWN
        # -----------------------------
        if z_axis[2] < 0:
            start_down = start_point - self.tcp_offset * z_axis
        else:
            start_down = start_point + self.tcp_offset * z_axis

        self.moveit2.move_to_pose(
            position=start_down.tolist(),
            quat_xyzw=quat,
            cartesian=True
        )
        if not self.wait_for_motion() or self.stop_was_requested():
            return msg

        # -----------------------------
        # 3. DRAW CONTINUOUSLY
        # -----------------------------
        drawn_path = []
        for pose_index, pose in enumerate(msg.poses):
            if self.stop_was_requested():
                return self.remaining_path_from(msg, pose_index)

            pixel_x = pose.pose.position.x
            pixel_y = pose.pose.position.y

            point = self.pixel_to_paper_point(pixel_x, pixel_y, mapping)

            if z_axis[2] < 0:
                real_point = point - self.tcp_offset * z_axis
            else:
                real_point = point + self.tcp_offset * z_axis

            self.moveit2.move_to_pose(
                position=real_point.tolist(),
                quat_xyzw=quat,
                cartesian=True
            )
            if not self.wait_for_motion() or self.stop_was_requested():
                return self.remaining_path_from(msg, pose_index)
            drawn_path.append(point)
            self.visualize_stroke_path(drawn_path)
            self.stroke_id += 1

        # # -----------------------------
        # # 4. PEN UP AFTER STROKE
        # # -----------------------------
        last_point = point

        if z_axis[2] < 0:
            end_up = last_point - (self.tcp_offset + lift_height) * z_axis
        else:
            end_up = last_point + (self.tcp_offset + lift_height) * z_axis

        self.moveit2.move_to_pose(
            position=end_up.tolist(),
            quat_xyzw=quat,
            cartesian=True
        )
        if not self.wait_for_motion() or self.stop_was_requested():
            return Path()
        return None
#---------------------------------------------5. Fundamental Functions (Home, Stop, Start, Resume)-------------------------------
    def go_home(self):
    #     self.moveit2.move_to_pose(
    #     position=[0.298, 0.113, 0.312],
    #     quat_xyzw=self.fixed_orientation,
    #     cartesian=True
    # )
        self.moveit2.move_to_configuration([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
        if not self.wait_for_motion():
            if self.stop_was_requested():
                return False
            raise RuntimeError("MoveIt planned the home motion, but execution did not complete")
        return True

    def handle_go_home(self, request, response):
        if self.is_drawing or self.drawing_requested or self.go_home_requested:
            response.success = False
            response.message = "Cannot go home while robot is drawing or another motion is queued"
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info("Go home service called")
        self.go_home_requested = True
        response.success = True
        response.message = "Go home sequence started"
        return response

    def handle_stop_drawing(self, request, response):
        if not self.is_drawing and not self.drawing_requested:
            response.success = False
            response.message = "Robot is not drawing"
            self.get_logger().warn(response.message)
            return response

        self.stop_requested = True
        self.drawing_requested = False
        self.get_logger().info("Stop drawing service called")

        try:
            self.moveit2.cancel_execution()
        except Exception as e:
            self.get_logger().warn(f"Stop requested, but active motion could not be cancelled: {e}")

        self.state_pub.publish(String(data="IDLE"))
        self.status_pub.publish(String(data="Drawing stopped. Remaining strokes kept for resume."))

        response.success = True
        response.message = "Drawing stopped. Press Start Drawing to resume remaining strokes."
        return response

    def start_go_home_sequence(self):
        self.is_drawing = True
        try:
            self.state_pub.publish(String(data="GOING_HOME"))
            self.status_pub.publish(String(data="Going home"))
            self.go_home()
            self.state_pub.publish(String(data="IDLE"))
            self.status_pub.publish(String(data="Robot moved home"))
        except Exception as e:
            self.get_logger().error(f"Go home sequence failed: {e}")
            self.state_pub.publish(String(data="ERROR"))
            self.status_pub.publish(String(data=f"Go home failed: {e}"))
        finally:
            self.is_drawing = False

    def start_utility_motion_sequence(self, request):
        self.is_drawing = True
        try:
            command = request.get("command")
            self.state_pub.publish(String(data="UTILITY_MOTION"))

            if command == "move_vertical":
                dist = float(request.get("dist", 0.0))
                self.status_pub.publish(String(data=f"Moving vertical by {dist:.4f} m"))
                success = self.move_vertical(dist)
            elif command == "rotate_end_effector":
                angle = float(request.get("angle", 0.0))
                degrees = bool(request.get("degrees", True))
                unit = "deg" if degrees else "rad"
                self.status_pub.publish(String(data=f"Rotating end effector by {angle:.4f} {unit}"))
                success = self.rotate_end_effector(angle, degrees=degrees)
            elif command == "attach_pen":
                pen_index = int(request.get("pen", 1))
                self.status_pub.publish(String(data=f"Attaching pen {pen_index}"))
                success = self.attach_pen(pen_index)
            elif command == "detach_pen":
                pen_index = int(request.get("pen", 1))
                self.status_pub.publish(String(data=f"Detaching pen {pen_index}"))
                success = self.detach_pen(pen_index)
            else:
                raise RuntimeError(f"Unknown utility motion command: {command}")

            if success and not self.stop_was_requested():
                self.state_pub.publish(String(data="IDLE"))
                self.status_pub.publish(String(data="Utility motion completed"))
            else:
                self.state_pub.publish(String(data="IDLE"))
                self.status_pub.publish(String(data="Utility motion stopped"))

        except Exception as e:
            self.get_logger().error(f"Utility motion failed: {e}")
            self.state_pub.publish(String(data="ERROR"))
            self.status_pub.publish(String(data=f"Utility motion failed: {e}"))
        finally:
            self.is_drawing = False

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
        self.stop_requested = False
        self.drawing_requested = True

        response.success = True
        response.message = "Drawing sequence started"
        return response


    def start_drawing_sequence(self):
        self.is_drawing = True
        try:
            self.state_pub.publish(String(data="DRAWING"))
            self.status_pub.publish(String(data="Drawing sequence started"))

            if self.portrait_in_progress:
                self.get_logger().info("Resuming portrait from remaining strokes")
            else:
                if not self.go_home() or self.stop_was_requested():
                    return
                time.sleep(1)

                if not self.draw_rectangle() or self.stop_was_requested():
                    return
                time.sleep(1)

            if not self.draw_portrait() or self.stop_was_requested():
                return
            time.sleep(1)

            if not self.go_home() or self.stop_was_requested():
                return
            time.sleep(1)

            self.state_pub.publish(String(data="IDLE"))
            self.status_pub.publish(String(data="Drawing sequence completed"))

        except Exception as e:
            self.get_logger().error(f"Drawing sequence failed: {e}")
            self.state_pub.publish(String(data="ERROR"))
            self.status_pub.publish(String(data=f"Drawing failed: {e}"))

        finally:
            self.is_drawing = False
#----------------------------- 6. Pen Attachment/Detachment --------------------------------
    def get_pen_ready_pose(self, pen_index):
            pen_index = int(pen_index)
            json_path = os.path.join(os.getcwd(), "data", "pen_storage_calibration.json")

            with open(json_path, "r") as f:
                data = json.load(f)

            key = f"pen_{pen_index}"
            if key not in data:
                raise RuntimeError(f"{key} is not saved in {json_path}")

            pose_data = data[key]
            return (
                np.array(pose_data["ready_position"], dtype=float),
                np.array(pose_data["ready_orientation"], dtype=float),
            )

    def detach_pen(self, pen_index):
        # Get Ready Pose from JSON
        ready_position, ready_quat = self.get_pen_ready_pose(pen_index)
        # Go home first to ensure a consistent starting pose for MoveIt planning to the pen ready pose
        self.go_home()
        # Move to ready pose above the pen + extra distance to ensure pentip not gonna collide with top of pen storage 
        ready_to_detach_position = ready_position + np.array([0.0, 0.0, 0.05])  # Add 5 cm in z to be safely above
        self.moveit2.move_to_pose(
            position=ready_to_detach_position.tolist(),
            quat_xyzw=ready_quat.tolist(),
            cartesian=True
        )
        if not self.wait_for_motion():
            if self.stop_was_requested():
                return False
            raise RuntimeError("MoveIt planned the wrist rotation, but execution did not complete")
        # The ready to detach position is above ready to attach position 5 cm. Move down to ready to attach position.
        self.move_vertical(-0.05)
        # Move down further to ready to detach position to drop the pen into the storage
        self.move_vertical(-0.10)
        # Twist while in contact to release the pen, then lift up
        self.rotate_end_effector(90.0, degrees=True)
        # Lift up 15 cm after detaching the pen
        self.move_vertical(0.15)
        # Twist back to original orientation after lifting
        self.rotate_end_effector(-90.0, degrees=True)
        # After detaching, the pen should be in the storage. Move back to home.
        self.go_home()


    def attach_pen(self, pen_index):
        #Get Ready Pose from JSON
        ready_position, ready_quat = self.get_pen_ready_pose(pen_index)
        # Go home first to ensure a consistent starting pose for MoveIt planning to the pen ready pose
        self.go_home()
        #Move to ready pose above the pen
        self.moveit2.move_to_pose(
            position=ready_position.tolist(),
            quat_xyzw=ready_quat.tolist(),
            cartesian=True
        )
        if not self.wait_for_motion():
            if self.stop_was_requested():
                return False
            raise RuntimeError("MoveIt planned the wrist rotation, but execution did not complete")
        #The ready to attach position is above the pen. Move down to grasp it.
        dist_tool_to_pen = ready_position[2] - self.pen_ready_to_attach_pos
        self.move_vertical(-dist_tool_to_pen)
        # Twist while in contact to secure the pen, then lift up with the pen
        self.rotate_end_effector(90.0, degrees=True)
        # Lift up 10 cm with the pen
        self.move_vertical(0.10 + abs(dist_tool_to_pen))
        # Twist back to original orientation after lifting
        self.rotate_end_effector(-90.0, degrees=True)
        # After attaching, the pen should be in the robot's end effector. Move back to home.
        self.go_home()
#---------------------------------------------7. Cartesian Utility Motion-------------------------------
    def move_vertical(self, dist):
        transform = self.tf_buffer.lookup_transform(
            "base_link",
            "tool0",
            rclpy.time.Time()
        )

        current_position = transform.transform.translation
        current_orientation = transform.transform.rotation

        target_position = [
            current_position.x,
            current_position.y,
            current_position.z + float(dist)
        ]

        target_quat = [
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ]

        direction = "up" if dist >= 0 else "down"
        self.get_logger().info(
            f"Moving {direction} by {abs(float(dist)):.3f} m to z={target_position[2]:.3f}"
        )

        self.moveit2.move_to_pose(
            position=target_position,
            quat_xyzw=target_quat,
            cartesian=True
        )

        if not self.wait_for_motion():
            if self.stop_was_requested():
                return False
            raise RuntimeError("MoveIt planned the vertical motion, but execution did not complete")

        return not self.stop_was_requested()

    def get_current_joint_positions(self):
        joint_state = self.moveit2.joint_state
        if joint_state is None:
            raise RuntimeError("Joint state is not ready yet")

        positions = []
        for joint_name in ur.joint_names():
            if joint_name not in joint_state.name:
                raise RuntimeError(f"Joint state does not contain {joint_name}")
            positions.append(joint_state.position[joint_state.name.index(joint_name)])

        return positions

    def normalize_angle(self, angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    # def rotate_end_effector(self, angle, degrees=True):
    #     angle_radians = np.deg2rad(float(angle)) if degrees else float(angle)
    #     if abs(angle_radians) < 1e-6:
    #         self.get_logger().info("Rotation angle is zero; skipping wrist rotation")
    #         return True

    #     direction = "positive" if angle_radians >= 0 else "negative"
    #     unit = "deg" if degrees else "rad"
    #     if abs(angle_radians) <= self.moveit_rotation_limit:
    #         joint_positions = self.get_current_joint_positions()
    #         joint_positions[-1] += angle_radians

    #         self.get_logger().info(
    #             f"Rotating wrist_3_joint {direction} by {abs(float(angle)):.3f} {unit} using MoveIt"
    #         )

    #         self.moveit2.move_to_configuration(joint_positions)

    #         if not self.wait_for_motion():
    #             if self.stop_was_requested():
    #                 return False
    #             raise RuntimeError("MoveIt planned the wrist rotation, but execution did not complete")

    #         return not self.stop_was_requested()

    #     self.get_logger().info(
    #         f"Rotating wrist_3_joint {direction} by {abs(float(angle)):.3f} {unit} using URScript speedj"
    #     )

    #     speed = np.sign(angle_radians) * abs(self.wrist_rotation_speed)
    #     duration = abs(angle_radians) / abs(self.wrist_rotation_speed)
    #     command = (
    #         "speedj("
    #         f"[0.0, 0.0, 0.0, 0.0, 0.0, {speed:.6f}], "
    #         f"{self.wrist_rotation_acceleration:.6f}, "
    #         f"{duration:.6f}"
    #         ")"
    #     )

    #     self.urscript_pub.publish(String(data=command))
    #     start_time = time.monotonic()
    #     while time.monotonic() - start_time < duration:
    #         if self.stop_was_requested():
    #             self.urscript_pub.publish(String(data="stopj(2.0)"))
    #             return False
    #         time.sleep(0.02)

    #     self.urscript_pub.publish(String(data="stopj(2.0)"))

    #     return not self.stop_was_requested()

    def rotate_end_effector(self, angle, degrees=True):
        angle_radians = np.deg2rad(float(angle)) if degrees else float(angle)
        if abs(angle_radians) < 1e-6:
            self.get_logger().info("Rotation angle is zero; skipping wrist rotation")
            return True

        direction = "positive" if angle_radians >= 0 else "negative"
        unit = "deg" if degrees else "rad"
        self.get_logger().info(
            f"Rotating wrist_3_joint {direction} by {abs(float(angle)):.3f} {unit} using MoveIt"
        )

        joint_positions = self.get_current_joint_positions()
        joint_positions[-1] += angle_radians

        self.moveit2.move_to_configuration(joint_positions)

        if not self.wait_for_motion():
            if self.stop_was_requested():
                return False
            raise RuntimeError("MoveIt planned the wrist rotation, but execution did not complete")

        return not self.stop_was_requested()

def main():

    rclpy.init()

    node = MotionNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.go_home_requested and not node.is_drawing:
                node.go_home_requested = False
                node.start_go_home_sequence()
            elif node.utility_motion_requested and not node.is_drawing:
                request = node.utility_motion_requested
                node.utility_motion_requested = None
                node.start_utility_motion_sequence(request)
            elif node.drawing_requested and not node.is_drawing:
                node.drawing_requested = False
                node.start_drawing_sequence()
    finally:
        if node.inactivity_timer:
            node.inactivity_timer.cancel()
        node.destroy_node()
        rclpy.shutdown()
