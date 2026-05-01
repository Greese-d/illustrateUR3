import json
import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from rclpy.node import Node


class GuiNode(Node):
    def __init__(self):
        super().__init__("illustrateur3_gui_node")

        self.current_state = "IDLE"
        self.tcp_offset = 0.12

        self.state_callback_fn = None
        self.camera_callback_fn = None
        self.preview_callback_fn = None
        self.live_drawing_callback_fn = None
        self.calibration_status_callback_fn = None

        self.calibration_pub = self.create_publisher(
            String, "/calibration/command", 10
        )

        self.create_portrait_client = self.create_client(
            Trigger, "/create_portrait"
        )
        self.start_drawing_client = self.create_client(
            Trigger, "/start_drawing"
        )
        self.stop_drawing_client = self.create_client(
            Trigger, "/stop_drawing"
        )
        self.go_home_client = self.create_client(
            Trigger, "/go_home"
        )
        self.clear_strokes_client = self.create_client(
            Trigger, "/clear_strokes"
        )
        self.calibration_status_sub = self.create_subscription(
            String,
            "/calibration/status",
            self.calibration_status_callback,
            10,
        )

        self.state_sub = self.create_subscription(
            String,
            "/state",
            self.state_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            "/gesture/debug_image", #/gesture/debug_image for Gestures overlay, /camera/image_raw for regular image
            self.camera_image_callback,
            10
        )

        self.preview_sub = self.create_subscription(
            Image,
            "/portrait/preview",
            self.preview_image_callback,
            10
        )

        self.get_logger().info("GUI node started")
        self.get_logger().info("Subscribed to /state")
        self.get_logger().info("Subscribed to /camera/image_raw")
        self.get_logger().info("Subscribed to /portrait/preview")
        self.get_logger().info("Publishing to /calibration/command")
        self.get_logger().info("Subscribed to /calibration/status")
        self.get_logger().info("Created client for /create_portrait")
        self.get_logger().info("Created client for /start_drawing")
        self.get_logger().info("Created client for /stop_drawing")
        self.get_logger().info("Created client for /go_home")

    def state_callback(self, msg):
        self.current_state = msg.data
        self.get_logger().info(f"Received state: {msg.data}")

        if self.state_callback_fn is not None:
            self.state_callback_fn(msg.data)

    def camera_image_callback(self, msg):
        try:
            frame_bgr = self.rosimg_to_bgr(msg)
            if self.camera_callback_fn is not None:
                self.camera_callback_fn(frame_bgr)
        except Exception as e:
            self.get_logger().warn(f"Failed to render camera image: {e}")

    def preview_image_callback(self, msg):
        try:
            frame_bgr = self.rosimg_to_bgr(msg)
            if self.preview_callback_fn is not None:
                self.preview_callback_fn(frame_bgr)
        except Exception as e:
            self.get_logger().warn(f"Failed to render preview image: {e}")

    def calibration_status_callback(self, msg):
        self.get_logger().info(f"Calibration status: {msg.data}")

        try:
            data = json.loads(msg.data)

            if "tcp_offset" in data:
                self.tcp_offset = float(data["tcp_offset"])

        except Exception as e:
            self.get_logger().warn(f"Failed to parse calibration status JSON: {e}")

        if self.calibration_status_callback_fn is not None:
            self.calibration_status_callback_fn(msg.data)

    def send_calibration_command(self, command: str):
        msg = String()
        msg.data = command
        self.calibration_pub.publish(msg)
        self.get_logger().info(f"Published calibration command: {command}")

    def send_calibration_payload(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload)
        self.calibration_pub.publish(msg)
        self.get_logger().info(f"Published calibration payload: {msg.data}")

    def set_point_1(self):
        self.send_calibration_command("set_p1")

    def set_point_2(self):
        self.send_calibration_command("set_p2")

    def set_point_3(self):
        self.send_calibration_command("set_p3")

    def on_confirm(self):
        self.send_calibration_command("confirm")

    def set_tcp_offset(self, tcp_offset: float):
        self.tcp_offset = float(tcp_offset)
        self.send_calibration_payload(
            {
                "command": "set_tcp_offset",
                "tcp_offset": self.tcp_offset,
            }
        )

    def toggle_paper_display(self, enabled: bool):
        self.send_calibration_command("show_paper" if enabled else "hide_paper")

    def toggle_axes_display(self, enabled: bool):
        self.send_calibration_command("show_axes" if enabled else "hide_axes")

    def rosimg_to_bgr(self, msg: Image):
        enc = msg.encoding.lower()

        if enc not in ("bgr8", "rgb8"):
            raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")

        frame = np.frombuffer(
            msg.data, dtype=np.uint8
        ).reshape((msg.height, msg.width, 3))

        if enc == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def create_portrait(self, gui_callback=None):
        if not self.create_portrait_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/create_portrait service not available")
            if gui_callback:
                gui_callback(False, "/create_portrait service not available")
            return

        request = Trigger.Request()
        future = self.create_portrait_client.call_async(request)

        def _handle_future_done(fut):
            try:
                response = fut.result()
                success = response.success
                message = response.message
                self.get_logger().info(
                    f"/create_portrait response: success={success}, message='{message}'"
                )
                if gui_callback:
                    gui_callback(success, message)
            except Exception as e:
                error_msg = f"/create_portrait call failed: {e}"
                self.get_logger().error(error_msg)
                if gui_callback:
                    gui_callback(False, error_msg)

        future.add_done_callback(_handle_future_done)

    def start_drawing(self, gui_callback=None):
        if not self.start_drawing_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/start_drawing service not available")
            if gui_callback:
                gui_callback(False, "/start_drawing service not available")
            return

        request = Trigger.Request()
        future = self.start_drawing_client.call_async(request)

        def _handle_future_done(fut):
            try:
                response = fut.result()
                success = response.success
                message = response.message
                self.get_logger().info(
                    f"/start_drawing response: success={success}, message='{message}'"
                )
                if gui_callback:
                    gui_callback(success, message)
            except Exception as e:
                error_msg = f"/start_drawing call failed: {e}"
                self.get_logger().error(error_msg)
                if gui_callback:
                    gui_callback(False, error_msg)

        future.add_done_callback(_handle_future_done)

    def stop_drawing(self, gui_callback=None):
        if not self.stop_drawing_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/stop_drawing service not available")
            if gui_callback:
                gui_callback(False, "/stop_drawing service not available")
            return

        request = Trigger.Request()
        future = self.stop_drawing_client.call_async(request)

        def _handle_future_done(fut):
            try:
                response = fut.result()
                success = response.success
                message = response.message
                self.get_logger().info(
                    f"/stop_drawing response: success={success}, message='{message}'"
                )
                if gui_callback:
                    gui_callback(success, message)
            except Exception as e:
                error_msg = f"/stop_drawing call failed: {e}"
                self.get_logger().error(error_msg)
                if gui_callback:
                    gui_callback(False, error_msg)

        future.add_done_callback(_handle_future_done)

    def go_home(self, gui_callback=None):
        if not self.go_home_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/go_home service not available")
            if gui_callback:
                gui_callback(False, "/go_home service not available")
            return

        request = Trigger.Request()
        future = self.go_home_client.call_async(request)

        def _handle_future_done(fut):
            try:
                response = fut.result()
                success = response.success
                message = response.message
                self.get_logger().info(
                    f"/go_home response: success={success}, message='{message}'"
                )
                if gui_callback:
                    gui_callback(success, message)
            except Exception as e:
                error_msg = f"/go_home call failed: {e}"
                self.get_logger().error(error_msg)
                if gui_callback:
                    gui_callback(False, error_msg)

        future.add_done_callback(_handle_future_done)

    def clear_strokes(self, gui_callback=None):
        self.get_logger().info("Calling /clear_strokes service...")

        if not self.clear_strokes_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/clear_strokes service not available")
            if gui_callback:
                gui_callback(False, "/clear_strokes service not available")
            return

        request = Trigger.Request()
        future = self.clear_strokes_client.call_async(request)

        def _done(fut):
            try:
                response = fut.result()
                self.get_logger().info(
                    f"/clear_strokes response: success={response.success}, message='{response.message}'"
                )
                if gui_callback:
                    gui_callback(response.success, response.message)
            except Exception as e:
                self.get_logger().error(f"/clear_strokes failed: {e}")
                if gui_callback:
                    gui_callback(False, f"/clear_strokes failed: {e}")

        future.add_done_callback(_done)
