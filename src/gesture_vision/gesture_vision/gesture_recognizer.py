import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import mediapipe as mp

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


def finger_extended(tip, pip, mcp, wrist):
    tip_d = np.linalg.norm(np.array([tip.x, tip.y]) - np.array([wrist.x, wrist.y]))
    pip_d = np.linalg.norm(np.array([pip.x, pip.y]) - np.array([wrist.x, wrist.y]))
    mcp_d = np.linalg.norm(np.array([mcp.x, mcp.y]) - np.array([wrist.x, wrist.y]))
    return tip_d > pip_d and tip_d > mcp_d

def thumb_extended(lm):
    # Use angle at THUMB_IP: MCP - IP - TIP
    mcp = lm[mp.solutions.hands.HandLandmark.THUMB_MCP]
    ip  = lm[mp.solutions.hands.HandLandmark.THUMB_IP]
    tip = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]

    a = np.array([mcp.x, mcp.y])
    b = np.array([ip.x,  ip.y])
    c = np.array([tip.x, tip.y])

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))

    # Straight thumb ~160-180, bent thumb much smaller
    return angle > 160.0

class GestureRecognizer(Node):
    def __init__(self):
        super().__init__("gesture_recognizer")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("gesture_topic", "/gesture")
        self.declare_parameter("debug_topic", "/gesture/debug_image")
        self.declare_parameter("publish_debug", True)

        # NEW: debounce/hold parameter
        self.declare_parameter("hold_time", 2)
        self.hold_time = float(self.get_parameter("hold_time").value)

        image_topic = str(self.get_parameter("image_topic").value)
        gesture_topic = str(self.get_parameter("gesture_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, image_topic, self.cb, 10)
        self.gesture_pub = self.create_publisher(String, gesture_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10) if self.publish_debug else None

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # NEW: debounce state
        self.pending = None
        self.pending_since = None
        self.last_published = "NONE"

        self.get_logger().info(f"Subscribing to {image_topic}, publishing gestures on {gesture_topic}")

    def classify(self, hand_lms):
        lm = hand_lms.landmark
        wrist = lm[self.mp_hands.HandLandmark.WRIST]
        thumb = thumb_extended(lm)

        idx = finger_extended(
            lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            lm[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            lm[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            wrist,
        )
        mid = finger_extended(
            lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            wrist,
        )
        ring = finger_extended(
            lm[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            lm[self.mp_hands.HandLandmark.RING_FINGER_PIP],
            lm[self.mp_hands.HandLandmark.RING_FINGER_MCP],
            wrist,
        )
        pinky = finger_extended(
            lm[self.mp_hands.HandLandmark.PINKY_TIP],
            lm[self.mp_hands.HandLandmark.PINKY_PIP],
            lm[self.mp_hands.HandLandmark.PINKY_MCP],
            wrist,
        )

        ext = [thumb, idx, mid, ring, pinky]
        n = sum(1 for e in ext if e)

        if n == 5:
            return "OPEN_PALM"
        if n == 0:
            return "FIST"
        if idx and (not mid) and (not ring) and (not pinky) and (not thumb):
            return "POINT"
        if thumb and (not idx) and (not mid) and (not ring) and (not pinky):
            return "THUMBS_UP"
        return "UNKNOWN"

    def cb(self, msg: Image):
        now = self.get_clock().now()

        # If gesture changed, reset timer
        if gesture != self.pending:
            self.pending = gesture
            self.pending_since = now
            return

        # Gesture is the same as last frame — check duration
        if self.pending_since is not None:
            elapsed = (now - self.pending_since).nanoseconds / 1e9

            if elapsed >= self.hold_time and gesture != self.last_published:
                self.last_published = gesture
                self.gesture_pub.publish(String(data=gesture))


def main():
    rclpy.init()
    node = GestureRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
