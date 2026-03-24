import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import mediapipe as mp

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


def landmark_xy(lm):
    return np.array([lm.x, lm.y], dtype=np.float32)


def distance(a, b):
    return np.linalg.norm(landmark_xy(a) - landmark_xy(b))


def finger_extended(tip, pip, mcp, wrist):
    tip_d = distance(tip, wrist)
    pip_d = distance(pip, wrist)
    mcp_d = distance(mcp, wrist)
    return tip_d > pip_d and tip_d > mcp_d


def thumb_extended(lm, mp_hands):
    wrist = lm[mp_hands.HandLandmark.WRIST]
    thumb_mcp = lm[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = lm[mp_hands.HandLandmark.PINKY_MCP]

    a = landmark_xy(thumb_mcp)
    b = landmark_xy(thumb_ip)
    c = landmark_xy(thumb_tip)

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))

    wrist_xy = landmark_xy(wrist)
    tip_d = np.linalg.norm(landmark_xy(thumb_tip) - wrist_xy)
    ip_d = np.linalg.norm(landmark_xy(thumb_ip) - wrist_xy)
    mcp_d = np.linalg.norm(landmark_xy(thumb_mcp) - wrist_xy)

    length_ok = (tip_d > ip_d) and (tip_d > mcp_d * 1.05)
    angle_ok = angle > 130

    palm_center = (
        landmark_xy(wrist) +
        landmark_xy(index_mcp) +
        landmark_xy(pinky_mcp)
    ) / 3.0

    thumb_vec = landmark_xy(thumb_tip) - landmark_xy(thumb_mcp)
    palm_vec = palm_center - landmark_xy(thumb_mcp)

    away_from_palm = np.dot(thumb_vec, palm_vec) < 0

    return angle_ok and length_ok and away_from_palm


class GestureRecognizer(Node):

    def __init__(self):
        super().__init__("gesture_recognizer")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("gesture_topic", "/gesture")
        self.declare_parameter("debug_topic", "/gesture/debug_image")
        self.declare_parameter("hold_time", 0.5)

        image_topic = self.get_parameter("image_topic").value
        gesture_topic = self.get_parameter("gesture_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value
        self.hold_time = float(self.get_parameter("hold_time").value)

        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, image_topic, self.cb, 10)
        self.pub = self.create_publisher(String, gesture_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.pending = None
        self.pending_since = None
        self.last_published = "NONE"

        self.get_logger().info(f"Running: {os.path.abspath(__file__)}")

    def classify(self, hand_lms):
        lm = hand_lms.landmark
        wrist = lm[self.mp_hands.HandLandmark.WRIST]

        thumb = thumb_extended(lm, self.mp_hands)

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

        n4 = sum([idx, mid, ring, pinky])

        if thumb and n4 == 0:
            tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
            ip = lm[self.mp_hands.HandLandmark.THUMB_IP]

            if tip.y < ip.y - 0.03:
                return "THUMBS_UP"

            if tip.y > ip.y + 0.03:
                return "THUMBS_DOWN"

        if n4 == 4 and thumb:
            return "OPEN_HAND"

        if n4 == 0 and not thumb:
            return "FIST"

        if idx and not mid and not ring and not pinky:
            return "POINT"

        if idx and mid and not ring and not pinky and not thumb:
            return "PEACE"

        if idx and mid and not ring and not pinky and thumb:
            return "GREEN GIANT"

        if n4 == 4 and not thumb:
            return "FOUR"

        if not idx and not mid and not ring and pinky and thumb:
            return "CALL ME"

        return "UNKNOWN"

    def cb(self, msg):

        now = self.get_clock().now()

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(str(e))
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        gesture = "NONE"
        hand_lms = None

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            gesture = self.classify(hand_lms)

        dbg = frame_bgr.copy()

        if hand_lms:
            self.mp_draw.draw_landmarks(
                dbg,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(
            dbg,
            f"Gesture: {gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
        dbg_msg.header = msg.header
        self.debug_pub.publish(dbg_msg)

        if gesture != self.pending:
            self.pending = gesture
            self.pending_since = now
            return

        if self.pending_since is None:
            return

        elapsed = (now - self.pending_since).nanoseconds / 1e9

        if elapsed < self.hold_time:
            return

        if gesture == self.last_published:
            return

        self.last_published = gesture
        self.pub.publish(String(data=gesture))

        self.get_logger().info(f"Gesture: {gesture}")


def main():
    rclpy.init()
    node = GestureRecognizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()