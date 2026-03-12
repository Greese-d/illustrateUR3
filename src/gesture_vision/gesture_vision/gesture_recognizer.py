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
    wrist = lm[mp.solutions.hands.HandLandmark.WRIST]
    thumb_cmc = lm[mp.solutions.hands.HandLandmark.THUMB_CMC]
    thumb_mcp = lm[mp.solutions.hands.HandLandmark.THUMB_MCP]
    thumb_ip  = lm[mp.solutions.hands.HandLandmark.THUMB_IP]
    thumb_tip = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]

    index_mcp = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = lm[mp.solutions.hands.HandLandmark.PINKY_MCP]

    # 1) Straightness at thumb IP joint
    a = np.array([thumb_mcp.x, thumb_mcp.y])
    b = np.array([thumb_ip.x, thumb_ip.y])
    c = np.array([thumb_tip.x, thumb_tip.y])

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))

    # 2) Thumb tip farther out than IP and MCP
    wrist_xy = np.array([wrist.x, wrist.y])
    tip_d = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - wrist_xy)
    ip_d  = np.linalg.norm(np.array([thumb_ip.x, thumb_ip.y]) - wrist_xy)
    mcp_d = np.linalg.norm(np.array([thumb_mcp.x, thumb_mcp.y]) - wrist_xy)

    extended_length = (tip_d > ip_d) and (tip_d > mcp_d * 1.05)

    # 3) Thumb should point away from the palm center
    palm_center = np.array([
        (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0,
        (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
    ])

    thumb_vec = np.array([thumb_tip.x, thumb_tip.y]) - np.array([thumb_mcp.x, thumb_mcp.y])
    palm_vec  = palm_center - np.array([thumb_mcp.x, thumb_mcp.y])

    # If dot product is negative, thumb points away from palm
    away_from_palm = np.dot(thumb_vec, palm_vec) < 0

    angle_ok = angle > 130.0

    return angle_ok and extended_length and away_from_palm


class GestureRecognizer(Node):
    def __init__(self):
        super().__init__("gesture_recognizer")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("gesture_topic", "/gesture")
        self.declare_parameter("debug_topic", "/gesture/debug_image")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("hold_time", 2.0)

        image_topic = str(self.get_parameter("image_topic").value)
        gesture_topic = str(self.get_parameter("gesture_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.hold_time = float(self.get_parameter("hold_time").value)

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

        # Debounce state
        self.pending = None
        self.pending_since = None
        self.last_published = "NONE"

        self.get_logger().info(
            f"Subscribing to {image_topic}, publishing gestures on {gesture_topic} (hold_time={self.hold_time}s)"
        )

    def classify(self, hand_lms, handedness=None):
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

        # Count only index..pinky separately so thumb noise doesn't break OPEN_PALM/FIST
        n4 = sum(1 for e in [idx, mid, ring, pinky] if e)

        if thumb and n4 == 0:
            tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
            ip  = lm[self.mp_hands.HandLandmark.THUMB_IP]

            # In image coordinates, smaller y means higher in image
            if tip.y < ip.y - 0.03:
                return "THUMBS_UP"

        # OPEN_PALM doesn't depend on thumb (more stable)
        if n4 == 4:
            return "OPEN_HAND"

        # FIST requires none of the 5 extended (stable)
        if n4 == 0 and (not thumb):
            return "FIST"

        # POINT: index only, thumb off
        if idx and (not mid) and (not ring) and (not pinky) and (not thumb):
            return "POINT"

        return "UNKNOWN"

    def cb(self, msg: Image):
        now = self.get_clock().now()

        # ROS Image -> OpenCV BGR
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        # BGR -> RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run MediaPipe
        results = self.hands.process(frame_rgb)

        # Compute gesture
        gesture = "NONE"
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            handedness = None
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            gesture = self.classify(hand_lms, handedness)

            # Optional debug drawing
            if self.publish_debug and self.debug_pub is not None:
                dbg = frame_bgr.copy()
                self.mp_draw.draw_landmarks(dbg, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
                dbg_msg.header = msg.header
                self.debug_pub.publish(dbg_msg)
        else:
            if self.publish_debug and self.debug_pub is not None:
                dbg_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
                dbg_msg.header = msg.header
                self.debug_pub.publish(dbg_msg)

        # Debounce/hold
        if gesture != self.pending:
            self.pending = gesture
            self.pending_since = now
            return

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