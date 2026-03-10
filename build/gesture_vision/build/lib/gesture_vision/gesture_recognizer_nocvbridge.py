import os
import urllib.request

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import String

from mediapipe.tasks.python import vision
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model(path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)


def imgmsg_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc not in ("bgr8", "rgb8"):
        raise RuntimeError(f"Unsupported encoding: {msg.encoding}")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    if enc == "bgr8":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def classify_simple(hand_landmarks) -> str:
    # hand_landmarks is a list of NormalizedLandmark objects with .x .y .z
    lm = hand_landmarks
    wrist = lm[0]

    tips = [lm[8], lm[12], lm[16], lm[20]]
    pips = [lm[6], lm[10], lm[14], lm[18]]
    mcps = [lm[5], lm[9], lm[13], lm[17]]

    def ext(tip, pip, mcp):
        tip_d = np.linalg.norm(np.array([tip.x, tip.y]) - np.array([wrist.x, wrist.y]))
        pip_d = np.linalg.norm(np.array([pip.x, pip.y]) - np.array([wrist.x, wrist.y]))
        mcp_d = np.linalg.norm(np.array([mcp.x, mcp.y]) - np.array([wrist.x, wrist.y]))
        return tip_d > pip_d and tip_d > mcp_d

    extended = [ext(tips[i], pips[i], mcps[i]) for i in range(4)]
    n = sum(1 for e in extended if e)

    if n == 4:
        return "OPEN_PALM"
    if n == 0:
        return "FIST"
    if extended[0] and not any(extended[1:]):
        return "POINT"
    return "UNKNOWN"


class GestureRecognizerNoCvBridge(Node):
    def __init__(self):
        super().__init__("gesture_recognizer")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("gesture_topic", "/gesture")
        self.declare_parameter("model_path", os.path.expanduser("~/.cache/mediapipe/hand_landmarker.task"))

        image_topic = str(self.get_parameter("image_topic").value)
        gesture_topic = str(self.get_parameter("gesture_topic").value)
        model_path = str(self.get_parameter("model_path").value)

        ensure_model(model_path)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.sub = self.create_subscription(Image, image_topic, self.cb, 10)
        self.pub = self.create_publisher(String, gesture_topic, 10)

        self.last = "NONE"
        self.get_logger().info(f"HandLandmarker ready. Subscribing {image_topic}, publishing {gesture_topic}")

    def cb(self, msg: Image):
        try:
            rgb = imgmsg_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(str(e))
            return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        gesture = "NONE"
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            gesture = classify_simple(result.hand_landmarks[0])

        if gesture != self.last:
            self.last = gesture
            self.pub.publish(String(data=gesture))


def main():
    rclpy.init()
    node = GestureRecognizerNoCvBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
