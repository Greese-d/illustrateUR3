import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult

import portrait_vectorisation.portrait_processor as pp


class ImgProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')

        # ------------------------------------------------------------------ #
        # Parameters                                                           #
        # ------------------------------------------------------------------ #
        self.declare_parameter('stroke_publish_delay', 0.05)   # seconds between strokes
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('snapshot_topic', '/camera/snapshot')
        self.declare_parameter('portrait_topic', '/portrait/preview')
        self.declare_parameter('strokes_topic', '/portrait/strokes')
        self.declare_parameter('markers_topic', '/portrait/markers')
        self.declare_parameter('emotion_topic', '/portrait/emotion')
        self.declare_parameter('emotion_scores_topic', '/portrait/emotion_scores')
        self.declare_parameter('mask_type', 'none')
        self.declare_parameter('masked_preview_topic', '/camera/masked_preview')
        self.declare_parameter('min_stroke_length', 20.0)
        self.declare_parameter('signature_scale', 0.40)
        self.declare_parameter('emotion_model_path', '')

        self._stroke_delay  = self.get_parameter('stroke_publish_delay').value
        camera_topic        = self.get_parameter('camera_topic').value
        snapshot_topic      = self.get_parameter('snapshot_topic').value
        portrait_topic      = self.get_parameter('portrait_topic').value
        strokes_topic       = self.get_parameter('strokes_topic').value
        markers_topic       = self.get_parameter('markers_topic').value
        emotion_topic       = self.get_parameter('emotion_topic').value
        emotion_scores_topic = self.get_parameter('emotion_scores_topic').value
        masked_preview_topic = self.get_parameter('masked_preview_topic').value
        min_stroke_length   = self.get_parameter('min_stroke_length').value
        self._publish_strokes = True
        emotion_model_path = self.get_parameter('emotion_model_path').value
        if isinstance(emotion_model_path, str):
            emotion_model_path = emotion_model_path.strip() or None
        else:
            emotion_model_path = None

        # ------------------------------------------------------------------ #
        # State                                                                #
        # ------------------------------------------------------------------ #
        self.latest_raw_image: Image | None = None      # most recent frame from camera
        self.latest_masked_image: Image | None = None   # most recent masked preview frame
        self.snapshot_raw:    Image | None = None       # frozen raw frame for emotion
        self.snapshot_masked: Image | None = None       # frozen masked frame for portrait
        self._snapshot_used = False              # True once snapshot has been processed

        self.bridge     = CvBridge()
        self.processor  = pp.PortraitProcessor(
            min_stroke_length=min_stroke_length,
            signature_scale=self.get_parameter('signature_scale').value,
            emotion_model_path=emotion_model_path,
        )
        self.allowed_masks = {"none"} | set(self.processor.masks.keys())
        self.mask_type = self.get_parameter('mask_type').value

        if not self.processor.emotion_available:
            self.get_logger().warn(
                "Emotion model not available; emotion detection disabled. "
                "Provide an ONNX model to enable emotion logs."
            )
        else:
            self.get_logger().info(
                f"Emotion model loaded from: {self.processor.emotion_model_path}"
            )

        self.add_on_set_parameters_callback(self.param_callback)

        # ------------------------------------------------------------------ #
        # Subscribers                                                          #
        # ------------------------------------------------------------------ #
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self._raw_image_callback,
            10,
        )
        self.masked_subscription = self.create_subscription(
            Image,
            masked_preview_topic,
            self._masked_image_callback,
            10,
        )

        # ------------------------------------------------------------------ #
        # Publishers                                                           #
        # ------------------------------------------------------------------ #
        self.snapshot_pub = self.create_publisher(Image,       snapshot_topic, 10)
        self.portrait_pub = self.create_publisher(Image,       portrait_topic, 10)
        self.stroke_pub   = self.create_publisher(Path,        strokes_topic,  50)
        self.marker_pub   = self.create_publisher(MarkerArray, markers_topic,  10)
        self.emotion_pub  = self.create_publisher(String,      emotion_topic,  10)
        self.emotion_scores_pub = self.create_publisher(String, emotion_scores_topic, 10)
        self.masked_pub   = self.create_publisher(Image,       masked_preview_topic, 10)

        # ------------------------------------------------------------------ #
        # Services                                                             #
        # ------------------------------------------------------------------ #
        self.snapshot_service = self.create_service(
            Trigger,
            '/capture_snapshot',
            self._capture_snapshot_callback,
        )
        self.portrait_service = self.create_service(
            Trigger,
            '/create_portrait',
            self._create_portrait_callback,
        )

        self.get_logger().info(
            f'Image processing node ready.\n'
            f'  Listening on : {camera_topic}\n'
            f'  Snapshot     : {snapshot_topic}\n'
            f'  Portrait     : {portrait_topic}\n'
            f'  Strokes      : {strokes_topic}\n'
            f'  Markers      : {markers_topic}\n'
            f'  Mask preview : {masked_preview_topic}\n'
            f'  Masked source: {masked_preview_topic}\n'
            f'  Stroke delay : {self._stroke_delay}s'
        )

    # ---------------------------------------------------------------------- #
    # Parameters                                                             #
    # ---------------------------------------------------------------------- #

    def param_callback(self, params):
        for p in params:
            if p.name == "mask_type":
                if p.value in self.allowed_masks:
                    self.mask_type = p.value
                    self.get_logger().info(f"Mask set to: {self.mask_type}")
                    return SetParametersResult(successful=True)

                self.get_logger().warn(
                    f"Unknown mask '{p.value}'. Valid options: {sorted(self.allowed_masks)}"
                )
                return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

    # ---------------------------------------------------------------------- #
    # Subscriber callback — only stores the latest frame                      #
    # ---------------------------------------------------------------------- #

    def _raw_image_callback(self, msg: Image) -> None:
        self.latest_raw_image = msg

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            masked = self.processor.apply_mask(cv_image, self.mask_type)
            masked_msg = self.bridge.cv2_to_imgmsg(masked, encoding='bgr8')
            masked_msg.header = msg.header
            self.masked_pub.publish(masked_msg)
        except Exception as e:
            self.get_logger().warn(f"Mask preview failed: {e}")

    def _masked_image_callback(self, msg: Image) -> None:
        self.latest_masked_image = msg

    # ---------------------------------------------------------------------- #
    # Service 1 — /capture_snapshot                                           #
    # ---------------------------------------------------------------------- #

    def _capture_snapshot_callback(self, request, response):
        """
        Freeze the most recent camera frame into the snapshot buffer and
        publish it on the snapshot topic.
        """
        if self.latest_raw_image is None or self.latest_masked_image is None:
            msg = (
                'No camera frame received yet. '
                'Is the camera publisher running and mask preview active?'
            )
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            return response

        self.snapshot_raw = self.latest_raw_image
        self.snapshot_masked = self.latest_masked_image
        self._snapshot_used = False

        self.snapshot_pub.publish(self.snapshot_masked)
        self.get_logger().info('Snapshot captured and published.')
        response.success = True
        response.message = 'Snapshot captured successfully.'
        return response

    # ---------------------------------------------------------------------- #
    # Service 2 — /create_portrait                                            #
    # ---------------------------------------------------------------------- #

    def _create_portrait_callback(self, request, response):
        """
        Process the current snapshot into a portrait and publish:
          1. The preview image on /portrait/preview
          2. Each stroke as a nav_msgs/Path on /portrait/strokes, with a small
             delay between publishes so the subscriber can receive them in order.

        If the snapshot buffer is empty or has already been processed,
        a fresh snapshot is taken automatically before proceeding.
        """
        # ---- ensure we have a fresh snapshot --------------------------------
        if self.snapshot_raw is None or self.snapshot_masked is None or self._snapshot_used:
            reason = 'empty' if (self.snapshot_raw is None or self.snapshot_masked is None) else 'already processed'
            self.get_logger().info(
                f'Snapshot buffer is {reason}. '
                f'Taking a fresh snapshot automatically.'
            )
            snap_response = self._capture_snapshot_callback(request, Trigger.Response())

            if not snap_response.success:
                msg = f'Aborted: could not obtain a snapshot — {snap_response.message}'
                self.get_logger().error(msg)
                response.success = False
                response.message = msg
                return response

        # ---- convert ROS image → OpenCV -------------------------------------
        try:
            cv_masked = self.bridge.imgmsg_to_cv2(
                self.snapshot_masked, desired_encoding='bgr8'
            )
            cv_raw = self.bridge.imgmsg_to_cv2(
                self.snapshot_raw, desired_encoding='bgr8'
            )
        except Exception as e:
            msg = f'Failed to decode snapshot image: {e}'
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response

        # ---- run portrait processing pipeline --------------------------------
        self.get_logger().info('Starting portrait processing pipeline...')
        try:
            canvas, strokes, emotion, emotion_scores = self.processor.process(
                cv_masked,
                emotion_image=cv_raw,
            )
        except Exception as e:
            msg = f'Portrait processing failed: {e}'
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response

        if emotion:
            msg = String()
            msg.data = emotion
            self.emotion_pub.publish(msg)

        if emotion_scores:
            score_msg = String()
            score_msg.data = ", ".join(
                f"{label}:{score:.3f}" for label, score in emotion_scores
            )
            self.emotion_scores_pub.publish(score_msg)

        if not strokes:
            msg = (
                'Processing produced zero strokes. '
                'The image may be blank or the subject was not detected.'
            )
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            return response

        self.get_logger().info(
            f'Processing complete: {len(strokes)} stroke(s) generated.'
        )

        # ---- publish preview image ------------------------------------------
        try:
            portrait_msg = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
            portrait_msg.header.stamp    = self.get_clock().now().to_msg()
            portrait_msg.header.frame_id = 'camera_frame'
            self.portrait_pub.publish(portrait_msg)
            self.get_logger().info('Portrait preview published.')
        except Exception as e:
            # Non-fatal — log and continue to stroke publishing
            self.get_logger().error(f'Failed to publish portrait preview: {e}')

        if not self._publish_strokes:
            self._snapshot_used = True
            msg = 'Stroke publishing disabled; skipping stroke and marker output.'
            self.get_logger().info(msg)
            response.success = True
            response.message = msg
            return response

        # ---- publish strokes in order with delay between each ---------------
        now = self.get_clock().now().to_msg()
        failed = 0

        for idx, path in enumerate(strokes):
            try:
                path.header.stamp = now
                for pose in path.poses:
                    pose.header.stamp = now

                self.stroke_pub.publish(path)

                if self._stroke_delay > 0.0:
                    time.sleep(self._stroke_delay)

            except Exception as e:
                self.get_logger().warn(
                    f'Failed to publish stroke {idx + 1}/{len(strokes)}: {e}'
                )
                failed += 1

        # ---- publish all strokes as a single MarkerArray for RViz2 ----------
        try:
            self.marker_pub.publish(self._strokes_to_marker_array(strokes, now))
            self.get_logger().info('Marker array published for RViz2.')
        except Exception as e:
            # Non-fatal — pipeline data was already sent above
            self.get_logger().warn(f'Failed to publish marker array: {e}')

        # Mark snapshot as consumed so the next call auto-refreshes
        self._snapshot_used = True

        if failed == 0:
            msg = (
                f'All {len(strokes)} stroke(s) published successfully '
                f'({self._stroke_delay * 1000:.0f}ms delay between each).'
            )
            self.get_logger().info(msg)
            response.success = True
            response.message = msg
        else:
            msg = f'{failed}/{len(strokes)} stroke(s) failed to publish.'
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg

        return response

    # ---------------------------------------------------------------------- #
    # RViz2 visualisation helper                                              #
    # ---------------------------------------------------------------------- #

    def _strokes_to_marker_array(self, strokes: list, stamp) -> MarkerArray:
        """
        Convert a list of nav_msgs/Path strokes into a single MarkerArray
        where each stroke is one LINE_STRIP marker with a unique id.
        Sending all strokes in one message means RViz2 renders them all at
        once rather than overwriting on each publish.
        """
        array = MarkerArray()

        for idx, path in enumerate(strokes):
            marker = Marker()
            marker.header.frame_id = path.header.frame_id
            marker.header.stamp    = stamp
            marker.ns              = 'portrait_strokes'
            marker.id              = idx
            marker.type            = Marker.LINE_STRIP
            marker.action          = Marker.ADD

            # Scale: line width in metres — adjust to taste in RViz2
            marker.scale.x = 1.0   # pixel-space units, scale in RViz2 display

            # White lines, fully opaque
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Persist until explicitly deleted (0 = forever)
            marker.lifetime.sec     = 0
            marker.lifetime.nanosec = 0

            for pose_stamped in path.poses:
                marker.points.append(pose_stamped.pose.position)

            array.markers.append(marker)

        return array

    # ---------------------------------------------------------------------- #
    # Cleanup                                                                  #
    # ---------------------------------------------------------------------- #

    def destroy_node(self):
        self.processor.close()
        self.get_logger().info('Portrait processor released.')
        super().destroy_node()


# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = ImgProcessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()