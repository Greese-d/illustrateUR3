# portrait_vectorisation

A ROS2 package that captures a person's portrait from a webcam, extracts its contours as ordered drawing strokes, and publishes them for a UR3 robotic arm to draw.

---

## Overview

```
webcam
  │
  ▼
camera_publisher          →  /camera/image_raw  (sensor_msgs/Image)
  │
  ▼
image_processing_node     ←  /capture_snapshot  (Trigger service)
  │                        ←  /create_portrait   (Trigger service)
  │
  ├──▶  /camera/snapshot    (sensor_msgs/Image)
  ├──▶  /portrait/preview   (sensor_msgs/Image)
  ├──▶  /portrait/strokes   (nav_msgs/Path, one per stroke)
  └──▶  /portrait/markers   (visualization_msgs/MarkerArray, all strokes)
```

The processing pipeline inside `image_processing_node` uses the `PortraitProcessor` library to:

1. Remove the background with MediaPipe Selfie Segmentation
2. Smooth the image with a bilateral filter
3. Detect edges with Canny
4. Extract and simplify contours
5. Chain nearby contours into continuous strokes to minimise pen lifts
6. Sort strokes with a nearest-neighbour heuristic to minimise arm travel
7. Flip individual stroke directions to further reduce travel between strokes

---

## Package Structure

```
portrait_vectorisation/
├── portrait_vectorisation/
│   ├── __init__.py
│   ├── camera_publisher.py       # Camera capture node
│   ├── img_processing_node.py    # Portrait processing node
│   └── portrait_processor.py     # PortraitProcessor library
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `rclpy` | ROS2 Python client library |
| `sensor_msgs` | `Image` message type |
| `nav_msgs` | `Path` message type for strokes |
| `visualization_msgs` | `MarkerArray` for RViz2 visualisation |
| `std_srvs` | `Trigger` service type |
| `cv_bridge` | Converts between ROS2 `Image` and OpenCV `ndarray` |
| `opencv-python` | Image processing (bilateral filter, Canny, contours) |
| `mediapipe` | Background removal via Selfie Segmentation |
| `onnxruntime` | Optional: emotion detection via ONNX model |
| `numpy` | Array manipulation |

Install Python dependencies:

```bash
pip install opencv-python
python3 -m pip install --user "numpy==1.26.4" "mediapipe==0.10.13"
```

Optional emotion detection dependency:

```bash
python3 -m pip install --user onnxruntime
```

To enable emotion detection, place an ONNX model at:

```
portrait_vectorisation/models/emotion-ferplus-8.onnx
```

If the model is missing, the node runs normally and emotion detection is disabled.

You can also point to a custom model path via the `emotion_model_path` parameter:

```bash
ros2 run portrait_vectorisation image_processing_node --ros-args \
  -p emotion_model_path:=/abs/path/to/emotion.onnx
```

---

## Nodes

### `camera_publisher`

Captures frames from a connected webcam and publishes them as `sensor_msgs/Image`.

**Published topics**

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Live camera frames (bgr8) |

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `device` | `int` | `1` | Camera device index passed to `cv2.VideoCapture` |
| `fps` | `float` | `30.0` | Target capture frame rate |
| `width` | `int` | `1920` | Requested capture width in pixels |
| `height` | `int` | `1080` | Requested capture height in pixels |
| `topic` | `string` | `/camera/image_raw` | Topic name to publish on |

**Run**

```bash
ros2 run portrait_vectorisation camera_publisher
```

With custom parameters:

```bash
ros2 run portrait_vectorisation camera_publisher --ros-args \
  -p device:=0 \
  -p width:=1280 \
  -p height:=720 \
  -p fps:=30.0
```

---

### `image_processing_node`

Processes snapshots into portrait strokes on demand. All operations are triggered by service calls — the node does **not** process every incoming camera frame.

**Subscribed topics**

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Incoming camera frames (buffered, not processed) |

**Published topics**

| Topic | Type | Description |
|---|---|---|
| `/camera/snapshot` | `sensor_msgs/Image` | The frozen frame most recently captured by `/capture_snapshot` |
| `/portrait/preview` | `sensor_msgs/Image` | Black-on-white edge preview of the processed portrait (mono8) |
| `/portrait/strokes` | `nav_msgs/Path` | One message per stroke; `pose.position.x/y` are pixel coordinates, `z = 0` |
| `/portrait/markers` | `visualization_msgs/MarkerArray` | All strokes in a single message for RViz2 visualisation |
| `/portrait/emotion` | `std_msgs/String` | Dominant emotion label per portrait (if enabled) |
| `/portrait/emotion_scores` | `std_msgs/String` | Top-3 emotion scores per portrait (if enabled) |

**Services**

| Service | Type | Description |
|---|---|---|
| `/capture_snapshot` | `std_srvs/Trigger` | Freezes the latest camera frame into the snapshot buffer and publishes it |
| `/create_portrait` | `std_srvs/Trigger` | Processes the snapshot and publishes the preview, strokes, and markers. Auto-captures a fresh snapshot if the buffer is empty or has already been processed |

Both services return `success: bool` and `message: string` in the response. On failure the message contains the exception or the reason processing was aborted.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera_topic` | `string` | `/camera/image_raw` | Camera topic to subscribe to |
| `snapshot_topic` | `string` | `/camera/snapshot` | Topic to publish the snapshot on |
| `portrait_topic` | `string` | `/portrait/preview` | Topic to publish the preview image on |
| `strokes_topic` | `string` | `/portrait/strokes` | Topic to publish individual stroke paths on |
| `markers_topic` | `string` | `/portrait/markers` | Topic to publish the RViz2 marker array on |
| `emotion_topic` | `string` | `/portrait/emotion` | Topic to publish the detected emotion label on |
| `emotion_scores_topic` | `string` | `/portrait/emotion_scores` | Topic to publish the top-3 emotion scores on |
| `stroke_publish_delay` | `float` | `0.05` | Delay in seconds between publishing consecutive strokes, to ensure ordered delivery |

**Run**

```bash
ros2 run portrait_vectorisation image_processing_node
```

With custom parameters:

```bash
ros2 run portrait_vectorisation image_processing_node --ros-args \
  -p stroke_publish_delay:=0.02 \
  -p strokes_topic:=/arm/strokes
```

---

## Typical Workflow

**1. Start both nodes**

```bash
# Terminal 1
ros2 run portrait_vectorisation camera_publisher

# Terminal 2
ros2 run portrait_vectorisation image_processing_node
```

**2. Position the subject in front of the webcam, then capture a snapshot**

```bash
ros2 service call /capture_snapshot std_srvs/srv/Trigger {}
```

Example success response:
```
success: True
message: Snapshot captured successfully.
```

**3. Generate the portrait and publish strokes**

```bash
ros2 service call /create_portrait std_srvs/srv/Trigger {}
```

Example success response:
```
success: True
message: All 82 stroke(s) published successfully (50ms delay between each).
```

Example failure response:
```
success: False
message: Portrait processing failed: Segmentation model returned empty mask.
```

**4. Pass strokes to the UR3 motion planning node**

The motion planning node should subscribe to `/portrait/strokes`. Each received `nav_msgs/Path` represents one continuous pen-down stroke. The sequence of strokes is pre-sorted to minimise arm travel. `pose.position.x` and `pose.position.y` are in pixel coordinates and must be transformed into the robot's workspace frame before execution.

---

## Visualising in RViz2

Add a **MarkerArray** display in RViz2 and set its topic to `/portrait/markers`.

- All strokes are sent in a single message so the full portrait is rendered at once.
- Each stroke is a `LINE_STRIP` marker in the `portrait_strokes` namespace.
- Markers persist indefinitely (`lifetime = 0`) and are replaced on the next `/create_portrait` call.
- Coordinates are in pixel space. Set your fixed frame accordingly or apply a scaling transform in your RViz2 configuration to map pixels to a physical drawing area.

---

## `PortraitProcessor` Library

The `PortraitProcessor` class in `portrait_processor.py` can be used independently of ROS2.

```python
from portrait_vectorisation.portrait_processor import PortraitProcessor
import cv2

processor = PortraitProcessor(
    chain_threshold=10.0,  # px — max gap to chain two contours
    line_thickness=6.0     # px — thickness of pen tip (USED FOR CHAIN THRESHOLD)
    sort_strokes=True,     # enable nearest-neighbour stroke ordering
)

image = cv2.imread('photo.jpg')
canvas, strokes = processor.process(image, frame_id='camera_frame')

# canvas  — np.ndarray, mono8 preview image
# strokes — List[nav_msgs/Path], ordered stroke paths

cv2.imwrite('preview.png', canvas)
processor.close()
```

**Constructor parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chain_threshold` | `float` | `10.0` | Maximum pixel distance between contour endpoints to merge them into one stroke. Lower = more pen lifts, higher = more aggressive joining |
| `line_thickness` | `float` | `6.0` | Thickness of lines related to dimensions of pen tip scaled onto A4 paper with Full HD resolution (1920 x 1080) |
| `sort_strokes` | `bool` | `True` | Whether to reorder strokes using nearest-neighbour to minimise arm travel |

**Processing pipeline stages**

| Stage | Method / Function | Description |
|---|---|---|
| Background removal | `remove_background()` | MediaPipe Selfie Segmentation with morphological smoothing |
| Grayscale | `cv2.cvtColor` | Collapse to single channel |
| Smoothing | `cv2.bilateralFilter(d=9, σ=120, 120)` | Reduce texture noise while preserving edges |
| Edge detection | `cv2.Canny(25, 60)` | Low/high thresholds — sensitive to soft facial gradients |
| Contour extraction | `cv2.findContours` + `approxPolyDP(ε=2.0)` | Trace and simplify edge pixel chains |
| Chaining | `_chain_strokes()` | Greedy merge of nearby stroke endpoints |
| Sorting + flipping | `_sort_strokes()` | Nearest-neighbour ordering with per-stroke direction selection |
| Conversion | `_raw_to_path()` | Convert `(x, y)` tuples to `nav_msgs/Path` with identity orientation |

---

## Notes for UR3 Integration

- **Coordinate transform** — pixel coordinates must be mapped to the robot's TCP workspace. Account for the physical drawing area dimensions and the camera's field of view.
- **Z-axis** — all stroke waypoints have `z = 0`. The motion planning node is responsible for lifting the pen between strokes (i.e. between consecutive Path messages) and pressing down at the start of each one.
- **Stroke ordering** — strokes are published in drawing order. The motion planning node should process them sequentially and not reorder them.
- **Tuning** — if the drawing contains too much detail or noise, increase the Canny thresholds or the `approxPolyDP` epsilon in `portrait_processor.py`. If strokes are too fragmented, increase `chain_threshold` in the `PortraitProcessor` constructor.