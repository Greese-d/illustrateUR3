# IllustrateUR3

A ROS2-based selfie drawing robot that uses a UR3 robotic arm to autonomously draw portraits on paper. The system captures a live webcam image, processes it into vector drawing strokes using computer vision, calibrates the drawing surface, and executes the portrait on physical paper using MoveIt2 motion planning.

The operator stands in front of the camera rather than sitting at a laptop — hand gestures provide full control over capture, mask selection, and drawing start/stop, with the GUI serving as a monitoring and advanced control interface.

---

## System Architecture

```
Webcam
  │
  ├──▶ gesture_vision ──────────────────────────▶ /gesture
  │      gesture_recognizer                            │
  │                                                    ▼
  └──▶ portrait_vectorisation                  illustrateur3_gui
         camera_publisher  →  /camera/image_raw   (operator interface)
         image_processing_node                         │
           ├── /capture_snapshot  (Trigger) ◀──────────┤
           ├── /create_portrait   (Trigger) ◀──────────┤
           ├──▶ /portrait/strokes (nav_msgs/Path)       │
           ├──▶ /portrait/preview (sensor_msgs/Image)   │
           └──▶ /portrait/markers (MarkerArray)         │
                      │                                 │
                      ▼                                 │
                 ur3_motion                             │
                   motion_node       ◀──────────────────┤  /start_drawing
                   calibration_node  ◀──────────────────┘  /calibration/command
                      │
                      └──▶ MoveIt2 → UR3 Robot
```

---

## Repository Structure

```
illustrateUR3/
├── src/
│   ├── portrait_vectorisation/   # Perception subsystem
│   ├── ur3_motion/               # Motion & calibration subsystem
│   ├── illustrateur3_gui/        # Operator GUI subsystem
│   ├── gesture_vision/           # Hand gesture recognition
│   └── pymoveit2/                # MoveIt2 Python bindings (vendored)
└── diagrams/
```

---

## Dependencies

### Hardware

| Component | Purpose | Picture |
|---|---|---|
| UR3 Robotic Arm | Main drawing platform | |
| Webcam / RGB Camera | Portrait capture and gesture control input | |
| Ethernet Cable | ROS2 / MoveIt2 communication with the robot | |
| Custom pen end-effector | Holds the drawing pen at the robot tool flange | |
| Pen storage rack | Holds up to 3 pens for automatic colour changing | |
| Whiteboard marker(s) | Drawing instrument attached to end-effector | |
| Custom paper holder | Fixes A4 paper to the drawing surface | |
| A4 paper | Drawing surface | |
| Fasteners | Secure paper to holder during drawing | |
| External PC / workstation | Runs all ROS2 nodes, MoveIt2, GUI, and vision | |

### Software

| Dependency | Purpose |
|---|---|
| Ubuntu 22.04 | Required OS |
| ROS2 Humble | Middleware |
| MoveIt2 | Motion planning |
| `ur_robot_driver` | UR3 hardware interface |
| `ur_moveit_config` | UR3 MoveIt2 configuration |
| `pymoveit2` | Python MoveIt2 bindings (included in `src/`) |
| `opencv-python` | Image processing |
| `mediapipe==0.10.14` | Background removal, face mesh, gesture recognition |
| `numpy==1.26.4` | Numerical operations (pinned for mediapipe compatibility) |
| `scipy` | Rotation transforms in motion node |
| `Pillow` | Image rendering in GUI (tkinter integration) |
| `cv_bridge` | ROS2 ↔ OpenCV image conversion |
| `tkinter` | GUI framework (included with Python) |

Install Python dependencies:

```bash
pip install opencv-python scipy Pillow
python3 -m pip install --user "numpy==1.26.4" "mediapipe==0.10.14"
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Greese-d/illustrateUR3.git ~/ros2_ws/src/illustrateUR3

# 2. Install ROS2 dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# 3. Build
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# 4. Source the workspace
source install/setup.bash
```

---

## Running the System

### Full system launch

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# Simulated robot only (default) — arm moves in RViz2, not in real life
ros2 launch illustrateur3_gui main.launch.py launch_camera:=true

# Real UR3 robot
ros2 launch illustrateur3_gui main.launch.py launch_camera:=true use_fake_hardware:=false

# With rosbag instead of live camera
ros2 launch illustrateur3_gui main.launch.py use_fake_hardware:=false
ros2 bag play <your_bag_file>
```

The launch file starts all nodes automatically: UR robot driver, MoveIt2 + RViz2, GUI, calibration node, motion node, image processing node, gesture recognizer, and optionally the camera publisher.

### Gesture control reference

The system is designed to be operated hands-free — the subject stands in front of the camera and uses gestures instead of touching the laptop.

| Gesture | Action |
|---|---|
| 👍 Thumbs Up | Start 3-second capture countdown, then capture portrait |
| 👎 Thumbs Down | Stop drawing |
| 🤙 Green Giant (index + middle + thumb) | Start drawing |
| ☝️ Point (index finger) | Apply **moustache** mask |
| ✌️ Peace (index + middle) | Apply **hat** mask |
| Three fingers (index + middle + ring) | Apply **glasses** mask |
| Four fingers (all except thumb) | Apply **nose** mask |

A 2-second cooldown applies between gestures to prevent accidental triggers. All gestures can also be triggered via the corresponding GUI buttons.

### Typical drawing workflow

**1. Launch the system** using the command above. RViz2 opens with the UR3 model.

**2. Paper calibration** — in the GUI, open the **Calibration** tab:
- Click **Go Home** to move the arm to its start position
- In RViz2, use the MoveIt2 interactive marker to manually position the pen tip at each of the three paper corners
- Click **Point 1** → **Confirm**, **Point 2** → **Confirm**, **Point 3** → **Confirm**
- Click **Display Paper** to verify — the calibrated paper boundary appears as a marker in RViz2
- The arm will draw a verification rectangle on the paper to confirm accuracy

**3. Position the subject** in front of the webcam. The live feed appears on the left panel of the GUI.

**4. Select a mask (optional)** — use gestures or the GUI Settings tab to apply glasses, hat, moustache, or nose overlays. The masked preview updates live.

**5. Capture portrait** — give a 👍 Thumbs Up gesture (3-second countdown then auto-capture), or click **Capture Portrait** in the GUI.

**6. Start drawing** — give a Green Giant gesture or click **Start Drawing**. The arm will:
- Move to home position
- Draw a verification rectangle around the paper boundary
- Execute all portrait strokes in order
- Return to home position

**7. Stop early if needed** — 👎 Thumbs Down or **Stop Drawing** button. Remaining strokes are kept and drawing can be resumed.

---

## Subsystem Specifics

---

### `portrait_vectorisation` — Perception

#### Purpose

Captures a live webcam portrait, isolates the subject with background removal, extracts facial edges as drawing contours, and publishes them as an ordered sequence of `nav_msgs/Path` strokes ready for the arm to execute. Supports face-anchored artistic overlays controlled by hand gestures, and appends a personalised SVG signature to every portrait.

#### Files

```
portrait_vectorisation/
├── portrait_vectorisation/
│   ├── camera_publisher.py       # Webcam capture node
│   ├── image_processing_node.py  # On-demand portrait processing node
│   ├── portrait_processor.py     # PortraitProcessor CV pipeline library
│   └── sig_gen.py                # SVG → stroke parser for signature overlay
├── masks/                        # RGBA PNG overlays (glasses, hat, moustache, nose)
└── signature/                    # SVG signature files
```

#### Nodes

**`camera_publisher`**

Captures frames from a webcam and publishes them as `sensor_msgs/Image`.

| | |
|---|---|
| Publishes | `/camera/image_raw` (`sensor_msgs/Image`, bgr8) |

| Parameter | Default | Description |
|---|---|---|
| `device` | `1` | Camera device index. Change to `0` if not detected |
| `fps` | `30.0` | Target capture frame rate |
| `width` / `height` | `1920` / `1080` | Requested capture resolution |

**`image_processing_node`**

Applies mask overlays to every incoming frame and publishes a live masked preview. Processes snapshots into portrait strokes on demand when triggered by a service call. All computationally heavy work (background removal, edge detection, stroke planning) only runs when `/create_portrait` is called — not on every frame.

| | |
|---|---|
| Subscribes | `/camera/image_raw`, `/camera/masked_preview` |
| Publishes | `/camera/snapshot`, `/camera/masked_preview`, `/portrait/preview`, `/portrait/strokes`, `/portrait/markers` |
| Services | `/capture_snapshot`, `/create_portrait` (both `std_srvs/Trigger`) |

Both services return `success: bool` and `message: string`. `/create_portrait` auto-captures a fresh snapshot if the buffer is empty or already consumed — calling it without a prior `/capture_snapshot` is safe.

| Parameter | Default | Description |
|---|---|---|
| `device` | `1` | Camera device index. Change to `0` if not detected |
| `stroke_publish_delay` | `0.05` s | Delay between consecutive stroke publishes to ensure ordered delivery |
| `mask_type` | `"none"` | Active overlay: `none`, `glasses`, `hat`, `moustache`, `nose`. Hot-swappable via gesture or `ros2 param set` |
| `min_stroke_length` | `20.0` px | Contours shorter than this are discarded as noise |
| `signature_scale` | `0.30` | Signature size as a fraction of image dimensions |

#### `PortraitProcessor` library (`portrait_processor.py`)

The core CV pipeline. Can be used independently of ROS2.

```python
from portrait_vectorisation.portrait_processor import PortraitProcessor
import cv2

processor = PortraitProcessor(
    line_thickness=4,      # px — rendered stroke width; also controls chain threshold (× 2)
    sort_strokes=True,     # nearest-neighbour stroke ordering
    min_stroke_length=20.0,
    signature_scale=0.40,
)

image = cv2.imread('photo.jpg')
canvas, strokes = processor.process(image, frame_id='camera_frame', mask_type='none')
# canvas  — np.ndarray BGR preview image
# strokes — List[nav_msgs/Path], ordered stroke paths ready for the arm

cv2.imwrite('preview.png', canvas)
processor.close()
```

**Processing pipeline:**

| Stage | Parameters | Notes |
|---|---|---|
| Background removal | MediaPipe `model_selection=1`, threshold `0.4` | Morphological open/close with 7×7 kernel; sets background to white |
| Mask overlay | Face-anchored via FaceMesh landmarks | Scales and rotates with head angle; silently skips if face not detected |
| Bilateral filter | `d=9, σColor=120, σSpace=120` | Smooths skin texture while preserving facial edges |
| Canny edge detection | `low=20, high=50` | Hysteresis connects weak edges to strong neighbours |
| Contour extraction | `approxPolyDP(ε=2.0)`, filter `< min_stroke_length` | `RETR_LIST + CHAIN_APPROX_NONE` |
| Stroke chaining | Threshold = `line_thickness × 2` px | Greedy merge, 4 endpoint combinations, direction flip on join |
| Signature injection | `signature_scale` fraction | SVG parsed by `sig_gen.py`; placed in least-occupied image corner |
| Sort + direction flip | Nearest-neighbour greedy | Minimises total arm travel between strokes |
| Render | BGR canvas, `line_thickness` px strokes | Extensible per-stroke colour via `_stroke_colour()` |

**`sig_gen.py`** — parses an SVG file into drawable strokes. Supports `path`, `circle`, `line`, and `polyline` elements including `transform` attributes. Used internally at startup to load `signature/signature.svg`.

#### Running independently

```bash
# With live camera
ros2 run portrait_vectorisation camera_publisher --ros-args -p device:=1
ros2 run portrait_vectorisation image_processing_node

# With rosbag
ros2 bag play <bag_file>
ros2 run portrait_vectorisation image_processing_node

# Trigger manually
ros2 service call /capture_snapshot std_srvs/srv/Trigger {}
ros2 service call /create_portrait std_srvs/srv/Trigger {}

# Change mask at runtime
ros2 param set /image_processing_node mask_type glasses
```

#### RViz2 visualisation

Add a **MarkerArray** display pointed at `/portrait/markers`. All strokes are sent in a single message so the full portrait renders simultaneously. Coordinates are in pixel space — scale your fixed frame accordingly.

#### Known limitations

- **Lighting** — MediaPipe degrades under inconsistent or low lighting. Canny thresholds (`20/50`) are tuned for controlled indoor use; adjust in `portrait_processor.py` for other environments.
- **Single subject** — multiple people in frame are segmented and processed together as one portrait.
- **Mask anchoring** — requires a near-frontal face visible to FaceMesh. At extreme angles or with partial occlusion, `apply_mask` silently returns the unmodified image.
- **Stroke count** — a single subject typically produces 50–120 strokes; maximum around 250. The stroke publisher queue is 50 with a 50 ms inter-stroke delay. The downstream subscriber queue must be sized to match.
- **Coordinate space** — all waypoints are in pixel coordinates. The motion node handles the transform to robot workspace.

---

### `ur3_motion` — Motion & Calibration

#### Purpose

Controls all physical robot motion: paper surface calibration, pixel-to-robot coordinate mapping, portrait stroke execution, pen attach/detach for colour changing, home positioning, and vertical movement. Receives stroke paths from `portrait_vectorisation` and executes them on paper using MoveIt2.

#### Files

```
ur3_motion/
└── ur3_motion/
    ├── motion_node.py       # Main drawing, motion, and pen-change node
    └── calibration_node.py  # Paper and pen-storage calibration node
```

#### Nodes

**`motion_node`**

Subscribes to `/portrait/strokes`, queues incoming strokes, and draws them on paper when `/start_drawing` is called. Also handles pen colour changes, home positioning, and RViz2 visualisation of the paper surface and drawing progress.

| | |
|---|---|
| Subscribes | `/portrait/strokes` (`nav_msgs/Path`), `/calibration/command` (`std_msgs/String`) |
| Publishes | `/drawing/status` (`std_msgs/String`), `/state` (`std_msgs/String`), `/paper_marker` (`visualization_msgs/Marker`), `/urscript_interface/script_command` (`std_msgs/String`) |
| Services | `/start_drawing`, `/stop_drawing`, `/go_home`, `/clear_strokes` (all `std_srvs/Trigger`) |

**`calibration_node`**

Guides the operator through three-point paper calibration and pen storage calibration.

| | |
|---|---|
| Subscribes | `/calibration/command` (`std_msgs/String`) |
| Publishes | `/calibration/status` (`std_msgs/String`) |

#### Calibration procedure

Paper calibration uses three touch points on the drawing surface:

| Point | Meaning |
|---|---|
| P1 | Origin corner of the paper |
| P2 | Defines the paper X-axis and width |
| P3 | Defines the paper Y-axis and height |

From these, the node computes paper width/height, centre position, X/Y/Z axes, orientation quaternion, and TCP offset. Results are saved to `data/paper_calibration.json` and loaded by the motion node at draw time.

In the GUI Calibration tab: click **Go Home**, then use the MoveIt2 interactive marker in RViz2 to position the pen tip at each corner — click **Point 1** → **Confirm**, then repeat for P2 and P3. Use **Display Paper** to verify — the calibrated boundary appears in RViz2. The arm draws a verification rectangle to confirm accuracy.

#### Pixel-to-paper coordinate mapping

When `/start_drawing` is called, `motion_node` computes the transform from pixel space to robot paper frame:

- Portrait bounding box is scaled to fill 90% of the drawable paper area (with a margin of `min(1.5 cm, 20%` of paper dimension))
- A fixed 90° rotation is applied to orient the portrait correctly on the paper
- Each pixel `(x, y)` maps to a 3D paper-plane point via calibrated axes and a metres-per-pixel scale
- The pen lifts 20 mm between strokes and descends at the start of each new stroke

#### Drawing sequence

When `/start_drawing` is triggered:

1. Robot moves to home joint position
2. Draws a verification rectangle around the calibrated paper boundary
3. Executes all queued portrait strokes in order
4. Returns to home position

#### Pen colour changing

Pen storage positions are calibrated independently (up to 3 pens). Once calibrated and saved to `data/pen_storage_calibration.json`, the arm automatically attaches and detaches pens. The GUI enforces valid state — attaching Pen 1 disables all other pen buttons until Pen 1 is detached.

#### Running independently

```bash
ros2 run ur3_motion motion_node
ros2 run ur3_motion calibration_node

ros2 service call /start_drawing std_srvs/srv/Trigger {}
ros2 service call /stop_drawing std_srvs/srv/Trigger {}
ros2 service call /go_home std_srvs/srv/Trigger {}
ros2 service call /clear_strokes std_srvs/srv/Trigger {}
```

#### Key parameters (in `motion_node.py`)

| Parameter | Value | Description |
|---|---|---|
| `tcp_offset` | `0.17 m` | Distance from `tool0` to pen tip. Must match the physical pen |
| `max_velocity` | `0.05` | MoveIt2 velocity scaling |
| `max_acceleration` | `0.05` | MoveIt2 acceleration scaling |
| Home joint pose | `[1.57, -1.57, 1.57, -1.57, -1.57, 0.0]` rad | Safe start/end position |
| Lift height | `0.02 m` | Pen lift between strokes |
| Portrait scale | `0.9` | Portrait fills 90% of drawable paper area |
| Portrait rotation | `90°` | Applied during pixel-to-paper mapping |
| Wrist rotation speed | `45 deg/s` | URScript rate for wrist rotation |
| Wrist rotation acceleration | `90 deg/s²` | URScript acceleration for wrist rotation |

#### Known limitations

- Paper calibration must be completed before any drawing sequence.
- Calibration JSON files must be present in the workspace `data/` folder at startup.
- `tcp_offset` must match the physical pen length precisely — errors cause the pen to miss the paper or press too hard.
- Pen attach/detach depends on accurately calibrated pen-ready poses. Always be ready to use the hardware E-stop during pen changes.
- `/stop_drawing` cancels active motion where possible, but completion depends on MoveIt2 and robot controller response time.
- The GUI E-stop button logs the request only. For safety-critical situations use the physical hardware E-stop.

---

### `illustrateur3_gui` — Operator Interface

#### Purpose

Provides the operator-facing control interface for the entire system. Displays the live camera feed, portrait preview, drawing status, and calibration feedback in a single window. Bridges all subsystem service calls, topic subscriptions, and parameter updates through a single ROS2 node, and forwards gesture commands received from `gesture_vision` to the appropriate system actions.

#### Files

```
illustrateur3_gui/
├── illustrateur3_gui/
│   ├── gui_app.py    # Tkinter UI — all tabs, buttons, image panels, gesture handling
│   ├── ros_node.py   # ROS2 node — all subscriptions, publishers, service clients
│   └── main.py       # Entry point
└── launch/
    └── main.launch.py
```

#### Node: `gui_node`

| | |
|---|---|
| Subscribes | `/gesture` (`std_msgs/String`), `/state` (`std_msgs/String`), `/camera/image_raw` (`sensor_msgs/Image`), `/portrait/preview` (`sensor_msgs/Image`), `/calibration/status` (`std_msgs/String`) |
| Publishes | `/calibration/command` (`std_msgs/String`) |
| Service clients | `/create_portrait`, `/start_drawing`, `/stop_drawing`, `/go_home`, `/clear_strokes` (all `std_srvs/Trigger`) |
| Parameter client | `/image_processing_node` — updates `mask_type` at runtime via `SetParameters` |

#### Interface layout

The GUI is divided into a persistent left panel (live camera feed) and a right panel with four switchable tabs:

**Preview tab** — shows the processed portrait preview image after `/create_portrait` completes.

**Calibration tab** — paper calibration controls:
- **Go Home** — moves robot to home position
- **Point 1 / Point 2 / Point 3** — record each calibration touch point (robot must be positioned manually in RViz2 first)
- **Confirm** — confirm the current point
- **Display Paper / Display Axes** — toggle RViz2 paper boundary and axis markers
- **TCP Offset** — set pen tip distance from flange
- **Move Vertical / Rotate End-Effector** — fine adjustment controls
- **Attach / Detach Pen 1–3** — pen colour change controls

**Live Drawing tab** — primary operation controls:
- **Capture Portrait** — triggers `/create_portrait`
- **Start Drawing** — triggers `/start_drawing`
- **Stop Drawing** — triggers `/stop_drawing`
- **E-STOP** — emergency stop button (logs request; use hardware E-stop for true safety)
- Status and message log panel

**Settings tab** — additional system configuration.

#### Gesture control

The GUI listens to `/gesture` published by `gesture_vision` and maps gestures to system actions with a 2-second cooldown between triggers:

| Gesture | GUI action |
|---|---|
| THUMBS_UP | 3-second countdown then **Capture Portrait** |
| THUMBS_DOWN | **Stop Drawing** |
| GREEN GIANT | **Start Drawing** |
| POINT | Set mask → `moustache` |
| PEACE | Set mask → `hat` |
| THREE | Set mask → `glasses` |
| FOUR | Set mask → `nose` |

Gesture-triggered actions respect button state — e.g. Start Drawing via gesture is ignored if the start button is currently disabled (arm already drawing).

#### Running independently

```bash
ros2 run illustrateur3_gui gui_main
```

---

### `gesture_vision` — Hand Gesture Recognition

#### Purpose

Publishes detected hand gestures from the webcam feed as string labels on `/gesture`. The gesture recognizer reads from `/camera/masked_preview` (so it sees the same view as the image processor) and uses MediaPipe Hands for landmark detection with a custom rule-based classifier.

> **Note:** `gesture_vision` requires a Python virtual environment because MediaPipe Tasks uses a different NumPy ABI to the system `cv_bridge`. See setup instructions below.

#### Nodes

**`gesture_recognizer`**

| | |
|---|---|
| Subscribes | `/camera/masked_preview` (`sensor_msgs/Image`) |
| Publishes | `/gesture` (`std_msgs/String`), `/gesture/debug_image` (`sensor_msgs/Image`) |

Gestures published: `THUMBS_UP`, `THUMBS_DOWN`, `OPEN_HAND`, `FIST`, `POINT`, `PEACE`, `GREEN GIANT`, `THREE`, `FOUR`, `CALL ME`, `UNKNOWN`, `NONE`.

A debounce mechanism requires a gesture to remain stable for two consecutive frames before it is published, and suppresses repeated publishing of the same gesture.

#### Setup

```bash
# Create virtual environment for gesture recognizer
python3 -m venv ~/venvs/mp_ros
source ~/venvs/mp_ros/bin/activate
pip install "numpy<2" mediapipe==0.10.14

# Terminal 1 — camera (system Python)
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run gesture_vision webcam_publisher --ros-args -p device:=0

# Terminal 2 — gesture recognizer (venv REQUIRED)
export ROS_DOMAIN_ID=0
source ~/venvs/mp_ros/bin/activate
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python -m gesture_vision.gesture_recognizer
```

---

## Troubleshooting & FAQs

**Camera not detected:**
Change `device` from `1` to `0`:
```bash
ros2 run portrait_vectorisation camera_publisher --ros-args -p device:=0
```

**Portrait preview is blank or contains only noise:**
The subject may not be detected by MediaPipe. Ensure controlled lighting with a distinct background. Try raising the Canny thresholds (`20/50` → `40/80`) in `portrait_processor.py`.

**Strokes are not received by the motion node:**
Increase `stroke_publish_delay` (try `0.1` s). Verify publishing with:
```bash
ros2 topic echo /portrait/strokes
```

**Robot does not move after `/start_drawing`:**
Ensure paper calibration is complete and `data/paper_calibration.json` exists. Check for errors:
```bash
ros2 topic echo /state
ros2 topic echo /drawing/status
```
If using the real robot, confirm `use_fake_hardware:=false` was passed to the launch command.

**Drawing is offset or distorted on paper:**
Re-run paper calibration. Verify `tcp_offset` in `motion_node.py` matches the physical pen length. Ensure P1, P2, P3 were recorded with the pen tip touching the paper surface.

**Mask overlay is not appearing:**
Ensure the face is clearly visible and near-frontal. Check the active mask:
```bash
ros2 param get /image_processing_node mask_type
```

**Gestures are not being detected:**
Confirm the gesture recognizer is running in the virtual environment (not system Python). Check `/gesture` topic:
```bash
ros2 topic echo /gesture
```

**RViz2 only shows one stroke at a time:**
Subscribe to `/portrait/markers` (MarkerArray) instead of `/portrait/strokes`. The MarkerArray sends all strokes in a single message.
