<<<<<<< HEAD
# gesture_vision (ROS 2 Humble) — Webcam + Hand Gesture Recognition

This package publishes webcam images to ROS 2 and runs a MediaPipe-based hand gesture recognizer that outputs a ROS topic.

Why two runtimes?
- The **webcam publisher** uses `cv_bridge` (compiled against NumPy 1.x) and is safest on **system Python**.
- The **gesture recognizer** uses **MediaPipe Tasks** and runs in a **Python venv** with `numpy<2`.

Tested on:
- Ubuntu 22.04 (Jammy)
- ROS 2 Humble

---
# gesture_vision# gesture_vision (ROS 2 Humble) — Webcam + Hand Gesture Recognition

This package publishes webcam images to ROS 2 and runs a MediaPipe-based hand gesture recognizer that outputs a ROS topic.

Why two runtimes?
- The **webcam publisher** uses `cv_bridge` (compiled against NumPy 1.x) and is safest on **system Python**.
- The **gesture recognizer** uses **MediaPipe Tasks** and runs in a **Python venv** with `numpy<2`.

Tested on:
- Ubuntu 22.04 (Jammy)
- ROS 2 Humble

---

## Topics

- `/camera/image_raw` — `sensor_msgs/msg/Image` (published by webcam node)
- `/gesture` — `std_msgs/msg/String` (published by gesture node)

Gestures currently published (simple heuristic):
- `OPEN_PALM`
- `FIST`
- `POINT`
- `UNKNOWN`
- `NONE`

---

## 1) Prerequisites (System)

Install ROS deps:
```bash
sudo apt update
sudo apt install -y ros-humble-cv-bridge python3-colcon-common-extensions
```

Webcam + Hand Gesture Recognition using **ROS 2 Humble** and **MediaPipe**

This package:
- Publishes webcam frames to `/camera/image_raw`
- Detects hand gestures using MediaPipe
- Publishes gesture labels to `/gesture`
- Optionally publishes a debug image with landmarks overlay

Tested on:
- Ubuntu 22.04 (Jammy)
- ROS 2 Humble

---

# Requirements

## System Requirements
- Ubuntu 22.04
- ROS 2 Humble

## ROS Packages Required

Install these on a new machine:

```bash
sudo apt update
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-rqt-image-view \
    python3-colcon-common-extensions
```
## Topics

- `/camera/image_raw` — `sensor_msgs/msg/Image` (published by webcam node)
- `/gesture` — `std_msgs/msg/String` (published by gesture node)

Gestures currently published (simple heuristic):
- `OPEN_PALM`
- `FIST`
- `POINT`
- `UNKNOWN`
- `NONE`

## Quick Start (Daily Run Instructions)

Use this section anytime you restart your computer or open new terminals.

This assumes:
- Workspace is at `~/ros2_ws`
- Virtual environment is at `~/venvs/mp_ros`
- Package is already built

You need **3 terminals**.

⚠ IMPORTANT: In ALL terminals, run this first:
```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
```


## 🟢 TERMINAL 1 — Start Webcam Publisher (System Python)

Copy and paste:
```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run gesture_vision webcam_publisher --ros-args -p device:=0 ## Change to device=1 if no camera is detected
```

## 🔵 TERMINAL 2 — Start Gesture Recognizer (MediaPipe venv REQUIRED)
⚠ This must use the virtual environment.

Copy and paste:
```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

source ~/venvs/mp_ros/bin/activate
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

python -m gesture_vision.gesture_recognizer
```
## TERMINAL 3 — Start GUI

Copy and paste:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run illustrateur3_gui gui_main
```

## 🛠 Build the Project
Copy and paste:
```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

colcon build --symlink-install
```
=======
>>>>>>> c914428a028a4c1df5802b951365dff7243293bd

