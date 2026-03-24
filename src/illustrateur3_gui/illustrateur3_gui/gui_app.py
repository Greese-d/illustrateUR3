import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage, ImageTk

import rclpy
import cv2


class GuiApp:
    def __init__(self, root, ros_node):
        self.root = root
        self.ros_node = ros_node

        self.root.title("illustrateUR3 GUI")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        self.camera_tk_image = None
        self.preview_tk_image = None
        self.freedrive_on = False

        self.setup_styles()
        self.build_layout()

        self.ros_node.state_callback_fn = self.on_state_update
        self.ros_node.camera_callback_fn = self.on_camera_frame
        self.ros_node.preview_callback_fn = self.on_preview_frame

        self.update_ui_for_state("IDLE")

        self.root.after(50, self.poll_ros)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Title.TLabel", font=("Arial", 20, "bold"))
        style.configure("Heading.TLabel", font=("Arial", 12, "bold"))
        style.configure("Status.TLabel", font=("Arial", 11))
        style.configure("Panel.TLabelframe", padding=10)
        style.configure("Panel.TLabelframe.Label", font=("Arial", 11, "bold"))
        style.configure("Big.TButton", font=("Arial", 11), padding=8)
        style.configure("TabSwitch.TButton", font=("Arial", 10, "bold"), padding=6)

        # Hide notebook tabs; we'll use our own buttons instead
        style.layout("TNotebook.Tab", [])

    def build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        # =========================
        # Top Bar
        # =========================
        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=0)

        title_label = ttk.Label(
            top_frame,
            text="Selfie Drawing Robot - Operator GUI",
            style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, sticky="w")

        self.state_label = ttk.Label(
            top_frame,
            text="System State: IDLE",
            style="Heading.TLabel"
        )
        self.state_label.grid(row=0, column=1, sticky="e", padx=(20, 0))

        # =========================
        # Main Content Area
        # =========================
        content_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # -------------------------
        # Camera Panel
        # -------------------------
        self.camera_frame = ttk.LabelFrame(
            content_frame,
            text="Live Camera",
            style="Panel.TLabelframe"
        )
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.camera_frame.columnconfigure(0, weight=1)
        self.camera_frame.rowconfigure(0, weight=1)

        self.camera_placeholder = tk.Label(
            self.camera_frame,
            text="Camera feed will appear here",
            bg="#2b2b2b",
            fg="white",
            font=("Arial", 14),
            relief="ridge",
            bd=2
        )
        self.camera_placeholder.grid(row=0, column=0, sticky="nsew")

        # -------------------------
        # Right Panel
        # -------------------------
        self.preview_frame = ttk.LabelFrame(
            content_frame,
            text="Output",
            style="Panel.TLabelframe"
        )
        self.preview_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(1, weight=1)

        # Custom header buttons
        self.preview_header = ttk.Frame(self.preview_frame)
        self.preview_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.preview_header.columnconfigure(0, weight=0)
        self.preview_header.columnconfigure(1, weight=0)
        self.preview_header.columnconfigure(2, weight=1)

        self.preview_button = ttk.Button(
            self.preview_header,
            text="Preview",
            style="TabSwitch.TButton",
            command=self.open_preview_tab
        )
        self.preview_button.grid(row=0, column=0, padx=(0, 6), sticky="w")

        self.calibration_button = ttk.Button(
            self.preview_header,
            text="Calibration",
            style="TabSwitch.TButton",
            command=self.open_calibration_tab
        )
        self.calibration_button.grid(row=0, column=1, sticky="w")

        self.preview_notebook = ttk.Notebook(self.preview_frame)
        self.preview_notebook.grid(row=1, column=0, sticky="nsew")

        # Preview page
        self.preview_tab = ttk.Frame(self.preview_notebook)
        self.preview_tab.columnconfigure(0, weight=1)
        self.preview_tab.rowconfigure(0, weight=1)

        self.preview_placeholder = tk.Label(
            self.preview_tab,
            text="Processed portrait preview will appear here",
            bg="#2b2b2b",
            fg="white",
            font=("Arial", 14),
            relief="ridge",
            bd=2
        )
        self.preview_placeholder.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self.preview_notebook.add(self.preview_tab, text="Preview")

        # Calibration page
        self.calibration_tab = ttk.Frame(self.preview_notebook, padding=10)
        self.calibration_tab.columnconfigure(0, weight=1)
        self.calibration_tab.columnconfigure(1, weight=1)
        self.calibration_tab.rowconfigure(0, weight=0)
        self.calibration_tab.rowconfigure(1, weight=0)
        self.calibration_tab.rowconfigure(2, weight=1)

        self.calibration_title = ttk.Label(
            self.calibration_tab,
            text="Calibration Tools",
            style="Heading.TLabel"
        )
        self.calibration_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.point1_button = ttk.Button(
            self.calibration_tab,
            text="Point 1",
            style="Big.TButton",
            command=self.on_point_1
        )
        self.point1_button.grid(row=1, column=0, padx=6, pady=6, sticky="ew")

        self.point2_button = ttk.Button(
            self.calibration_tab,
            text="Point 2",
            style="Big.TButton",
            command=self.on_point_2
        )
        self.point2_button.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        self.point3_button = ttk.Button(
            self.calibration_tab,
            text="Point 3",
            style="Big.TButton",
            command=self.on_point_3
        )
        self.point3_button.grid(row=2, column=0, padx=6, pady=6, sticky="new")

        self.freedrive_button = ttk.Button(
            self.calibration_tab,
            text="Free Drive: OFF",
            style="Big.TButton",
            command=self.on_toggle_freedrive
        )
        self.freedrive_button.grid(row=2, column=1, padx=6, pady=6, sticky="new")

        self.preview_notebook.add(self.calibration_tab, text="Calibration")

        # =========================
        # Bottom Area
        # =========================
        bottom_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        bottom_frame.grid(row=2, column=0, sticky="ew")
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        # Controls Panel
        controls_frame = ttk.LabelFrame(
            bottom_frame,
            text="Controls",
            style="Panel.TLabelframe"
        )
        controls_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)

        self.capture_button = ttk.Button(
            controls_frame,
            text="Capture Portrait",
            style="Big.TButton",
            command=self.on_capture
        )
        self.capture_button.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        self.start_button = ttk.Button(
            controls_frame,
            text="Start Drawing",
            style="Big.TButton",
            command=self.on_start
        )
        self.start_button.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        self.stop_button = ttk.Button(
            controls_frame,
            text="Stop Drawing",
            style="Big.TButton",
            command=self.on_stop
        )
        self.stop_button.grid(row=1, column=0, padx=6, pady=6, sticky="ew")

        self.estop_button = tk.Button(
            controls_frame,
            text="E-STOP",
            font=("Arial", 12, "bold"),
            bg="#cc3333",
            fg="white",
            activebackground="#aa2222",
            activeforeground="white",
            command=self.on_estop
        )
        self.estop_button.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")

        # Status Panel
        status_frame = ttk.LabelFrame(
            bottom_frame,
            text="Status & Messages",
            style="Panel.TLabelframe"
        )
        status_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        status_frame.columnconfigure(0, weight=1)

        self.status_text = ttk.Label(
            status_frame,
            text="Ready. Waiting for backend connections.",
            style="Status.TLabel"
        )
        self.status_text.grid(row=0, column=0, sticky="w", padx=4, pady=(4, 8))

        self.progress_label = ttk.Label(
            status_frame,
            text="Drawing Progress: 0%",
            style="Status.TLabel"
        )
        self.progress_label.grid(row=1, column=0, sticky="w", padx=4, pady=4)

        self.log_box = tk.Text(
            status_frame,
            height=8,
            wrap="word",
            state="disabled",
            bg="#f4f4f4",
            font=("Arial", 10)
        )
        self.log_box.grid(row=2, column=0, sticky="ew", padx=4, pady=(8, 4))

        self.add_log("GUI started successfully.")
        self.add_log("Waiting for camera, preview, and robot status topics.")

    def add_log(self, message):
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def on_state_update(self, new_state):
        self.state_label.config(text=f"System State: {new_state}")
        self.status_text.config(text=f"Backend state changed to: {new_state}")
        self.add_log(f"State update received: {new_state}")
        self.update_ui_for_state(new_state)

    def update_ui_for_state(self, state):
        if state == "IDLE":
            self.capture_button.config(state="normal")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
        elif state == "PREVIEW_READY":
            self.capture_button.config(state="normal")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
        elif state == "DRAWING":
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
        elif state == "PROCESSING":
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
        elif state == "ESTOP":
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
        elif state == "ERROR":
            self.capture_button.config(state="normal")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
        else:
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")

    def open_calibration_tab(self):
        self.preview_notebook.select(self.calibration_tab)
        self.status_text.config(text="Calibration view opened.")
        self.add_log("Switched to Calibration view.")

    def open_preview_tab(self):
        self.preview_notebook.select(self.preview_tab)
        self.status_text.config(text="Preview view opened.")
        self.add_log("Switched to Preview view.")

    def on_capture(self):
        self.status_text.config(text="Capture requested.")
        self.add_log("Capture Portrait button pressed.")
        self.ros_node.get_logger().info("Capture Portrait requested")

    def on_start(self):
        self.status_text.config(text="Start drawing requested.")
        self.add_log("Start Drawing button pressed.")
        self.ros_node.get_logger().info("Start Drawing requested")

    def on_stop(self):
        self.status_text.config(text="Stop drawing requested.")
        self.add_log("Stop Drawing button pressed.")
        self.ros_node.get_logger().info("Stop Drawing requested")

    def on_estop(self):
        self.status_text.config(text="Emergency stop requested.")
        self.add_log("E-STOP button pressed.")
        self.ros_node.get_logger().warn("Emergency stop requested")

    def render_frame_to_label(self, frame_bgr, target_label, which):
        label_w = max(target_label.winfo_width(), 320)
        label_h = max(target_label.winfo_height(), 240)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        h, w = frame_rgb.shape[:2]
        scale = min(label_w / w, label_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = PILImage.fromarray(resized)
        tk_img = ImageTk.PhotoImage(image=pil_img)

        target_label.config(image=tk_img, text="")
        target_label.image = tk_img

        if which == "camera":
            self.camera_tk_image = tk_img
        elif which == "preview":
            self.preview_tk_image = tk_img

    def on_camera_frame(self, frame_bgr):
        self.render_frame_to_label(frame_bgr, self.camera_placeholder, "camera")

    def on_preview_frame(self, frame_bgr):
        self.render_frame_to_label(frame_bgr, self.preview_placeholder, "preview")

    def poll_ros(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0.0)
        self.root.after(30, self.poll_ros)

    def on_point_1(self):
        self.status_text.config(text="Calibration Point 1 requested.")
        self.add_log("Calibration Point 1 button pressed.")
        self.ros_node.get_logger().info("Calibration Point 1 requested")

    def on_point_2(self):
        self.status_text.config(text="Calibration Point 2 requested.")
        self.add_log("Calibration Point 2 button pressed.")
        self.ros_node.get_logger().info("Calibration Point 2 requested")

    def on_point_3(self):
        self.status_text.config(text="Calibration Point 3 requested.")
        self.add_log("Calibration Point 3 button pressed.")
        self.ros_node.get_logger().info("Calibration Point 3 requested")

    def on_toggle_freedrive(self):
        self.freedrive_on = not self.freedrive_on
        state_text = "ON" if self.freedrive_on else "OFF"
        self.freedrive_button.config(text=f"Free Drive: {state_text}")
        self.status_text.config(text=f"Free Drive toggled {state_text}.")
        self.add_log(f"Free Drive toggled {state_text}.")
        self.ros_node.get_logger().info(f"Free Drive toggled {state_text}")