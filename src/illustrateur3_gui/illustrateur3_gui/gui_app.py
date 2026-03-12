import tkinter as tk
from tkinter import ttk
import rclpy

#ros2 run illustrateur3_gui gui_main


class GuiApp:
    def __init__(self, root, ros_node):
        self.root = root
        self.ros_node = ros_node

        self.root.title("illustrateUR3 GUI")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        self.setup_styles()
        self.build_layout()

        self.ros_node.state_callback_fn = self.on_state_update
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

        # Camera Panel
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
            width=40,
            height=20,
            relief="ridge",
            bd=2
        )
        self.camera_placeholder.grid(row=0, column=0, sticky="nsew")

        # Preview Panel
        self.preview_frame = ttk.LabelFrame(
            content_frame,
            text="Drawing Preview",
            style="Panel.TLabelframe"
        )
        self.preview_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)

        self.preview_placeholder = tk.Label(
            self.preview_frame,
            text="Processed portrait preview will appear here",
            bg="#2b2b2b",
            fg="white",
            font=("Arial", 14),
            width=40,
            height=20,
            relief="ridge",
            bd=2
        )
        self.preview_placeholder.grid(row=0, column=0, sticky="nsew")

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

        self.start_button.config(state="disabled") #Enable these when i actually have something to draw
        self.stop_button.config(state="disabled") #Enable these when i actually have something to draw

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

    def poll_ros(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0.0)
        self.root.after(50, self.poll_ros)