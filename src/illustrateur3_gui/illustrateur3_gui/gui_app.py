import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage, ImageTk
import json

import rclpy
import cv2
import time

class GuiApp:
    def __init__(self, root, ros_node):
        self.root = root
        self.ros_node = ros_node

        self.root.title("illustrateUR3 GUI")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        self.camera_tk_image = None
        self.preview_tk_image = None
        self.live_drawing_tk_image = None
        self.last_gesture_time = 0.0
        self.gesture_cooldown_sec = 2.0
        self.show_paper_var = tk.BooleanVar(value=True)
        self.show_axes_var = tk.BooleanVar(value=False)
        self.change_color_var = tk.BooleanVar(value=False)
        self.tcp_offset_var = tk.StringVar(value="0.12")
        self.move_vertical_var = tk.StringVar(value="0.02")
        self.rotate_end_effector_var = tk.StringVar(value="90")
        self.pending_capture = False
        self.go_home_pending = False
        self.stop_pending = False
        self.can_start_drawing = False
        self.resume_available = False
        self.displayed_state = "IDLE"
        self.attached_pen_index = None

        self.setup_styles()
        self.build_layout()

        self.ros_node.state_callback_fn = self.on_state_update
        self.ros_node.camera_callback_fn = self.on_camera_frame
        self.ros_node.preview_callback_fn = self.on_preview_frame
        self.ros_node.live_drawing_callback_fn = self.on_live_drawing_frame
        self.ros_node.calibration_status_callback_fn = self.on_calibration_status
        self.ros_node.gesture_callback_fn = self.on_gesture_detected

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
        self.preview_header.columnconfigure(2, weight=0)
        self.preview_header.columnconfigure(3, weight=0)
        self.preview_header.columnconfigure(4, weight=1)

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

        self.live_drawing_button = ttk.Button(
            self.preview_header,
            text="Live-Drawing",
            style="TabSwitch.TButton",
            command=self.open_live_drawing_tab
        )
        self.live_drawing_button.grid(row=0, column=2, padx=(6, 0), sticky="w")

        self.settings_button = ttk.Button(
            self.preview_header,
            text="Settings",
            style="TabSwitch.TButton",
            command=self.open_settings_tab
        )
        self.settings_button.grid(row=0, column=3, padx=(6, 0), sticky="w")

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
        self.calibration_tab.columnconfigure(0, weight=1, uniform="calibration")
        self.calibration_tab.columnconfigure(1, weight=1, uniform="calibration")
        self.calibration_tab.rowconfigure(0, weight=0)
        self.calibration_tab.rowconfigure(1, weight=0)
        self.calibration_tab.rowconfigure(2, weight=0)
        self.calibration_tab.rowconfigure(3, weight=0)
        self.calibration_tab.rowconfigure(4, weight=0)
        self.calibration_tab.rowconfigure(5, weight=0)
        self.calibration_tab.rowconfigure(6, weight=0)
        self.calibration_title = ttk.Label(
            self.calibration_tab,
            text="Calibration Tools",
            style="Heading.TLabel"
        )
        self.calibration_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Live-Drawing page
        self.live_drawing_tab = ttk.Frame(self.preview_notebook)
        self.live_drawing_tab.columnconfigure(0, weight=1)
        self.live_drawing_tab.rowconfigure(0, weight=1)

        self.live_drawing_placeholder = tk.Label(
            self.live_drawing_tab,
            text="Live drawing will appear here in real time",
            bg="#2b2b2b",
            fg="white",
            font=("Arial", 14),
            relief="ridge",
            bd=2
        )
        self.live_drawing_placeholder.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self.preview_notebook.add(self.live_drawing_tab, text="Live-Drawing")

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
        self.point2_button.grid(row=2, column=0, padx=6, pady=6, sticky="ew")

        self.point3_button = ttk.Button(
            self.calibration_tab,
            text="Point 3",
            style="Big.TButton",
            command=self.on_point_3
        )
        self.point3_button.grid(row=3, column=0, padx=6, pady=6, sticky="ew")

        self.confirm_button = ttk.Button(
            self.calibration_tab,
            text="Confirm Point",
            style="Big.TButton",
            command=self.on_confirm
        )
        self.confirm_button.grid(row=4, column=0, padx=6, pady=6, sticky="ew")

        self.pen1_button = ttk.Button(
            self.calibration_tab,
            text="Pen 1",
            style="Big.TButton",
            command=lambda: self.on_pen_selected(1)
        )
        self.pen1_button.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        self.pen2_button = ttk.Button(
            self.calibration_tab,
            text="Pen 2",
            style="Big.TButton",
            command=lambda: self.on_pen_selected(2)
        )
        self.pen2_button.grid(row=2, column=1, padx=6, pady=6, sticky="ew")

        self.pen3_button = ttk.Button(
            self.calibration_tab,
            text="Pen 3",
            style="Big.TButton",
            command=lambda: self.on_pen_selected(3)
        )
        self.pen3_button.grid(row=3, column=1, padx=6, pady=6, sticky="ew")

        self.go_home_button = ttk.Button(
            self.calibration_tab,
            text="Go Home",
            style="Big.TButton",
            command=self.on_go_home
        )
        self.go_home_button.grid(row=4, column=1, padx=6, pady=6, sticky="ew")

        self.show_paper_check = ttk.Checkbutton(
            self.calibration_tab,
            text="Display Paper in Simulation",
            variable=self.show_paper_var,
            command=self.on_toggle_show_paper
        )
        self.show_paper_check.grid(row=5, column=0, columnspan=2, padx=6, pady=(10, 4), sticky="w")

        self.show_axes_check = ttk.Checkbutton(
            self.calibration_tab,
            text="Display Paper XYZ Axes",
            variable=self.show_axes_var,
            command=self.on_toggle_show_axes
        )
        self.show_axes_check.grid(row=6, column=0, columnspan=2, padx=6, pady=4, sticky="w")

        self.preview_notebook.add(self.calibration_tab, text="Calibration")

        # Settings page
        self.settings_tab = ttk.Frame(self.preview_notebook, padding=10)
        self.settings_tab.columnconfigure(0, weight=0)
        self.settings_tab.columnconfigure(1, weight=0)
        self.settings_tab.columnconfigure(2, weight=1)

        self.settings_title = ttk.Label(
            self.settings_tab,
            text="Settings",
            style="Heading.TLabel"
        )
        self.settings_title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        self.tcp_offset_label = ttk.Label(
            self.settings_tab,
            text="TCP Offset (m)",
            style="Status.TLabel"
        )
        self.tcp_offset_label.grid(row=1, column=0, padx=(6, 8), pady=6, sticky="w")

        self.tcp_offset_entry = ttk.Entry(
            self.settings_tab,
            textvariable=self.tcp_offset_var,
            width=12
        )
        self.tcp_offset_entry.grid(row=1, column=1, padx=(0, 6), pady=6, sticky="w", ipady=7)

        self.tcp_offset_send_button = ttk.Button(
            self.settings_tab,
            text="Send",
            style="Big.TButton",
            command=self.on_send_tcp_offset,
            width=12
        )
        self.tcp_offset_send_button.grid(row=1, column=2, padx=(0, 6), pady=6, sticky="w")

        self.move_vertical_label = ttk.Label(
            self.settings_tab,
            text="Move Vertical (m)",
            style="Status.TLabel"
        )
        self.move_vertical_label.grid(row=2, column=0, padx=(6, 8), pady=6, sticky="w")

        self.move_vertical_entry = ttk.Entry(
            self.settings_tab,
            textvariable=self.move_vertical_var,
            width=12
        )
        self.move_vertical_entry.grid(row=2, column=1, padx=(0, 6), pady=6, sticky="w", ipady=7)

        self.move_vertical_send_button = ttk.Button(
            self.settings_tab,
            text="Send",
            style="Big.TButton",
            command=self.on_send_move_vertical,
            width=12
        )
        self.move_vertical_send_button.grid(row=2, column=2, padx=(0, 6), pady=6, sticky="w")

        self.rotate_end_effector_label = ttk.Label(
            self.settings_tab,
            text="Rotate End Effector (deg)",
            style="Status.TLabel"
        )
        self.rotate_end_effector_label.grid(row=3, column=0, padx=(6, 8), pady=6, sticky="w")

        self.rotate_end_effector_entry = ttk.Entry(
            self.settings_tab,
            textvariable=self.rotate_end_effector_var,
            width=12
        )
        self.rotate_end_effector_entry.grid(row=3, column=1, padx=(0, 6), pady=6, sticky="w", ipady=7)

        self.rotate_end_effector_send_button = ttk.Button(
            self.settings_tab,
            text="Send",
            style="Big.TButton",
            command=self.on_send_rotate_end_effector,
            width=12
        )
        self.rotate_end_effector_send_button.grid(row=3, column=2, padx=(0, 6), pady=6, sticky="w")

        self.change_color_check = ttk.Checkbutton(
            self.settings_tab,
            text="Change Color",
            variable=self.change_color_var,
            command=self.on_toggle_change_color
        )
        self.change_color_check.grid(row=4, column=0, columnspan=3, padx=6, pady=(14, 6), sticky="w")

        self.change_color_frame = ttk.Frame(self.settings_tab)
        self.change_color_frame.columnconfigure(0, weight=1)
        self.change_color_frame.columnconfigure(1, weight=1)

        self.pen_change_buttons = []
        self.attach_pen_buttons = {}
        self.detach_pen_buttons = {}
        for row, pen_index in enumerate((1, 2, 3)):
            attach_button = ttk.Button(
                self.change_color_frame,
                text=f"Attach Pen {pen_index}",
                style="Big.TButton",
                command=lambda index=pen_index: self.on_attach_pen(index)
            )
            attach_button.grid(row=row, column=0, padx=(6, 4), pady=6, sticky="ew")

            detach_button = ttk.Button(
                self.change_color_frame,
                text=f"Detach Pen {pen_index}",
                style="Big.TButton",
                command=lambda index=pen_index: self.on_detach_pen(index)
            )
            detach_button.grid(row=row, column=1, padx=(4, 6), pady=6, sticky="ew")

            self.pen_change_buttons.extend((attach_button, detach_button))
            self.attach_pen_buttons[pen_index] = attach_button
            self.detach_pen_buttons[pen_index] = detach_button

        self.preview_notebook.add(self.settings_tab, text="Settings")

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

    def on_toggle_show_paper(self):
        enabled = self.show_paper_var.get()
        self.ros_node.toggle_paper_display(enabled)

        state_text = "ON" if enabled else "OFF"
        self.status_text.config(text=f"Paper display: {state_text}")
        self.add_log(f"Paper display toggled {state_text}.")
        self.ros_node.get_logger().info(f"Paper display toggled {state_text}")

    def on_toggle_show_axes(self):
        enabled = self.show_axes_var.get()
        self.ros_node.toggle_axes_display(enabled)

        state_text = "ON" if enabled else "OFF"
        self.status_text.config(text=f"Paper XYZ axes display: {state_text}")
        self.add_log(f"Paper XYZ axes display toggled {state_text}.")
        self.ros_node.get_logger().info(f"Paper XYZ axes display toggled {state_text}")

    def on_mask_response(self, success, message):
        self.root.after(0, lambda: self._handle_mask_response(success, message))

    def _handle_mask_response(self, success, message):
        self.add_log(f"Set mask response: success={success}, message='{message}'")
        if success:
            self.status_text.config(text=message if message else "Mask updated.")
        else:
            self.status_text.config(text=message if message else "Failed to update mask.")

    def on_state_update(self, new_state):
        previous_state = self.displayed_state
        if new_state == "IDLE" and previous_state == "DRAWING":
            self.can_start_drawing = self.stop_pending or self.resume_available

        self.state_label.config(text=f"System State: {new_state}")
        self.status_text.config(text=f"Backend state changed to: {new_state}")
        self.add_log(f"State update received: {new_state}")
        self.update_ui_for_state(new_state)
        self.displayed_state = new_state

    def update_ui_for_state(self, state):
        if state == "IDLE":
            self.capture_button.config(state="normal")
            self.start_button.config(
                state="normal" if self.can_start_drawing else "disabled"
            )
            self.stop_button.config(state="disabled")
            self.go_home_button.config(state="normal")
            self.tcp_offset_send_button.config(state="normal")
            self.move_vertical_send_button.config(state="normal")
            self.rotate_end_effector_send_button.config(state="normal")
            self.update_pen_change_button_states(True)
        elif state == "PREVIEW_READY":
            self.capture_button.config(state="normal")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.go_home_button.config(state="normal")
            self.tcp_offset_send_button.config(state="normal")
            self.move_vertical_send_button.config(state="normal")
            self.rotate_end_effector_send_button.config(state="normal")
            self.update_pen_change_button_states(True)
        elif state == "DRAWING":
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.go_home_button.config(state="disabled")
            self.tcp_offset_send_button.config(state="disabled")
            self.move_vertical_send_button.config(state="disabled")
            self.rotate_end_effector_send_button.config(state="disabled")
            self.update_pen_change_button_states(False)
        elif state == "GOING_HOME":
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.go_home_button.config(state="disabled")
            self.tcp_offset_send_button.config(state="disabled")
            self.move_vertical_send_button.config(state="disabled")
            self.rotate_end_effector_send_button.config(state="disabled")
            self.update_pen_change_button_states(False)
        elif state == "PROCESSING":
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.go_home_button.config(state="disabled")
            self.tcp_offset_send_button.config(state="disabled")
            self.move_vertical_send_button.config(state="disabled")
            self.rotate_end_effector_send_button.config(state="disabled")
            self.update_pen_change_button_states(False)
        elif state == "ESTOP":
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.go_home_button.config(state="disabled")
            self.tcp_offset_send_button.config(state="disabled")
            self.move_vertical_send_button.config(state="disabled")
            self.rotate_end_effector_send_button.config(state="disabled")
            self.update_pen_change_button_states(False)
        elif state == "ERROR":
            self.capture_button.config(state="normal")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.go_home_button.config(state="normal")
            self.tcp_offset_send_button.config(state="normal")
            self.move_vertical_send_button.config(state="normal")
            self.rotate_end_effector_send_button.config(state="normal")
            self.update_pen_change_button_states(True)
        else:
            self.capture_button.config(state="disabled")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.go_home_button.config(state="disabled")
            self.tcp_offset_send_button.config(state="disabled")
            self.move_vertical_send_button.config(state="disabled")
            self.rotate_end_effector_send_button.config(state="disabled")
            self.update_pen_change_button_states(False)

    def set_pen_change_buttons_state(self, state):
        for button in self.pen_change_buttons:
            button.config(state=state)

    def update_pen_change_button_states(self, controls_enabled=True):
        if not controls_enabled:
            self.set_pen_change_buttons_state("disabled")
            return

        if self.attached_pen_index is None:
            for pen_index in (1, 2, 3):
                self.attach_pen_buttons[pen_index].config(state="normal")
                self.detach_pen_buttons[pen_index].config(state="disabled")
            return

        for pen_index in (1, 2, 3):
            self.attach_pen_buttons[pen_index].config(state="disabled")
            detach_state = "normal" if pen_index == self.attached_pen_index else "disabled"
            self.detach_pen_buttons[pen_index].config(state=detach_state)

    def open_calibration_tab(self):
        self.preview_notebook.select(self.calibration_tab)
        self.status_text.config(text="Calibration view opened.")
        self.add_log("Switched to Calibration view.")

    def open_preview_tab(self):
        self.preview_notebook.select(self.preview_tab)
        self.status_text.config(text="Preview view opened.")
        self.add_log("Switched to Preview view.")

    def open_live_drawing_tab(self):
        self.preview_notebook.select(self.live_drawing_tab)
        self.status_text.config(text="Live-Drawing view opened.")
        self.add_log("Switched to Live-Drawing view.")

    def open_settings_tab(self):
        self.preview_notebook.select(self.settings_tab)
        self.status_text.config(text="Settings view opened.")
        self.add_log("Switched to Settings view.")

    def trigger_capture(self, source="button"):
        self.status_text.config(text="Capture requested...")
        self.add_log(f"Capture Portrait requested from {source}.")
        self.ros_node.get_logger().info(f"Capture Portrait requested from {source}")

        self.can_start_drawing = False
        self.resume_available = False
        self.capture_button.config(state="disabled")  # stop double-click spam
        self.pending_capture = True
        self.ros_node.clear_strokes(self.on_clear_strokes_response)

    def on_capture(self):
        self.trigger_capture(source="button")

    def on_gesture_detected(self, gesture_name):
        gesture_name = gesture_name.strip().upper()
        now = time.time()

        if now - self.last_gesture_time < self.gesture_cooldown_sec:
            return

        mask_gesture_map = {
            "POINT": "moustache",   # mask 1
            "PEACE": "hat",         # mask 2
            "THREE": "glasses",     # mask 3
            "FOUR": "nose",         # mask 4
        }

        if gesture_name in mask_gesture_map:
            self.last_gesture_time = now

            mask_type = mask_gesture_map[gesture_name]
            self.status_text.config(
                text=f"{gesture_name} detected. Applying {mask_type} mask..."
            )
            self.add_log(f"Gesture detected: {gesture_name} -> mask: {mask_type}")

            self.ros_node.set_mask_type(mask_type, self.on_mask_response)
            return

        if gesture_name == "THUMBS_UP":
            self.last_gesture_time = now

            self.status_text.config(text="Thumbs up detected. Capturing portrait...")
            self.add_log("Gesture detected: THUMBS_UP -> capture portrait")

            self.trigger_capture(source="gesture: THUMBS_UP")
            return

        if gesture_name == "THUMBS_DOWN":
            self.last_gesture_time = now

            self.status_text.config(text="Thumbs down detected. Stopping drawing...")
            self.add_log("Gesture detected: THUMBS_DOWN -> stop drawing")

            if str(self.stop_button.cget("state")) != "disabled":
                self.on_stop()
            else:
                self.add_log("Stop ignored because Stop Drawing is currently disabled.")
            return

        if gesture_name == "GREEN GIANT":
            self.last_gesture_time = now

            self.status_text.config(text="Green Giant detected. Starting drawing...")
            self.add_log("Gesture detected: GREEN GIANT -> start drawing")

            if str(self.start_button.cget("state")) != "disabled":
                self.on_start()
            else:
                self.add_log("Start ignored because Start Drawing is currently disabled.")
            return

    def on_start(self):
        self.status_text.config(text="Start drawing requested...")
        self.add_log("Start Drawing button pressed.")
        self.ros_node.get_logger().info("Start Drawing requested")

        self.start_button.config(state="disabled")
        self.ros_node.start_drawing(self.on_start_response)

    def on_stop(self):
        self.status_text.config(text="Stop drawing requested.")
        self.add_log("Stop Drawing button pressed.")
        self.ros_node.get_logger().info("Stop Drawing requested")
        self.stop_pending = True
        self.resume_available = True
        self.stop_button.config(state="disabled")
        self.ros_node.stop_drawing(self.on_stop_response)

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
        elif which == "live_drawing":
            self.live_drawing_tk_image = tk_img

    def on_camera_frame(self, frame_bgr):
        self.render_frame_to_label(frame_bgr, self.camera_placeholder, "camera")

    def on_preview_frame(self, frame_bgr):
        self.render_frame_to_label(frame_bgr, self.preview_placeholder, "preview")

    def on_live_drawing_frame(self, frame_bgr):
        self.render_frame_to_label(
            frame_bgr,
            self.live_drawing_placeholder,
            "live_drawing"
        )

    def poll_ros(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0.0)
        self.root.after(30, self.poll_ros)

    def on_point_1(self):
        self.ros_node.set_point_1()
        self.status_text.config(text="Calibration Point 1 requested.")
        self.add_log("Calibration Point 1 button pressed.")
        self.ros_node.get_logger().info("Calibration Point 1 requested")

    def on_point_2(self):
        self.ros_node.set_point_2()
        self.status_text.config(text="Calibration Point 2 requested.")
        self.add_log("Calibration Point 2 button pressed.")
        self.ros_node.get_logger().info("Calibration Point 2 requested")


    def on_point_3(self):
        self.ros_node.set_point_3()
        self.status_text.config(text="Calibration Point 3 requested.")
        self.add_log("Calibration Point 3 button pressed.")
        self.ros_node.get_logger().info("Calibration Point 3 requested")

    def on_confirm(self):
        self.ros_node.send_calibration_command("confirm")
        self.status_text.config(text="Confirming calibration point...")
        self.add_log("Confirm button pressed.")
        self.ros_node.get_logger().info("Confirm calibration point")

    def on_go_home(self):
        self.status_text.config(text="Go Home requested...")
        self.add_log("Go Home button pressed.")
        self.ros_node.get_logger().info("Go Home requested")

        self.go_home_pending = True
        self.go_home_button.config(state="disabled")
        self.ros_node.go_home(self.on_go_home_response)
        self.root.after(10000, self.on_go_home_timeout)

    def on_go_home_timeout(self):
        if not self.go_home_pending:
            return

        self.go_home_pending = False
        self.status_text.config(text="Go Home is still waiting for the robot/controller.")
        self.add_log("Go Home timed out waiting for a backend response.")

        if self.ros_node.current_state != "DRAWING":
            self.go_home_button.config(state="normal")

    def on_pen_selected(self, pen_index):
        self.ros_node.save_pen_ready_pose(pen_index)
        self.status_text.config(
            text=f"Previewing Pen {pen_index} ready pose. Press Confirm Point to save."
        )
        self.add_log(f"Pen {pen_index} ready pose preview requested.")
        self.ros_node.get_logger().info(f"Pen {pen_index} ready pose preview requested")

    def on_toggle_change_color(self):
        if self.change_color_var.get():
            self.change_color_frame.grid(row=5, column=0, columnspan=3, padx=0, pady=(0, 6), sticky="ew")
            self.update_pen_change_button_states(self.displayed_state in ("IDLE", "PREVIEW_READY", "ERROR"))
            self.status_text.config(text="Pen changing controls shown.")
            self.add_log("Change Color controls shown.")
        else:
            self.change_color_frame.grid_remove()
            self.status_text.config(text="Pen changing controls hidden.")
            self.add_log("Change Color controls hidden.")

    def on_attach_pen(self, pen_index):
        self.ros_node.attach_pen(pen_index)
        self.attached_pen_index = pen_index
        self.update_pen_change_button_states(True)
        self.status_text.config(text=f"Attaching Pen {pen_index}...")
        self.add_log(f"Attach Pen {pen_index} requested.")
        self.ros_node.get_logger().info(f"Attach Pen {pen_index} requested")

    def on_detach_pen(self, pen_index):
        self.ros_node.detach_pen(pen_index)
        if self.attached_pen_index == pen_index:
            self.attached_pen_index = None
        self.update_pen_change_button_states(True)
        self.status_text.config(text=f"Detaching Pen {pen_index}...")
        self.add_log(f"Detach Pen {pen_index} requested.")
        self.ros_node.get_logger().info(f"Detach Pen {pen_index} requested")

    def on_send_tcp_offset(self):
        raw_value = self.tcp_offset_var.get().strip()
        try:
            tcp_offset = float(raw_value)
        except ValueError:
            self.status_text.config(text="TCP offset must be a number.")
            self.add_log(f"Rejected TCP offset input: '{raw_value}'")
            return

        if tcp_offset <= 0.0:
            self.status_text.config(text="TCP offset must be positive.")
            self.add_log(f"Rejected non-positive TCP offset: {tcp_offset}")
            return

        self.ros_node.set_tcp_offset(tcp_offset)
        self.status_text.config(text=f"Sending TCP offset: {tcp_offset:.4f} m")
        self.add_log(f"Sent TCP offset {tcp_offset:.4f} m to calibration and motion nodes.")
        self.ros_node.get_logger().info(f"Sent TCP offset {tcp_offset:.4f} m")

    def on_send_move_vertical(self):
        raw_value = self.move_vertical_var.get().strip()
        try:
            dist = float(raw_value)
        except ValueError:
            self.status_text.config(text="Move vertical distance must be a number.")
            self.add_log(f"Rejected move vertical input: '{raw_value}'")
            return

        self.ros_node.move_vertical(dist)
        direction = "up" if dist >= 0 else "down"
        self.status_text.config(text=f"Moving {direction} by {abs(dist):.4f} m")
        self.add_log(f"Sent move vertical command: {dist:.4f} m.")
        self.ros_node.get_logger().info(f"Sent move vertical command: {dist:.4f} m")

    def on_send_rotate_end_effector(self):
        raw_value = self.rotate_end_effector_var.get().strip()
        try:
            angle = float(raw_value)
        except ValueError:
            self.status_text.config(text="End-effector rotation must be a number.")
            self.add_log(f"Rejected end-effector rotation input: '{raw_value}'")
            return

        self.ros_node.rotate_end_effector(angle, degrees=True)
        direction = "positive" if angle >= 0 else "negative"
        self.status_text.config(text=f"Rotating end effector {direction} by {abs(angle):.2f} deg")
        self.add_log(f"Sent end-effector rotation command: {angle:.2f} deg.")
        self.ros_node.get_logger().info(f"Sent end-effector rotation command: {angle:.2f} deg")

    def on_calibration_status(self, msg_data):
        try:
            data = json.loads(msg_data)
        except json.JSONDecodeError:
            return

        if "tcp_offset" in data:
            tcp_offset = float(data["tcp_offset"])
            self.tcp_offset_var.set(f"{tcp_offset:.4f}")

        if data.get("command") == "tcp_offset_status":
            message = data.get("message", f"TCP offset is {float(data['tcp_offset']):.4f} m")
            self.status_text.config(text=message)
            self.add_log(message)

        if data.get("command") in ("pen_ready_pose_preview", "pen_ready_pose_saved"):
            message = data.get("message", "Pen ready pose updated.")
            self.status_text.config(text=message)
            self.add_log(message)

    def on_toggle_show_paper(self):
        enabled = self.show_paper_var.get()

        # call into backend / ros bridge
        self.ros_node.toggle_paper_display(enabled)

        state_text = "ON" if enabled else "OFF"
        self.status_text.config(text=f"Paper display: {state_text}")
        self.add_log(f"Paper display toggled {state_text}.")
        self.ros_node.get_logger().info(f"Paper display toggled {state_text}")

    def on_toggle_show_axes(self):
        enabled = self.show_axes_var.get()

        # call into backend / ros bridge
        self.ros_node.toggle_axes_display(enabled)

        state_text = "ON" if enabled else "OFF"
        self.status_text.config(text=f"Paper XYZ axes display: {state_text}")
        self.add_log(f"Paper XYZ axes display toggled {state_text}.")
        self.ros_node.get_logger().info(f"Paper XYZ axes display toggled {state_text}")

    def on_capture_response(self, success, message):
        self.root.after(0, lambda: self._handle_capture_response(success, message))  # safely update Tkinter from callback

    def _handle_capture_response(self, success, message):
        self.add_log(f"/create_portrait response: success={success}, message='{message}'")  # log service result
        self.status_text.config(text=message if message else "Portrait capture completed.")  # show backend message

        if success:
            self.can_start_drawing = True
            self.add_log("Portrait capture succeeded.")
            self.update_ui_for_state("PREVIEW_READY")
            self.displayed_state = "PREVIEW_READY"
        else:
            self.can_start_drawing = False
            self.add_log("Portrait capture failed.")
            self.update_ui_for_state("ERROR")
            self.displayed_state = "ERROR"

    def on_start_response(self, success, message):
        self.root.after(0, lambda: self._handle_start_response(success, message))

    def _handle_start_response(self, success, message):
        self.add_log(f"/start_drawing response: success={success}, message='{message}'")
        self.status_text.config(text=message if message else "Start drawing response received.")

        if success:
            self.add_log("Drawing sequence accepted by motion node.")
            self.resume_available = False
            self.update_ui_for_state("DRAWING")
            self.displayed_state = "DRAWING"
        else:
            self.add_log("Drawing sequence rejected.")
            self.start_button.config(state="normal")

    def on_stop_response(self, success, message):
        self.root.after(0, lambda: self._handle_stop_response(success, message))

    def _handle_stop_response(self, success, message):
        self.stop_pending = False
        self.add_log(f"/stop_drawing response: success={success}, message='{message}'")
        self.status_text.config(text=message if message else "Stop drawing response received.")

        if success:
            self.can_start_drawing = True
            self.resume_available = True
            self.add_log("Drawing stopped. Start Drawing will resume remaining strokes.")
            self.update_ui_for_state("IDLE")
            self.displayed_state = "IDLE"
        else:
            self.resume_available = False
            self.add_log("Stop drawing request rejected or failed.")
            self.update_ui_for_state(self.ros_node.current_state)

    def on_go_home_response(self, success, message):
        self.root.after(0, lambda: self._handle_go_home_response(success, message))

    def _handle_go_home_response(self, success, message):
        self.go_home_pending = False
        self.add_log(f"/go_home response: success={success}, message='{message}'")
        self.status_text.config(text=message if message else "Go Home response received.")

        if success:
            self.add_log("Go Home sequence accepted by motion node.")
            self.go_home_button.config(state="disabled")
        else:
            self.add_log("Go Home request rejected or failed.")
            if self.ros_node.current_state not in ("DRAWING", "GOING_HOME"):
                self.go_home_button.config(state="normal")
    
    def on_clear_strokes_response(self, success, message):
        self.root.after(0, lambda: self._handle_clear_strokes_response(success, message))

    def _handle_clear_strokes_response(self, success, message):
        self.add_log(f"/clear_strokes response: success={success}, message='{message}'")

        if not success:
            self.status_text.config(text=message if message else "Failed to clear old strokes.")
            self.capture_button.config(state="normal")
            return

        self.status_text.config(text="Capture requested...")
        self.ros_node.create_portrait(self.on_capture_response)
