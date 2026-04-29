import unittest
from unittest.mock import MagicMock
import tkinter as tk
import numpy as np

from gui_app import GuiApp


class FakeLogger:
    def info(self, msg): pass      # dummy logger (no output)
    def warn(self, msg): pass      # dummy logger (no output)


class FakeRosNode:
    def __init__(self):
        self.state_callback_fn = None
        self.camera_callback_fn = None
        self.preview_callback_fn = None
        self.live_drawing_callback_fn = None

        # mock backend functions so we can check if they were called
        self.set_point_1 = MagicMock()
        self.set_point_2 = MagicMock()
        self.set_point_3 = MagicMock()
        self.send_calibration_command = MagicMock()

        self._logger = FakeLogger()

    def get_logger(self):
        return self._logger


class TestGuiApp(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # hide GUI during tests
        self.ros_node = FakeRosNode()

        self.root.after = lambda *args, **kwargs: None  # disable poll_ros loop

        self.app = GuiApp(self.root, self.ros_node)  # create GUI with fake ROS
        self.root.update()

    def tearDown(self):
        self.root.destroy()  # clean up window after each test

    # --------------------------------------------------
    # Test 1: Calibration buttons
    # --------------------------------------------------
    def test_calibration_controls_trigger_expected_ros_calls(self):
        self.app.on_point_1()  
        self.ros_node.set_point_1.assert_called_once()  # check Point 1 button works

        self.app.on_point_2()
        self.ros_node.set_point_2.assert_called_once()  # check Point 2 button works

        self.app.on_point_3()
        self.ros_node.set_point_3.assert_called_once()  # check Point 3 button works

        self.app.on_confirm()
        self.ros_node.send_calibration_command.assert_called_once_with("confirm")  # check confirm sends correct command

    # --------------------------------------------------
    # Test 2: Button state logic
    # --------------------------------------------------
    def test_state_updates_enable_and_disable_buttons_correctly(self):
        self.app.update_ui_for_state("IDLE")
        self.assertEqual(str(self.app.capture_button["state"]), "normal")    # capture enabled in IDLE
        self.assertEqual(str(self.app.start_button["state"]), "disabled")    # start disabled
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")     # stop disabled

        self.app.update_ui_for_state("PREVIEW_READY")
        self.assertEqual(str(self.app.capture_button["state"]), "normal")    # capture still enabled
        self.assertEqual(str(self.app.start_button["state"]), "normal")      # start now enabled
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")     # stop still disabled

        self.app.update_ui_for_state("DRAWING")
        self.assertEqual(str(self.app.capture_button["state"]), "disabled")  # capture disabled while drawing
        self.assertEqual(str(self.app.start_button["state"]), "disabled")    # start disabled
        self.assertEqual(str(self.app.stop_button["state"]), "normal")       # stop enabled

        self.app.update_ui_for_state("ESTOP")
        self.assertEqual(str(self.app.capture_button["state"]), "disabled")  # all controls disabled
        self.assertEqual(str(self.app.start_button["state"]), "disabled")
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")

    # --------------------------------------------------
    # Test 3: Image display (preview + live drawing)
    # --------------------------------------------------
    def test_preview_and_live_drawing_frames_are_rendered(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # fake image frame

        self.app.preview_placeholder.config(width=640, height=480)
        self.app.live_drawing_placeholder.config(width=640, height=480)
        self.root.update()  # ensure widgets have size

        self.app.on_preview_frame(frame)
        self.assertIsNotNone(self.app.preview_tk_image)  # check preview image created
        self.assertEqual(str(self.app.preview_placeholder.cget("text")), "")  # placeholder text removed

        self.app.on_live_drawing_frame(frame)
        self.assertIsNotNone(self.app.live_drawing_tk_image)  # check live drawing image created
        self.assertEqual(str(self.app.live_drawing_placeholder.cget("text")), "")  # placeholder cleared


if __name__ == "__main__":
    unittest.main()
