"""Video changes must preserve ownership of the current worker."""

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from pose_project.video.app_config import SourceConfig
from pose_project.apps.pose_app import PoseMainWindow


class PoseAppLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch.object(PoseMainWindow, "_init_models"), patch.object(
            PoseMainWindow, "_load_rois"
        ):
            self.window = PoseMainWindow()

    def tearDown(self):
        self.window.thread = None
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_stop_keeps_worker_when_shutdown_is_incomplete(self):
        worker = Mock()
        worker.stop.return_value = False
        self.window.thread = worker
        with patch("pose_project.apps.pose_app.QMessageBox.warning"):
            self.assertFalse(self.window._stop())
        self.assertIs(worker, self.window.thread)

    def test_opening_another_video_stops_current_worker_before_playing(self):
        window = self.window
        window.pose_estimator = window.person_detector = object()
        window.source_config = SourceConfig()
        window.video_path = "old.mp4"
        order = []
        worker = Mock()
        worker.stop.side_effect = lambda: order.append("stop") or True
        window.thread = worker
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 100
        with patch("pose_project.apps.pose_app.QFileDialog.getOpenFileName", return_value=("new.mp4", "")), \
                patch("pose_project.apps.pose_app.save_source_config"), \
                patch("pose_project.apps.pose_app.cv.VideoCapture", return_value=capture), \
                patch.object(window, "_play", side_effect=lambda: order.append("play")):
            window._open_video()
        self.assertEqual(["stop", "play"], order)
        self.assertEqual("new.mp4", window.video_path)


if __name__ == "__main__":
    unittest.main()
