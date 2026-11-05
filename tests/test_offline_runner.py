import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pose_project.apps.run_put_luggage_offline import run


class OfflineRunnerTest(unittest.TestCase):
    def test_invalid_stride_fails_before_opening_any_video(self):
        with self.assertRaisesRegex(ValueError, "stride"):
            run(SimpleNamespace(stride=0, max_frames=1))

    def test_model_initialization_error_releases_video_resources(self):
        capture, writer = Mock(), Mock()
        capture.isOpened.return_value = writer.isOpened.return_value = True
        capture.get.return_value = 10
        with patch("pose_project.apps.run_put_luggage_offline.os.path.isfile", return_value=True), \
                patch("pose_project.apps.run_put_luggage_offline.os.makedirs"), \
                patch("pose_project.apps.run_put_luggage_offline.cv.VideoCapture", return_value=capture), \
                patch("pose_project.apps.run_put_luggage_offline.cv.VideoWriter", return_value=writer), \
                patch("pose_project.apps.run_put_luggage_offline.MPPersonDet", side_effect=RuntimeError("model")):
            with self.assertRaisesRegex(RuntimeError, "model"):
                run(SimpleNamespace(video="video.mp4", output_dir="out", stride=3, max_frames=1))
        capture.release.assert_called_once()
        writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
