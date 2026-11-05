import tempfile
import unittest
from pathlib import Path

from pose_project.apps.pose_app import PoseVideoThread


class FakePublisher:
    def __init__(self):
        self.calls = []
        self.closed = False

    def publish_action(self, event, roi_id=None, video_path=""):
        self.calls.append((dict(event), roi_id, video_path))
        return True

    def close(self):
        self.closed = True


class PoseAppMqttEventTest(unittest.TestCase):
    def test_write_action_event_publishes_roi_id_to_mqtt(self):
        publisher = FakePublisher()
        thread = PoseVideoThread(
            "rtsp://camera/main",
            pose_estimator=None,
            person_detector=None,
            mqtt_publisher=publisher,
        )
        with tempfile.TemporaryDirectory() as tmp:
            thread.event_log_path = str(Path(tmp) / "events.jsonl")
            thread._open_event_log()
            try:
                thread._write_action_event({
                    "event": "put_luggage",
                    "roi_id": 3,
                    "confidence": 0.91,
                })
            finally:
                thread._close_event_log()

        self.assertEqual(1, len(publisher.calls))
        event, roi_id, video_path = publisher.calls[0]
        self.assertEqual("put_luggage", event["event"])
        self.assertEqual(3, roi_id)
        self.assertEqual("rtsp://camera/main", video_path)

if __name__ == "__main__":
    unittest.main()
