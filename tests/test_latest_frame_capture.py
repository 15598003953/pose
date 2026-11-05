import time
import threading
import unittest

import numpy as np

from pose_project.video.latest_frame_capture import LatestFrameCapture


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            time.sleep(0.01)
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop):
        return 0

    def set(self, prop, value):
        return True

    def release(self):
        self.released = True


class LatestFrameCaptureTest(unittest.TestCase):
    def test_stop_does_not_release_capture_while_read_is_active(self):
        entered, resume = threading.Event(), threading.Event()
        fake = FakeCapture([])

        def read():
            entered.set()
            resume.wait(timeout=5)
            return False, None

        fake.read = read
        capture = LatestFrameCapture("rtsp://camera/main", capture_factory=lambda _: fake)
        capture.start()
        worker = capture._thread
        try:
            self.assertTrue(entered.wait(timeout=1))
            self.assertFalse(capture.stop())
            self.assertFalse(fake.released)
            self.assertIs(worker, capture._thread)
        finally:
            resume.set()
            worker.join(timeout=2)
            capture.stop()
        self.assertTrue(fake.released)
        self.assertFalse(worker.is_alive())

    def test_failed_open_releases_capture(self):
        fake = FakeCapture([])
        fake.opened = False
        capture = LatestFrameCapture("bad", capture_factory=lambda _: fake)
        self.assertFalse(capture.open())
        self.assertTrue(fake.released)
        self.assertIsNone(capture.cap)

    def test_background_reader_keeps_latest_frame_without_queue(self):
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
        fake = FakeCapture(frames)
        capture = LatestFrameCapture("rtsp://camera/main", capture_factory=lambda _source: fake)

        self.assertTrue(capture.open())
        capture.start()
        try:
            deadline = time.time() + 1.0
            latest = None
            while time.time() < deadline:
                latest = capture.get_latest()
                if latest is not None and latest.frame_idx == 2:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(latest)
            self.assertEqual(2, latest.frame_idx)
            self.assertEqual(3, int(latest.frame[0, 0, 0]))
        finally:
            capture.stop()

        self.assertTrue(fake.released)


if __name__ == "__main__":
    unittest.main()
