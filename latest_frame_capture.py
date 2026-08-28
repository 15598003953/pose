import threading
import time
from dataclasses import dataclass

import cv2 as cv


@dataclass
class LatestFrame:
    frame: object
    frame_idx: int
    read_time: float


class LatestFrameCapture:
    """Read a stream in the background and keep only the newest frame."""

    def __init__(self, source, capture_factory=None):
        self.source = source
        self.capture_factory = capture_factory or cv.VideoCapture
        self.cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest = None
        self._frame_idx = -1
        self._error = ""

    @property
    def error(self):
        return self._error

    def open(self):
        self.cap = self.capture_factory(self.source)
        if not self.cap or not self.cap.isOpened():
            self._error = "Cannot open stream: {}".format(self.source)
            return False
        try:
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return True

    def start(self):
        if self.cap is None and not self.open():
            return False
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def get_latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get(self, prop):
        if self.cap is None:
            return 0
        return self.cap.get(prop)

    def _reader_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame_idx += 1
                self._latest = LatestFrame(
                    frame=frame,
                    frame_idx=self._frame_idx,
                    read_time=time.time(),
                )
