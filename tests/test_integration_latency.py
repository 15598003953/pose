"""Regressions for live preview, inference startup and visible recognition results."""

from copy import deepcopy
from importlib import import_module
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from pose_project.apps.integration_app import IntegrationWindow
from pose_project.integration.configuration import defaults, validate
from pose_project.integration.controller import IntegrationController
from pose_project.integration.workers import RecognitionWorker


class IntegrationLatencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = defaults()
        self.settings["output_dir"] = self.temp.name
        self.settings["reid"].update(image_size=[64, 64])
        self.settings["binding"]["regions"] = {
            "A": {"roi_xyxy": [0, 0, 64, 64]}}

    def tearDown(self):
        self.app.processEvents()
        self.temp.cleanup()

    def test_preview_mailbox_keeps_one_frame_instead_of_qt_event_per_frame(self):
        controller = IntegrationController()
        controller.start(self.settings, start_workers=False)
        controller.timer.stop()
        frames = []
        controller.frame.connect(lambda stage, image: frames.append((stage, int(image[0, 0, 0]))))
        try:
            for number in range(100):
                controller._store_frame("gate", np.full((2, 2, 3), number, dtype=np.uint8))
            controller._flush_previews()
            self.assertEqual([("gate", 99)], frames)
            controller._flush_previews()
            self.assertEqual(1, len(frames))
        finally:
            controller.stop()

    def test_startup_warms_tracking_and_reid_before_accepting_stream(self):
        detector = SimpleNamespace(detect=Mock(), detect_track=Mock(), reset_tracker=Mock(),
                                   device="cpu")
        extractor = SimpleNamespace(extract=Mock(), device="cpu")
        module = SimpleNamespace(PersonDetector=Mock(return_value=detector),
                                 ReidExtractor=Mock(return_value=extractor),
                                 set_phase_threads=Mock(), EXT_THREADS=20, DET_THREADS=8,
                                 is_stream_source=lambda source: True)
        worker = RecognitionWorker(self.settings["reid"], self.settings["binding"]["regions"])

        def stream(_):
            self.assertTrue(detector.detect_track.called)
            self.assertTrue(extractor.extract.called)
            self.assertTrue(detector.reset_tracker.called)

        worker._run_stream = stream
        failures = []
        worker.failed.connect(failures.append)
        with patch("pose_project.integration.workers.import_module", return_value=module):
            worker.run()
        self.assertFalse(failures)
        detector.detect_track.assert_called_once()

    def test_stream_drains_gallery_before_selecting_fresh_frame(self):
        worker = RecognitionWorker(self.settings["reid"], self.settings["binding"]["regions"])
        events = []
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        def frame(*args, **kwargs):
            events.append("capture")
            return image, 8, 1000

        capture = SimpleNamespace(start=Mock(), stop=Mock(), _thread=None, wait_frame=frame)
        worker.module = SimpleNamespace(StreamGrabber=lambda *a, **k: capture)
        worker._drain_gallery = lambda: events.append("gallery")
        worker._process = lambda *a: (events.append("process"), worker.cancelled.set())
        worker._run_stream("rtsp://fixture")
        self.assertEqual(["gallery", "capture", "process"], events)

    def test_slow_inference_does_not_block_live_preview(self):
        self.settings["reid"]["source"] = "rtsp://fixture"
        worker = RecognitionWorker(self.settings["reid"], self.settings["binding"]["regions"])
        started = time.monotonic()

        def frame(sequence, **kwargs):
            current = int((time.monotonic() - started) * 40)
            image = np.full((64, 64, 3), current % 255, dtype=np.uint8)
            return (image if current > sequence else None, current, int(time.time() * 1000))

        capture = SimpleNamespace(start=Mock(), stop=Mock(), interrupt=Mock(), _thread=None,
                                  failed=False, wait_frame=frame)
        worker.module = SimpleNamespace(StreamGrabber=lambda *a, **k: capture)
        processed, previews = [], []

        def slow_process(*args):
            time.sleep(.3)
            processed.append(1)

        worker._process = slow_process
        worker.run = lambda: worker._run_stream("rtsp://fixture")
        controller = IntegrationController()
        controller.start(self.settings, start_workers=False)
        controller.workers["reid"] = worker
        controller.frame.connect(lambda *args: previews.append(1))
        try:
            worker.start()
            QTest.qWait(1100)
            self.assertGreaterEqual(len(previews), 8)
            self.assertLessEqual(len(processed), 4)
            self.assertGreater(len(previews), len(processed) * 2)
        finally:
            controller.stop()
            worker.wait(3000)
            controller.tick()
            self.assertFalse(controller.running)

    def test_unmatched_candidates_visible_but_not_sent_as_binding_identity(self):
        module = import_module("pose_project.reid.search_worker")
        detector = SimpleNamespace(detect_track=lambda image, **kw:
            (np.array([[1, 1, 30, 50]]), np.array([.9]), [7]))
        extractor = SimpleNamespace(extract=lambda crops, **kw: np.array([[0., 1.]]))
        matcher = module.TrackMatcher([{"label": "person", "feature": np.array([1., 0.])}],
                                      detector, extractor, .65)
        matcher.annotate(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual([], matcher.last_detections)
        self.assertEqual(1, len(matcher.last_candidates))
        self.assertEqual("below_threshold", matcher.last_candidates[0]["status"])
        self.assertGreaterEqual(matcher.last_timing["detection_ms"], 0)

    def test_region_workers_confirm_together_and_preserve_passenger_lock(self):
        module = import_module("pose_project.reid.search_worker")
        self.settings["reid"]["enabled_regions"] = ["A", "B"]
        self.settings["binding"]["regions"]["B"] = {"roi_xyxy": [0, 0, 64, 64]}
        worker = RecognitionWorker(self.settings["reid"], self.settings["binding"]["regions"])
        worker.module = module
        worker.extractor = SimpleNamespace(device="cpu", extract=lambda crops, **kw:
                                           np.tile([1., 0.], (len(crops), 1)))
        worker.gallery["person"] = {
            "label": "person", "feature": np.array([1., 0.]), "display_label": "P001"}
        for region in ("A", "B"):
            worker.detectors[region] = SimpleNamespace(detect_track=lambda image, **kw:
                (np.array([[1, 1, 30, 50]]), np.array([.9]), [7]))
        snapshots, reports = [], []
        worker.snapshot.connect(snapshots.append)
        worker.result.connect(reports.append)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for index in range(3):
            worker._process(frame, index)
        self.assertEqual({}, worker.passenger_locks)
        self.assertTrue(all(not s["detections"] for s in snapshots))
        self.assertEqual(["person"], reports[-1]["regions"]["A"]["conflicts"])
        self.assertEqual(["person"], reports[-1]["regions"]["B"]["conflicts"])
        detect_b = worker.detectors["B"].detect_track
        worker.detectors["B"].detect_track = lambda image, **kw: (
            np.zeros((0, 4)), np.zeros(0), [])
        worker._process(frame, 4)
        self.assertEqual("person", snapshots[-2]["detections"][0]["passenger_id"])
        worker.detectors["B"].detect_track = detect_b
        worker._process(frame, 5)
        self.assertEqual([], snapshots[-1]["detections"])
        self.assertEqual("passenger_locked",
                         reports[-1]["regions"]["B"]["candidates"][0]["status"])
        self.assertTrue(reports[-1]["regions"]["A"]["candidates"][0]["identity_locked"])

    def test_inference_controls_validate_and_old_configuration_still_loads(self):
        validate(self.settings)
        for key, value in (("sample_interval_ms", 0), ("detector_image_size", 555),
                           ("preview_fps", 0), ("flip_tta", "yes")):
            with self.subTest(key=key):
                settings = deepcopy(self.settings)
                settings["reid"][key] = value
                with self.assertRaises(ValueError):
                    validate(settings)

    def test_result_panel_shows_rejected_candidates_and_clears_old_people(self):
        window = IntegrationWindow(self.settings)
        report = {"captured_at_ms": 1000, "completed_at_ms": 1050, "regions": {
            "A": {"candidates": [{"label": "candidate", "score": .4, "tid": 3,
                                 "bbox_xyxy": [1, 2, 30, 50], "status": "below_threshold"}],
                  "detections": []}}, "metrics": {"processing_ms": 50}}
        try:
            window._recognition(report)
            self.assertEqual(1, window.recognition_table.rowCount())
            self.assertIn("未达阈值", window.recognition_table.item(0, 4).text())
            report["regions"]["A"]["candidates"] = []
            window._recognition(report)
            self.assertEqual(0, window.recognition_table.rowCount())
        finally:
            window.close()

    def test_result_age_includes_time_waiting_for_gui_delivery(self):
        window = IntegrationWindow(self.settings)
        try:
            window._recognition({
                "captured_at_ms": 1000, "completed_at_ms": 1050, "regions": {},
                "metrics": {"ui_delivery_ms": 5000}})
            self.assertIn("5050", window.recognition_age.text())
            self.assertIn("已过期", window.recognition_age.text())
            self.assertEqual([], window.canvases["reid"].detections)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
