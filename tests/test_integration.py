"""Focused regression coverage for the combined workbench and message path."""

from copy import deepcopy
from importlib import import_module
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from pose_project.apps.integration_app import (
    IntegrationWindow, configure_application,
)
from pose_project.integration.configuration import (
    active_binding, defaults, load_settings, save_settings, validate,
)
from pose_project.integration.controller import IntegrationController, message_preview
from pose_project.integration.widgets import VideoCanvas
from pose_project.integration.workers import (
    GateWorker, PreviewWorker, RecognitionWorker, StreamCapture, encode_image,
    unique_detections,
)


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_application(cls.app)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = defaults()
        self.settings["output_dir"] = self.temp.name
        self.controllers = []
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        for controller in self.controllers:
            controller.stop()
            controller.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def controller(self):
        controller = IntegrationController()
        self.controllers.append(controller)
        controller.start(self.settings, start_workers=False)
        return controller

    def wait_for(self, predicate, timeout=5000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate(), "Qt worker/transport timed out")

    def test_config_save_backup_utf8_and_password_exclusion(self):
        path = Path(self.temp.name) / "调试配置.json"
        self.settings["mqtt"]["password"] = "secret"
        save_settings(self.settings, path)
        loaded = load_settings(path)
        self.assertNotIn("password", loaded["mqtt"])
        save_settings(self.settings, path)
        self.assertEqual(1, len(list(path.parent.glob(path.name + ".*.bak"))))
        self.assertEqual(loaded, json.loads(path.read_bytes().decode("utf-8", errors="strict")))

    def test_old_two_region_config_loads_as_one_region(self):
        path = Path(self.temp.name) / "旧配置.json"
        legacy = deepcopy(self.settings)
        legacy["binding"]["regions"]["A"]["operation_roi_xyxy"] = [300, 250, 500, 500]
        path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        loaded = load_settings(path)
        self.assertEqual([0, 0, 960, 1080],
                         loaded["binding"]["regions"]["A"]["roi_xyxy"])
        self.assertNotIn("operation_roi_xyxy", loaded["binding"]["regions"]["A"])
        save_settings(loaded, path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("operation_roi_xyxy", saved["binding"]["regions"]["A"])

    def test_relative_config_paths_resolve_against_config_directory(self):
        path = Path(self.temp.name) / "configs" / "integration.json"
        self.settings["output_dir"] = "../result"
        self.settings["reid"]["detector_weights"] = "../models/person.pt"
        self.settings["reid"]["reid_weights"] = "../models/reid.pth"
        self.settings["gate"]["source"] = "../clips/gate.mp4"
        self.settings["reid"]["source"] = "rtsp://127.0.0.1/tray"
        save_settings(self.settings, path)
        loaded = load_settings(path)
        self.assertEqual(str(Path(self.temp.name) / "result"), loaded["output_dir"])
        self.assertEqual(str(Path(self.temp.name) / "clips/gate.mp4"), loaded["gate"]["source"])
        self.assertEqual("rtsp://127.0.0.1/tray", loaded["reid"]["source"])
        self.assertEqual(str(Path(self.temp.name) / "models/person.pt"),
                         loaded["reid"]["detector_weights"])

    def test_packaged_config_stays_portable_after_save(self):
        from pose_project.integration import configuration
        base = Path(self.temp.name)
        path = base / "configs" / "integration.json"
        self.settings["output_dir"] = str(base / "result")
        self.settings["reid"]["detector_weights"] = str(base / "_internal/person.pt")
        with patch.object(configuration.sys, "frozen", True, create=True), \
             patch.object(configuration, "APP_DIR", base):
            save_settings(self.settings, path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(Path(saved["output_dir"]).is_absolute())
        self.assertFalse(Path(saved["reid"]["detector_weights"]).is_absolute())
        self.assertEqual(self.settings, load_settings(path))

    def test_bad_topics_regions_and_source_fail_validation(self):
        for change in (
            lambda cfg: cfg["mqtt"].update(person_topic="reidtest"),
            lambda cfg: cfg["binding"]["regions"]["A"].update(roi_xyxy=[0, 0, 2000, 1080]),
            lambda cfg: cfg["gate"]["roi"].update(w=2),
            lambda cfg: cfg["reid"].update(enabled_regions=[]),
        ):
            config = deepcopy(self.settings)
            change(config)
            with self.assertRaises(ValueError):
                validate(config)
        with self.assertRaisesRegex(ValueError, "请选择"):
            validate(self.settings, sources=True)

    def region_window(self):
        window = IntegrationWindow(self.settings, Path(self.temp.name) / "regions.json")
        self.windows.append(window)
        window._preview_frame("reid", np.zeros((1080, 1920, 3), dtype=np.uint8))
        window._preview_frame("gate", np.zeros((1080, 1920, 3), dtype=np.uint8))
        return window

    def test_station_uses_one_region_for_recognition_and_binding(self):
        window = self.region_window()
        window._rectangle("A.roi_xyxy", [600, 100, 950, 900])
        validate(window.collect())
        self.assertEqual([600, 100, 950, 900],
                         window.settings["binding"]["regions"]["A"]["roi_xyxy"])
        self.assertNotIn("operation_roi_xyxy", window.settings["binding"]["regions"]["A"])
        binding = active_binding(window.collect())
        self.assertEqual([600, 100, 950, 900],
                         binding["regions"]["A"]["operation_roi_xyxy"])
        self.assertEqual(["A"], [window.rfid_region.itemText(i)
                                for i in range(window.rfid_region.count())])

    def test_front_mode_is_saved_and_restored_per_region(self):
        window = self.region_window()
        window.region_front_modes["A"].setCurrentIndex(5)
        self.assertEqual("area", window.collect()["binding"]["regions"]["A"]["front_mode"])
        saved = load_settings(window.config_path)
        self.assertEqual("area", saved["binding"]["regions"]["A"]["front_mode"])
        reopened = IntegrationWindow(saved, window.config_path)
        self.windows.append(reopened)
        self.assertEqual("area", reopened.region_front_modes["A"].currentData())

    def test_old_front_mode_defaults_and_explicit_off_survive_load(self):
        path = Path(self.temp.name) / "queue.json"
        del self.settings["binding"]["regions"]["A"]["front_mode"]
        self.settings["binding"]["regions"]["B"]["front_mode"] = None
        save_settings(self.settings, path)
        loaded = load_settings(path)
        self.assertEqual("bottom", loaded["binding"]["regions"]["A"]["front_mode"])
        self.assertIsNone(loaded["binding"]["regions"]["B"]["front_mode"])
        loaded["binding"]["regions"]["A"]["front_mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "队首模式无效"):
            validate(loaded)

    def test_rectangle_edit_is_saved_before_reopening(self):
        path = Path(self.temp.name) / "regions.json"
        window = IntegrationWindow(self.settings, path)
        self.windows.append(window)
        window._preview_frame("reid", np.zeros((1080, 1920, 3), dtype=np.uint8))
        with patch.object(window, "error"):
            window._rectangle("A.roi_xyxy", [0, 0, 1920, 1080])
        saved = load_settings(path)
        self.assertEqual([0, 0, 1920, 1080],
                         saved["binding"]["regions"]["A"]["roi_xyxy"])

    def test_region_editor_has_no_operation_area(self):
        window = self.region_window()
        self.assertNotIn("A.operation_roi_xyxy", window.draw_buttons)
        self.assertNotIn("A.operation_roi_xyxy", window.delete_buttons)
        self.assertEqual(["A"], [window.rfid_region.itemText(i)
                                 for i in range(window.rfid_region.count())])

    def test_delete_station_and_gate_then_save_and_reopen_draft(self):
        window = self.region_window()
        window.delete_buttons["A.roi_xyxy"].click()
        self.assertEqual({"front_mode": "bottom"}, window.settings["binding"]["regions"]["A"])
        self.assertIsNone(window.canvases["reid"].target)
        self.assertNotIn("A 区域", window.canvases["reid"].overlays)
        window.delete_buttons["gate"].click()
        self.assertIsNone(window.settings["gate"]["roi"])
        self.assertEqual({}, window.canvases["gate"].overlays)
        window.save_config()
        reopened = load_settings(window.config_path)
        self.assertEqual({"front_mode": "bottom"}, reopened["binding"]["regions"]["A"])
        self.assertIsNone(reopened["gate"]["roi"])
        another = IntegrationWindow(reopened, window.config_path)
        self.windows.append(another)
        another._preview_frame("reid", np.zeros((540, 960, 3), dtype=np.uint8))
        self.assertEqual({"front_mode": "bottom"}, another.settings["binding"]["regions"]["A"])
        window._rectangle("A.roi_xyxy", [0, 0, 960, 1080])
        with self.assertRaisesRegex(ValueError, "缺少闸机抓拍区"):
            validate(window.collect())
        window._rectangle("gate", [100, 100, 1500, 900])
        validate(window.collect())

    def test_add_and_delete_whole_region_updates_runtime_and_rfid_choices(self):
        window = self.region_window()
        window.add_region_button.click()
        self.assertIn("C", window.region_enabled)
        self.assertTrue(window.region_enabled["C"].isChecked())
        window._rectangle("C.roi_xyxy", [100, 100, 800, 900])
        config = window.collect()
        validate(config)
        self.assertEqual(["A", "C"], [window.rfid_region.itemText(i)
                                     for i in range(window.rfid_region.count())])
        controller = window.controller
        controller.start(config, start_workers=False)
        self.controllers.append(controller)
        self.assertEqual({"A", "C"}, set(controller.binder.history))
        self.assertNotIn("B", controller.settings["binding"]["regions"])
        with self.assertRaisesRegex(ValueError, "已配置的区域"):
            controller.inject_rfid("B", "disabled")
        controller.stop()
        window.delete_buttons["C"].click()
        self.assertNotIn("C", window.settings["binding"]["regions"])
        self.assertNotIn("C", window.collect()["reid"]["enabled_regions"])
        self.assertEqual(["A"], [window.rfid_region.itemText(i)
                                for i in range(window.rfid_region.count())])
        window.add_region_button.click()
        self.assertEqual({}, window.settings["binding"]["regions"]["C"])

    def test_disabled_invalid_region_is_editable_but_not_used_by_binder(self):
        self.settings["binding"]["regions"]["B"]["roi_xyxy"] = [1500, 100, 1900, 900]
        validate(self.settings)
        controller = self.controller()
        self.assertEqual({"A"}, set(controller.binder.history))
        window = self.region_window()
        self.assertIn("未启用", window.region_status["B"].text())
        window.region_enabled["B"].setChecked(True)
        validate(window.collect())
        save_settings(window.collect(), window.config_path)
        self.assertEqual(["A", "B"], load_settings(window.config_path)["reid"]["enabled_regions"])

    def test_empty_region_draft_roundtrips_but_start_is_rejected_before_workers(self):
        window = self.region_window()
        for name in list(window.region_enabled):
            window.delete_region(name)
        window.save_config()
        draft = load_settings(window.config_path)
        self.assertEqual({}, draft["binding"]["regions"])
        with self.assertRaisesRegex(ValueError, "至少新增并启用"):
            window.controller.start(draft)
        self.assertFalse(window.controller.running)
        self.assertIsNone(window.controller.run_dir)
        with patch.object(window, "error") as warning:
            window.start_session()
        self.assertIn("至少新增并启用", warning.call_args.args[0])
        window.add_region()
        self.assertEqual(["A"], list(window.region_enabled))
        with self.assertRaisesRegex(ValueError, "A 区缺少区域"):
            validate(window.collect())

    def test_partial_drafts_still_validate_coordinates_and_binding_parameters(self):
        self.settings["binding"]["regions"] = {"C": {}}
        self.settings["reid"]["enabled_regions"] = ["C"]
        self.settings["gate"]["roi"] = None
        validate(self.settings, require_regions=False)
        self.settings["binding"]["binding_parameters"]["min_dwell_ms"] = -1
        with self.assertRaises(ValueError):
            validate(self.settings, require_regions=False)
        self.settings["binding"]["binding_parameters"]["min_dwell_ms"] = 5000
        self.settings["binding"]["regions"]["C"]["roi_xyxy"] = [0, 0, float("nan"), 100]
        with self.assertRaisesRegex(ValueError, "C 区坐标无效"):
            validate(self.settings, require_regions=False)

    def test_both_sources_accept_rtsp_and_reject_malformed_urls(self):
        self.settings["gate"]["source"] = "rtsp://user:password@127.0.0.1:8554/gate"
        self.settings["reid"]["source"] = "RTSP://127.0.0.1:8554/tray"
        validate(self.settings, sources=True)
        local = Path(self.temp.name) / "video.mp4"
        local.touch()
        self.settings["reid"]["source"] = str(local)
        validate(self.settings, sources=True)
        for source in ("rtsp://", "rtsp://host:bad/live", "rtsp://host:0/live",
                       "rtsp://host:99999/live", "rtsp://host/live stream", "ftp://host/live"):
            self.settings["gate"]["source"] = source
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "流地址无效"):
                validate(self.settings, sources=True)

    def test_source_modes_follow_pasted_urls_and_reload_without_losing_them(self):
        self.settings["gate"]["source"] = "rtsp://127.0.0.1:8554/gate"
        self.settings["reid"]["source"] = "rtsp://127.0.0.1:8554/tray"
        window = IntegrationWindow(self.settings, Path(self.temp.name) / "settings.json")
        self.windows.append(window)
        for stage in ("gate", "reid"):
            self.assertEqual("stream", window.source_modes[stage].currentData())
            self.assertFalse(window.source_browse[stage].isEnabled())
            self.assertEqual(self.settings[stage]["source"], window.collect()[stage]["source"])
        window.source_modes["gate"].setCurrentIndex(0)
        self.assertEqual("", window.sources["gate"].text())
        self.assertTrue(window.source_browse["gate"].isEnabled())
        window.sources["gate"].setText("rtsp://127.0.0.1:8554/new")
        self.assertEqual("stream", window.source_modes["gate"].currentData())
        save_settings(window.collect(), window.config_path)
        self.assertEqual("rtsp://127.0.0.1:8554/new",
                         load_settings(window.config_path)["gate"]["source"])
        window.sources["reid"].clear()
        QTest.keyClicks(window.sources["reid"], "rtsp")
        self.assertEqual("stream", window.source_modes["reid"].currentData())
        self.assertEqual("rtsp", window.sources["reid"].text())

    def test_stream_adapter_preserves_received_time_and_cleans_up(self):
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        grabber = SimpleNamespace(start=Mock(), interrupt=Mock(), stop=Mock(),
            wait_frame=Mock(return_value=(image, 9, 1234567)), failed=False,
            fps=25, _thread=SimpleNamespace(join=Mock()))
        module = SimpleNamespace(StreamGrabber=Mock(return_value=grabber))
        with patch("pose_project.integration.workers.import_module", return_value=module):
            capture = StreamCapture("rtsp://127.0.0.1/gate")
        ok, received = capture.read()
        self.assertTrue(ok)
        self.assertIs(image, received)
        self.assertEqual(1234567, capture.captured_at_ms)
        capture.interrupt()
        self.assertEqual((False, None), capture.read())
        capture.release()
        grabber.stop.assert_called_once()
        grabber._thread.join.assert_called_once()
        self.assertFalse(capture.isOpened())

    def test_stream_adapter_reports_failure_and_preview_releases_capture(self):
        grabber = SimpleNamespace(start=Mock(), interrupt=Mock(), stop=Mock(),
            wait_frame=Mock(return_value=(None, -1, 0)), failed=True, _thread=None)
        with patch("pose_project.integration.workers.import_module",
                   return_value=SimpleNamespace(StreamGrabber=lambda *a, **k: grabber)):
            capture = StreamCapture("rtsp://127.0.0.1/bad")
        try:
            with self.assertRaisesRegex(ValueError, "视频流连接失败"):
                capture.read()
        finally:
            capture.release()
        image = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
        capture = SimpleNamespace(read=lambda: (True, image), release=Mock())
        preview = PreviewWorker("rtsp://127.0.0.1/preview", rotate=True)
        frames = []
        preview.frame.connect(frames.append)
        with patch("pose_project.integration.workers.StreamCapture", return_value=capture):
            preview.run()
        self.assertEqual(1, len(frames))
        self.assertTrue(np.array_equal(cv2.rotate(image, cv2.ROTATE_180), frames[0]))
        capture.release.assert_called_once()

    def test_gate_stream_uses_received_clock_and_releases_on_detection_error(self):
        gate = import_module("pose_project.gate.processor")
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        captured_at = 1793671800123
        capture = SimpleNamespace(isOpened=lambda: True, get=lambda prop: 25 if prop == cv2.CAP_PROP_FPS else 0,
            read=Mock(side_effect=[(True, image), (False, None)]),
            captured_at_ms=captured_at, release=Mock())
        processor = gate.ProcessorThread(
            ["rtsp://127.0.0.1/gate"], self.settings["gate"]["roi"],
            dict(self.settings["gate"]["opts"], require_approach=False, min_hits=1),
            self.temp.name, mqtt_enabled=False, stream_factory=lambda _: capture)
        track = SimpleNamespace(box=(1, 1, 8, 12), face=None, missed=0, hits=2,
            tid=1, best=None, face_hist=[], size_hist=[], approaching=lambda *a: (True, .1))
        tracker = SimpleNamespace(tracks=[track], update=lambda *a: [], finalize_all=lambda: [])
        times = []
        processor._make_candidate = lambda frame, t, ts, *a: times.append(ts)
        detector = SimpleNamespace(detect=lambda frame: [], face_engine=None)
        with patch.object(gate, "SimpleTracker", return_value=tracker):
            processor._process_video(0, processor.files[0], datetime(2000, 1, 1), detector)
        self.assertEqual(datetime.fromtimestamp(captured_at / 1000), times[0])
        capture.release.assert_called_once()
        capture.read = Mock(return_value=(True, image))
        capture.release.reset_mock()
        detector.detect = Mock(side_effect=RuntimeError("detector failed"))
        with self.assertRaisesRegex(RuntimeError, "detector failed"):
            processor._process_video(0, processor.files[0], datetime(2000, 1, 1), detector)
        capture.release.assert_called_once()

    def test_gate_stream_does_not_parse_url_as_file_and_propagates_failure(self):
        gate = import_module("pose_project.gate.processor")
        processor = gate.ProcessorThread(
            ["rtsp://127.0.0.1/gate"], self.settings["gate"]["roi"],
            self.settings["gate"]["opts"], self.temp.name, mqtt_enabled=False,
            stream_factory=Mock())
        processor._process_video = Mock(side_effect=RuntimeError("stream failed"))
        with patch.object(gate, "parse_start_time", side_effect=RuntimeError("file stat called")), \
             patch.object(gate, "PersonDetector", return_value=SimpleNamespace(warnings=[])):
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                processor._run()

    def test_two_streams_restore_environment_and_keep_separate_transports(self):
        module = import_module("pose_project.video.stream_capture")
        observed = []

        def open_capture(source, backend, params):
            observed.append((source, os.environ[module._FFMPEG_ENV_KEY]))
            time.sleep(.02)
            return SimpleNamespace(isOpened=lambda: True, set=Mock())

        with patch.dict(os.environ, {module._FFMPEG_ENV_KEY: "original;value"}), \
             patch.object(module.cv2, "VideoCapture", side_effect=open_capture):
            captures = [module.StreamGrabber("rtsp://127.0.0.1/" + name)
                        for name in ("gate", "tray")]
            threads = [threading.Thread(target=capture._create_capture, args=(transport,))
                       for capture, transport in zip(captures, ("tcp", "udp"))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual("original;value", os.environ[module._FFMPEG_ENV_KEY])
        options = dict(observed)
        self.assertIn("rtsp_transport;tcp", options["rtsp://127.0.0.1/gate"])
        self.assertIn("rtsp_transport;udp", options["rtsp://127.0.0.1/tray"])

    def test_gate_payload_keeps_stable_id_and_old_positional_api(self):
        gate = import_module("pose_project.gate.mqtt_publisher")
        publisher = gate.MqttPublisher(enabled=False)
        image = np.zeros((80, 40, 3), dtype=np.uint8)
        payload = publisher.build_payload("P001", "2026-11-03 09:00:00.000", upper_img=image)
        updated = publisher.build_payload("P001", "2026-11-03 09:00:01.000", upper_img=image)
        self.assertEqual(payload["passenger_id"], updated["passenger_id"])
        self.assertNotEqual("P001", payload["passenger_id"])
        self.assertEqual(payload["person_id"], payload["passenger_id"])
        publisher.enabled = True
        publisher._client = Mock()
        self.assertTrue(publisher.publish_person("P001", "2026-11-03 09:00:00", upper_img=image))
        self.assertEqual(1, publisher._client.publish.call_count)

    def test_gate_injected_publisher_does_not_open_mqtt(self):
        gate = import_module("pose_project.gate.processor")
        injected = object()
        with patch.object(gate, "MqttPublisher", side_effect=RuntimeError("must not connect")):
            worker = gate.ProcessorThread([], self.settings["gate"]["roi"],
                self.settings["gate"]["opts"], self.temp.name, publisher=injected)
        self.assertIs(injected, worker.mqtt)

    def test_reid_gallery_uses_full_id_updates_without_resetting_tracks(self):
        worker = RecognitionWorker(self.settings["reid"], self.settings["binding"]["regions"])
        image = np.zeros((80, 40, 3), dtype=np.uint8)
        worker.module = SimpleNamespace(
            decode_base64_image=lambda value: image if value else None,
            set_phase_threads=Mock(), EXT_THREADS=20)
        worker.extractor = SimpleNamespace(extract=lambda imgs, tta: np.ones((1, 4)))
        detector = SimpleNamespace(reset_tracker=Mock())
        matcher = SimpleNamespace(detector=detector, update_gallery=Mock())
        worker.matchers = {"A": matcher}
        payload = {"passenger_id": "stable-identity", "local_id": "P001",
                   "images": {"upper": "encoded"}}
        worker.add_person(payload)
        worker._drain_gallery()
        self.assertEqual(["stable-identity"], list(worker.gallery))
        self.assertIs(matcher, worker.matchers["A"])
        detector.reset_tracker.assert_not_called()
        matcher.update_gallery.assert_called_once_with(
            list(worker.gallery.values()), {"stable-identity": "P001"})
        worker.add_person(payload)
        worker._drain_gallery()
        self.assertEqual(1, len(worker.gallery))
        worker.add_person({"local_id": "P002", "images": {"upper": "encoded"}})
        worker._drain_gallery()
        self.assertEqual(1, len(worker.gallery))
        worker.add_person({"passenger_id": "bad", "images": ["not", "an", "object"]})
        worker._drain_gallery()
        self.assertEqual(1, len(worker.gallery))

    def test_duplicate_identity_is_omitted_and_valid_box_clipped(self):
        detections = [
            {"label": "a", "score": .9, "bbox_xyxy": [1, 2, 30, 70]},
            {"label": "a", "score": .8, "bbox_xyxy": [40, 2, 70, 70]},
            {"label": "b", "score": .9, "bbox_xyxy": [-10, 2, 300, 500]},
            {"label": "c", "score": .3, "bbox_xyxy": [1, 2, 30, 70]},
        ]
        people, duplicates = unique_detections(detections, 100, 100, .65)
        self.assertEqual(["a"], duplicates)
        self.assertEqual([{"passenger_id": "b", "bbox_xyxy": [0, 2, 100, 100]}], people)

    def test_local_path_binds_full_id_with_new_frames(self):
        controller = self.controller()
        controller.timer.stop()
        passenger = "0123456789abcdef0123456789abcdef-2026-11-03"
        base = controller.inbox.now() - 6200
        topic = self.settings["binding"]["topics"]["detections"]
        decisions = []
        controller.decision.connect(decisions.append)
        for offset in range(0, 6001, 200):
            now = base + offset
            controller._accept(topic, {"roi_id": "A", "captured_at_ms": now,
                "detections": [{"passenger_id": passenger,
                                "bbox_xyxy": [320, 255, 480, 495]}]}, now)
        rfid_topic = self.settings["binding"]["topics"]["rfid"]
        controller._accept(rfid_topic, {"roi_id": "A", "rfid": "TRAY", "event_id": "one"}, base + 6010)
        controller._accept(rfid_topic, {"roi_id": "A", "rfid": "TRAY", "event_id": "one"}, base + 6010)
        controller._record_decisions(controller.binder.finish())
        self.assertEqual(1, len(decisions))
        self.assertEqual(passenger, decisions[0]["binding"]["passenger_id"])
        run_dir = controller.run_dir
        controller.stop()
        self.assertEqual(1, json.loads((run_dir / "summary.json").read_text())["bound"])
        self.assertTrue((run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip())

    def test_field_test_session_records_metadata_events_and_summary(self):
        settings = deepcopy(self.settings)
        settings["debug"] = {
            "enabled": False,
            "field_test": {
                "enabled": True,
                "test_id": "FT-001",
                "operator": "tester",
                "site": "gate-A",
                "scenario": "single passenger",
                "notes": "field check",
            },
        }
        controller = IntegrationController()
        self.controllers.append(controller)
        controller.start(settings, start_workers=False)
        now = controller.inbox.now()
        controller._accept(
            settings["mqtt"]["person_topic"],
            {"passenger_id": "passenger-1", "local_id": "P001"},
            now,
        )
        controller._record_decisions([{
            "roi_id": "A", "rfid": "TRAY-1", "status": "bound",
            "reason": "unique_candidate", "passenger_id": "passenger-1",
            "binding": {"roi_id": "A", "rfid": "TRAY-1",
                        "passenger_id": "passenger-1"},
            "candidates": [],
        }])
        controller.record_field_observation(
            "binding", "pass", "绑定 passenger-1",
            "绑定 passenger-1", "现场记录", {"selected_passenger_id": "passenger-1"})
        run_dir = controller.run_dir
        controller.stop()

        metadata = json.loads((run_dir / "field_test.json").read_text(encoding="utf-8"))
        events = [
            json.loads(line) for line in
            (run_dir / "field_test_events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual("finished", metadata["status"])
        self.assertEqual("FT-001", metadata["test"]["test_id"])
        self.assertTrue(metadata["duration_ms"] >= 0)
        self.assertEqual({
            "session_started", "person_collected", "binding_decision",
            "manual_observation", "session_stopped",
        }, {event["event_type"] for event in events})
        self.assertEqual(1, summary["field_test"]["manual_observations"])
        self.assertEqual(1, summary["field_test"]["event_counts"]["binding_decision"])
        self.assertTrue((run_dir / "performance.jsonl").is_file())

    def test_stale_positions_never_gain_dwell_when_repeated(self):
        controller = self.controller()
        controller.timer.stop()
        base = controller.inbox.now() - 7000
        topic = self.settings["binding"]["topics"]["detections"]
        payload = {"roi_id": "A", "captured_at_ms": base,
                   "detections": [{"passenger_id": "stable", "bbox_xyxy": [320,255,480,495]}]}
        controller._accept(topic, payload, base)
        controller._accept(topic, payload, base + 6000)
        controller._accept(self.settings["binding"]["topics"]["rfid"],
                           {"roi_id": "A", "rfid": "TRAY"}, base + 6010)
        result = controller.binder.finish()[0]
        self.assertIsNone(result["binding"])
        self.assertEqual("stale_position_data", result["reason"])

    def test_message_logs_omit_image_content(self):
        payload = {"passenger_id": "abc", "images": {"upper": "123456", "face": None}}
        result = message_preview(payload)
        self.assertEqual("<base64: 6 chars>", result["images"]["upper"])
        self.assertEqual("123456", payload["images"]["upper"])

    def test_restart_has_distinct_run_directory_and_no_old_history(self):
        controller = self.controller()
        first = controller.run_dir
        controller.stop()
        controller.start(self.settings, start_workers=False)
        self.assertNotEqual(first, controller.run_dir)
        self.assertEqual(0, len(controller.binder.pending))
        self.assertEqual({}, controller.binder.history["A"])

    def test_stop_keeps_active_worker_until_it_really_exits(self):
        controller = self.controller()
        active = [True]
        worker = SimpleNamespace(
            stop=Mock(), isRunning=lambda: active[0], deleteLater=Mock())
        controller.workers = {"gate": worker}
        controller.stop()
        self.assertTrue(controller.running)
        self.assertTrue(controller.stopping)
        worker.stop.assert_called_once()
        worker.deleteLater.assert_not_called()
        active[0] = False
        controller.tick()
        self.assertFalse(controller.running)
        worker.deleteLater.assert_called_once()

    def test_canvas_converts_letterboxed_drag_to_source_pixels(self):
        canvas = VideoCanvas("test")
        canvas.resize(400, 400)
        canvas.set_frame(np.zeros((100, 200, 3), dtype=np.uint8))
        canvas.show()
        self.app.processEvents()
        drawn = []
        canvas.rectangle_drawn.connect(lambda target, box: drawn.append((target, box)))
        canvas.begin_draw("A.roi_xyxy")
        QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(40, 120))
        QTest.mouseMove(canvas, QPoint(360, 280))
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(360, 280))
        self.assertEqual([("A.roi_xyxy", [20, 10, 180, 90])], drawn)
        canvas.close()

    def test_window_collects_one_config_and_contains_no_legacy_windows(self):
        window = IntegrationWindow(self.settings, Path(self.temp.name) / "config.json")
        self.windows.append(window)
        window.show()
        self.app.processEvents()
        self.assertEqual("local", window.collect()["transport"])
        self.assertEqual(2, len(window.canvases))
        self.assertEqual(8, window.event_tabs.count())
        self.assertFalse(window.stop_button.isEnabled())
        for size in ((1440, 940), (1000, 720)):
            window.resize(*size)
            self.app.processEvents()
            self.assertGreaterEqual(window.canvases["gate"].width(), 240)
            self.assertGreaterEqual(window.canvases["reid"].width(), 240)
        window._preview_frame("reid", np.zeros((540, 960, 3), dtype=np.uint8))
        self.assertEqual([960, 540], window.collect()["reid"]["image_size"])
        self.assertEqual([0, 0, 480, 540],
                         window.collect()["binding"]["regions"]["A"]["roi_xyxy"])

    def test_mqtt_uses_actual_broker_routes_for_person_position_and_rfid(self):
        from test_binding_app import LoopbackBroker
        broker = LoopbackBroker()
        self.settings["transport"] = "mqtt"
        self.settings["mqtt"]["port"] = broker.server_address[1]
        controller = self.controller()
        try:
            self.wait_for(lambda: controller.bridge.connected)
            received = []
            controller.message.connect(received.append)
            controller.receive_collected({"passenger_id": "stable", "images": {"upper": "x"}})
            self.wait_for(lambda: bool(broker.received))
            self.assertEqual("check_gate/person/collected", broker.received[0][0])
            broker.send_message("check_gate/person/collected",
                                {"passenger_id": "stable", "images": {"upper": "x"}})
            self.wait_for(lambda: controller.counts["persons"] == 1)
            now = controller.inbox.now()
            broker.send_message("reidtest", {"roi_id": "A", "captured_at_ms": now,
                                           "detections": []})
            self.wait_for(lambda: len(received) >= 2)
            broker.send_message("pose/rfid", {"roi_id": "A", "rfid": "T", "event_id": "t"})
            self.wait_for(lambda: controller.counts["unbound"] == 1)
            self.assertEqual(0, controller.counts["invalid"])
        finally:
            controller.stop()
            broker.close()

    def test_mqtt_disconnected_publish_is_queued_and_stop_reports_unsent(self):
        controller = self.controller()
        controller.bridge = SimpleNamespace(
            connected=False, waiting={}, close=lambda: None, deleteLater=lambda: None)
        controller.receive_collected({"passenger_id": "stable"})
        self.assertEqual(1, len(controller.pending_sends))
        controller.stop()
        self.assertEqual(1, controller.counts["unsent"])

    def test_stream_time_is_returned_atomically_with_frame(self):
        module = import_module("pose_project.video.stream_capture")
        capture = module.StreamGrabber("rtsp://unused")
        capture._frame = np.zeros((16, 16, 3), dtype=np.uint8)
        capture._seq, capture._captured_at_ms = 5, 1234
        image, sequence, captured = capture.wait_frame(-1, with_timestamp=True)
        self.assertEqual((5, 1234), (sequence, captured))
        self.assertEqual(2, len(capture.wait_frame(-1)))

    def test_stream_capture_clock_can_share_the_receiver_time(self):
        module = import_module("pose_project.video.stream_capture")
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        grabber = module.StreamGrabber("rtsp://unused", clock=lambda: 1234)

        def receive():
            grabber.interrupt()
            return True, image

        cap = SimpleNamespace(get=lambda _: 25, grab=lambda: True,
                              retrieve=receive, release=Mock())
        grabber._create_capture = lambda transport: cap
        grabber._loop()
        self.assertEqual(1234, grabber.wait_frame(-1, with_timestamp=True)[2])
        cap.release.assert_called_once()

    def test_file_replay_skips_late_frames_without_changing_capture_time(self):
        path = Path(self.temp.name) / "short.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 64))
        for _ in range(24):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        worker = RecognitionWorker(self.settings["reid"], self.settings["binding"]["regions"],
                                   clock=lambda: 1000 + int(time.monotonic() * 1000))
        worker.module = SimpleNamespace(
            open_file_capture=lambda source: (cv2.VideoCapture(source), None))
        samples = []

        def slow_process(image, captured):
            samples.append(captured)
            time.sleep(.32)

        worker._process = slow_process
        started = time.monotonic()
        worker._run_file(str(path))
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertGreaterEqual(len(samples), 2)
        self.assertGreater(samples[1] - samples[0], 200)
        self.assertFalse(worker.capture.isOpened())

    def test_file_replay_never_publishes_a_future_timestamp(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        capture = SimpleNamespace(isOpened=lambda: True, get=lambda _: 25, grab=lambda: True,
            read=Mock(side_effect=[(True, image), (True, image), (False, None)]), release=Mock())
        worker = RecognitionWorker(self.settings["reid"], self.settings["binding"]["regions"],
                                   clock=lambda: 5000)
        worker.module = SimpleNamespace(open_file_capture=lambda _: (capture, None))
        times = []
        worker._process = lambda image, captured: times.append(captured)
        worker._run_file("unused")
        self.assertTrue(times)
        self.assertTrue(all(timestamp <= 5000 for timestamp in times))


if __name__ == "__main__":
    unittest.main()
