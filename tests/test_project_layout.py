"""目录迁移回归：检查源码入口、资源定位和独立界面的配置读写。"""

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PyQt5.QtWidgets import QApplication

from pose_project.__main__ import COMMANDS
from pose_project.binding import generate_binding_simulation, replay_binding
from pose_project.common.project_paths import (
    CONFIG_DIR, DATA_DIR, MODEL_DIR, RESULT_DIR, ROOT,
    GATE_MODEL_DIR, REID_MODEL_DIR, RESOURCE_CONFIG_DIR,
)
from pose_project.integration.configuration import defaults, environment_report


class ProjectLayoutTest(unittest.TestCase):
    def test_resources_resolve_to_project_directories(self):
        self.assertEqual(Path(__file__).resolve().parents[1], ROOT)
        self.assertEqual(ROOT / "models", MODEL_DIR)
        for path in (generate_binding_simulation.DEFAULT_CONFIG, replay_binding.DEFAULT_CONFIG):
            with self.subTest(path=path):
                self.assertEqual(CONFIG_DIR, Path(path).parent)

    def test_binding_defaults_keep_existing_data_and_output_locations(self):
        self.assertEqual(DATA_DIR / "binding_simulation",
                         generate_binding_simulation.DEFAULT_OUTPUT)
        self.assertEqual(DATA_DIR / "binding_simulation" / "replay.jsonl",
                         replay_binding.DEFAULT_INPUT)
        self.assertEqual(RESULT_DIR / "binding_replay", replay_binding.DEFAULT_OUTPUT)

    def test_primary_commands_only_cover_gate_reid_and_binding(self):
        self.assertEqual({"integrate", "gate", "reid", "binding", "simulate", "replay"},
                         set(COMMANDS))
        for folder in ("actions", "vision", "training"):
            self.assertFalse((ROOT / "src/pose_project" / folder).exists())

    def test_three_stage_resources_are_in_central_directories(self):
        self.assertEqual(MODEL_DIR / "gate", GATE_MODEL_DIR)
        self.assertEqual(MODEL_DIR / "reid", REID_MODEL_DIR)
        self.assertEqual(CONFIG_DIR, RESOURCE_CONFIG_DIR)
        settings = defaults()
        self.assertEqual(REID_MODEL_DIR / "yolov8s.pt",
                         Path(settings["reid"]["detector_weights"]))
        self.assertEqual(REID_MODEL_DIR / "msmt_sbs_R50.pth",
                         Path(settings["reid"]["reid_weights"]))
        report = environment_report(settings)
        self.assertTrue(all(item["exists"] for item in report["weights"].values()))

    def test_model_defaults_work_from_an_unrelated_directory(self):
        from pose_project.reid.core import PersonDetector, ReidExtractor
        current = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(str(REID_MODEL_DIR / "msmt_sbs_R50.pth"),
                                 ReidExtractor._find_weights())
                self.assertEqual(str(REID_MODEL_DIR / "yolov8s.pt"),
                                 PersonDetector._pick_weights())
            finally:
                os.chdir(current)

    def test_frozen_paths_separate_resources_and_writable_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            resource = bundle / "_internal"
            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "_MEIPASS", str(resource), create=True), \
                 patch.object(sys, "executable", str(bundle / "PoseIntegration.exe")):
                paths = runpy.run_path(str(ROOT / "src/pose_project/common/project_paths.py"))
            self.assertEqual(resource / "models/gate", paths["GATE_MODEL_DIR"])
            self.assertEqual(resource / "models/reid", paths["REID_MODEL_DIR"])
            self.assertEqual(resource / "configs", paths["RESOURCE_CONFIG_DIR"])
            self.assertEqual(bundle / "configs", paths["CONFIG_DIR"])

    def test_standalone_windows_restore_regions_without_opening_mqtt(self):
        from pose_project.apps import gate_app, reid_app
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            reid_path = config_dir / "reid_roi_config.json"
            reid_path.write_text(json.dumps({"rois": {
                "A": {"enabled": True, "rect": [.1, .2, .8, .9]}}}), encoding="utf-8")
            with patch.object(gate_app, "CONFIG_DIR", config_dir), \
                 patch.object(gate_app, "FaceEngine",
                              return_value=SimpleNamespace(available=False)), \
                 patch.object(reid_app, "CONFIG_PATH", str(reid_path)), \
                 patch.object(reid_app, "PersonDetector"), \
                 patch.object(reid_app, "ReidExtractor",
                              return_value=SimpleNamespace(device="cpu")), \
                 patch.object(reid_app.MqttReceiver, "start") as connect:
                gate_window = gate_app.MainWindow()
                reid_window = reid_app.MainWindow()
                try:
                    self.assertEqual(str(config_dir / "gate_config.json"), gate_window.config_path)
                    self.assertTrue(reid_window.rois["A"]["enabled"])
                    self.assertEqual((.1, .2, .8, .9), reid_window.rois["A"]["rect"])
                    self.assertIsNone(reid_window.mqtt)
                    self.assertIsNone(reid_window.roi_publisher)
                    connect.assert_not_called()
                finally:
                    gate_window.close()
                    reid_window.close()
                    app.processEvents()
                self.assertTrue((config_dir / "gate_config.json").is_file())
                self.assertTrue(json.loads(reid_path.read_text(encoding="utf-8"))
                                ["rois"]["A"]["enabled"])


if __name__ == "__main__":
    unittest.main()
