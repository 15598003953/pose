"""Regression coverage for the live passenger evidence view."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from pose_project.apps.integration_app import IntegrationWindow, configure_application
from pose_project.integration.configuration import defaults


class LiveEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_application(cls.app)

    def test_gate_reid_and_binding_share_one_evidence_row(self):
        window = IntegrationWindow(defaults())
        try:
            window._person({
                "passenger_id": "passenger-1",
                "local_id": "P001",
                "capture_time": "2026-11-09T10:00:00",
                "image": np.zeros((20, 12, 3), dtype=np.uint8),
            })
            self.assertEqual(1, window.evidence_table.rowCount())
            self.assertEqual("等待重识别",
                             window.evidence_table.item(0, 3).text())

            window._recognition({
                "captured_at_ms": 2000,
                "completed_at_ms": 2050,
                "regions": {
                    "A": {
                        "candidates": [{
                            "label": "passenger-1",
                            "display_label": "P001",
                            "score": .91,
                            "tid": 7,
                            "bbox_xyxy": [1, 2, 10, 18],
                            "status": "matched",
                        }],
                        "detections": [],
                    }
                },
                "metrics": {},
            })
            self.assertEqual(1, window.evidence_table.rowCount())
            self.assertEqual("已匹配",
                             window.evidence_table.item(0, 3).text())
            self.assertEqual("0.910",
                             window.evidence_table.item(0, 6).text())

            window._decision({
                "roi_id": "A",
                "passenger_id": "passenger-1",
                "rfid": "TRAY-1",
                "status": "bound",
                "reason": "unique_candidate",
            })
            self.assertEqual("已绑定",
                             window.evidence_table.item(0, 7).text())
            window.evidence_table.selectRow(0)
            self.assertIn("passenger-1", window.evidence_details.toPlainText())
            self.assertIn("TRAY-1", window.evidence_details.toPlainText())
        finally:
            window.close()

    def test_evidence_images_keep_a_large_display_area(self):
        window = IntegrationWindow(defaults())
        try:
            window.resize(1000, 720)
            window.show()
            window.event_tabs.setCurrentIndex(6)
            self.app.processEvents()
            self.assertGreaterEqual(window.evidence_gate_image.height(), 240)
            self.assertGreaterEqual(window.evidence_reid_image.height(), 240)
        finally:
            window.close()

    def test_completed_passenger_evidence_is_not_replaced_by_another_target(self):
        window = IntegrationWindow(defaults())
        try:
            original = np.zeros((30, 30, 3), dtype=np.uint8)
            candidate = {
                "label": "passenger-1", "score": .9, "tid": 7,
                "bbox_xyxy": [1, 2, 10, 18], "status": "matched",
                "identity_locked": True}
            report = {
                "captured_at_ms": 1000, "completed_at_ms": 1050,
                "_frame": original, "metrics": {},
                "regions": {"A": {"candidates": [candidate], "detections": []}}}
            window._recognition(report)
            self.assertEqual("已匹配，任务锁定", window.evidence_table.item(0, 3).text())
            report["_frame"] = np.full_like(original, 255)
            report["regions"] = {
                "B": {"candidates": [dict(candidate, tid=8, status="passenger_locked",
                                          identity_locked=False)], "detections": []}}
            window._recognition(report)
            evidence = window._evidence["passenger-1"]["recognition"]
            self.assertEqual("A", evidence["region"])
            self.assertEqual(7, evidence["tid"])
            self.assertIs(original, evidence["frame"])
            self.assertIn("不再分配", window.recognition_table.item(0, 4).text())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
