"""用本地视频和真实模型验证两路后台任务、旅客入库及联合界面。"""

import argparse
from copy import deepcopy
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from pose_project.apps.integration_app import IntegrationWindow, configure_application
from pose_project.common.project_paths import ROOT
from pose_project.integration.configuration import defaults, new_run_directory
from pose_project.integration.workers import encode_image


def make_clip(source, output, count, rotate=False):
    capture = cv2.VideoCapture(str(source))
    writer = None
    frames = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        while frames < count:
            ok, image = capture.read()
            if not ok:
                break
            if rotate:
                image = cv2.rotate(image, cv2.ROTATE_180)
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps, (image.shape[1], image.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Could not create isolated video fixture")
            writer.write(image)
            frames += 1
        if not frames:
            raise RuntimeError("Could not decode " + str(source))
        return frames
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def wait_until(predicate, seconds):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(25)
    if not predicate():
        raise RuntimeError("Timed out waiting for integration workers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = defaults()
    output = args.output or new_run_directory(settings)
    if args.output and output.exists() and any(output.iterdir()):
        raise ValueError("Verification output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    gate_clip, reid_clip = output / "gate_clip.mp4", output / "reid_clip.mp4"
    gate_frames = make_clip(
        ROOT / "data/BC5496691_1_20260923T180406Z_20260923T182102Z.mp4", gate_clip, 30)
    reid_frames = make_clip(ROOT / "data/verification/reid_sample.mp4", reid_clip, 100)
    capture = cv2.VideoCapture(str(reid_clip))
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not decode Re-ID fixture")
    module = import_module("pose_project.reid.search_worker")
    detector = module.PersonDetector(weights=settings["reid"]["detector_weights"],
                                     allow_download=False)
    boxes, scores = detector.detect(image)
    if not len(boxes):
        raise RuntimeError("No person detected in the local Re-ID fixture")
    box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
    crop = module.crop_from_frame(image, box)
    cv2.imwrite(str(output / "reference_person.jpg"), crop)
    height, width = image.shape[:2]
    settings["gate"]["source"] = str(gate_clip)
    settings["reid"].update(source=str(reid_clip), image_size=[width, height])
    settings["binding"]["regions"] = {
        "A": {"roi_xyxy": [0, 0, width, height],
              "operation_roi_xyxy": [0, 0, width, height]}}
    settings["output_dir"] = str(output / "sessions")
    settings["debug"]["field_test"].update(enabled=True, test_id="LOCAL-RECORDING-CHECK")
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    window = IntegrationWindow(settings, output / "saved_settings.json")
    window.show()
    frames = {"gate": 0, "reid": 0}
    position_messages = []
    window.controller.frame.connect(
        lambda stage, image: frames.__setitem__(stage, frames[stage] + 1))
    window.controller.message.connect(
        lambda record: position_messages.append(record["payload"])
        if "detections" in record["payload"] else None)
    try:
        window.start_session()
        if not window.controller.running:
            raise RuntimeError("Combined session did not start")
        stable = "0123456789abcdef0123456789abcdef-2026-11-03"
        window.controller.receive_collected({
            "passenger_id": stable, "person_id": stable, "local_id": "P001",
            "capture_time": "2026-11-03 09:00:00.000",
            "images": {"upper": encode_image(crop), "face": None},
        })
        wait_until(lambda: frames["gate"] > 0 and frames["reid"] >= 2, 180)
        window.controller.inject_rfid("A", "MODEL-PIPELINE-TEST")
        QTest.qWait(800)
        for size in ((1440, 940), (1000, 720)):
            window.resize(*size)
            app.processEvents()
            window.grab().save(str(output / ("running_{}x{}.png".format(*size))))
        for index, name in ((1, "messages"), (2, "gallery"), (3, "logs")):
            window.event_tabs.setCurrentIndex(index)
            app.processEvents()
            window.grab().save(str(output / (name + ".png")))
        run_dir = window.controller.run_dir
        gallery_count = window.people_table.rowCount()
        window.stop_session()
        wait_until(lambda: not window.controller.running, 60)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        results = [
            json.loads(line)["data"] for line in
            (run_dir / "test_results.jsonl").read_text(encoding="utf-8").splitlines()]
        image_paths = set()
        candidates = []
        for result in results:
            image_paths.update(result["images"].get(key) for key in
                               ("source_frame", "annotated_frame"))
            for region in result["regions"].values():
                for candidate in region["candidates"]:
                    candidates.append(candidate)
                    image_paths.add(candidate.get("image_path"))
                    image_paths.update(candidate.get("reference_images", {}).values())
                    for ranked in candidate.get("ranked_candidates", []):
                        image_paths.update(ranked.get("reference_images", {}).values())
        image_paths.discard(None)
        images_ok = bool(image_paths) and all(
            cv2.imdecode(np.fromfile(run_dir / path, np.uint8), cv2.IMREAD_COLOR)
            is not None for path in image_paths)
        recording_ok = bool(results) and bool(candidates) and images_ok and all(
            "ranked_candidates" in row and "accepted_label" in row for row in candidates)
        report = {
            "ok": frames["gate"] > 0 and frames["reid"] > 0 and gallery_count == 1
                  and any(row["detections"] for row in position_messages)
                  and not summary.get("gate_errors") and not summary.get("reid_errors")
                  and recording_ok,
            "gate_fixture_frames": gate_frames, "reid_fixture_frames": reid_frames,
            "preview_frames": frames, "gallery_count": gallery_count,
            "recognized_position_messages": sum(bool(row["detections"]) for row in position_messages),
            "summary": summary, "run_dir": str(run_dir),
            "recorded_results": len(results), "recorded_images": len(image_paths),
            "recording_ok": recording_ok,
            "scope": "Local clips, real gate/YOLO/FastReID models, local transport, Qt controls",
            "not_verified": "Live cameras, real RFID, production broker, field accuracy",
        }
        (output / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["ok"] else 1
    finally:
        if window.controller.running:
            window.controller.stop()
            wait_until(lambda: not window.controller.running, 90)
        window.close()


if __name__ == "__main__":
    raise SystemExit(main())
