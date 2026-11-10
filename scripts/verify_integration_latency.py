"""启动本机 RTSP 测试源，测量真实模型的画面更新、推理和位置消息延迟。"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from pose_project.apps.integration_app import IntegrationWindow, configure_application
from pose_project.integration.configuration import defaults
from pose_project.integration.workers import RecognitionWorker, encode_image


def distribution(values):
    if len(values) == 0:
        return {"count": 0}
    return dict(count=len(values), median=round(float(np.median(values)), 2),
                p95=round(float(np.percentile(values, 95)), 2),
                maximum=round(float(max(values)), 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rtsp-server", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--detector-image-size", type=int, default=960)
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(args.video.resolve()))
    ok, image = capture.read()
    capture.release()
    reference = cv2.imdecode(np.fromfile(args.reference, dtype=np.uint8), cv2.IMREAD_COLOR)
    if not ok or reference is None:
        raise RuntimeError("Missing video or reference image")
    height, width = image.shape[:2]
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    settings = defaults()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_config = output / "rtsp.yml"
    server_config.write_text(
        "rtsp: yes\nrtspAddress: 127.0.0.1:{}\nrtspTransports: [tcp]\n"
        "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\npaths:\n  tray:\n".format(port),
        encoding="utf-8")
    url = "rtsp://127.0.0.1:{}/tray".format(port)
    settings["gate"].update(source=url)
    settings["gate"]["opts"].update(rotate=0, require_approach=True)
    settings["reid"].update(source=url, image_size=[width, height],
                            detector_image_size=args.detector_image_size,
                            flip_tta=not args.no_flip)
    settings["binding"]["regions"] = {
        "A": {"roi_xyxy": [0, 0, width, height],
              "operation_roi_xyxy": [0, 0, width, height]}}
    settings["output_dir"] = str(output / "sessions")
    settings["debug"] = {"enabled": True}
    window = IntegrationWindow(settings, output / "config.json")
    window.show()
    timings, previews, ages, reports = [], defaultdict(list), [], []
    original = RecognitionWorker._process

    def measure(worker, frame, captured):
        started = time.monotonic()
        try:
            return original(worker, frame, captured)
        finally:
            timings.append((time.monotonic(), (time.monotonic() - started) * 1000))

    RecognitionWorker._process = measure
    window.controller.frame.connect(
        lambda stage, frame: previews[stage].append(time.monotonic()))
    window.controller.message.connect(
        lambda record: ages.append((time.monotonic(), record["received_at_ms"]
                                    - record["payload"]["captured_at_ms"]))
        if "captured_at_ms" in record["payload"] else None)
    if hasattr(window.controller, "recognition"):
        window.controller.recognition.connect(reports.append)
    processes, handles = [], []
    report = {"ok": False}
    try:
        for name, command in (
            ("server", [str(args.rtsp_server.resolve()), str(server_config)]),
            ("publisher", [str(args.ffmpeg.resolve()), "-hide_banner", "-loglevel", "error",
                           "-re", "-stream_loop", "-1", "-i", str(args.video.resolve()),
                           "-an", "-c:v", "libx264", "-preset", "ultrafast",
                           "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", "15",
                           "-f", "rtsp", "-rtsp_transport", "tcp", url])):
            handle = (output / (name + ".log")).open("wb")
            handles.append(handle)
            processes.append(subprocess.Popen(command, stdout=handle,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW))
            QTest.qWait(1500)
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("RTSP fixture did not start")
        window.start_session()
        if not window.controller.running:
            raise RuntimeError("Session did not start")
        window.controller.receive_collected({
            "passenger_id": "LATENCY-REFERENCE", "local_id": "P001",
            "capture_time": "fixture", "images": {"upper": encode_image(reference)}})
        deadline = time.monotonic() + 180
        while not timings and time.monotonic() < deadline:
            QTest.qWait(25)
        if not timings:
            raise RuntimeError("No recognition frames")
        started = time.monotonic()
        for block in range(args.seconds):
            QTest.qWait(1000)
            if not window.controller.running or window.controller.counts["reid_errors"]:
                raise RuntimeError("Recognition worker failed during latency verification")
            if block % 15 == 14:
                print("elapsed={}s frames={} latest_ms={}".format(
                    block + 1, len(timings), ages[-1][1] if ages else None), flush=True)
        for size in ((1440, 940), (1000, 720)):
            window.resize(*size)
            for name in ("重识别结果", "性能调试"):
                for index in range(window.event_tabs.count()):
                    if window.event_tabs.tabText(index).startswith(name):
                        window.event_tabs.setCurrentIndex(index)
                app.processEvents()
                window.grab().save(str(output / ("{}_{}x{}.png".format(name, *size))))
        window.stop_session()
        deadline = time.monotonic() + 60
        while window.controller.running and time.monotonic() < deadline:
            QTest.qWait(25)
        if window.controller.running:
            raise RuntimeError("Workers did not stop")
        report.update(
            ok=bool(ages and not window.controller.counts["reid_errors"]
                    and not window.controller.counts["gate_errors"]),
            seconds=args.seconds, processing_ms=distribution([value for _, value in timings]),
            position_age_ms=distribution([value for _, value in ages]),
            last_30s_age_ms=distribution([value for when, value in ages
                                        if when >= started + args.seconds - 30]),
            preview_gap_ms={stage: distribution(np.diff(values) * 1000)
                            for stage, values in previews.items()},
            preview_counts={stage: len(values) for stage, values in previews.items()},
            result_reports=len(reports),
            recognized_reports=sum(any(region["detections"] for region in row["regions"].values())
                                   for row in reports),
            cpu_requested=args.cpu, detector_image_size=args.detector_image_size,
            flip_tta=not args.no_flip,
            run_dir=str(window.controller.run_dir),
            scope="Loopback RTSP, real models, one seeded reference, both workers, offscreen GUI",
            not_verified="Field cameras, production MQTT, actual RFID, field identity accuracy")
    finally:
        if window.controller.running:
            window.controller.stop()
            while window.controller.running:
                QTest.qWait(50)
        window.close()
        RecognitionWorker._process = original
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()
        (output / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
