"""Check installed commands, real video inference, training and GUI playback."""

import argparse
import ast
from datetime import datetime
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2 as cv
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from pose_project.__main__ import COMMANDS
from pose_project.actions.mqtt_publisher import MqttConfig
from pose_project.actions.pose_action_recognizer import PutLuggageModelRecognizer
from pose_project.apps.pose_app import PoseMainWindow, PoseVideoThread
from pose_project.common.project_paths import DATA_DIR, ROOT
from pose_project.video.app_config import SourceConfig


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_command(output, cwd, name, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "pose_project", *arguments],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
    check(result.returncode == 0, name + " failed: " + result.stderr)


def check_playback(output, video):
    app = QApplication.instance() or QApplication([])
    errors, frames = [], []
    started = time.monotonic()
    progress = {}
    with patch("pose_project.apps.pose_app.QMessageBox.critical",
               side_effect=lambda parent, title, text: errors.append(text)), \
            patch.object(PoseVideoThread, "_make_event_log_path",
                         return_value=str(output / "gui_video_events.jsonl")):
        window = PoseMainWindow()
        window.video_path = str(video)
        window.source_config = SourceConfig(infer_stride=3)
        window.mqtt_config = MqttConfig()
        window.rois = []
        window.resize(1320, 800)
        window.show()
        try:
            check(window.pose_estimator is not None and window.person_detector is not None,
                  "GUI models did not initialize")
            window._play()
            worker = window.thread
            check(worker is not None, "video thread did not start")
            worker.frame_ready.connect(lambda frame, fps, index, total: frames.append(index))
            deadline = time.monotonic() + 30
            while not errors and (not frames or frames[-1] < 15) and time.monotonic() < deadline:
                QTest.qWait(20)
            check(not errors and frames and frames[-1] >= 15, "GUI playback failed")
            progress["seek_requested_after_sec"] = round(time.monotonic() - started, 3)
            generation = worker._recognizer_generation
            worker.seek_to(45)
            deadline = time.monotonic() + 30
            while not errors and frames[-1] < 66 and time.monotonic() < deadline:
                QTest.qWait(20)
            check(not errors and frames[-1] >= 66,
                  "GUI seek/playback failed; last frames: " + str(frames[-5:]))
            check(worker._recognizer_generation > generation, "seek retained old action history")
            check(window.grab().save(str(output / "video_playback.png")), "GUI capture failed")
            check(window._stop() and not worker.isRunning(), "video thread did not stop")
            return {
                "models_loaded": True, "displayed_frames": len(frames),
                "last_displayed_frame": frames[-1], "seek_target": 45,
                "seek_history_reset": True, "worker_stopped": True, "errors": errors,
            }
        finally:
            progress.update({
                "elapsed_sec": round(time.monotonic() - started, 3),
                "frames": frames, "errors": errors,
                "worker_running": window.thread is not None and window.thread.isRunning(),
            })
            (output / "video_progress.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            if window.thread is not None:
                window.thread.stop()
                window.thread.wait(10000)
            window.close()
            app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new or empty output directory")
    args = parser.parse_args()
    output = args.output.resolve()
    check(not output.exists() or (output.is_dir() and not any(output.iterdir())),
          "choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "project_root": str(ROOT),
        "runtime_versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "opencv-contrib-python", "PyQt5", "onnxruntime", "paho-mqtt",
        )},
    }
    sources = sorted((ROOT / "src").rglob("*.py"))
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        parts = source.relative_to(ROOT / "src").with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        importlib.import_module(".".join(parts))
    evidence["parsed_and_imported_modules"] = len(sources)

    with tempfile.TemporaryDirectory(prefix="pose_verify_") as unrelated:
        for command in COMMANDS:
            run_command(output, unrelated, "help_" + command, command, "--help")
        console = Path(sys.executable).parent / "Scripts" / ("pose.exe" if os.name == "nt" else "pose")
        if os.name != "nt":
            console = Path(sys.executable).parent / "pose"
        result = subprocess.run([str(console), "--help"], cwd=unrelated,
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        (output / "console_help.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        check(result.returncode == 0, "installed pose command failed")
        evidence["cli_help_checks"] = len(COMMANDS)
        evidence["installed_console_command"] = str(console)
        evidence["commands_work_from_unrelated_directory"] = True

        run_command(output, unrelated, "simulate", "simulate", "--output", str(output / "simulation"))
        run_command(
            output, unrelated, "replay", "replay",
            "--input", str(output / "simulation" / "replay.jsonl"),
            "--expected", str(output / "simulation" / "expected_bindings.jsonl"),
            "--output", str(output / "replay"),
        )
        evaluation = json.loads((output / "replay" / "evaluation.json").read_text(encoding="utf-8"))
        check(evaluation["all_events_match"], "binding results differ from independent labels")
        evidence["binding"] = {key: value for key, value in evaluation.items() if key != "cases"}

        run_command(output, unrelated, "extract", "extract", "--stride_frames", "10",
                    "--max_clips", "2", "--output_dir", str(output / "features"))
        records = [json.loads(line) for line in
                   (output / "features" / "features.jsonl").read_text(encoding="utf-8").splitlines()]
        check(len(records) == 2 and all(row["valid_pose_frames"] > 0 for row in records),
              "feature extraction did not produce two valid clips")
        evidence["real_feature_clips"] = [
            {"clip_id": row["clip_id"], "valid_pose_frames": row["valid_pose_frames"]}
            for row in records
        ]
        run_command(output, unrelated, "train", "train", "--output_dir", str(output / "training"))
        PutLuggageModelRecognizer(str(output / "training" / "model.json"))
        evidence["training"] = json.loads((output / "training" / "report.json").read_text(encoding="utf-8"))

        video = DATA_DIR / "2026-08-12_102107_553.mp4"
        run_command(output, unrelated, "offline", "offline", "--video", str(video),
                    "--output_dir", str(output / "offline"), "--max_frames", "40")
    preview = next((output / "offline").glob("*_preview.mp4"))
    capture = cv.VideoCapture(str(preview))
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            check(frame.size > 0, "preview contains an empty frame")
            decoded += 1
    finally:
        capture.release()
    check(decoded == 40, "preview did not decode all 40 frames")
    evidence["preview_decoded_frames"] = decoded
    evidence["video_gui"] = check_playback(output, video)
    (output / "verification.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
