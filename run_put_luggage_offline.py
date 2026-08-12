import argparse
import json
import os
import time

import cv2 as cv

from mp_persondet import MPPersonDet
from mp_pose import MPPose
from pose_action_recognizer import PutLuggageRecognizer
from pose_app import (
    POSE_MODEL_PATH,
    PUT_LUGGAGE_DISPLAY_TEXT,
    _PERSON_MODEL_PATH,
    draw_stickman,
)


def _draw_overlay(frame, result):
    color = (0, 0, 255) if result.state == "put_luggage" else (0, 255, 255)
    cv.rectangle(frame, (8, 64), (470, 118), (0, 0, 0), -1)
    cv.putText(frame, result.debug_text,
               (14, 86), cv.FONT_HERSHEY_SIMPLEX, 0.55,
               color, 1, cv.LINE_AA)
    if result.state == "put_luggage":
        cv.putText(frame, PUT_LUGGAGE_DISPLAY_TEXT,
                   (14, 112), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 0, 255), 2, cv.LINE_AA)


def run(args):
    if not os.path.isfile(args.video):
        raise FileNotFoundError(args.video)
    if not os.path.isfile(POSE_MODEL_PATH):
        raise FileNotFoundError(POSE_MODEL_PATH)
    if not _PERSON_MODEL_PATH or not os.path.isfile(_PERSON_MODEL_PATH):
        raise FileNotFoundError("person detection model not found")

    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    event_path = os.path.join(args.output_dir, video_name + "_put_luggage_events.jsonl")
    debug_path = os.path.join(args.output_dir, video_name + "_put_luggage_debug.jsonl")
    preview_path = os.path.join(args.output_dir, video_name + "_put_luggage_preview.mp4")

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("cannot open video: " + args.video)

    fps = cap.get(cv.CAP_PROP_FPS) or 12.5
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    writer = cv.VideoWriter(
        preview_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        fps / max(args.stride, 1),
        (width, height),
    )

    person_detector = MPPersonDet(
        modelPath=_PERSON_MODEL_PATH,
        nmsThreshold=0.3,
        scoreThreshold=0.3,
        topK=5000,
        backendId=cv.dnn.DNN_BACKEND_OPENCV,
        targetId=cv.dnn.DNN_TARGET_CPU,
    )
    pose_estimator = MPPose(
        modelPath=POSE_MODEL_PATH,
        confThreshold=0.3,
        backendId=cv.dnn.DNN_BACKEND_OPENCV,
        targetId=cv.dnn.DNN_TARGET_CPU,
    )
    recognizer = PutLuggageRecognizer(fps=fps / max(args.stride, 1))

    frame_idx = 0
    processed = 0
    events = 0
    started = time.time()
    with open(event_path, "w", encoding="utf-8") as event_log, \
            open(debug_path, "w", encoding="utf-8") as debug_log:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            persons = person_detector.infer(frame)
            poses = []
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose is not None:
                    poses.append(pose)

            display = draw_stickman(frame, poses)
            timestamp_ms = int(frame_idx * 1000.0 / fps)
            result = recognizer.update(poses, processed, timestamp_ms=timestamp_ms)
            debug_log.write(json.dumps({
                "processed_frame": processed,
                "source_frame_idx": int(frame_idx),
                "timestamp_ms": timestamp_ms,
                "state": result.state,
                "confidence": round(float(result.confidence), 3),
                "debug_text": result.debug_text,
            }, ensure_ascii=False) + "\n")
            for event in result.events:
                event["source_frame_idx"] = int(frame_idx)
                event_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                events += 1
            _draw_overlay(display, result)
            writer.write(display)

            processed += 1
            frame_idx += 1
            if args.max_frames and processed >= args.max_frames:
                break

    cap.release()
    writer.release()
    elapsed = time.time() - started
    print("processed_frames", processed)
    print("source_frames", total)
    print("events", events)
    print("event_path", event_path)
    print("debug_path", debug_path)
    print("preview_path", preview_path)
    print("elapsed_sec", round(elapsed, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", default=os.path.join("result", "offline_put_luggage"))
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
