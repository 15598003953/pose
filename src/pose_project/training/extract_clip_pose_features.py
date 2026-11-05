import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import cv2 as cv
from pose_project.vision.mp_persondet import MPPersonDet
from pose_project.vision.mp_pose import MPPose
from pose_project.actions.pose_action_recognizer import PoseWindowFeatures
from pose_project.common.project_paths import DATA_DIR, POSE_MODEL_PATH, PERSON_MODEL_PATH


class ClipPoseAccumulator:
    def __init__(self, clip_id, label, start_frame, end_frame, fps, video_path=""):
        self.clip_id = clip_id
        self.label = label
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.fps = float(fps) if fps else 0.0
        self.video_path = video_path
        self.total_frames = max(0, self.end_frame - self.start_frame)
        self.window = PoseWindowFeatures(max(1, self.total_frames))

    def add_frame(self, poses):
        self.window.add(poses)

    def to_record(self):
        return {
            "video_path": self.video_path,
            "clip_id": self.clip_id,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sec": round(self.start_frame / self.fps, 4) if self.fps else 0.0,
            "end_sec": round(self.end_frame / self.fps, 4) if self.fps else 0.0,
            "fps": round(self.fps, 6),
            "total_frames": self.total_frames,
            **self.window.to_record(),
        }


def read_clip_labels(path: Path, include_skip=False):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "")
            if label == "skip" and not include_skip:
                continue
            if label not in ("put_luggage", "not_put_luggage", "skip"):
                continue
            yield {
                "video_path": row["video_path"],
                "clip_id": row["clip_id"],
                "label": label,
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "fps": float(row.get("fps") or 0),
            }


def detect_poses(frame, person_detector, pose_estimator):
    persons = person_detector.infer(frame)
    poses = []
    if persons is None:
        return poses
    for person in list(persons):
        pose = pose_estimator.infer(frame, person)
        if pose is not None:
            poses.append(pose)
    return poses


def extract_records(labels, stride_frames=1, max_clips=0):
    if not os.path.isfile(POSE_MODEL_PATH):
        raise FileNotFoundError(POSE_MODEL_PATH)
    if not os.path.isfile(PERSON_MODEL_PATH):
        raise FileNotFoundError("person detection model not found")

    person_detector = MPPersonDet(
        modelPath=PERSON_MODEL_PATH,
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

    labels = list(labels)
    if max_clips > 0:
        labels = labels[:max_clips]

    caps = {}
    records = []
    try:
        for idx, item in enumerate(labels, start=1):
            video_path = item["video_path"]
            cap = caps.get(video_path)
            if cap is None:
                cap = cv.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError("cannot open video: {}".format(video_path))
                caps[video_path] = cap

            acc = ClipPoseAccumulator(
                item["clip_id"], item["label"],
                item["start_frame"], item["end_frame"], item["fps"],
                video_path=video_path,
            )
            for frame_idx in range(item["start_frame"], item["end_frame"], max(1, int(stride_frames))):
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    acc.add_frame([])
                    continue
                acc.add_frame(detect_poses(frame, person_detector, pose_estimator))

            record = acc.to_record()
            records.append(record)
            print("[{}/{}] {} {} valid_pose_ratio={:.3f}".format(
                idx, len(labels), item["label"], item["clip_id"], record["valid_pose_ratio"]))
    finally:
        for cap in caps.values():
            cap.release()

    return records


def make_quality_report(records):
    records = list(records)
    label_counts = Counter(record["label"] for record in records)
    ratios_by_label = defaultdict(list)
    visibility_by_label = defaultdict(lambda: defaultdict(list))

    for record in records:
        label = record["label"]
        ratios_by_label[label].append(float(record.get("valid_pose_ratio", 0.0)))
        for keypoint, score in record.get("keypoint_visibility", {}).items():
            visibility_by_label[label][keypoint].append(float(score))

    return {
        "total_clips": len(records),
        "label_counts": dict(label_counts),
        "valid_pose_ratio_mean": round(
            float(mean([record.get("valid_pose_ratio", 0.0) for record in records])) if records else 0.0,
            4,
        ),
        "valid_pose_ratio_by_label": {
            label: round(float(mean(values)), 4) if values else 0.0
            for label, values in ratios_by_label.items()
        },
        "keypoint_visibility_by_label": {
            label: {
                keypoint: round(float(mean(values)), 4) if values else 0.0
                for keypoint, values in keypoints.items()
            }
            for label, keypoints in visibility_by_label.items()
        },
    }


def write_jsonl(path: Path, records: Iterable[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract pose quality and upper-body features from labeled clips.")
    parser.add_argument(
        "--labels_csv",
        default=str(DATA_DIR / "clip_labels" / "labels.csv"),
        help="CSV created by label_video_clips.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DATA_DIR / "clip_features"),
        help="Directory for features.jsonl and quality_report.json.",
    )
    parser.add_argument("--include_skip", action="store_true", help="Also extract clips labeled skip.")
    parser.add_argument("--stride_frames", type=int, default=3, help="Frame sampling stride inside each clip.")
    parser.add_argument("--max_clips", type=int, default=0, help="Limit clips for a quick run; 0 means all.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    labels = list(read_clip_labels(Path(args.labels_csv), include_skip=args.include_skip))
    records = extract_records(labels, stride_frames=args.stride_frames, max_clips=args.max_clips)
    output_dir = Path(args.output_dir)
    features_path = output_dir / "features.jsonl"
    report_path = output_dir / "quality_report.json"
    write_jsonl(features_path, records)
    write_json(report_path, make_quality_report(records))
    print("features_path", features_path)
    print("quality_report_path", report_path)


if __name__ == "__main__":
    main()
