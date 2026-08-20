import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import cv2 as cv
import numpy as np

from mp_persondet import MPPersonDet
from mp_pose import MPPose
from pose_app import POSE_MODEL_PATH, _PERSON_MODEL_PATH


KEYPOINTS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


class ClipPoseAccumulator:
    def __init__(self, clip_id, label, start_frame, end_frame, fps, video_path=""):
        self.clip_id = clip_id
        self.label = label
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.fps = float(fps) if fps else 0.0
        self.video_path = video_path
        self.total_frames = max(0, self.end_frame - self.start_frame)
        self.sampled_frames = 0
        self.valid_pose_frames = 0
        self.keypoint_scores: Dict[str, List[float]] = {name: [] for name in KEYPOINTS}
        self.coords: Dict[str, List[List[float]]] = {name: [] for name in KEYPOINTS}

    def add_frame(self, poses):
        self.sampled_frames += 1
        pose = self._select_pose(poses)
        if pose is None:
            for name in KEYPOINTS:
                self.keypoint_scores[name].append(0.0)
            return

        landmarks = np.asarray(pose[1])
        self.valid_pose_frames += 1
        for name, idx in KEYPOINTS.items():
            score = float(min(landmarks[idx, 3], landmarks[idx, 4]))
            self.keypoint_scores[name].append(score)
            if score > 0.1:
                self.coords[name].append([
                    float(landmarks[idx, 0]),
                    float(landmarks[idx, 1]),
                ])

    def to_record(self):
        visibility = {
            name: round(float(mean(values)) if values else 0.0, 4)
            for name, values in self.keypoint_scores.items()
        }
        features = {}
        for name, points in self.coords.items():
            arr = np.asarray(points, dtype=np.float32)
            if arr.size == 0:
                features[name + "_x_mean"] = None
                features[name + "_y_mean"] = None
                features[name + "_x_range"] = None
                features[name + "_y_range"] = None
                continue
            features[name + "_x_mean"] = round(float(np.mean(arr[:, 0])), 4)
            features[name + "_y_mean"] = round(float(np.mean(arr[:, 1])), 4)
            features[name + "_x_range"] = round(float(np.max(arr[:, 0]) - np.min(arr[:, 0])), 4)
            features[name + "_y_range"] = round(float(np.max(arr[:, 1]) - np.min(arr[:, 1])), 4)

        features.update(self._temporal_wrist_features())

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
            "sampled_frames": self.sampled_frames,
            "valid_pose_frames": self.valid_pose_frames,
            "valid_pose_ratio": round(
                self.valid_pose_frames / self.sampled_frames if self.sampled_frames else 0.0, 4),
            "keypoint_visibility": visibility,
            "features": features,
        }

    @staticmethod
    def _select_pose(poses):
        best = None
        best_score = -1.0
        for pose in poses or []:
            if len(pose) < 2:
                continue
            landmarks = np.asarray(pose[1])
            if landmarks.shape[0] <= max(KEYPOINTS.values()) or landmarks.shape[1] < 5:
                continue
            scores = [
                float(min(landmarks[idx, 3], landmarks[idx, 4]))
                for idx in KEYPOINTS.values()
            ]
            score = float(mean(scores))
            if score > best_score:
                best = pose
                best_score = score
        return best

    def _temporal_wrist_features(self):
        features = {}
        for side in ("left", "right"):
            name = side + "_wrist"
            features.update(self._single_wrist_temporal_features(name))
        features["wrist_down_sync_ratio"] = round(self._wrist_down_sync_ratio(), 4)
        return features

    def _single_wrist_temporal_features(self, name):
        points = self.coords.get(name, [])
        prefix = name + "_"
        if len(points) < 2:
            return {
                prefix + "y_displacement": 0.0,
                prefix + "downward_range": 0.0,
                prefix + "upward_recovery": 0.0,
                prefix + "mean_abs_y_velocity": 0.0,
                prefix + "max_down_y_velocity": 0.0,
                prefix + "low_hold_ratio": 0.0,
                prefix + "first_second_half_y_delta": 0.0,
            }

        arr = np.asarray(points, dtype=np.float32)
        y = arr[:, 1]
        dy = np.diff(y)
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        downward_range = y_max - y_min
        low_threshold = y_min + 0.8 * downward_range
        mid = max(1, len(y) // 2)
        first_half = y[:mid]
        second_half = y[mid:] if mid < len(y) else y[mid - 1:]
        return {
            prefix + "y_displacement": round(float(y[-1] - y[0]), 4),
            prefix + "downward_range": round(float(downward_range), 4),
            prefix + "upward_recovery": round(float(y_max - y[-1]), 4),
            prefix + "mean_abs_y_velocity": round(float(np.mean(np.abs(dy))) if dy.size else 0.0, 4),
            prefix + "max_down_y_velocity": round(float(np.max(np.maximum(dy, 0.0))) if dy.size else 0.0, 4),
            prefix + "low_hold_ratio": round(float(np.mean(y >= low_threshold)) if downward_range > 0 else 0.0, 4),
            prefix + "first_second_half_y_delta": round(float(np.mean(second_half) - np.mean(first_half)), 4),
        }

    def _wrist_down_sync_ratio(self):
        left = np.asarray(self.coords.get("left_wrist", []), dtype=np.float32)
        right = np.asarray(self.coords.get("right_wrist", []), dtype=np.float32)
        count = min(len(left), len(right))
        if count < 2:
            return 0.0
        left_dy = np.diff(left[:count, 1])
        right_dy = np.diff(right[:count, 1])
        either_down = (left_dy > 0) | (right_dy > 0)
        if not np.any(either_down):
            return 0.0
        both_down = (left_dy > 0) & (right_dy > 0)
        return float(np.sum(both_down) / np.sum(either_down))


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
    if not _PERSON_MODEL_PATH or not os.path.isfile(_PERSON_MODEL_PATH):
        raise FileNotFoundError("person detection model not found")

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
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract pose quality and upper-body features from labeled clips.")
    parser.add_argument(
        "--labels_csv",
        default=str(base_dir / "data" / "clip_labels" / "labels.csv"),
        help="CSV created by label_video_clips.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(base_dir / "data" / "clip_features"),
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
