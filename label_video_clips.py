import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2 as cv


VALID_LABELS = {"put_luggage", "not_put_luggage", "skip"}
FIELDNAMES = [
    "video_path",
    "clip_id",
    "label",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "fps",
    "notes",
]


@dataclass
class ClipLabel:
    video_path: str
    clip_id: str
    label: str
    start_frame: int
    end_frame: int
    fps: float
    notes: str = ""

    @property
    def start_sec(self):
        return self.start_frame / self.fps if self.fps else 0.0

    @property
    def end_sec(self):
        return self.end_frame / self.fps if self.fps else 0.0

    def to_row(self):
        return {
            "video_path": self.video_path,
            "clip_id": self.clip_id,
            "label": self.label,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "start_sec": "{:.3f}".format(self.start_sec),
            "end_sec": "{:.3f}".format(self.end_sec),
            "fps": "{:.6f}".format(float(self.fps)),
            "notes": self.notes,
        }


def generate_windows(total_frames: int, fps: float, clip_seconds: float, stride_seconds: float) -> List[Tuple[int, int]]:
    clip_frames = max(1, int(round(float(fps) * float(clip_seconds))))
    stride_frames = max(1, int(round(float(fps) * float(stride_seconds))))
    total_frames = max(0, int(total_frames))
    if total_frames <= 0:
        return []
    if total_frames <= clip_frames:
        return [(0, total_frames)]

    windows = []
    start = 0
    while start + clip_frames <= total_frames:
        windows.append((start, start + clip_frames))
        start += stride_frames
    return windows


def clip_id_for(video_path: str, start_frame: int, end_frame: int) -> str:
    stem = Path(video_path).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return "{}_{:06d}_{:06d}".format(safe_stem or "video", int(start_frame), int(end_frame))


def write_labels_csv(path: Path, labels: Iterable[ClipLabel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for label in labels:
            writer.writerow(label.to_row())


def load_existing_labels(path: Path) -> List[ClipLabel]:
    path = Path(path)
    if not path.is_file():
        return []
    labels = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "")
            if label not in VALID_LABELS:
                continue
            try:
                labels.append(ClipLabel(
                    video_path=row.get("video_path", ""),
                    clip_id=row.get("clip_id", ""),
                    label=label,
                    start_frame=int(row.get("start_frame", 0)),
                    end_frame=int(row.get("end_frame", 0)),
                    fps=float(row.get("fps", 0) or 0),
                    notes=row.get("notes", ""),
                ))
            except ValueError:
                continue
    return labels


def _read_frame(cap, frame_idx):
    cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _draw_info(frame, idx, total, start_frame, end_frame, fps, existing_label=None):
    display = frame.copy()
    start_sec = start_frame / fps if fps else 0.0
    end_sec = end_frame / fps if fps else 0.0
    lines = [
        "Clip {}/{} | frames {}-{} | {:.2f}-{:.2f}s".format(
            idx + 1, total, start_frame, end_frame, start_sec, end_sec),
        "p=put_luggage  n=not_put_luggage  s=skip  b=back  r=replay  q=save&quit",
    ]
    if existing_label:
        lines.append("current label: {}".format(existing_label))
    height = 28 + 24 * len(lines)
    cv.rectangle(display, (8, 8), (min(display.shape[1] - 8, 1080), height), (0, 0, 0), -1)
    for line_idx, line in enumerate(lines):
        cv.putText(display, line, (16, 34 + 24 * line_idx),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv.LINE_AA)
    return display


def _play_clip(cap, window_name, start_frame, end_frame, fps, clip_index, total_clips, existing_label):
    delay = max(1, int(round(1000.0 / fps))) if fps else 80
    frame_idx = int(start_frame)
    while frame_idx < int(end_frame):
        frame = _read_frame(cap, frame_idx)
        if frame is None:
            break
        display = _draw_info(frame, clip_index, total_clips, start_frame, end_frame, fps, existing_label)
        cv.imshow(window_name, display)
        key = cv.waitKey(delay) & 0xFF
        if key in (ord("p"), ord("n"), ord("s"), ord("b"), ord("r"), ord("q")):
            return chr(key)
        frame_idx += 1

    frame = _read_frame(cap, max(start_frame, end_frame - 1))
    if frame is not None:
        display = _draw_info(frame, clip_index, total_clips, start_frame, end_frame, fps, existing_label)
        cv.imshow(window_name, display)
    while True:
        key = cv.waitKey(0) & 0xFF
        if key in (ord("p"), ord("n"), ord("s"), ord("b"), ord("r"), ord("q")):
            return chr(key)


def label_video(args) -> Sequence[ClipLabel]:
    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video: {}".format(video_path))

    fps = float(cap.get(cv.CAP_PROP_FPS) or 0) or 12.5
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    windows = generate_windows(total_frames, fps, args.clip_seconds, args.stride_seconds)
    if args.max_clips > 0:
        windows = windows[:args.max_clips]

    output_csv = Path(args.output_csv)
    labels = load_existing_labels(output_csv) if args.resume else []
    label_by_clip = {label.clip_id: label for label in labels}

    window_name = "Put Luggage Clip Labeler"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, args.window_width, args.window_height)

    idx = 0
    try:
        while idx < len(windows):
            start_frame, end_frame = windows[idx]
            clip_id = clip_id_for(video_path, start_frame, end_frame)
            existing = label_by_clip.get(clip_id)
            key = _play_clip(
                cap, window_name, start_frame, end_frame, fps,
                idx, len(windows), existing.label if existing else None)

            if key == "q":
                break
            if key == "b":
                idx = max(0, idx - 1)
                continue
            if key == "r":
                continue

            label = {
                "p": "put_luggage",
                "n": "not_put_luggage",
                "s": "skip",
            }[key]
            label_by_clip[clip_id] = ClipLabel(
                video_path=video_path,
                clip_id=clip_id,
                label=label,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                notes="",
            )
            write_labels_csv(output_csv, sorted(label_by_clip.values(), key=lambda item: item.start_frame))
            print("saved {} -> {}".format(clip_id, label))
            idx += 1
    finally:
        cap.release()
        cv.destroyAllWindows()

    final_labels = sorted(label_by_clip.values(), key=lambda item: item.start_frame)
    write_labels_csv(output_csv, final_labels)
    return final_labels


def build_arg_parser():
    default_output = Path(__file__).resolve().parent / "data" / "clip_labels" / "labels.csv"
    parser = argparse.ArgumentParser(description="Label fixed-length video clips for put-luggage classification.")
    parser.add_argument("--video", required=True, help="Path to the source video.")
    parser.add_argument("--output_csv", default=str(default_output), help="CSV label output path.")
    parser.add_argument("--clip_seconds", type=float, default=4.0, help="Clip window length in seconds.")
    parser.add_argument("--stride_seconds", type=float, default=2.0, help="Stride between clip windows in seconds.")
    parser.add_argument("--max_clips", type=int, default=0, help="Limit clips for a quick labeling pass; 0 means all.")
    parser.add_argument("--resume", action="store_true", help="Load existing labels.csv and continue editing.")
    parser.add_argument("--window_width", type=int, default=1280)
    parser.add_argument("--window_height", type=int, default=720)
    return parser


def main():
    parser = build_arg_parser()
    labels = label_video(parser.parse_args())
    print("labels_saved", len(labels))


if __name__ == "__main__":
    main()
