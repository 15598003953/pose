import csv
import tempfile
import unittest
from pathlib import Path

from pose_project.training.label_video_clips import ClipLabel, generate_windows, write_labels_csv


class LabelVideoClipsTest(unittest.TestCase):
    def test_generates_fixed_length_windows_with_stride(self):
        windows = generate_windows(total_frames=125, fps=12.5, clip_seconds=4.0, stride_seconds=2.0)

        self.assertEqual([(0, 50), (25, 75), (50, 100), (75, 125)], windows)

    def test_writes_labels_csv_with_frame_and_time_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "labels.csv"
            labels = [
                ClipLabel(
                    video_path="D:/video.mp4",
                    clip_id="video_000000_000050",
                    label="put_luggage",
                    start_frame=0,
                    end_frame=50,
                    fps=12.5,
                    notes="",
                )
            ]

            write_labels_csv(output, labels)

            with output.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(1, len(rows))
        self.assertEqual("put_luggage", rows[0]["label"])
        self.assertEqual("0", rows[0]["start_frame"])
        self.assertEqual("50", rows[0]["end_frame"])
        self.assertEqual("0.000", rows[0]["start_sec"])
        self.assertEqual("4.000", rows[0]["end_sec"])


if __name__ == "__main__":
    unittest.main()
