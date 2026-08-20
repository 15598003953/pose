import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extract_clip_pose_features import (
    ClipPoseAccumulator,
    detect_poses,
    make_quality_report,
    write_jsonl,
)


def _pose(visibility=0.9, presence=0.8, left_wrist=(10, 50), right_wrist=(30, 52)):
    landmarks = np.zeros((39, 5), dtype=np.float32)
    for idx, xy in {
        11: (10, 20),
        12: (30, 20),
        13: (10, 35),
        14: (30, 36),
        15: left_wrist,
        16: right_wrist,
    }.items():
        landmarks[idx] = [xy[0], xy[1], 0.0, visibility, presence]
    return [np.array([[0, 0], [40, 80]], dtype=np.float32), landmarks, None, None, None, 0.95]


class ExtractClipPoseFeaturesTest(unittest.TestCase):
    def test_accumulator_summarizes_upper_body_visibility_and_wrist_motion(self):
        acc = ClipPoseAccumulator("clip_001", "put_luggage", 0, 3, 10.0)

        acc.add_frame([_pose(left_wrist=(10, 50), right_wrist=(30, 52))])
        acc.add_frame([_pose(left_wrist=(10, 58), right_wrist=(30, 60))])
        acc.add_frame([])

        row = acc.to_record()

        self.assertEqual("clip_001", row["clip_id"])
        self.assertEqual("put_luggage", row["label"])
        self.assertEqual(3, row["total_frames"])
        self.assertEqual(3, row["sampled_frames"])
        self.assertEqual(2, row["valid_pose_frames"])
        self.assertAlmostEqual(2 / 3, row["valid_pose_ratio"], places=3)
        self.assertAlmostEqual(0.5333, row["keypoint_visibility"]["left_wrist"], places=3)
        self.assertEqual(8.0, row["features"]["left_wrist_y_range"])
        self.assertEqual(8.0, row["features"]["right_wrist_y_range"])

    def test_accumulator_adds_temporal_wrist_features(self):
        acc = ClipPoseAccumulator("clip_002", "put_luggage", 0, 4, 10.0)

        acc.add_frame([_pose(left_wrist=(10, 40), right_wrist=(30, 42))])
        acc.add_frame([_pose(left_wrist=(10, 55), right_wrist=(30, 57))])
        acc.add_frame([_pose(left_wrist=(10, 65), right_wrist=(30, 67))])
        acc.add_frame([_pose(left_wrist=(10, 62), right_wrist=(30, 64))])

        features = acc.to_record()["features"]

        self.assertEqual(22.0, features["left_wrist_y_displacement"])
        self.assertEqual(25.0, features["left_wrist_downward_range"])
        self.assertAlmostEqual(0.5, features["left_wrist_low_hold_ratio"], places=3)
        self.assertAlmostEqual(1.0, features["wrist_down_sync_ratio"], places=3)

    def test_quality_report_counts_labels_and_valid_pose_ratios(self):
        records = [
            {"label": "put_luggage", "valid_pose_ratio": 0.5},
            {"label": "put_luggage", "valid_pose_ratio": 1.0},
            {"label": "not_put_luggage", "valid_pose_ratio": 0.25},
        ]

        report = make_quality_report(records)

        self.assertEqual(3, report["total_clips"])
        self.assertEqual({"put_luggage": 2, "not_put_luggage": 1}, report["label_counts"])
        self.assertAlmostEqual(0.75, report["valid_pose_ratio_by_label"]["put_luggage"], places=3)

    def test_write_jsonl_outputs_one_json_object_per_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "features.jsonl"
            write_jsonl(output, [{"clip_id": "a"}, {"clip_id": "b"}])

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(["a", "b"], [json.loads(line)["clip_id"] for line in lines])

    def test_detect_poses_accepts_numpy_person_detector_output(self):
        class Detector:
            def infer(self, _frame):
                return np.array([[1.0, 2.0, 3.0]])

        class Estimator:
            def infer(self, _frame, _person):
                return _pose()

        poses = detect_poses(np.zeros((8, 8, 3), dtype=np.uint8), Detector(), Estimator())

        self.assertEqual(1, len(poses))


if __name__ == "__main__":
    unittest.main()
