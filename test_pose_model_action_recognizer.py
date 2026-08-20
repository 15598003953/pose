import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_action_recognizer import PutLuggageModelRecognizer


def _pose(left_wrist_y, right_wrist_y=None, visibility=0.95, presence=0.95):
    if right_wrist_y is None:
        right_wrist_y = left_wrist_y
    landmarks = np.zeros((39, 5), dtype=np.float32)
    points = {
        11: (10, 20),
        12: (30, 20),
        13: (10, 35),
        14: (30, 35),
        15: (10, left_wrist_y),
        16: (30, right_wrist_y),
    }
    for idx, (x, y) in points.items():
        landmarks[idx] = [x, y, 0.0, visibility, presence]
    return [np.array([[0, 0], [40, 80]], dtype=np.float32), landmarks, None, None, None, 0.9]


def _write_model(path):
    model = {
        "model_type": "nearest_centroid_numpy",
        "positive_label": "put_luggage",
        "negative_label": "not_put_luggage",
        "feature_names": ["feat:left_wrist_y_displacement"],
        "feature_mean": [10.0],
        "feature_std": [10.0],
        "positive_centroid": [1.0],
        "negative_centroid": [-1.0],
    }
    Path(path).write_text(json.dumps(model), encoding="utf-8")


class PutLuggageModelRecognizerTest(unittest.TestCase):
    def test_warms_up_until_window_has_enough_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            _write_model(model_path)
            recognizer = PutLuggageModelRecognizer(
                model_path=str(model_path), fps=10.0, clip_seconds=0.3,
                sample_stride_frames=1)

            result = recognizer.update([_pose(40)], frame_idx=0)

        self.assertEqual("warming_up", result.state)
        self.assertIn("warming_up", result.debug_text)

    def test_emits_put_luggage_event_when_model_probability_crosses_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            _write_model(model_path)
            recognizer = PutLuggageModelRecognizer(
                model_path=str(model_path), fps=10.0, clip_seconds=0.3,
                sample_stride_frames=1, threshold=0.5, cooldown_seconds=1.0)

            recognizer.update([_pose(40)], frame_idx=0, timestamp_ms=0)
            recognizer.update([_pose(50)], frame_idx=1, timestamp_ms=100)
            result = recognizer.update([_pose(60)], frame_idx=2, timestamp_ms=200)

        self.assertEqual("put_luggage", result.state)
        self.assertTrue(result.events)
        self.assertEqual("model_clip_classifier", result.events[0]["reason"])
        self.assertGreater(result.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
