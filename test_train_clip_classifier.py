import tempfile
import unittest
from pathlib import Path

from train_clip_classifier import (
    CentroidClassifier,
    build_feature_matrix,
    load_feature_records,
    write_predictions_csv,
)


class TrainClipClassifierTest(unittest.TestCase):
    def test_build_feature_matrix_flattens_visibility_and_numeric_features(self):
        records = [
            {
                "clip_id": "a",
                "label": "put_luggage",
                "keypoint_visibility": {"left_wrist": 0.9},
                "features": {"left_wrist_y_range": 30.0, "right_wrist_y_range": None},
            },
            {
                "clip_id": "b",
                "label": "not_put_luggage",
                "keypoint_visibility": {"left_wrist": 0.4},
                "features": {"left_wrist_y_range": 10.0, "right_wrist_y_range": 5.0},
            },
        ]

        matrix, labels, clip_ids, names = build_feature_matrix(records)

        self.assertEqual(["a", "b"], clip_ids)
        self.assertEqual(["put_luggage", "not_put_luggage"], labels)
        self.assertIn("vis:left_wrist", names)
        self.assertIn("feat:left_wrist_y_range", names)
        self.assertEqual((2, len(names)), matrix.shape)
        self.assertEqual(0.0, matrix[0, names.index("feat:right_wrist_y_range")])

    def test_centroid_classifier_scores_samples_near_positive_centroid_higher(self):
        records = [
            {"clip_id": "p1", "label": "put_luggage", "keypoint_visibility": {}, "features": {"x": 10.0}},
            {"clip_id": "p2", "label": "put_luggage", "keypoint_visibility": {}, "features": {"x": 12.0}},
            {"clip_id": "n1", "label": "not_put_luggage", "keypoint_visibility": {}, "features": {"x": 1.0}},
            {"clip_id": "n2", "label": "not_put_luggage", "keypoint_visibility": {}, "features": {"x": 2.0}},
        ]
        matrix, labels, _clip_ids, _names = build_feature_matrix(records)

        clf = CentroidClassifier().fit(matrix, labels)
        probs = clf.predict_proba(matrix)
        preds = clf.predict(matrix)

        self.assertGreater(probs[0], 0.5)
        self.assertLess(probs[2], 0.5)
        self.assertEqual(["put_luggage", "put_luggage", "not_put_luggage", "not_put_luggage"], preds)

    def test_load_and_write_prediction_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            features_path = Path(tmp) / "features.jsonl"
            features_path.write_text(
                '{"clip_id":"a","label":"put_luggage","keypoint_visibility":{},"features":{"x":1}}\n',
                encoding="utf-8",
            )
            rows = load_feature_records(features_path)
            output = Path(tmp) / "predictions.csv"
            write_predictions_csv(output, [
                {"clip_id": "a", "label": "put_luggage", "prediction": "put_luggage", "put_probability": 0.9}
            ])

            text = output.read_text(encoding="utf-8-sig")

        self.assertEqual("a", rows[0]["clip_id"])
        self.assertIn("put_probability", text)


if __name__ == "__main__":
    unittest.main()
