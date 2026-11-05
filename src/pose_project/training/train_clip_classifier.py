import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from pose_project.common.project_paths import DATA_DIR


POSITIVE_LABEL = "put_luggage"
NEGATIVE_LABEL = "not_put_luggage"


class CentroidClassifier:
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None
        self.positive_centroid = None
        self.negative_centroid = None

    def fit(self, matrix, labels):
        matrix = np.asarray(matrix, dtype=np.float32)
        labels = list(labels)
        if matrix.ndim != 2 or 0 in matrix.shape:
            raise ValueError("feature matrix is empty")
        if len(labels) != matrix.shape[0]:
            raise ValueError("sample and label counts must match")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must be finite numbers")
        if any(label not in (POSITIVE_LABEL, NEGATIVE_LABEL) for label in labels):
            raise ValueError("unknown training label")
        if POSITIVE_LABEL not in labels or NEGATIVE_LABEL not in labels:
            raise ValueError("both put_luggage and not_put_luggage samples are required")

        self.feature_mean = np.mean(matrix, axis=0)
        self.feature_std = np.std(matrix, axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        normalized = self._normalize(matrix)

        labels_arr = np.asarray(labels)
        self.positive_centroid = np.mean(normalized[labels_arr == POSITIVE_LABEL], axis=0)
        self.negative_centroid = np.mean(normalized[labels_arr == NEGATIVE_LABEL], axis=0)
        return self

    def predict_proba(self, matrix):
        normalized = self._normalize(np.asarray(matrix, dtype=np.float32))
        pos_dist = np.linalg.norm(normalized - self.positive_centroid, axis=1)
        neg_dist = np.linalg.norm(normalized - self.negative_centroid, axis=1)
        logits = neg_dist - pos_dist
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))

    def predict(self, matrix, threshold=0.5):
        probs = self.predict_proba(matrix)
        return [POSITIVE_LABEL if prob >= threshold else NEGATIVE_LABEL for prob in probs]

    def _normalize(self, matrix):
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("classifier is not fitted")
        return (matrix - self.feature_mean) / self.feature_std

    def to_model_dict(self, feature_names):
        return {
            "model_type": "nearest_centroid_numpy",
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "feature_names": list(feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "positive_centroid": self.positive_centroid.tolist(),
            "negative_centroid": self.negative_centroid.tolist(),
        }


def load_feature_records(path: Path) -> List[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_feature_matrix(records: Sequence[dict]) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    feature_names = _collect_feature_names(records)
    rows = []
    labels = []
    clip_ids = []
    for record in records:
        label = record.get("label")
        if label not in (POSITIVE_LABEL, NEGATIVE_LABEL):
            continue
        values = []
        visibility = record.get("keypoint_visibility") or {}
        features = record.get("features") or {}
        for name in feature_names:
            group, key = name.split(":", 1)
            source = visibility if group == "vis" else features
            value = source.get(key, 0.0)
            values.append(0.0 if value is None else float(value))
        rows.append(values)
        labels.append(label)
        clip_ids.append(record.get("clip_id", ""))
    return np.asarray(rows, dtype=np.float32), labels, clip_ids, feature_names


def _collect_feature_names(records: Sequence[dict]) -> List[str]:
    visibility_names = set()
    feature_names = set()
    for record in records:
        visibility_names.update((record.get("keypoint_visibility") or {}).keys())
        feature_names.update((record.get("features") or {}).keys())
    return (
        ["vis:" + name for name in sorted(visibility_names)]
        + ["feat:" + name for name in sorted(feature_names)]
    )


def evaluate(labels: Sequence[str], predictions: Sequence[str], probabilities: Sequence[float]) -> dict:
    labels = list(labels)
    predictions = list(predictions)
    probabilities = [float(prob) for prob in probabilities]
    total = len(labels)
    correct = sum(1 for label, pred in zip(labels, predictions) if label == pred)
    tp = sum(1 for label, pred in zip(labels, predictions) if label == POSITIVE_LABEL and pred == POSITIVE_LABEL)
    tn = sum(1 for label, pred in zip(labels, predictions) if label == NEGATIVE_LABEL and pred == NEGATIVE_LABEL)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == NEGATIVE_LABEL and pred == POSITIVE_LABEL)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == POSITIVE_LABEL and pred == NEGATIVE_LABEL)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "total": total,
        "label_counts": dict(Counter(labels)),
        "accuracy": round(correct / total if total else 0.0, 4),
        "precision_put_luggage": round(precision, 4),
        "recall_put_luggage": round(recall, 4),
        "confusion_matrix": {
            "true_put_pred_put": tp,
            "true_put_pred_not": fn,
            "true_not_pred_put": fp,
            "true_not_pred_not": tn,
        },
        "mean_put_probability": round(float(np.mean(probabilities)) if probabilities else 0.0, 4),
    }


def write_predictions_csv(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip_id", "label", "prediction", "put_probability"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "clip_id": row["clip_id"],
                "label": row["label"],
                "prediction": row["prediction"],
                "put_probability": "{:.6f}".format(float(row["put_probability"])),
            })


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train_and_write(features_path: Path, output_dir: Path) -> dict:
    records = load_feature_records(features_path)
    matrix, labels, clip_ids, feature_names = build_feature_matrix(records)
    classifier = CentroidClassifier().fit(matrix, labels)
    probabilities = classifier.predict_proba(matrix)
    predictions = classifier.predict(matrix)

    prediction_rows = [
        {
            "clip_id": clip_id,
            "label": label,
            "prediction": pred,
            "put_probability": prob,
        }
        for clip_id, label, pred, prob in zip(clip_ids, labels, predictions, probabilities)
    ]
    report = evaluate(labels, predictions, probabilities)
    report["warning"] = (
        "This is a training-set diagnostic baseline on a very small, overlapping dataset; "
        "do not treat it as real deployment accuracy."
    )
    report["feature_count"] = len(feature_names)

    output_dir = Path(output_dir)
    write_predictions_csv(output_dir / "predictions.csv", prediction_rows)
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "model.json", classifier.to_model_dict(feature_names))
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a small put-luggage clip classifier baseline.")
    parser.add_argument(
        "--features_jsonl",
        default=str(DATA_DIR / "clip_features" / "features.jsonl"),
        help="Feature file created by extract_clip_pose_features.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DATA_DIR / "clip_classifier_baseline"),
        help="Output directory for model, report, and predictions.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    report = train_and_write(Path(args.features_jsonl), Path(args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
