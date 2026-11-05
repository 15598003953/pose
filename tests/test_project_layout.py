"""Regression coverage for paths after the source-directory migration."""

from pathlib import Path
import unittest

from pose_project.apps.pose_app import MQTT_CONFIG_PATH, ROI_CONFIG_PATH, SOURCE_CONFIG_PATH
from pose_project.binding import generate_binding_simulation, replay_binding
from pose_project.common.project_paths import (
    CONFIG_DIR, DATA_DIR, MODEL_DIR, RESULT_DIR, ROOT, POSE_MODEL_PATH,
    PUT_LUGGAGE_MODEL_PATH,
)
from pose_project.training import extract_clip_pose_features, train_clip_classifier


class ProjectLayoutTest(unittest.TestCase):
    def test_resources_resolve_to_project_directories(self):
        self.assertEqual(Path(__file__).resolve().parents[1], ROOT)
        self.assertEqual(ROOT / "models", MODEL_DIR)
        self.assertEqual(MODEL_DIR, Path(POSE_MODEL_PATH).parent)
        self.assertEqual(MODEL_DIR, Path(PUT_LUGGAGE_MODEL_PATH).parent)
        for path in (MQTT_CONFIG_PATH, ROI_CONFIG_PATH, SOURCE_CONFIG_PATH,
                     generate_binding_simulation.DEFAULT_CONFIG, replay_binding.DEFAULT_CONFIG):
            with self.subTest(path=path):
                self.assertEqual(CONFIG_DIR, Path(path).parent)

    def test_binding_defaults_keep_existing_data_and_output_locations(self):
        self.assertEqual(DATA_DIR / "binding_simulation",
                         generate_binding_simulation.DEFAULT_OUTPUT)
        self.assertEqual(DATA_DIR / "binding_simulation" / "replay.jsonl",
                         replay_binding.DEFAULT_INPUT)
        self.assertEqual(RESULT_DIR / "binding_replay", replay_binding.DEFAULT_OUTPUT)

    def test_training_defaults_keep_existing_dataset_locations(self):
        extraction = extract_clip_pose_features.build_arg_parser().parse_args([])
        training = train_clip_classifier.build_arg_parser().parse_args([])
        self.assertEqual(DATA_DIR / "clip_labels" / "labels.csv", Path(extraction.labels_csv))
        self.assertEqual(DATA_DIR / "clip_features", Path(extraction.output_dir))
        self.assertEqual(DATA_DIR / "clip_features" / "features.jsonl",
                         Path(training.features_jsonl))
        self.assertEqual(DATA_DIR / "clip_classifier_baseline", Path(training.output_dir))


if __name__ == "__main__":
    unittest.main()
