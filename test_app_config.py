import tempfile
import unittest
from pathlib import Path

from app_config import (
    SourceConfig,
    load_source_config,
    save_source_config,
)


class SourceConfigTest(unittest.TestCase):
    def test_load_missing_source_config_defaults_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_source_config(Path(tmp) / "source_config.json")

        self.assertEqual("file", config.source_type)
        self.assertEqual("", config.path_or_url)

    def test_save_and_load_rtsp_source_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_config.json"
            expected = SourceConfig(
                source_type="rtsp",
                path_or_url="rtsp://127.0.0.1/live",
                low_latency=True,
                infer_stride=3,
            )

            save_source_config(expected, path)
            actual = load_source_config(path)

        self.assertEqual(expected, actual)

    def test_invalid_source_type_falls_back_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_config.json"
            path.write_text('{"source_type":"bad","path_or_url":"x"}', encoding="utf-8")

            config = load_source_config(path)

        self.assertEqual("file", config.source_type)
        self.assertEqual("", config.path_or_url)

    def test_low_latency_flags_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_config.json"
            expected = SourceConfig(
                source_type="rtsp",
                path_or_url="rtsp://127.0.0.1/live",
                low_latency=False,
                infer_stride=5,
            )

            save_source_config(expected, path)
            actual = load_source_config(path)

        self.assertFalse(actual.low_latency)
        self.assertEqual(5, actual.infer_stride)


if __name__ == "__main__":
    unittest.main()
