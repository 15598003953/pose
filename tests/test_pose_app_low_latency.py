import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from pose_project.apps.pose_app import PoseVideoThread
from pose_project.apps.pose_app import reset_thread_action_recognizers
import pose_project.apps.pose_app as pose_app_module


class PoseVideoThreadLowLatencyTest(unittest.TestCase):
    def test_roi_reset_during_inference_discards_events_and_recognizers(self):
        thread = PoseVideoThread("rtsp://camera/main", None, None)
        generation = thread._recognizer_generation
        thread._write_action_event = Mock()

        def infer(*args):
            reset_thread_action_recognizers(thread)
            return [], {"FULL": []}, 0

        thread._infer_frame = infer
        thread._async_inference_worker(None, 0, 30, 100, 100, generation)

        thread._write_action_event.assert_not_called()
        self.assertEqual({}, thread.action_recognizers)

    def test_async_worker_reports_exceptions_and_releases_busy_state(self):
        thread = PoseVideoThread("rtsp://camera/main", None, None)
        thread._infer_running = True
        thread._infer_frame = Mock(side_effect=RuntimeError("test inference failure"))
        errors = []
        thread.error.connect(errors.append)
        thread._async_inference_worker(None, 0, 30, 100, 100, 0)
        self.assertEqual(["test inference failure"], errors)
        self.assertFalse(thread._infer_running)

    def test_app_uses_the_canonical_clip_model_path(self):
        self.assertEqual(
            "put_luggage_centroid_model.json",
            os.path.basename(pose_app_module.PUT_LUGGAGE_MODEL_PATH),
        )

    def test_app_has_no_posec3d_runtime_path(self):
        self.assertFalse(hasattr(pose_app_module, "PoseC3DRecognizer"))

    def test_thread_keeps_low_latency_and_stride_settings(self):
        thread = PoseVideoThread(
            "rtsp://camera/main",
            pose_estimator=None,
            person_detector=None,
            low_latency=True,
            infer_stride=4,
            recognizer_options={
                "display_fps": 20,
                "infer_interval_ms": 500,
            },
        )

        self.assertTrue(thread.low_latency)
        self.assertEqual(4, thread.infer_stride)
        self.assertEqual(20, thread.display_fps)
        self.assertEqual(500, thread.infer_interval_ms)

    def test_infer_frame_ignores_none_pose_results(self):
        thread = PoseVideoThread(
            "D:/video/test.mp4",
            pose_estimator=None,
            person_detector=None,
        )
        thread.rois = []
        persons = [object(), object()]
        thread._detect_persons = lambda frame: persons
        thread._infer_pose = lambda frame, person: None if person is persons[0] else [
            None, None, None, None, None, 0.9
        ]

        poses, poses_by_roi, total_persons = thread._infer_frame(
            frame=None,
            frame_h=100,
            frame_w=100,
        )

        self.assertEqual(2, total_persons)
        self.assertEqual(1, len(poses))
        self.assertIsNotNone(poses[0])
        self.assertIn("FULL", poses_by_roi)

    def test_reset_thread_action_recognizers_clears_cached_state(self):
        thread = PoseVideoThread(
            "D:/video/test.mp4",
            pose_estimator=None,
            person_detector=None,
        )
        thread.action_recognizers = {"FULL": object(), "1": object()}
        thread.action_recognizer = object()
        thread._last_overlay_poses = [object()]
        thread._last_action_result = pose_app_module.ActionResult(
            "put_luggage", 0.9, "old result")
        thread._last_total_persons = 2
        thread._last_infer_ms = 123.0

        reset_thread_action_recognizers(thread)

        self.assertEqual({}, thread.action_recognizers)
        self.assertIsNone(thread.action_recognizer)
        self.assertEqual([], thread._last_overlay_poses)
        self.assertEqual("idle", thread._last_action_result.state)
        self.assertEqual(0, thread._last_total_persons)
        self.assertEqual(0.0, thread._last_infer_ms)

    def test_reset_thread_action_recognizers_invalidates_inflight_results(self):
        thread = PoseVideoThread(
            "rtsp://camera/main",
            pose_estimator=None,
            person_detector=None,
        )
        generation = thread._recognizer_generation

        reset_thread_action_recognizers(thread)

        self.assertGreater(thread._recognizer_generation, generation)
        self.assertFalse(thread._is_recognizer_generation_current(generation))

    def test_invalid_clip_model_falls_back_to_rule_recognizer(self):
        thread = PoseVideoThread(
            "D:/video/test.mp4",
            pose_estimator=None,
            person_detector=None,
            recognizer_kind="clip_model",
        )

        with tempfile.TemporaryDirectory() as tmp:
            bad_model = Path(tmp) / "bad_model.json"
            bad_model.write_text("{", encoding="utf-8")
            old_path = pose_app_module.PUT_LUGGAGE_MODEL_PATH
            old_model_cls = pose_app_module.PutLuggageModelRecognizer
            try:
                pose_app_module.PUT_LUGGAGE_MODEL_PATH = str(bad_model)

                def _boom(*_args, **_kwargs):
                    raise ValueError("invalid model")

                pose_app_module.PutLuggageModelRecognizer = _boom
                recognizer = thread._build_recognizer(12.5)
            finally:
                pose_app_module.PUT_LUGGAGE_MODEL_PATH = old_path
                pose_app_module.PutLuggageModelRecognizer = old_model_cls

        self.assertIsInstance(recognizer, pose_app_module.PutLuggageRecognizer)


if __name__ == "__main__":
    unittest.main()
