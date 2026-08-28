import unittest

from pose_app import PoseVideoThread


class PoseVideoThreadLowLatencyTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
