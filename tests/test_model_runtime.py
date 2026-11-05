"""Regression checks for detector geometry and pose input ownership."""

import unittest

import numpy as np

from pose_project.vision.mp_persondet import MPPersonDet
from pose_project.vision.mp_pose import MPPose


class ModelRuntimeTest(unittest.TestCase):
    def test_nms_preserves_separate_boxes_far_from_origin(self):
        detector = MPPersonDet.__new__(MPPersonDet)
        detector.input_size = np.array([224, 224])
        detector.anchors = np.zeros((2, 2))
        detector.score_threshold = 0.5
        detector.nms_threshold = 0.3
        detector.topK = 5000
        boxes = np.zeros((1, 2, 12), dtype=np.float32)
        boxes[0, :, :4] = [[110, 110, 20, 20], [140, 110, 20, 20]]
        scores = np.array([[[5.0], [4.0]]], dtype=np.float32)

        result = detector._postprocess([boxes, scores], np.array([224, 224]), [0, 0])

        self.assertEqual((2, 13), result.shape)
        np.testing.assert_allclose(result[:, :4], [[100, 100, 120, 120],
                                                   [130, 100, 150, 120]], atol=1e-5)

    def test_pose_preprocessing_does_not_modify_detection(self):
        estimator = MPPose.__new__(MPPose)
        estimator.input_size = np.array([256, 256])
        estimator.PERSON_BOX_PRE_ENLARGE_FACTOR = 1
        person = np.array([20, 20, 80, 100, 70, 90, 70, 40, 70, 60, 70, 30, .9],
                          dtype=np.float32)
        original = person.copy()

        self.assertIsNotNone(estimator._preprocess(np.zeros((180, 180, 3), np.uint8),
                                                  person))

        np.testing.assert_array_equal(person, original)

    def test_onnx_failure_reaches_enabled_opencv_fallback(self):
        from unittest.mock import Mock

        estimator = MPPose.__new__(MPPose)
        estimator.ort_session = Mock()
        estimator.ort_session.run.side_effect = RuntimeError("ORT failed")
        estimator.ort_output_names = []
        estimator.ort_input_name = "input"
        estimator._ort_failed_warned = False
        estimator.allow_opencv_fallback = True
        estimator.model = Mock()
        expected = [np.zeros((1, size)) for size in (195, 1, 65536, 159744, 117)]
        estimator.model.forward.return_value = expected

        self.assertIs(expected, estimator._forward_pose(np.zeros((1, 256, 256, 3))))
        estimator.model.forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
