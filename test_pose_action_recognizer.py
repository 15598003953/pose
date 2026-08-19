import unittest

import numpy as np

from pose_action_recognizer import PutLuggageRecognizer
from pose_app import should_show_put_luggage_business_label


def _pose(
        left_wrist_y,
        right_wrist_y=None,
        nose_y=70,
        shoulder_y=100,
        hip_y=220,
        knee_y=330,
        visibility=0.9,
        presence=0.9):
    if right_wrist_y is None:
        right_wrist_y = left_wrist_y
    landmarks = np.zeros((39, 5), dtype=np.float32)

    # shoulders
    landmarks[0] = [110, nose_y, 0, visibility, presence]
    landmarks[11] = [90, shoulder_y, 0, visibility, presence]
    landmarks[12] = [130, shoulder_y, 0, visibility, presence]
    # elbows
    landmarks[13] = [90, (shoulder_y + left_wrist_y) / 2, 0, visibility, presence]
    landmarks[14] = [130, (shoulder_y + right_wrist_y) / 2, 0, visibility, presence]
    # wrists
    landmarks[15] = [90, left_wrist_y, 0, visibility, presence]
    landmarks[16] = [130, right_wrist_y, 0, visibility, presence]
    # hips
    landmarks[23] = [95, hip_y, 0, visibility, presence]
    landmarks[24] = [125, hip_y, 0, visibility, presence]
    # knees
    landmarks[25] = [95, knee_y, 0, visibility, presence]
    landmarks[26] = [125, knee_y, 0, visibility, presence]

    return [np.array([[80, 80], [150, 360]], dtype=np.float32),
            landmarks,
            np.zeros((39, 3), dtype=np.float32),
            None,
            None,
            0.95]


class PutLuggageRecognizerTest(unittest.TestCase):
    def test_posec3d_related_k400_labels_show_business_label(self):
        for label in (
                "garbage collecting",
                "pushing cart",
                "setting table",
                "unboxing",
                "canoeing or kayaking"):
            self.assertTrue(
                should_show_put_luggage_business_label(label, 0.3),
                label)

    def test_posec3d_unrelated_or_low_score_labels_do_not_show_business_label(self):
        self.assertFalse(
            should_show_put_luggage_business_label("playing guitar", 0.9))
        self.assertFalse(
            should_show_put_luggage_business_label("unboxing", 0.29))

    def test_detects_put_luggage_from_table_operation_posture(self):
        recognizer = PutLuggageRecognizer(fps=12.5)
        events = []

        # Upright frames before operating at the security table.
        for frame_idx in range(4):
            result = recognizer.update([_pose(155, 158)], frame_idx=frame_idx)
            events.extend(result.events)

        # Low head, compressed torso and both hands in the front-lower
        # operating area, matching the real checkpoint table scene.
        for offset in range(4, 18):
            result = recognizer.update([
                _pose(250, 258, nose_y=128, shoulder_y=132, hip_y=215, knee_y=322)
            ], frame_idx=offset)
            events.extend(result.events)

        # Body and hands recover from the operating posture.
        for offset in range(18, 24):
            result = recognizer.update([_pose(165, 168)], frame_idx=offset)
            events.extend(result.events)

        self.assertTrue(events)
        self.assertEqual("put_luggage", events[-1]["event"])
        self.assertEqual("front_table_operation_posture", events[-1]["reason"])
        self.assertGreaterEqual(events[-1]["confidence"], 0.5)
        self.assertIn("evidence", events[-1])

    def test_does_not_detect_wrist_drop_without_table_operation_posture(self):
        recognizer = PutLuggageRecognizer(fps=12.5)
        events = []

        # A pure wrist drop while the head and torso stay upright should not
        # be enough to call it a put-luggage action.
        for frame_idx, y in enumerate([150, 180, 215, 250, 270, 260, 220, 180, 155]):
            result = recognizer.update([
                _pose(y, y + 2, nose_y=70, shoulder_y=100, hip_y=220, knee_y=330)
            ], frame_idx=frame_idx)
            events.extend(result.events)

        self.assertEqual([], events)


if __name__ == "__main__":
    unittest.main()
