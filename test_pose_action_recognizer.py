import unittest

import numpy as np

from pose_action_recognizer import PutLuggageRecognizer


def _pose(left_wrist_y, right_wrist_y=None, visibility=0.9, presence=0.9):
    if right_wrist_y is None:
        right_wrist_y = left_wrist_y
    landmarks = np.zeros((39, 5), dtype=np.float32)

    # shoulders
    landmarks[11] = [90, 100, 0, visibility, presence]
    landmarks[12] = [130, 100, 0, visibility, presence]
    # elbows
    landmarks[13] = [90, 145, 0, visibility, presence]
    landmarks[14] = [130, 145, 0, visibility, presence]
    # wrists
    landmarks[15] = [90, left_wrist_y, 0, visibility, presence]
    landmarks[16] = [130, right_wrist_y, 0, visibility, presence]
    # hips
    landmarks[23] = [95, 220, 0, visibility, presence]
    landmarks[24] = [125, 220, 0, visibility, presence]
    # knees
    landmarks[25] = [95, 330, 0, visibility, presence]
    landmarks[26] = [125, 330, 0, visibility, presence]

    return [np.array([[80, 80], [150, 360]], dtype=np.float32),
            landmarks,
            np.zeros((39, 3), dtype=np.float32),
            None,
            None,
            0.95]


class PutLuggageRecognizerTest(unittest.TestCase):
    def test_detects_put_luggage_after_down_hold_and_release(self):
        recognizer = PutLuggageRecognizer(fps=30.0)
        events = []

        # Standing / preparation.
        for frame_idx, y in enumerate([150, 155, 160, 165]):
            result = recognizer.update([_pose(y)], frame_idx=frame_idx)
            events.extend(result.events)

        # Hand moves down below the hips.
        for offset, y in enumerate([185, 210, 238, 265, 288], start=4):
            result = recognizer.update([_pose(y)], frame_idx=offset)
            events.extend(result.events)

        # Low hold near luggage-height area.
        for offset, y in enumerate([292, 294, 293, 295, 294, 293], start=9):
            result = recognizer.update([_pose(y)], frame_idx=offset)
            events.extend(result.events)

        # Release / hand returns upward.
        for offset, y in enumerate([270, 240, 205, 175], start=15):
            result = recognizer.update([_pose(y)], frame_idx=offset)
            events.extend(result.events)

        self.assertTrue(events)
        self.assertEqual("put_luggage", events[-1]["event"])
        self.assertGreaterEqual(events[-1]["confidence"], 0.5)

    def test_does_not_detect_without_low_hold(self):
        recognizer = PutLuggageRecognizer(fps=30.0)
        events = []

        for frame_idx, y in enumerate([150, 170, 195, 225, 260, 225, 190, 160]):
            result = recognizer.update([_pose(y)], frame_idx=frame_idx)
            events.extend(result.events)

        self.assertEqual([], events)


if __name__ == "__main__":
    unittest.main()
