import unittest

import numpy as np

from pose_project.reid.matcher import TrackMatcher


class _Detector:
    def detect_track(self, frame, imgsz=960):
        return (
            np.array([[10, 10, 50, 90]], dtype=np.float32),
            np.array([0.99], dtype=np.float32),
            [7],
        )


class _Extractor:
    def __init__(self, vectors):
        self.vectors = iter(vectors)
        self.feat_dim = 2
        self.calls = 0

    def extract(self, crops, tta=True):
        self.calls += 1
        return np.asarray([next(self.vectors) for _ in crops], dtype=np.float32)


class _ScriptDetector:
    def __init__(self, script):
        self.script = script
        self.index = 0

    def detect_track(self, frame, imgsz=960):
        boxes, tids = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return (
            np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
            np.ones(len(boxes), dtype=np.float32),
            list(tids),
        )


class _ConstantExtractor:
    feat_dim = 2

    def __init__(self):
        self.calls = 0

    def extract(self, crops, tta=True):
        self.calls += 1
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32),
                       (len(crops), 1))


class TrackMatcherTest(unittest.TestCase):
    def test_rejected_match_keeps_ranked_candidates_without_an_accepted_identity(self):
        matcher = TrackMatcher(
            [{"label": "third", "feature": np.array([.2, .98])},
             {"label": "first", "feature": np.array([.58, .81])},
             {"label": "second", "feature": np.array([.5, .86])},
             {"label": "fourth", "feature": np.array([.1, .99])}],
            _Detector(), _ConstantExtractor(), threshold=.65)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            matcher.annotate(frame)
        candidate = matcher.last_candidates[0]
        self.assertEqual("below_threshold", candidate["status"])
        self.assertIsNone(candidate["accepted_label"])
        self.assertEqual(
            ["first", "second", "third"],
            [row["label"] for row in candidate["ranked_candidates"]])
        self.assertEqual([.58, .5, .2],
                         [row["score"] for row in candidate["ranked_candidates"]])
        self.assertEqual("track_history", candidate["score_source"])
        self.assertEqual(2, candidate["frame_index"])
        self.assertEqual(0, candidate["feature_frame_index"])
        self.assertEqual(1, matcher.extractor.calls)
        self.assertEqual([], matcher.last_detections)
        self.assertEqual({}, matcher.passenger_locks)

    def test_locked_result_keeps_confirmation_ranking_and_marks_score_source(self):
        matcher = TrackMatcher(
            [{"label": "first", "feature": np.array([.9, .44])},
             {"label": "second", "feature": np.array([.7, .71])}],
            _Detector(), _ConstantExtractor(), threshold=.65)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            matcher.annotate(frame)
        confirmed = matcher.last_candidates[0]
        self.assertEqual("first", confirmed["accepted_label"])
        matcher.annotate(frame)
        locked = matcher.last_candidates[0]
        self.assertEqual("locked_confirmation", locked["score_source"])
        self.assertEqual(confirmed["ranked_candidates"], locked["ranked_candidates"])
        self.assertEqual(confirmed["second_score"], locked["second_score"])
        self.assertEqual(confirmed["match_margin"], locked["match_margin"])

    def test_track_keeps_first_confirmed_identity(self):
        frames = [
            [1.00, 0.00],
            [0.98, 0.20],
            [0.96, 0.28],
            [0.94, 0.34],
            [0.90, 0.44],
            [0.86, 0.51],
            [0.82, 0.57],
            [0.78, 0.63],
        ]
        matcher = TrackMatcher(
            [
                {"label": "gate-person", "feature": np.array([1.00, 0.00])},
                {"label": "other-person", "feature": np.array([0.75, 0.66])},
            ],
            _Detector(),
            _Extractor(frames),
            threshold=0.65,
        )
        matcher.FEAT_REFRESH = 0

        labels = []
        for _ in frames:
            matcher.annotate(np.zeros((120, 80, 3), dtype=np.uint8))
            labels.append(matcher.last_candidates[0]["label"])

        self.assertEqual(["gate-person"] * len(frames), labels)
        self.assertEqual("gate-person", matcher.last_detections[0]["label"])
        self.assertEqual(3, matcher.extractor.calls)

    def test_locked_track_keeps_position_without_reextracting_or_rematching(self):
        box_a = [[10, 10, 50, 90]]
        box_b = [[20, 12, 60, 92]]
        matcher = TrackMatcher(
            [{"label": "person", "feature": np.array([1.0, 0.0])}],
            _ScriptDetector([(box_a, [7])] * 3 + [(box_b, [7])]),
            _ConstantExtractor(), threshold=0.65)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)

        for _ in range(3):
            matcher.annotate(frame)
        calls = matcher.extractor.calls
        self.assertTrue(matcher.last_candidates[0]["identity_locked"])

        matcher.annotate(frame)
        self.assertEqual(calls, matcher.extractor.calls)
        self.assertEqual([20, 12, 60, 92],
                         matcher.last_detections[0]["bbox_xyxy"])
        self.assertTrue(matcher.last_candidates[0]["identity_frozen"])
        self.assertEqual("matched", matcher.last_candidates[0]["status"])

    def test_close_second_candidate_does_not_lock_identity(self):
        matcher = TrackMatcher(
            [{"label": "person", "feature": np.array([1.0, 0.0])},
             {"label": "similar", "feature": np.array([.99, .1])}],
            _Detector(), _ConstantExtractor(), threshold=0.65)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)

        for _ in range(3):
            matcher.annotate(frame)
        self.assertEqual({}, matcher.passenger_locks)
        self.assertEqual("ambiguous_match", matcher.last_candidates[0]["status"])
        self.assertLess(matcher.last_candidates[0]["match_margin"],
                        matcher.IDENTITY_MARGIN)

    def test_track_feature_cache_refreshes_every_ten_frames(self):
        extractor = _Extractor([[1.0, 0.0], [1.0, 0.0]])
        matcher = TrackMatcher(
            [{"label": "person", "feature": np.array([1.0, 0.0])}],
            _Detector(), extractor, threshold=0.65)
        matcher.MIN_FRAMES = 100
        matcher.TRACK_HIST = 200
        frame = np.zeros((120, 80, 3), dtype=np.uint8)

        for _ in range(10):
            matcher.annotate(frame)
        self.assertEqual(1, extractor.calls)
        matcher.annotate(frame)
        self.assertEqual(2, extractor.calls)

    def test_front_mode_keeps_leader_and_waits_before_replacing(self):
        initial = ([[10, 10, 50, 100], [100, 120, 150, 200]], [1, 2])
        rear_only = ([[10, 10, 50, 100]], [1])
        detector = _ScriptDetector([initial] * 4 + [rear_only] * 8)
        matcher = TrackMatcher(
            [{"label": "person", "feature": np.array([1.0, 0.0])}],
            detector, _ConstantExtractor(), threshold=0.65, front_mode="bottom")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        for _ in range(4):
            matcher.annotate(frame)
        self.assertEqual(2, matcher._leader_tid)
        self.assertEqual([100, 120, 150, 200],
                         matcher.last_detections[0]["bbox_xyxy"])

        for _ in range(7):
            matcher.annotate(frame)
            self.assertEqual([], matcher.last_detections)
        matcher.annotate(frame)
        self.assertEqual(1, matcher._leader_tid)
        self.assertEqual([], matcher.last_detections)
        self.assertEqual("passenger_locked", matcher.last_candidates[0]["status"])
        for _ in range(matcher.MIN_FRAMES - 1):
            matcher.annotate(frame)
        self.assertEqual([], matcher.last_detections)
        self.assertEqual(1, matcher.last_candidates[0]["tid"])

    def make_matcher(self, script, front_mode=None):
        return TrackMatcher(
            [{"label": "person", "feature": np.array([1.0, 0.0])}],
            _ScriptDetector(script), _ConstantExtractor(), .65,
            front_mode=front_mode)

    def test_confirmed_passenger_cannot_move_to_a_new_track(self):
        box = [[10, 10, 50, 90]]
        matcher = self.make_matcher([(box, [7])] * 3 + [(box, [8])] * 3)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            matcher.annotate(frame)
        self.assertEqual(7, matcher.last_detections[0]["tid"])
        for _ in range(3):
            matcher.annotate(frame)
            self.assertEqual([], matcher.last_detections)
        self.assertEqual("passenger_locked", matcher.last_candidates[0]["status"])

    def test_expired_track_id_does_not_reopen_completed_passenger(self):
        box = [[10, 10, 50, 90]]
        matcher = self.make_matcher(
            [(box, [7])] * 3 + [([], [])] * 62 + [(box, [7])] * 3)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(65):
            matcher.annotate(frame)
        self.assertEqual({}, matcher.tracks)
        for _ in range(3):
            matcher.annotate(frame)
        self.assertEqual([], matcher.last_detections)

    def test_multiple_candidates_do_not_lock_first_detection(self):
        boxes = [[10, 10, 40, 90], [45, 10, 75, 90]]
        matcher = self.make_matcher([(boxes, [7, 8])])
        for _ in range(3):
            matcher.annotate(np.zeros((120, 80, 3), dtype=np.uint8))
        self.assertEqual([], matcher.last_detections)
        self.assertTrue(all(c["status"] == "duplicate" for c in matcher.last_candidates))

    def test_missing_track_id_cannot_complete_passenger(self):
        matcher = self.make_matcher([([[10, 10, 50, 90]], [None])])
        matcher.annotate(np.zeros((120, 80, 3), dtype=np.uint8))
        self.assertEqual([], matcher.last_detections)
        self.assertEqual("waiting_track", matcher.last_candidates[0]["status"])

    def test_original_track_returns_without_reassigning_passenger(self):
        box = [[10, 10, 50, 90]]
        matcher = self.make_matcher(
            [(box, [7])] * 3 + [(box, [8])] * 3 + [(box, [7])])
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(6):
            matcher.annotate(frame)
        self.assertEqual([], matcher.last_detections)
        matcher.annotate(frame)
        self.assertEqual(7, matcher.last_detections[0]["tid"])
        self.assertTrue(matcher.last_candidates[0]["identity_locked"])

    def test_uncompleted_passenger_can_match_new_track_after_gallery_update(self):
        box = [[10, 10, 50, 90]]
        matcher = self.make_matcher([(box, [7])] * 3 + [(box, [8])])
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            matcher.annotate(frame)
        matcher.update_gallery([
            {"label": "person", "feature": np.array([1., 0.])},
            {"label": "next", "feature": np.array([0., 1.])}])
        matcher.extractor = _Extractor([[0., 1.]])
        for _ in range(3):
            matcher.annotate(frame)
        self.assertEqual("next", matcher.last_detections[0]["label"])
        self.assertEqual({"person", "next"}, set(matcher.passenger_locks))

    def test_shared_lock_blocks_same_numeric_track_in_another_region(self):
        locks = {}
        persons = [{"label": "person", "feature": np.array([1., 0.])}]
        matchers = [TrackMatcher(persons, _Detector(), _ConstantExtractor(), .65,
                                 passenger_locks=locks) for _ in range(2)]
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            matchers[0].annotate(frame)
        for _ in range(3):
            matchers[1].annotate(frame)
        self.assertEqual([], matchers[1].last_detections)
        self.assertEqual("passenger_locked", matchers[1].last_candidates[0]["status"])
        matchers[1].update_gallery(persons)
        for _ in range(3):
            matchers[1].annotate(frame)
        self.assertEqual([], matchers[1].last_detections)
        fresh = TrackMatcher(persons, _Detector(), _ConstantExtractor(), .65)
        for _ in range(3):
            fresh.annotate(frame)
        self.assertEqual("person", fresh.last_detections[0]["label"])

    def test_cross_region_conflict_does_not_complete_either_task(self):
        locks = {}
        persons = [{"label": "person", "feature": np.array([1., 0.])}]
        matchers = [TrackMatcher(persons, _Detector(), _ConstantExtractor(), .65,
                                 passenger_locks=locks) for _ in range(2)]
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            for matcher in matchers:
                matcher.annotate(frame, defer_confirmation=True)
            TrackMatcher.confirm_frame(matchers, frame.copy())
        self.assertEqual({}, locks)
        for matcher in matchers:
            self.assertEqual([], matcher.last_detections)
            self.assertEqual("duplicate", matcher.last_candidates[0]["status"])
        matchers[1].detector = _ScriptDetector([([], [])])
        for matcher in matchers:
            matcher.annotate(frame, defer_confirmation=True)
        TrackMatcher.confirm_frame(matchers, frame.copy())
        self.assertEqual("person", matchers[0].last_detections[0]["label"])

    def test_new_claim_blocks_immature_candidate_in_same_frame(self):
        locks = {}
        persons = [{"label": "person", "feature": np.array([1., 0.])}]
        ready = TrackMatcher(persons, _Detector(), _ConstantExtractor(), .65,
                             passenger_locks=locks)
        pending = TrackMatcher(persons, _Detector(), _ConstantExtractor(), .65,
                               passenger_locks=locks)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(2):
            ready.annotate(frame)
        pending.annotate(frame, defer_confirmation=True)
        ready.annotate(frame, defer_confirmation=True)
        TrackMatcher.confirm_frame([pending, ready], frame.copy())
        self.assertEqual("passenger_locked", pending.last_candidates[0]["status"])
        self.assertEqual([], pending.last_detections)
        self.assertEqual("person", ready.last_detections[0]["label"])

    def test_locked_top_candidate_does_not_force_second_best_identity(self):
        box = [[10, 10, 50, 90]]
        matcher = self.make_matcher([(box, [7])] * 3 + [(box, [8])])
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            matcher.annotate(frame)
        matcher.update_gallery([
            {"label": "person", "feature": np.array([1., 0.])},
            {"label": "other", "feature": np.array([.8, .6])}])
        for _ in range(3):
            matcher.annotate(frame)
        self.assertEqual([], matcher.last_detections)
        self.assertNotIn("other", matcher.passenger_locks)

    def test_all_front_directions_choose_one_person(self):
        boxes = [[20, 10, 60, 80], [100, 100, 180, 220]]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for mode, expected in (("bottom", 2), ("top", 1), ("left", 1),
                               ("right", 2), ("area", 2)):
            with self.subTest(mode=mode):
                matcher = self.make_matcher([(boxes, [1, 2])], mode)
                matcher.annotate(frame)
                self.assertEqual(expected, matcher._leader_tid)
                self.assertEqual([expected], [c["tid"] for c in matcher.last_candidates])
                self.assertEqual(2, matcher.last_queue_count)
                self.assertEqual(1, matcher.last_skipped)

    def test_leader_margin_keeps_small_changes_and_accepts_boundary(self):
        for mode, current, boundary in (
                ("bottom", 100, 120), ("top", -100, -80),
                ("left", -100, -60), ("right", 100, 140),
                ("area", 10000, 12500)):
            with self.subTest(mode=mode):
                matcher = self.make_matcher([([], [])], mode)
                self.assertFalse(matcher._leader_better(boundary - 1, current, (250, 500)))
                self.assertTrue(matcher._leader_better(boundary, current, (250, 500)))

    def test_leader_follows_id_when_detection_order_changes(self):
        matcher = self.make_matcher([
            ([[10, 10, 50, 100], [100, 100, 150, 180]], [1, 2]),
            ([[100, 90, 150, 170], [10, 90, 50, 180]], [2, 1]),
            ([[100, 90, 150, 170], [10, 90, 50, 200]], [2, 1]),
        ], "bottom")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for expected in (2, 2, 1):
            matcher.annotate(frame)
            self.assertEqual(expected, matcher._leader_tid)

    def test_leader_return_resets_wait_without_extracting_rear_person(self):
        initial = ([[10, 10, 50, 100], [100, 120, 150, 200]], [1, 2])
        rear = ([[10, 10, 50, 100]], [1])
        matcher = self.make_matcher([initial, rear, ([], []), initial, rear], "bottom")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for expected_miss in (0, 1, 2, 0, 1):
            matcher.annotate(frame)
            self.assertEqual(2, matcher._leader_tid)
            self.assertEqual(expected_miss, matcher._leader_miss)
            if expected_miss:
                self.assertEqual([], matcher.last_detections)
                self.assertEqual([], matcher.last_candidates)
        self.assertEqual(1, matcher.extractor.calls)

    def test_features_without_track_ids_are_never_cached(self):
        matcher = self.make_matcher([([[10, 10, 50, 90]], [None])])
        for _ in range(3):
            matcher.annotate(np.zeros((120, 80, 3), dtype=np.uint8))
        self.assertEqual(3, matcher.extractor.calls)
        self.assertEqual({}, matcher._feat_cache)

    def test_track_and_feature_cache_expire_during_empty_frames(self):
        matcher = self.make_matcher([([[10, 10, 50, 90]], [7]), ([], [])])
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        matcher.annotate(frame)
        self.assertIn(7, matcher._feat_cache)
        for _ in range(matcher.TRACK_TTL + 1):
            matcher.annotate(frame)
        self.assertEqual({}, matcher.tracks)
        self.assertEqual({}, matcher._feat_cache)

    def test_visible_nonleader_keeps_track_state(self):
        initial = ([[10, 10, 50, 100], [100, 100, 150, 200]], [1, 2])
        overtaken = ([[10, 100, 50, 230], [100, 100, 150, 200]], [1, 2])
        matcher = self.make_matcher([initial] * 3 + [overtaken], "bottom")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(matcher.TRACK_TTL + 5):
            matcher.annotate(frame)
        self.assertEqual(1, matcher._leader_tid)
        self.assertIn(2, matcher.tracks)
        self.assertIn(2, matcher._feat_cache)

    def test_gallery_update_keeps_leader_cache_and_identity_by_label(self):
        matcher = self.make_matcher([([[10, 10, 50, 90]], [7])], "bottom")
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        for _ in range(3):
            matcher.annotate(frame)
        cached = matcher._feat_cache[7]["feat"]
        matcher.update_gallery([
            {"label": "new-person", "feature": np.array([1.0, 0.0])},
            {"label": "person", "feature": np.array([.9, .1])},
        ], {"person": "P001"})
        self.assertEqual(7, matcher._leader_tid)
        self.assertIs(cached, matcher._feat_cache[7]["feat"])
        self.assertEqual(0, len(matcher.tracks[7]["hist"]))
        for _ in range(3):
            matcher.annotate(frame)
        self.assertEqual("person", matcher.last_detections[0]["label"])
        self.assertTrue(matcher.last_candidates[0]["identity_locked"])
        self.assertEqual(1, matcher.extractor.calls)


if __name__ == "__main__":
    unittest.main()
