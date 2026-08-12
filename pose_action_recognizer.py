from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ActionResult:
    state: str
    confidence: float
    debug_text: str
    events: List[Dict] = field(default_factory=list)


class PutLuggageRecognizer:
    """Pose-only recognizer for checkpoint-table put-luggage posture.

    This recognizer intentionally does not detect luggage, ROI, or identity. It
    looks for a checkpoint-table operation posture: head/torso lean forward,
    hands stay in the lower-front operating area, and then the posture recovers.
    """

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(self, fps=30.0, min_visibility=0.5):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.min_visibility = min_visibility
        self.state = "idle"
        self._start_frame: Optional[int] = None
        self._operation_frames = 0
        self._best_score = 0.0
        self._best_evidence: Dict[str, float] = {}
        self._last_event_frame = -100000

        self.enter_threshold = 0.62
        self.release_threshold = 0.60
        self.min_operation_frames = max(6, int(round(self.fps * 0.8)))
        self.max_operation_frames = max(36, int(round(self.fps * 6.0)))
        self.cooldown_frames = max(12, int(round(self.fps * 2.0)))

    def update(self, poses, frame_idx, timestamp_ms=None):
        sample = self._select_pose_sample(poses)
        if sample is None:
            self._reset()
            return ActionResult("idle", 0.0, "ACTION idle no_valid_pose")

        score, evidence = self._score_operation_posture(sample)
        events = []

        if self.state == "idle":
            if score >= self.enter_threshold:
                self._start_operation(frame_idx, score, evidence)
            else:
                self.state = "ready"

        elif self.state == "ready":
            if score >= self.enter_threshold:
                self._start_operation(frame_idx, score, evidence)

        elif self.state in ("operating_posture", "stable_operation"):
            if score >= self.enter_threshold:
                self._operation_frames += 1
                self._record_best(score, evidence)
                if self._operation_frames >= self.min_operation_frames:
                    self.state = "stable_operation"
            elif self._is_confirmed_release(score, frame_idx):
                confidence = self._confidence()
                event = {
                    "event": "put_luggage",
                    "frame_idx": int(frame_idx),
                    "timestamp_ms": int(timestamp_ms) if timestamp_ms is not None else None,
                    "confidence": round(confidence, 3),
                    "state": "put_luggage",
                    "reason": "front_table_operation_posture",
                    "evidence": self._rounded_evidence(),
                }
                events.append(event)
                self._last_event_frame = frame_idx
                self.state = "put_luggage"
            elif self._expired(frame_idx):
                if (self.state == "stable_operation"
                        and frame_idx - self._last_event_frame >= self.cooldown_frames):
                    confidence = self._confidence()
                    event = {
                        "event": "put_luggage",
                        "frame_idx": int(frame_idx),
                        "timestamp_ms": int(timestamp_ms) if timestamp_ms is not None else None,
                        "confidence": round(confidence, 3),
                        "state": "put_luggage",
                        "reason": "front_table_operation_posture_timeout",
                        "evidence": self._rounded_evidence(),
                    }
                    events.append(event)
                    self._last_event_frame = frame_idx
                    self.state = "put_luggage"
                else:
                    self._reset_to_ready()

        elif self.state == "put_luggage":
            if score < self.release_threshold:
                self._reset_to_ready()

        debug = (
            "ACTION {} conf:{:.2f} score:{:.2f} hold:{} head:{:.2f} hand:{:.2f}"
            .format(
                self.state,
                self._confidence() if self.state in ("stable_operation", "put_luggage") else score,
                score,
                self._operation_frames,
                evidence.get("head_down", 0.0),
                evidence.get("hands_low", 0.0),
            )
        )
        return ActionResult(self.state, self._confidence(), debug, events)

    def _start_operation(self, frame_idx, score, evidence):
        self.state = "operating_posture"
        self._start_frame = frame_idx
        self._operation_frames = 1
        self._best_score = score
        self._best_evidence = dict(evidence)

    def _record_best(self, score, evidence):
        if score >= self._best_score:
            self._best_score = score
            self._best_evidence = dict(evidence)

    def _is_confirmed_release(self, score, frame_idx):
        if self.state != "stable_operation":
            return False
        if self._operation_frames < self.min_operation_frames:
            return False
        if score >= self.release_threshold:
            return False
        if frame_idx - self._last_event_frame < self.cooldown_frames:
            return False
        return True

    def _expired(self, frame_idx):
        return self._start_frame is not None and frame_idx - self._start_frame > self.max_operation_frames

    def _confidence(self):
        hold_score = min(1.0, self._operation_frames / max(self.min_operation_frames, 1))
        return min(1.0, 0.65 * self._best_score + 0.35 * hold_score)

    def _rounded_evidence(self):
        evidence = {k: round(float(v), 3) for k, v in self._best_evidence.items()}
        evidence["hold_frames"] = int(self._operation_frames)
        evidence["best_score"] = round(float(self._best_score), 3)
        return evidence

    def _reset(self):
        self.state = "idle"
        self._start_frame = None
        self._operation_frames = 0
        self._best_score = 0.0
        self._best_evidence = {}

    def _reset_to_ready(self):
        self.state = "ready"
        self._start_frame = None
        self._operation_frames = 0
        self._best_score = 0.0
        self._best_evidence = {}

    def _score_operation_posture(self, sample):
        shoulder_y = sample["shoulder_y"]
        hip_y = sample["hip_y"]
        knee_y = sample["knee_y"]
        nose_y = sample["nose_y"]
        left_wrist_y = sample["left_wrist_y"]
        right_wrist_y = sample["right_wrist_y"]
        left_elbow_y = sample["left_elbow_y"]
        right_elbow_y = sample["right_elbow_y"]
        body_scale = max(knee_y - shoulder_y, 1.0)
        torso_len = max(hip_y - shoulder_y, 1.0)

        head_down = self._clamp((nose_y - (shoulder_y - 0.18 * body_scale)) / (0.28 * body_scale))
        torso_compression = self._clamp((0.62 * body_scale - torso_len) / (0.28 * body_scale))

        left_hand_low = self._hand_low_score(left_wrist_y, left_elbow_y, hip_y, body_scale)
        right_hand_low = self._hand_low_score(right_wrist_y, right_elbow_y, hip_y, body_scale)
        hands_low = max(left_hand_low, right_hand_low)

        both_hands_low = min(left_hand_low, right_hand_low)
        hand_balance = self._clamp(1.0 - abs(left_wrist_y - right_wrist_y) / (0.35 * body_scale))
        two_hand_operation = both_hands_low * hand_balance

        wrist_center_y = (left_wrist_y + right_wrist_y) / 2.0
        operating_band_top = shoulder_y + 0.22 * body_scale
        operating_band_bottom = knee_y + 0.10 * body_scale
        front_operating = (
            self._clamp((wrist_center_y - operating_band_top) / (0.28 * body_scale))
            * self._clamp((operating_band_bottom - wrist_center_y) / (0.35 * body_scale))
        )

        score = (
            0.25 * head_down
            + 0.20 * torso_compression
            + 0.25 * hands_low
            + 0.20 * front_operating
            + 0.10 * two_hand_operation
        )
        evidence = {
            "head_down": head_down,
            "torso_compression": torso_compression,
            "hands_low": hands_low,
            "front_operating": front_operating,
            "two_hand_operation": two_hand_operation,
        }
        return score, evidence

    def _hand_low_score(self, wrist_y, elbow_y, hip_y, body_scale):
        below_elbow = self._clamp((wrist_y - elbow_y) / (0.20 * body_scale))
        near_hip = self._clamp((wrist_y - (hip_y - 0.18 * body_scale)) / (0.30 * body_scale))
        return 0.55 * below_elbow + 0.45 * near_hip

    def _select_pose_sample(self, poses):
        best = None
        best_score = -1.0
        for pose in poses or []:
            if len(pose) < 2:
                continue
            landmarks = np.asarray(pose[1])
            if landmarks.shape[0] < 27 or landmarks.shape[1] < 5:
                continue
            visibility_score = self._visibility_score(landmarks)
            if visibility_score < self.min_visibility:
                continue
            sample = {
                "nose_y": float(landmarks[self.NOSE, 1]),
                "shoulder_y": self._mean_y(landmarks, self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
                "hip_y": self._mean_y(landmarks, self.LEFT_HIP, self.RIGHT_HIP),
                "knee_y": self._mean_y(landmarks, self.LEFT_KNEE, self.RIGHT_KNEE),
                "left_elbow_y": float(landmarks[self.LEFT_ELBOW, 1]),
                "right_elbow_y": float(landmarks[self.RIGHT_ELBOW, 1]),
                "left_wrist_y": float(landmarks[self.LEFT_WRIST, 1]),
                "right_wrist_y": float(landmarks[self.RIGHT_WRIST, 1]),
            }
            score, _evidence = self._score_operation_posture(sample)
            combined_score = visibility_score + score
            if best is None or combined_score > best_score:
                best = sample
                best_score = combined_score
        return best

    def _visibility_score(self, landmarks):
        indices = [
            self.NOSE,
            self.LEFT_SHOULDER,
            self.RIGHT_SHOULDER,
            self.LEFT_ELBOW,
            self.RIGHT_ELBOW,
            self.LEFT_WRIST,
            self.RIGHT_WRIST,
            self.LEFT_HIP,
            self.RIGHT_HIP,
        ]
        scores = []
        for idx in indices:
            scores.append(float(min(landmarks[idx, 3], landmarks[idx, 4])))
        return float(np.mean(scores))

    @staticmethod
    def _mean_y(landmarks, left_idx, right_idx):
        return float((landmarks[left_idx, 1] + landmarks[right_idx, 1]) / 2.0)

    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))
