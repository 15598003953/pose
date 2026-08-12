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
    """Rule-based recognizer for coarse put-luggage body posture.

    The recognizer uses MediaPipe Pose landmarks only. It detects a sequence:
    hand starts around upper body, moves below hips, stays low briefly, then
    moves upward. It does not verify that a physical bag exists.
    """

    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(self, fps=30.0, min_visibility=0.5):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.min_visibility = min_visibility
        self.state = "idle"
        self._start_y: Optional[float] = None
        self._low_y: Optional[float] = None
        self._start_frame: Optional[int] = None
        self._low_hold_frames = 0
        self._last_event_frame = -100000
        self._last_wrist_y: Optional[float] = None

        self.min_drop_ratio = 0.32
        self.low_hold_required = max(4, int(round(self.fps * 0.16)))
        self.release_rise_ratio = 0.18
        self.cooldown_frames = max(12, int(round(self.fps * 1.0)))
        self.max_action_frames = max(24, int(round(self.fps * 4.0)))

    def update(self, poses, frame_idx, timestamp_ms=None):
        sample = self._select_pose_sample(poses)
        if sample is None:
            self._reset()
            return ActionResult("idle", 0.0, "ACTION idle no_valid_pose")

        wrist_y = sample["wrist_y"]
        hip_y = sample["hip_y"]
        knee_y = sample["knee_y"]
        shoulder_y = sample["shoulder_y"]
        body_scale = max(knee_y - shoulder_y, 1.0)
        low_threshold = hip_y + 0.20 * body_scale
        drop_threshold = self.min_drop_ratio * body_scale
        release_rise = self.release_rise_ratio * body_scale
        events = []

        if self.state == "idle":
            self._start_y = wrist_y
            self._start_frame = frame_idx
            self._low_y = wrist_y
            self._low_hold_frames = 0
            if wrist_y > low_threshold:
                self.state = "holding_low"
                self._low_hold_frames = 1
            else:
                self.state = "ready"

        elif self.state == "ready":
            if self._start_y is None:
                self._start_y = wrist_y
                self._start_frame = frame_idx
            self._low_y = wrist_y if self._low_y is None else max(self._low_y, wrist_y)
            if wrist_y - self._start_y >= drop_threshold and wrist_y > low_threshold:
                self.state = "reaching_down"
            elif self._expired(frame_idx):
                self._reset_to_ready(wrist_y, frame_idx)

        elif self.state == "reaching_down":
            self._low_y = wrist_y if self._low_y is None else max(self._low_y, wrist_y)
            if wrist_y > low_threshold:
                self.state = "holding_low"
                self._low_hold_frames = 1
            elif self._expired(frame_idx):
                self._reset_to_ready(wrist_y, frame_idx)

        elif self.state == "holding_low":
            self._low_y = wrist_y if self._low_y is None else max(self._low_y, wrist_y)
            if wrist_y > low_threshold:
                self._low_hold_frames += 1
            if (self._low_hold_frames >= self.low_hold_required
                    and self._low_y is not None
                    and self._low_y - wrist_y >= release_rise
                    and frame_idx - self._last_event_frame >= self.cooldown_frames):
                confidence = self._confidence(body_scale)
                event = {
                    "event": "put_luggage",
                    "frame_idx": int(frame_idx),
                    "timestamp_ms": int(timestamp_ms) if timestamp_ms is not None else None,
                    "confidence": round(confidence, 3),
                    "state": "put_luggage",
                    "reason": "wrist_down_hold_release",
                }
                events.append(event)
                self._last_event_frame = frame_idx
                self.state = "put_luggage"
            elif self._expired(frame_idx):
                self._reset_to_ready(wrist_y, frame_idx)

        elif self.state == "put_luggage":
            if wrist_y <= low_threshold:
                self._reset_to_ready(wrist_y, frame_idx)

        self._last_wrist_y = wrist_y
        confidence = self._confidence(body_scale) if self.state in ("holding_low", "put_luggage") else 0.0
        debug = (
            "ACTION {} conf:{:.2f} wrist_y:{:.0f} hip_y:{:.0f} hold:{}"
            .format(self.state, confidence, wrist_y, hip_y, self._low_hold_frames)
        )
        return ActionResult(self.state, confidence, debug, events)

    def _select_pose_sample(self, poses):
        best = None
        best_score = -1.0
        for pose in poses or []:
            if len(pose) < 2:
                continue
            landmarks = np.asarray(pose[1])
            if landmarks.shape[0] < 27 or landmarks.shape[1] < 5:
                continue
            for wrist_idx in (self.LEFT_WRIST, self.RIGHT_WRIST):
                wrist = landmarks[wrist_idx]
                visibility = float(wrist[3])
                presence = float(wrist[4])
                score = visibility * presence
                if visibility < self.min_visibility or presence < self.min_visibility:
                    continue
                shoulder_y = self._mean_y(landmarks, self.LEFT_SHOULDER, self.RIGHT_SHOULDER)
                hip_y = self._mean_y(landmarks, self.LEFT_HIP, self.RIGHT_HIP)
                knee_y = self._mean_y(landmarks, self.LEFT_KNEE, self.RIGHT_KNEE)
                if best is None or score > best_score:
                    best_score = score
                    best = {
                        "wrist_y": float(wrist[1]),
                        "shoulder_y": shoulder_y,
                        "hip_y": hip_y,
                        "knee_y": knee_y,
                    }
        return best

    @staticmethod
    def _mean_y(landmarks, left_idx, right_idx):
        return float((landmarks[left_idx, 1] + landmarks[right_idx, 1]) / 2.0)

    def _confidence(self, body_scale):
        if self._start_y is None or self._low_y is None:
            return 0.0
        drop_score = min(1.0, max(0.0, (self._low_y - self._start_y) / max(body_scale * 0.55, 1.0)))
        hold_score = min(1.0, self._low_hold_frames / max(self.low_hold_required, 1))
        return 0.35 + 0.4 * drop_score + 0.25 * hold_score

    def _expired(self, frame_idx):
        return self._start_frame is not None and frame_idx - self._start_frame > self.max_action_frames

    def _reset(self):
        self.state = "idle"
        self._start_y = None
        self._low_y = None
        self._start_frame = None
        self._low_hold_frames = 0
        self._last_wrist_y = None

    def _reset_to_ready(self, wrist_y, frame_idx):
        self.state = "ready"
        self._start_y = wrist_y
        self._low_y = wrist_y
        self._start_frame = frame_idx
        self._low_hold_frames = 0
