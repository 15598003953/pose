from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pose_action_recognizer import ActionResult

if not hasattr(np, "Inf"):
    np.Inf = np.inf


DEFAULT_CONFIG_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/"
    "configs/skeleton/posec3d/"
    "slowonly_r50_8xb32-u48-240e_k400-keypoint.py"
)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "posec3d_cache"
DEFAULT_CONFIG_LOCAL = DEFAULT_CACHE_DIR / "slowonly_r50_8xb32-u48-240e_k400-keypoint.py"
DEFAULT_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/"
    "slowonly_r50_8xb32-u48-240e_k400-keypoint/"
    "slowonly_r50_8xb32-u48-240e_k400-keypoint_20230731-7f498b55.pth"
)
DEFAULT_CHECKPOINT_LOCAL = (
    DEFAULT_CACHE_DIR
    / "slowonly_r50_8xb32-u48-240e_k400-keypoint_20230731-7f498b55.pth"
)
DEFAULT_LABEL_MAP_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/"
    "tools/data/kinetics/label_map_k400.txt"
)
DEFAULT_LABEL_MAP_LOCAL = DEFAULT_CACHE_DIR / "label_map_k400.txt"


COCO17_FROM_MEDIAPIPE33 = {
    0: 0,    # nose
    1: 2,    # left_eye
    2: 5,    # right_eye
    3: 7,    # left_ear
    4: 8,    # right_ear
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}


def _mean_score(landmarks, indices: Sequence[int]) -> float:
    scores = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        scores.append(float(min(landmarks[idx, 3], landmarks[idx, 4])))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _make_coco17_from_mediapipe(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map MediaPipe 33-landmark output to COCO-17 layout.

    Returns:
        keypoints: (17, 2)
        keypoint_scores: (17,)
    """
    keypoints = np.zeros((17, 2), dtype=np.float32)
    keypoint_scores = np.zeros((17,), dtype=np.float32)
    for coco_idx, mp_idx in COCO17_FROM_MEDIAPIPE33.items():
        if mp_idx >= len(landmarks):
            continue
        keypoints[coco_idx] = landmarks[mp_idx, :2]
        keypoint_scores[coco_idx] = float(
            min(landmarks[mp_idx, 3], landmarks[mp_idx, 4]))
    return keypoints, keypoint_scores


def _make_frame_pose_result(poses, max_persons: int = 1) -> Dict[str, np.ndarray]:
    selected = []
    ranked = []
    for pose in poses or []:
        if len(pose) < 2:
            continue
        landmarks = np.asarray(pose[1])
        if landmarks.ndim != 2 or landmarks.shape[0] < 29 or landmarks.shape[1] < 5:
            continue
        if np.allclose(landmarks[:17, :2], 0):
            continue
        score = _mean_score(
            landmarks,
            [0, 11, 12, 13, 14, 15, 16, 23, 24],
        )
        ranked.append((score, landmarks))
    ranked.sort(key=lambda x: x[0], reverse=True)
    for _score, landmarks in ranked[:max_persons]:
        keypoints, keypoint_scores = _make_coco17_from_mediapipe(landmarks)
        selected.append((keypoints, keypoint_scores))
    if not selected:
        return {
            "keypoints": np.zeros((0, 17, 2), dtype=np.float32),
            "keypoint_scores": np.zeros((0, 17), dtype=np.float32),
        }
    return {
        "keypoints": np.stack([item[0] for item in selected]).astype(np.float32),
        "keypoint_scores": np.stack([item[1] for item in selected]).astype(np.float32),
    }


def _ensure_file(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    try:
        from urllib.request import urlretrieve
        urlretrieve(url, str(path))
        return path
    except Exception as exc:
        raise RuntimeError(f"failed to download {url} -> {path}: {exc}") from exc


class PoseC3DRecognizer:
    """Online PoseC3D recognizer on top of MediaPipe pose output."""

    def __init__(
        self,
        clip_len: int = 48,
        predict_interval: int = 4,
        max_persons: int = 1,
        config: Optional[str] = None,
        checkpoint: Optional[str] = None,
        label_map: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.clip_len = int(clip_len)
        self.predict_interval = max(1, int(predict_interval))
        self.max_persons = max(1, int(max_persons))
        self.device = device

        cache_root = Path(cache_dir).resolve() if cache_dir else DEFAULT_CACHE_DIR
        if config:
            self.config_path = Path(config)
        else:
            try:
                import mmaction
                pkg_config = (
                    Path(mmaction.__file__).resolve().parent
                    / ".mim" / "configs" / "skeleton" / "posec3d"
                    / DEFAULT_CONFIG_LOCAL.name
                )
                self.config_path = pkg_config if pkg_config.is_file() else _ensure_file(
                    cache_root / DEFAULT_CONFIG_LOCAL.name,
                    DEFAULT_CONFIG_URL,
                )
            except Exception:
                self.config_path = _ensure_file(
                    cache_root / DEFAULT_CONFIG_LOCAL.name,
                    DEFAULT_CONFIG_URL,
                )
        self.checkpoint = checkpoint or (
            str(cache_root / DEFAULT_CHECKPOINT_LOCAL.name)
            if (cache_root / DEFAULT_CHECKPOINT_LOCAL.name).is_file()
            else DEFAULT_CHECKPOINT_URL
        )
        self.label_map_path = _ensure_file(
            Path(label_map) if label_map else cache_root / DEFAULT_LABEL_MAP_LOCAL.name,
            DEFAULT_LABEL_MAP_URL,
        )

        self._labels = self._load_labels(self.label_map_path)
        self._buffer: Deque[Dict[str, np.ndarray]] = deque(maxlen=self.clip_len)
        self._frame_count = 0
        self._last_result: Optional[ActionResult] = None

        try:
            from mmaction.apis import init_recognizer, inference_skeleton
            from mmengine.dataset import Compose
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "MMAction2 is required for PoseC3D inference. "
                "Install mmaction2/mmengine/mmcv/torch first."
            ) from exc

        self._init_recognizer = init_recognizer
        self._inference_skeleton = inference_skeleton
        self.model = init_recognizer(self.config_path, self.checkpoint, self.device)
        test_pipeline_cfg = [
            item for item in self.model.cfg.test_pipeline
            if item.get("type") != "DecompressPose"
        ]
        self._test_pipeline = Compose(test_pipeline_cfg)

    @staticmethod
    def _load_labels(path: Path) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def update(self, poses, frame_idx: int, img_shape: Tuple[int, int], timestamp_ms: Optional[int] = None):
        self._frame_count += 1
        frame_poses = _make_frame_pose_result(poses, max_persons=self.max_persons)
        self._buffer.append(frame_poses)

        if len(self._buffer) < self.clip_len:
            return ActionResult(
                state="warming_up",
                confidence=0.0,
                debug_text="PoseC3D warming_up {}/{}".format(len(self._buffer), self.clip_len),
                events=[],
            )

        if (self._frame_count - self.clip_len) % self.predict_interval != 0 and self._last_result is not None:
            return self._last_result

        pose_results = list(self._buffer)
        if max(len(item["keypoints"]) for item in pose_results) <= 0:
            return ActionResult(
                state="no_pose",
                confidence=0.0,
                debug_text="PoseC3D no valid pose in current window",
                events=[],
            )
        result = self._inference_skeleton(
            self.model, pose_results, img_shape, self._test_pipeline)
        pred_score = result.pred_score
        if hasattr(pred_score, "detach"):
            pred_score = pred_score.detach().cpu().numpy()
        else:
            pred_score = np.asarray(pred_score)
        top_idx = int(np.argmax(pred_score))
        top_score = float(pred_score[top_idx])
        top_label = self._labels[top_idx] if top_idx < len(self._labels) else str(top_idx)

        debug_text = (
            "PoseC3D top1:{} score:{:.3f} frames:{} persons:{}"
            .format(top_label, top_score, len(pose_results), self.max_persons)
        )
        self._last_result = ActionResult(
            state=top_label,
            confidence=top_score,
            debug_text=debug_text,
            events=[],
        )
        return self._last_result
