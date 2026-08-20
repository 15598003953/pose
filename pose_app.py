"""
PyQt5 GUI program for MediaPipe pose estimation on video.

- Uses the local ONNX pose model (pose_estimation_mediapipe_2023mar.onnx)
  via the existing MPPose class.
- Person detection model is downloaded from HuggingFace on first run
  (same behavior as demo.py).
- Draws a clean "stickman" overlay on each frame.
"""

import os
import importlib.util
import sys
import time
import json
import uuid
import numpy as np
import cv2 as cv

# OpenCV version guard (same as demo.py)
_opencv_version = lambda s: tuple(map(int, s.split(".")))
assert _opencv_version(cv.__version__) >= _opencv_version("4.10.0"), \
    "Please install latest opencv-python: python -m pip install --upgrade opencv-python"

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QStatusBar, QCheckBox,
    QComboBox,
    QGroupBox, QMessageBox, QDialog
)

from mp_pose import MPPose
from mp_persondet import MPPersonDet
from pose_action_recognizer import PutLuggageModelRecognizer, PutLuggageRecognizer

POSEC3D_AVAILABLE = importlib.util.find_spec("mmaction") is not None
try:
    from posec3d_action_recognizer import PoseC3DRecognizer
except Exception:
    PoseC3DRecognizer = None
    POSEC3D_AVAILABLE = False


PUT_LUGGAGE_DISPLAY_TEXT = "旅客正在放置行李"
POSEC3D_PUT_LUGGAGE_LABELS = {
    "garbage collecting",
    "pushing cart",
    "setting table",
    "unboxing",
    "canoeing or kayaking",
}
POSEC3D_BUSINESS_LABEL_THRESHOLD = 0.3


def _find_person_detection_model():
    """Locate the person detection ONNX model in several likely places.

    Search order:
      1. PERSON_DET_MODEL env var
      2. script dir (any *person*.onnx)
      3. parent dir and sibling dirs (Desktop level) - non-recursive
      4. HuggingFace cache / download
    """
    # 1. env var
    env_path = os.environ.get("PERSON_DET_MODEL", "")
    if env_path and os.path.isfile(env_path):
        return env_path

    here = os.path.dirname(os.path.abspath(__file__))

    # 2. script directory - any onnx containing 'person'
    if os.path.isdir(here):
        for name in os.listdir(here):
            ln = name.lower()
            if ln.endswith(".onnx") and "person" in ln:
                return os.path.join(here, name)

    # 3. parent directory and its sibling folders (one level deep)
    parent = os.path.dirname(here)
    if os.path.isdir(parent):
        candidates = []
        # direct children of parent
        for name in os.listdir(parent):
            ln = name.lower()
            if ln.endswith(".onnx") and "person" in ln:
                candidates.append(os.path.join(parent, name))
        if not candidates:
            # one level into each sibling folder
            for name in os.listdir(parent):
                sub = os.path.join(parent, name)
                if not os.path.isdir(sub):
                    continue
                try:
                    for sub_name in os.listdir(sub):
                        ln = sub_name.lower()
                        if ln.endswith(".onnx") and "person" in ln:
                            candidates.append(os.path.join(sub, sub_name))
                except OSError:
                    continue
        if candidates:
            # prefer the non-quantized (no 'int8' / 'bq') version
            non_q = [c for c in candidates if "int8" not in os.path.basename(c).lower()
                     and "_bq" not in os.path.basename(c).lower()]
            return (non_q or candidates)[0]

    # 4. try huggingface hub download as last resort
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id="opencv/person_detection_mediapipe",
            filename="person_detection_mediapipe_2023mar.onnx"
        )
    except Exception as e:
        print("WARNING: could not download person detection model "
              "and no local file found. Reason:", e)
        return ""


_PERSON_MODEL_PATH = _find_person_detection_model()


POSE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_estimation_mediapipe_2023mar.onnx"
)

POSEC3D_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "posec3d_cache",
)


def default_posec3d_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def should_show_put_luggage_business_label(
        state, confidence,
        threshold=POSEC3D_BUSINESS_LABEL_THRESHOLD):
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.0
    return str(state).strip().lower() in POSEC3D_PUT_LUGGAGE_LABELS and score >= threshold

# JSON file that stores the ROI rectangle (x, y, w, h). Loaded on startup.
ROI_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "roi_config.json"
)

PUT_LUGGAGE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "put_luggage_centroid_model.json"
)

# BlazePose 33-keypoint connections (excluding auxiliary points)
POSE_CONNECTIONS = [
    # Face outline and ears
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

BODY_INDICES = set(range(33))


def draw_stickman(image, poses, show_skeleton=True, show_points=True,
                  show_mask=False, show_bbox=False):
    """Draw a stickman overlay on the image.

    poses: list returned by MPPose.infer
           [bbox, landmarks_screen, landmarks_world, mask, heatmap, conf]
    """
    out = image.copy()
    overlay = out.copy()

    for pose in poses:
        bbox, landmarks_screen, _landmarks_world, mask, _heatmap, conf = pose

        # Optional segmentation mask (green edge)
        if show_mask and mask is not None:
            edges = cv.Canny(mask, 100, 200)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv.dilate(edges, kernel, iterations=1)
            edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            edges_bgr[edges == 255] = [0, 255, 0]
            overlay = cv.add(edges_bgr, overlay)

        # Drop the 6 auxiliary points (keep first 33 BlazePose keypoints)
        landmarks = landmarks_screen[:-6, :]
        keep = landmarks[:, 4] > 0.5  # presence threshold
        xy = landmarks[:, 0:2].astype(np.int32)

        # Bounding box
        if show_bbox:
            bb = bbox.astype(np.int32)
            cv.rectangle(overlay, tuple(bb[0]), tuple(bb[1]), (0, 200, 0), 1)
            cv.putText(overlay, "conf {:.2f}".format(float(conf)),
                        tuple(bb[0]), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv.LINE_AA)

        # Skeleton lines
        if show_skeleton:
            for a, b in POSE_CONNECTIONS:
                if keep[a] and keep[b]:
                    cv.line(overlay, tuple(xy[a]), tuple(xy[b]),
                            (0, 255, 0), 3, cv.LINE_AA)

        # Joint points
        if show_points:
            for i in BODY_INDICES:
                if i < len(xy) and keep[i]:
                    cv.circle(overlay, tuple(xy[i]), 4,
                              (0, 0, 255), -1, cv.LINE_AA)

    cv.addWeighted(overlay, 0.9, out, 0.1, 0, out)
    return out


def cv_img_to_qpix(img_bgr):
    """Convert a BGR numpy image to QPixmap (RGB)."""
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class RoiDialog(QDialog):
    """Modal dialog to manage multiple numbered ROIs.

    Displays the given frame scaled to fit the screen. The user can:
      - drag with the left mouse button on an empty area to add a new ROI
      - click on an existing ROI to select it
      - delete the selected ROI or clear all via the buttons

    The resulting ROI list is exposed via ``self.rois`` as a list of
    ``(x, y, w, h)`` tuples in original frame coordinates.
    """

    def __init__(self, frame, rois=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("框选ROI - 左键拖动添加, 点击选中, 右键取消选择")
        self.setModal(True)

        self._frame = frame.copy()
        h, w = self._frame.shape[:2]

        # Scale the frame to fit within a reasonable on-screen area.
        max_w, max_h = 1280, 720
        self._scale = min(max_w / w, max_h / h, 1.0)
        self._disp_w = max(1, int(round(w * self._scale)))
        self._disp_h = max(1, int(round(h * self._scale)))

        # rois: list of (x, y, w, h) in original frame coords
        self.rois = []
        if rois:
            for r in rois:
                if r is not None and len(r) == 4:
                    try:
                        self.rois.append(tuple(int(v) for v in r))
                    except (TypeError, ValueError):
                        pass

        self._drawing = False
        self._start = None  # QPoint in display coords (while drawing)
        self._end = None
        self._selected = None  # index into self.rois, or None

        self._label = QLabel()
        self._label.setFixedSize(self._disp_w, self._disp_h)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.installEventFilter(self)

        btn_row = QHBoxLayout()
        self._btn_del = QPushButton("删除选中")
        self._btn_clear = QPushButton("全部清除")
        self._btn_ok = QPushButton("确定")
        self._btn_cancel = QPushButton("取消")
        self._btn_del.clicked.connect(self._on_delete)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_ok.clicked.connect(self.accept)
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_del)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(self._btn_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addLayout(btn_row)

        self._redraw()

    def _to_display(self, rect):
        """Convert (x, y, w, h) original coords to (x1, y1, x2, y2) display coords."""
        x, y, w, h = rect
        return (int(round(x * self._scale)),
                int(round(y * self._scale)),
                int(round((x + w) * self._scale)),
                int(round((y + h) * self._scale)))

    def _hit_test(self, pos):
        """Return index of the topmost ROI at display position pos, or None."""
        px, py = pos.x(), pos.y()
        for i in range(len(self.rois) - 1, -1, -1):  # last drawn is on top
            x1, y1, x2, y2 = self._to_display(self.rois[i])
            if x1 <= px <= x2 and y1 <= py <= y2:
                return i
        return None

    def eventFilter(self, obj, event):
        if obj is self._label:
            t = event.type()
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                hit = self._hit_test(event.pos())
                if hit is not None:
                    # Select an existing ROI instead of drawing.
                    self._selected = hit
                    self._drawing = False
                    self._start = None
                    self._end = None
                else:
                    # Start drawing a new ROI.
                    self._selected = None
                    self._drawing = True
                    self._start = event.pos()
                    self._end = event.pos()
                self._redraw()
                return True
            elif t == QEvent.MouseMove and self._drawing:
                self._end = event.pos()
                self._redraw()
                return True
            elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton \
                    and self._drawing:
                self._drawing = False
                self._end = event.pos()
                self._add_new()
                self._redraw()
                return True
            elif t == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self._selected = None
                self._drawing = False
                self._start = None
                self._end = None
                self._redraw()
                return True
        return super().eventFilter(obj, event)

    def _add_new(self):
        if self._start is None or self._end is None:
            return
        x1 = min(self._start.x(), self._end.x())
        y1 = min(self._start.y(), self._end.y())
        x2 = max(self._start.x(), self._end.x())
        y2 = max(self._start.y(), self._end.y())
        # Clamp to the display area.
        x1 = max(0, min(x1, self._disp_w))
        y1 = max(0, min(y1, self._disp_h))
        x2 = max(0, min(x2, self._disp_w))
        y2 = max(0, min(y2, self._disp_h))
        # Reject tiny selections.
        if x2 - x1 < 5 or y2 - y1 < 5:
            self._start = None
            self._end = None
            return
        # Convert back to original frame coordinates.
        ox = int(round(x1 / self._scale))
        oy = int(round(y1 / self._scale))
        ow = int(round((x2 - x1) / self._scale))
        oh = int(round((y2 - y1) / self._scale))
        self.rois.append((ox, oy, ow, oh))
        self._start = None
        self._end = None

    def _on_delete(self):
        if self._selected is not None and 0 <= self._selected < len(self.rois):
            del self.rois[self._selected]
            self._selected = None
            self._redraw()

    def _on_clear(self):
        self.rois = []
        self._selected = None
        self._start = None
        self._end = None
        self._redraw()

    def _redraw(self):
        pix = cv_img_to_qpix(self._frame).scaled(
            self._disp_w, self._disp_h,
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        canvas = QPixmap(self._disp_w, self._disp_h)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, pix)

        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        # Draw all existing ROIs with their 1-based numbers.
        for i, r in enumerate(self.rois):
            x1, y1, x2, y2 = self._to_display(r)
            if i == self._selected:
                painter.setPen(QPen(QColor(255, 200, 0), 3))
            else:
                painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
            # Number label background + text.
            painter.fillRect(x1 + 2, y1 + 2, 32, 18, QColor(0, 0, 0, 160))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x1 + 6, y1 + 16, "#{}".format(i + 1))

        # Draw the in-progress rectangle.
        if self._drawing and self._start is not None and self._end is not None:
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            x2 = max(self._start.x(), self._end.x())
            y2 = max(self._start.y(), self._end.y())
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

        # Info bar.
        painter.fillRect(0, 0, self._disp_w, 22, QColor(0, 0, 0, 140))
        painter.setPen(QColor(200, 255, 200))
        info = "ROI数量: {}  |  左键拖动添加, 点击选中, 右键取消选择".format(len(self.rois))
        painter.drawText(8, 16, info)

        painter.end()
        self._label.setPixmap(canvas)



class PoseVideoThread(QThread):
    """Background thread that reads frames and runs pose estimation."""
    frame_ready = pyqtSignal(np.ndarray, float, int, int)  # frame, fps, frame_idx, total
    finished_video = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_path, pose_estimator, person_detector,
                 conf_threshold=0.5, show_skeleton=True, show_points=True,
                 show_mask=False, show_bbox=False, rois=None,
                 recognizer_kind="rule", recognizer_options=None,
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.pose_estimator = pose_estimator
        self.person_detector = person_detector
        self.conf_threshold = conf_threshold
        self.show_skeleton = show_skeleton
        self.show_points = show_points
        self.show_mask = show_mask
        self.show_bbox = show_bbox
        # List of ROI rectangles (x, y, w, h) in original frame coords.
        # May be updated by the main window while the thread is running.
        self.rois = list(rois) if rois else []
        self.recognizer_kind = recognizer_kind or "rule"
        self.recognizer_options = dict(recognizer_options or {})

        self._running = True
        self._pause = False
        self.cap = None
        self._seek_frame = None  # frame index to seek to (None = no pending seek)
        self.action_recognizer = None
        self.event_log_path = self._make_event_log_path(video_path)
        self._event_log = None
        self._recognizer_device = self.recognizer_options.get("device", "")

    def seek_to(self, frame_idx):
        """Request the thread to jump to a given frame index."""
        self._seek_frame = max(int(frame_idx), 0)

    def run(self):
        try:
            self.cap = cv.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error.emit("Cannot open video: {}".format(self.video_path))
                return
            total = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            # Frame interval in ms; fall back to 30 fps if metadata missing.
            fps = self.cap.get(cv.CAP_PROP_FPS)
            try:
                fps = float(fps) if fps and float(fps) > 0 else 30.0
            except (TypeError, ValueError):
                fps = 30.0
            self.action_recognizer = self._build_recognizer(fps)
            frame_interval_ms = int(1000.0 / fps)
            self._open_event_log()

            tm = cv.TickMeter()
            frame_idx = 0
            while self._running:
                # Handle pending seek
                if self._seek_frame is not None:
                    target = self._seek_frame
                    self._seek_frame = None
                    if total > 0 and target < total:
                        self.cap.set(cv.CAP_PROP_POS_FRAMES, target)
                        frame_idx = target
                    elif total <= 0:
                        self.cap.set(cv.CAP_PROP_POS_FRAMES, target)
                        frame_idx = target

                if self._pause:
                    self.msleep(30)
                    continue

                ok, frame = self.cap.read()
                if not ok:
                    break

                iter_start = time.time()
                tm.start()

                # Apply ROIs: detection/pose estimation only runs inside the
                # regions of interest. When multiple ROIs are set, each one is
                # processed independently and the resulting poses are offset
                # back to full-frame coordinates and merged. When no ROI is
                # set, the whole frame is processed.
                frame_h, frame_w = frame.shape[:2]
                rois_val = list(self.rois) if self.rois else []
                poses = []
                total_persons = 0
                # List of (roi_number, (x1, y1, x2, y2)) for drawing.
                active_roi_rects = []

                if not rois_val:
                    # No ROI: process the full frame.
                    persons = self._detect_persons(frame)
                    total_persons += len(persons)
                    for person in persons:
                        pose = self._infer_pose(frame, person)
                        if pose is not None:
                            poses.append(pose)
                else:
                    for idx, roi_val in enumerate(rois_val):
                        if roi_val is None or len(roi_val) != 4:
                            continue
                        rx, ry, rw, rh = [int(v) for v in roi_val]
                        x1 = max(0, min(rx, frame_w - 1))
                        y1 = max(0, min(ry, frame_h - 1))
                        x2 = max(0, min(rx + rw, frame_w))
                        y2 = max(0, min(ry + rh, frame_h))
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        detect_frame = frame[y1:y2, x1:x2]
                        roi_rect = (x1, y1, x2, y2)
                        active_roi_rects.append((idx + 1, roi_rect))
                        persons = self._detect_persons(detect_frame)
                        total_persons += len(persons)
                        for person in persons:
                            pose = self._infer_pose(detect_frame, person)
                            if pose is not None:
                                pose = self._offset_pose(pose, roi_rect, frame_h, frame_w)
                                poses.append(pose)
                tm.stop()
                infer_ms = (time.time() - iter_start) * 1000.0

                display = draw_stickman(
                    frame, poses,
                    show_skeleton=self.show_skeleton,
                    show_points=self.show_points,
                    show_mask=self.show_mask,
                    show_bbox=self.show_bbox,
                )
                timestamp_ms = int(frame_idx * 1000.0 / fps)
                if self.recognizer_kind == "posec3d":
                    action_result = self.action_recognizer.update(
                        poses,
                        frame_idx=frame_idx,
                        img_shape=(frame_h, frame_w),
                        timestamp_ms=timestamp_ms,
                    )
                else:
                    action_result = self.action_recognizer.update(
                        poses, frame_idx=frame_idx, timestamp_ms=timestamp_ms)
                for event in action_result.events:
                    self._write_action_event(event)
                self._draw_action_overlay(display, action_result)

                # Draw every active ROI rectangle with its 1-based number.
                for roi_num, (rx1, ry1, rx2, ry2) in active_roi_rects:
                    cv.rectangle(display, (rx1, ry1), (rx2 - 1, ry2 - 1),
                                 (255, 0, 0), 2, cv.LINE_AA)
                    cv.putText(display, "ROI{}".format(roi_num),
                               (rx1 + 4, ry1 + 18),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6,
                               (255, 0, 0), 1, cv.LINE_AA)

                if total > 0:
                    cv.putText(display, "Frame {}/{}".format(frame_idx + 1, total),
                               (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 255, 255), 1, cv.LINE_AA)
                cv.putText(display,
                           "detected:{}  poses:{}".format(total_persons, len(poses)),
                           (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 255), 1, cv.LINE_AA)
                roi_text = "ROI:{}".format(len(rois_val)) if rois_val else "ROI:FULL"
                recognizer_text = self.recognizer_kind
                if self.recognizer_kind == "posec3d" and self._recognizer_device:
                    recognizer_text += " {}".format(self._recognizer_device)
                cv.putText(
                    display,
                    "{}  {}  infer:{:.0f}ms".format(
                        roi_text, recognizer_text, infer_ms),
                    (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv.LINE_AA)

                self.frame_ready.emit(display, tm.getFPS(), frame_idx, total)
                # Pace playback to the video's own FPS so it doesn't run too fast.
                # Measure the full iteration (inference + draw + emit), not just inference.
                iter_elapsed_ms = (time.time() - iter_start) * 1000.0
                wait_ms = int(frame_interval_ms - iter_elapsed_ms)
                if wait_ms > 0:
                    self.msleep(wait_ms)
                frame_idx += 1
                tm.reset()

            self.finished_video.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._close_event_log()
            if self.cap is not None:
                self.cap.release()

    def _build_recognizer(self, fps):
        if self.recognizer_kind == "clip_model":
            if not os.path.isfile(PUT_LUGGAGE_MODEL_PATH):
                print("[pose_app] put-luggage clip model missing, fallback to rule:", PUT_LUGGAGE_MODEL_PATH)
                return PutLuggageRecognizer(fps=fps)
            return PutLuggageModelRecognizer(
                model_path=PUT_LUGGAGE_MODEL_PATH,
                fps=fps,
                clip_seconds=4.0,
                sample_stride_frames=10,
                threshold=0.5,
                cooldown_seconds=2.0,
            )
        if self.recognizer_kind == "posec3d":
            if PoseC3DRecognizer is None:
                raise RuntimeError(
                    "PoseC3D is not available in this environment."
                )
            return PoseC3DRecognizer(
                clip_len=int(self.recognizer_options.get("clip_len", 48)),
                predict_interval=int(self.recognizer_options.get("predict_interval", 12)),
                max_persons=int(self.recognizer_options.get("max_persons", 1)),
                config=self.recognizer_options.get("config") or None,
                checkpoint=self.recognizer_options.get("checkpoint") or None,
                label_map=self.recognizer_options.get("label_map") or None,
                device=self.recognizer_options.get("device", default_posec3d_device()),
                cache_dir=self.recognizer_options.get("cache_dir") or POSEC3D_CACHE_DIR,
            )
        return PutLuggageRecognizer(fps=fps)

    @staticmethod
    def _make_event_log_path(video_path):
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
        os.makedirs(result_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path or "video"))[0]
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in video_name)
        return os.path.join(result_dir, "{}_put_luggage_events.jsonl".format(safe_name or "video"))

    def _open_event_log(self):
        self._close_event_log()
        self._event_log = open(self.event_log_path, "a", encoding="utf-8")

    def _close_event_log(self):
        if self._event_log is not None:
            self._event_log.close()
            self._event_log = None

    def _write_action_event(self, event):
        payload = dict(event)
        payload["event_id"] = str(uuid.uuid4())
        payload["video_path"] = self.video_path
        payload["log_time_ms"] = int(time.time() * 1000)
        if self._event_log is not None:
            self._event_log.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._event_log.flush()

    @staticmethod
    def _draw_action_overlay(display, action_result):
        color = (0, 0, 255) if action_result.state == "put_luggage" else (0, 255, 255)
        cv.rectangle(display, (8, 64), (420, 116), (0, 0, 0), -1)
        cv.putText(display, action_result.debug_text,
                   (14, 86), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                   color, 1, cv.LINE_AA)
        if action_result.state == "put_luggage":
            PoseVideoThread._draw_chinese_text(
                display, PUT_LUGGAGE_DISPLAY_TEXT,
                x=14, y=94, color=QColor(255, 40, 40), point_size=16)
        if should_show_put_luggage_business_label(
                action_result.state, action_result.confidence):
            cv.rectangle(
                display,
                (max(0, display.shape[1] - 300), 8),
                (display.shape[1] - 8, 54),
                (0, 0, 0),
                -1)
            PoseVideoThread._draw_chinese_text(
                display, PUT_LUGGAGE_DISPLAY_TEXT,
                x=max(14, display.shape[1] - 286), y=18,
                color=QColor(255, 40, 40), point_size=18)

    @staticmethod
    def _draw_chinese_text(display, text, x, y, color, point_size=16):
        image_rgb = cv.cvtColor(display, cv.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        qimg = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        painter = QPainter(qimg)
        painter.setPen(color)
        font = painter.font()
        font.setPointSize(point_size)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x, y + point_size, text)
        painter.end()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        rendered = np.array(ptr).reshape(h, w, ch)
        display[:, :, :] = cv.cvtColor(rendered, cv.COLOR_RGB2BGR)

    def _detect_persons(self, frame):
        """Run person detection on the given (possibly cropped) frame."""
        persons = []
        try:
            det = self.person_detector.infer(frame)
            # MPPersonDet returns numpy array (N, 13); guard against
            # unexpected return types.
            if det is not None and hasattr(det, "shape") and len(det.shape) == 2:
                persons = list(det)
            elif det is not None and hasattr(det, "__iter__"):
                persons = list(det)
        except Exception as e:
            import sys, traceback
            print("[pose_app] person detector error:", e, file=sys.stderr)
            traceback.print_exc()
        return persons

    def _infer_pose(self, frame, person):
        """Run pose estimation for one detected person."""
        try:
            return self.pose_estimator.infer(frame, person)
        except Exception as e:
            import sys, traceback
            print("[pose_app] pose estimator error:", e, file=sys.stderr)
            traceback.print_exc()
            return None

    @staticmethod
    def _offset_pose(pose, roi_rect, frame_h, frame_w):
        """Translate a pose computed on an ROI-cropped frame back into the
        full-frame coordinate system.

        pose: [bbox, landmarks_screen, landmarks_world, mask, heatmap, conf]
        roi_rect: (x1, y1, x2, y2) crop offsets in full-frame pixels.
        """
        bbox, landmarks_screen, landmarks_world, mask, heatmap, conf = pose
        ox, oy, _rx2, _ry2 = roi_rect

        # Offset bounding box (2x2 array: top-left, bottom-right).
        bbox = np.array(bbox, copy=True)
        bbox[0, 0] += ox
        bbox[0, 1] += oy
        bbox[1, 0] += ox
        bbox[1, 1] += oy

        # Offset screen-space landmarks (x, y columns).
        landmarks_screen = np.array(landmarks_screen, copy=True)
        landmarks_screen[:, 0] += ox
        landmarks_screen[:, 1] += oy

        # World landmarks are in a normalized 3D space; leave them untouched.
        # Pad the segmentation mask back to full-frame size so it aligns.
        if mask is not None and mask.size > 0:
            mh, mw = mask.shape[:2]
            if mh > 0 and mw > 0:
                full_mask = np.zeros((frame_h, frame_w), dtype=mask.dtype)
                full_mask[oy:oy + mh, ox:ox + mw] = mask[:frame_h - oy, :frame_w - ox]
                mask = full_mask

        return [bbox, landmarks_screen, landmarks_world, mask, heatmap, conf]

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False

    def is_paused(self):
        return self._pause

    def stop(self):
        self._running = False
        self._pause = False
        self.wait(3000)


class PoseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediaPipe Pose - Stickman Viewer")
        self.resize(1100, 700)

        self.pose_estimator = None
        self.person_detector = None
        self.pose_model_path = POSE_MODEL_PATH
        self.thread = None
        self.video_path = None
        self.video_frame_size = None
        self.current_frame = None
        self.recognizer_kind = "clip_model" if os.path.isfile(PUT_LUGGAGE_MODEL_PATH) else "rule"
        # List of ROI rectangles (x, y, w, h) in original frame coords.
        self.rois = []
        self.roi_frame_size = None

        self._build_ui()
        self._init_models()
        self._load_rois()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Open Video")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_select_model = QPushButton("Select Pose ONNX")
        self.cbo_recognizer = QComboBox()
        self.btn_select_roi = QPushButton("框选ROI")
        self.btn_save_roi = QPushButton("保存ROI")
        self.btn_clear_roi = QPushButton("清除ROI")
        for b in (self.btn_play, self.btn_pause, self.btn_stop, self.btn_select_roi):
            b.setEnabled(False)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_play)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_select_model)
        self.cbo_recognizer.addItem("ClipModel")
        self.cbo_recognizer.addItem("Rule")
        self.cbo_recognizer.addItem("PoseC3D")
        self.cbo_recognizer.setCurrentIndex(0 if self.recognizer_kind == "clip_model" else 1)
        if not os.path.isfile(PUT_LUGGAGE_MODEL_PATH):
            self.cbo_recognizer.setItemData(0, "models/put_luggage_centroid_model.json is missing", Qt.ToolTipRole)
            self.cbo_recognizer.model().item(0).setEnabled(False)
        if not POSEC3D_AVAILABLE:
            self.cbo_recognizer.setItemData(2, "PoseC3D requires mmaction2", Qt.ToolTipRole)
            self.cbo_recognizer.model().item(2).setEnabled(False)
        bar.addWidget(self.cbo_recognizer)
        bar.addWidget(self.btn_select_roi)
        bar.addWidget(self.btn_save_roi)
        bar.addWidget(self.btn_clear_roi)
        bar.addStretch(1)
        self.lbl_info = QLabel("No video loaded")
        bar.addWidget(self.lbl_info)
        root.addLayout(bar)

        # Display
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(640, 360)
        self.display.setStyleSheet("background-color:#202020; color:#888;")
        self.display.setText("Open a video to start pose analysis")
        root.addWidget(self.display, 1)

        # Seek bar
        seek_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        seek_row.addWidget(self.slider, 1)
        root.addLayout(seek_row)

        # Options
        opts = QGroupBox("Display Options")
        opts_row = QHBoxLayout(opts)
        self.chk_skeleton = QCheckBox("Skeleton")
        self.chk_points = QCheckBox("Joints")
        self.chk_mask = QCheckBox("Body Mask")
        self.chk_bbox = QCheckBox("Bounding Box")
        for c in (self.chk_skeleton, self.chk_points):
            c.setChecked(True)
        for c in (self.chk_mask, self.chk_bbox):
            c.setChecked(False)
        for c in (self.chk_skeleton, self.chk_points, self.chk_mask, self.chk_bbox):
            c.toggled.connect(self._on_options_changed)
            opts_row.addWidget(c)
        opts_row.addStretch(1)
        root.addWidget(opts)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # Signals
        self.btn_open.clicked.connect(self._open_video)
        self.btn_play.clicked.connect(self._play)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_select_model.clicked.connect(self._select_pose_model)
        self.cbo_recognizer.currentIndexChanged.connect(self._on_recognizer_changed)
        self.btn_select_roi.clicked.connect(self._select_roi)
        self.btn_save_roi.clicked.connect(self._save_rois)
        self.btn_clear_roi.clicked.connect(self._clear_rois)


    def _on_recognizer_changed(self, index):
        if index == 0:
            kind = "clip_model"
        elif index == 2:
            kind = "posec3d"
        else:
            kind = "rule"
        self.recognizer_kind = kind
        if kind == "clip_model":
            self.status.showMessage("Recognizer: ClipModel")
        elif kind == "posec3d" and not POSEC3D_AVAILABLE:
            self.status.showMessage("PoseC3D is unavailable in this environment")
        elif kind == "posec3d":
            self.status.showMessage("Recognizer: PoseC3D")
        else:
            self.status.showMessage("Recognizer: Rule")

    def _init_models(self):
        if not os.path.isfile(self.pose_model_path):
            QMessageBox.critical(
                self, "Model missing",
                "Pose model not found:\n{}\n\n"
                "Please place pose_estimation_mediapipe_2023mar.onnx "
                "in the project folder or choose another ONNX model.".format(
                    self.pose_model_path))
            self.btn_open.setEnabled(False)
            return
        if not _PERSON_MODEL_PATH or not os.path.isfile(_PERSON_MODEL_PATH):
            QMessageBox.warning(
                self, "Person detector missing",
                "Person detection model not found.\n"
                "Please either:\n"
                "  - install huggingface_hub and re-run with internet, or\n"
                "  - place person_detection_mediapipe_2023mar.onnx next to "
                "pose_app.py\n"
                "  - set PERSON_DET_MODEL env var to its path.")
        try:
            print("[pose_app] person det model:", _PERSON_MODEL_PATH)
            print("[pose_app] pose model:", self.pose_model_path)
            self.person_detector = MPPersonDet(
                modelPath=_PERSON_MODEL_PATH,
                nmsThreshold=0.3,
                scoreThreshold=0.3,   # more permissive
                topK=5000,
                backendId=cv.dnn.DNN_BACKEND_OPENCV,
                targetId=cv.dnn.DNN_TARGET_CPU,
            )
            self.pose_estimator = MPPose(
                modelPath=self.pose_model_path,
                confThreshold=0.3,    # more permissive
                backendId=cv.dnn.DNN_BACKEND_OPENCV,
                targetId=cv.dnn.DNN_TARGET_CPU,
            )
            self.btn_open.setEnabled(True)
            self.status.showMessage(
                "Models loaded. Pose: {}".format(
                    os.path.basename(self.pose_model_path)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Model init failed", str(e))
            self.btn_open.setEnabled(False)

    def _select_pose_model(self):
        start_dir = os.path.dirname(self.pose_model_path) \
            if self.pose_model_path else os.path.dirname(os.path.abspath(__file__))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pose ONNX", start_dir,
            "ONNX Models (*.onnx)")
        if not path:
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Model missing",
                                "Selected model does not exist:\n{}".format(path))
            return
        if not path.lower().endswith(".onnx"):
            QMessageBox.warning(
                self, "Invalid model",
                "This button only accepts the MediaPipe Pose ONNX file.\n"
                "PoseC3D is selected from the recognizer dropdown.")
            return
        lower_name = os.path.basename(path).lower()
        if "person" in lower_name or "detect" in lower_name:
            QMessageBox.warning(
                self, "Invalid pose model",
                "Selected file looks like a person detection model.\n"
                "Please choose pose_estimation_mediapipe_2023mar.onnx.")
            return

        if self.thread is not None and self.thread.isRunning():
            self._stop()

        old_estimator = self.pose_estimator
        old_path = self.pose_model_path
        try:
            new_estimator = MPPose(
                modelPath=path,
                confThreshold=0.3,
                backendId=cv.dnn.DNN_BACKEND_OPENCV,
                targetId=cv.dnn.DNN_TARGET_CPU,
            )
        except Exception as e:
            self.pose_estimator = old_estimator
            self.pose_model_path = old_path
            QMessageBox.critical(self, "Model init failed",
                                 "Cannot load pose model:\n{}\n\n{}".format(path, e))
            return

        self.pose_estimator = new_estimator
        self.pose_model_path = path
        self.btn_open.setEnabled(self.person_detector is not None)
        self.status.showMessage(
            "Pose model changed: {}".format(os.path.basename(path)))

    # ---- Slots ----------------------------------------------------------
    def _open_video(self):
        if self.pose_estimator is None or self.person_detector is None:
            QMessageBox.warning(self, "Not ready", "Models are not loaded.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.webm);;All Files (*)")
        if not path:
            return
        self.video_path = path
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Open failed",
                                 "Cannot open video:\n{}".format(path))
            return
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.video_frame_size = (frame_w, frame_h)
        self._validate_rois_for_video(frame_w, frame_h)
        self.slider.setEnabled(total > 0)
        self.slider.setMaximum(max(total - 1, 0))
        self.slider.setValue(0)
        self.lbl_info.setText(os.path.basename(path))
        self.btn_select_roi.setEnabled(True)
        if self.rois:
            self.status.showMessage("Loaded: {} | ROI: {} 个".format(path, len(self.rois)))
        else:
            self.status.showMessage("Loaded: {}".format(path))
        self._play()

    def _play(self):
        if not self.video_path:
            return
        if self.thread is not None and self.thread.isRunning():
            if self.thread.is_paused():
                self.thread.resume()
            return
        self.thread = PoseVideoThread(
            self.video_path,
            self.pose_estimator,
            self.person_detector,
            show_skeleton=self.chk_skeleton.isChecked(),
            show_points=self.chk_points.isChecked(),
            show_mask=self.chk_mask.isChecked(),
            show_bbox=self.chk_bbox.isChecked(),
            rois=self.rois,
            recognizer_kind=self.recognizer_kind,
            recognizer_options={
                "device": default_posec3d_device(),
                "predict_interval": 12,
            },
        )
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.finished_video.connect(self._on_finished)
        self.thread.error.connect(self._on_error)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.cbo_recognizer.setEnabled(False)
        self.status.showMessage("Playing...")
        self.thread.start()

    def _pause(self):
        if self.thread and self.thread.isRunning():
            self.thread.pause()
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.status.showMessage("Paused")

    def _stop(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.btn_play.setEnabled(bool(self.video_path))
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.cbo_recognizer.setEnabled(True)
        self.status.showMessage("Stopped")

    def _on_frame(self, frame, fps, idx, total):
        self.current_frame = frame
        pix = cv_img_to_qpix(frame)
        scaled = pix.scaled(
            self.display.size(), Qt.KeepAspectRatio,
            Qt.FastTransformation)
        self.display.setPixmap(scaled)
        self.status.showMessage(
            "Frame {}/{}  |  FPS {:.1f}".format(idx + 1, total or idx + 1, fps))
        # update slider without triggering a seek
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _on_finished(self):
        self.btn_play.setEnabled(bool(self.video_path))
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.cbo_recognizer.setEnabled(True)
        self.status.showMessage("Finished")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self._stop()

    def _on_slider_changed(self, value):
        if not self.thread or not self.video_path:
            return
        self.thread.seek_to(value)
        self.status.showMessage("Seeking to frame {}".format(value + 1))

    def _on_options_changed(self):
        if self.thread:
            self.thread.show_skeleton = self.chk_skeleton.isChecked()
            self.thread.show_points = self.chk_points.isChecked()
            self.thread.show_mask = self.chk_mask.isChecked()
            self.thread.show_bbox = self.chk_bbox.isChecked()

    # ---- ROI ------------------------------------------------------------
    def _load_rois(self):
        """Load the ROI list from the JSON config file, if present.

        Supports both the new ``{"rois": [[x,y,w,h], ...]}`` format and the
        legacy single-ROI ``{"roi": [x,y,w,h]}`` format.
        """
        if not os.path.isfile(ROI_CONFIG_PATH):
            return
        try:
            with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = []
            rois_data = data.get("rois")
            if isinstance(rois_data, list):
                for r in rois_data:
                    if r and len(r) == 4:
                        loaded.append(tuple(int(v) for v in r))
            # Legacy single-ROI format.
            if not loaded:
                legacy = data.get("roi")
                if legacy and len(legacy) == 4:
                    loaded.append(tuple(int(v) for v in legacy))
            if loaded:
                self.rois = loaded
                frame_size = data.get("frame_size")
                if isinstance(frame_size, list) and len(frame_size) == 2:
                    self.roi_frame_size = tuple(int(v) for v in frame_size)
                print("[pose_app] loaded {} ROI(s) from {}".format(
                    len(self.rois), ROI_CONFIG_PATH))
                self.status.showMessage("ROI loaded: {} 个".format(len(self.rois)))
        except Exception as e:
            print("[pose_app] failed to load ROI:", e, file=sys.stderr)

    def _save_rois(self):
        """Persist the current ROI list to the JSON config file."""
        if not self.rois:
            QMessageBox.warning(self, "保存ROI", "当前未设置ROI，请先框选。")
            return
        try:
            data = {"rois": [list(r) for r in self.rois]}
            if self.video_frame_size is not None:
                data["frame_size"] = list(self.video_frame_size)
            with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.roi_frame_size = self.video_frame_size
            self.status.showMessage("ROI 已保存: {} 个".format(len(self.rois)))
            QMessageBox.information(
                self, "保存ROI",
                "已保存 {} 个 ROI 到:\n{}\n下次启动程序将自动加载。".format(
                    len(self.rois), ROI_CONFIG_PATH))
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _clear_rois(self):
        """Clear all ROI rectangles (does not delete the saved file)."""
        self.rois = []
        self.roi_frame_size = None
        if self.thread is not None:
            self.thread.rois = []
        self.status.showMessage("ROI 已全部清除")

    def _validate_rois_for_video(self, frame_w, frame_h):
        """Drop saved ROIs when they clearly belong to another video size."""
        if not self.rois:
            return
        expected = self.roi_frame_size
        if expected is not None and tuple(expected) == (frame_w, frame_h):
            return
        self.rois = []
        self.roi_frame_size = None
        print(
            "[pose_app] cleared saved ROI because video size differs or "
            "roi_config has no frame_size. current_size=({}, {})".format(
                frame_w, frame_h)
        )

    def _get_frame_for_roi(self):
        """Read a raw frame from the current video for ROI selection."""
        if not self.video_path:
            QMessageBox.warning(self, "框选ROI", "请先打开视频。")
            return None
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "框选ROI", "无法打开视频。")
            return None
        idx = self.slider.value() if self.slider.isEnabled() else 0
        try:
            cap.set(cv.CAP_PROP_POS_FRAMES, max(idx, 0))
        except Exception:
            pass
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QMessageBox.warning(self, "框选ROI", "无法读取视频帧。")
            return None
        return frame

    def _select_roi(self):
        """Open the ROI management dialog and apply the result."""
        frame = self._get_frame_for_roi()
        if frame is None:
            return
        # Pause playback while selecting so the underlying frame is stable.
        was_playing = (self.thread is not None
                       and self.thread.isRunning()
                       and not self.thread.is_paused())
        if was_playing:
            self.thread.pause()

        dlg = RoiDialog(frame, rois=self.rois, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.rois = list(dlg.rois)
            self.roi_frame_size = self.video_frame_size
            if self.thread is not None:
                self.thread.rois = list(self.rois)
            if not self.rois:
                self.status.showMessage("ROI 已全部清除")
            else:
                self.status.showMessage("ROI: {} 个".format(len(self.rois)))

        if was_playing and self.thread is not None:
            self.thread.resume()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            pix = cv_img_to_qpix(self.current_frame)
            self.display.setPixmap(pix.scaled(
                self.display.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PoseMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
