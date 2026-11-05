# -*- coding: utf-8 -*-
"""人物重识别可视化程序（PyQt5）

流程：
1. 通过 MQTT（topic: check_gate/person/collected）接收人物消息（JSON），
   base64 解码 images.upper 得到上半身图像，自动以 local_id 为标签建立人物库
   （最多 10 张，同一人物多条消息共用同一 local_id 标签）；
   也可点 Step1 的「手动添加」从本地选图片、手动输入 ID 号入库。
2. 设置匹配阈值。
3. 选择视频来源开始检索（两种来源共用同一套匹配逻辑）：
   - 本地文件：直接读取 .ts / .mp4 / .mkv 等视频文件逐帧检索
     （可选按原速预览——以墙钟为准，处理跟不上时自动跳帧，画面始终与
     原片同步；相机倒装时可勾选「旋转180°」，检测前先整帧转正）；
   - RTSP 实时流：后台 TCP 抓帧线程持续取“最新帧”再检索，避免解码缓冲堆积
     和 UDP 丢包造成的花屏、延迟越来越大的问题。
   只在 ROI 内检测行人，提取特征并与人物库比对，结合轨迹历史做 top-k 时序投票后
   高亮匹配者并显示其标签（实时预览）。
   识别仅在 ROI 内进行：勾选并框选 A/B 感兴趣区域后，只在框内检测与重识别
   （框外不识别）；未启用任何 ROI 时只预览画面、不进行任何识别；
   检索过程中框选 ROI，下一帧立即开始框内识别；
   框内达到匹配阈值的人物显示绿色框、标签 + 置信度；低于阈值的结果不显示、
   不进入 MQTT detections；
   ROI 位置与启用状态保存在 roi_config.json，下次启动自动恢复；
   命中结果按 300ms 间隔以 JSON 发送到 MQTT topic「reidtest」——
   置信度 > 0.65 上报、< 0.6 清空（滞回防抖）。
4. 检索结果不保存 mp4：仅实时预览标注画面（结束后保留最后一帧），
   ROI 命中结果通过 MQTT 上报。

界面布局（1920×1080）：
- 左侧贴边：Step1（MQTT 接收 / 手动添加人物）在上、Step2（匹配阈值）、
  ROI 感兴趣区域（A/B）在下；
- 右侧全部区域：Step3（视频文件 / RTSP 实时流检索与回放）。

MQTT ROI 上报消息格式（topic: reidtest，每 300ms 一条）：
{
  "roi_id": "A",
  "captured_at_ms": 1792630800000,
  "detections": [
    {"passenger_id": "P048", "bbox_xyxy": [320, 255, 480, 495]}
  ]
}
（ROI 内未识别到人物库成员或置信度过低时 detections 为空列表）

MQTT 消息格式（仅需 images.upper）：
{
  "person_id": "...", "local_id": "P048", "passenger_id": "...",
  "capture_time": "...", "timestamp": "...",
  "source_video": "...", "score": 7.2, "has_face": true,
  "images": {"face": "<base64>", "upper": "<base64>"}
}

运行： python main.py
"""
import base64
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np

import paho.mqtt.client as mqtt

from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QScrollArea, QSlider, QGroupBox,
    QFrame, QMessageBox, QSizePolicy, QLineEdit, QComboBox, QFileDialog,
    QProgressBar, QCheckBox, QDialog,
)

if __package__:
    from .reid_core import (
        PersonDetector, ReidExtractor, crop_from_frame, valid_crop,
    )
else:
    from reid_core import (
        PersonDetector, ReidExtractor, crop_from_frame, valid_crop,
    )

MAX_PERSONS = 10  # 最多上传的人物图像数

# ROI 配置文件（保存 A/B 框位置与启用状态，下次启动自动恢复；归一化坐标，
# 与视频分辨率无关，换视频/换分辨率也依然适用）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "roi_config.json")

# Step3 支持的本地视频后缀（仅用于文件选择框过滤，实际以 FFmpeg 能否解码为准）
VIDEO_EXTS = (".ts", ".mp4", ".mkv", ".avi", ".mov", ".flv", ".m4v",
              ".mpg", ".mpeg", ".wmv", ".h264", ".264", ".m2ts", ".mts")

# RTSP/HTTP 等网络流的协议前缀
STREAM_SCHEME_RE = re.compile(r"^(rtsp|rtsps|rtmp|rtmps|http|https|udp|tcp)://", re.I)

# FFmpeg 打开 RTSP 的参数（OpenCV 通过该环境变量透传）：
# - rtsp_transport=tcp：默认 UDP 一旦丢包，H.264 参考帧缺失就会整片花屏，
#   强制 TCP（interleaved）可彻底避免丢包花屏，这是花屏问题的根因；
# - buffer_size / max_delay / reorder_queue_size：加大接收缓冲并允许按序重排；
# - stimeout / timeout：socket 超时（微秒），避免读阻塞把线程卡死。
#   旧版 FFmpeg 只认 stimeout，新版改名为 timeout，两个都写，
#   不认识的那个会被忽略（不影响打开）。
# - flags=discardcorrupt：解码器层面直接丢弃损坏帧（不输出带马赛克的画面，
#   本程序只取最新帧，丢一帧不影响观感，比显示花屏好）。
# - fflags=discardcorrupt：封装层面的同类开关，两者同时开启更保险。
FFMPEG_RTSP_OPTS = (
    "rtsp_transport;{transport}|buffer_size;1048576|max_delay;500000|"
    "reorder_queue_size;0|timeout;5000000|stimeout;5000000|"
    "fflags;discardcorrupt|flags;discardcorrupt"
)


# ----------------------------- 工具函数 -----------------------------
def cv2_to_pixmap(frame):
    """BGR numpy -> QPixmap（安全拷贝）。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def draw_text(frame, text, org, color=(0, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def imread_unicode(path):
    """支持中文路径的图像读取。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def is_stream_source(src):
    """判断来源是网络流（RTSP/HTTP…）还是本地文件。"""
    return bool(STREAM_SCHEME_RE.match((src or "").strip()))


def probe_real_fps(cap, allow_seek=True):
    """返回视频的真实 FPS。

    优先用元数据 FPS；若与“总帧数/实际时长”推算值偏差超过 10%
    （如可变帧率或元数据错误的视频），则采用推算值，
    保证检测节奏与实际播放时间同步。

    allow_seek=False 时不定位到尾帧校验（TS/M2TS 这类无索引容器 seek
    代价高且常失败，直接用元数据 FPS，缺失时按 25 处理）。
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    real = None
    if allow_seek and total > 1:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            ok, _ = cap.read()
            dur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if ok and dur > 0:
                real = total / dur
        except Exception:
            real = None
    if real and (not fps or fps <= 0 or abs(real - fps) > 0.1 * max(real, fps)):
        return real
    return fps if fps and fps > 0 else 25.0


def open_file_capture(path):
    """打开本地视频文件（TS/MP4 等），返回 (cap, tmp_path)。

    Windows 下 OpenCV 对非 ASCII（中文）路径支持不佳：直接打开失败时，
    在源文件同目录建一个 ASCII 文件名的硬链接（同盘零拷贝）再打开，
    tmp_path 供调用方结束后删除，避免复制大文件。
    """
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            return cap, None
        cap.release()

    if all(ord(ch) < 128 for ch in path):
        return cv2.VideoCapture(path), None

    tmp = None
    try:
        folder, name = os.path.split(path)
        ext = os.path.splitext(name)[1]
        # 文件名唯一，避免同一文件被同时打开多次时互相覆盖/删除
        tmp = os.path.join(folder or ".",
                           f".reid_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}{ext}")
        if os.path.exists(tmp):
            os.remove(tmp)
        try:
            os.link(path, tmp)  # NTFS 硬链接，不占额外空间、瞬间完成
        except OSError:
            import shutil
            shutil.copy2(path, tmp)
        for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
            cap = cv2.VideoCapture(tmp, backend)
            if cap.isOpened():
                return cap, tmp
            cap.release()
    except Exception:
        pass
    if tmp and os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return cv2.VideoCapture(path), None


# ----------------------------- RTSP 抓帧线程 -----------------------------
class StreamGrabber:
    """RTSP 后台抓帧器：独立线程持续 grab，只保留最新一帧。

    为什么需要它（花屏 + 延迟增长的根因）：
    1) OpenCV 默认用 UDP 传输 RTSP，网络抖动导致丢包后，H.264 的参考帧不完整，
       解码出来就是花屏/绿块 —— 这里强制 TCP 传输。
    2) 主循环要做检测 + 特征提取，通常慢于视频帧率；直接在主循环里 cap.read()
       会让解码缓冲持续堆积，既越播越延迟，也更容易因缓冲溢出丢包而花屏 ——
       这里让抓帧线程始终把缓冲“抽空”，主循环永远只消费最新一帧。
    """

    def __init__(self, source, log=None, clock=None):
        self.source = source
        self._log = log or (lambda m: None)
        self._clock = clock or (lambda: time.time_ns() // 1000000)
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0          # 累计抓到的帧数（也作为帧序号）
        self._captured_at_ms = 0
        self._stopped = False
        self._thread = None
        self.fps = 25.0
        self.transport = "tcp"
        self.reconnects = 0
        self.failed = False    # 彻底连不上（重连次数超限）
        self.size = (0, 0)

    # ----- 对外接口 -----
    def start(self):
        self._thread = threading.Thread(target=self._loop, name="rtsp-grabber",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def interrupt(self):
        """只请求停止、不 join（供 UI 线程调用，立即打断阻塞中的 wait_frame）。"""
        self._stopped = True
        with self._cond:
            self._cond.notify_all()

    def wait_frame(self, last_seq=-1, timeout=5.0, with_timestamp=False):
        """等待一帧序号大于 last_seq 的新帧；超时/已停止返回 (None, last_seq)。

        last_seq 传 -1 表示“首帧”：一直等到抓到第一帧为止（此时 seq 从 0 开始，
        不能只比较序号，否则会立刻返回 None）。
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                if self._frame is not None and self._seq > last_seq:
                    if with_timestamp:
                        return self._frame, self._seq, self._captured_at_ms
                    return self._frame, self._seq
                if self._stopped:
                    return (None, last_seq, 0) if with_timestamp else (None, last_seq)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return (None, last_seq, 0) if with_timestamp else (None, last_seq)
                self._cond.wait(remaining)

    @property
    def grabbed(self):
        return self._seq

    # ----- 抓帧线程 -----
    def _create_capture(self, transport):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000]
        # Both cameras open concurrently; each open must use its own transport.
        with _FFMPEG_OPEN_LOCK:
            if self._stopped:
                return None
            try:
                _enable_ffmpeg_rtsp_opts(transport)
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
            finally:
                _restore_ffmpeg_env()
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _loop(self):
        transport = "tcp"
        first_attempt = True
        while not self._stopped:
            try:
                cap = self._create_capture(transport)
            except Exception as exc:
                self._log("视频流连接异常：" + str(exc))
                cap = None
            if self._stopped:
                if cap is not None:
                    cap.release()
                return
            if cap is None and first_attempt and transport == "tcp":
                # 少数设备/中间件不支持 RTSP over TCP，回退 UDP 再试一次
                self._log("RTSP TCP 方式打开失败，回退 UDP 重试…")
                transport = "udp"
                first_attempt = False
                continue
            first_attempt = False
            if cap is None:
                self._log(f"无法打开视频流：{self.source}")
                self.failed = True
                with self._cond:
                    self._cond.notify_all()
                return

            self.transport = transport
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.size = (w, h)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and 1 <= fps <= 120:
                self.fps = float(fps)
            kind = "RTSP" if (self.source or "").lower().startswith("rtsp") else "视频流"
            self._log(f"{kind} 已连接（{transport.upper()} 传输，{w}x{h} @ "
                      f"{self.fps:.1f}fps）")

            fails = 0
            while not self._stopped:
                if not cap.grab():
                    fails += 1
                    if fails > 50:  # ≈2.5s 完全读不到数据 -> 认为断流，重连
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                ok, frame = cap.retrieve()
                if not ok or frame is None or frame.size == 0:
                    continue
                if frame.ndim != 3 or frame.shape[2] != 3:
                    continue
                with self._cond:
                    self._frame = frame
                    self._captured_at_ms = self._clock()
                    self._seq += 1
                    self._cond.notify_all()

            cap.release()
            if self._stopped:
                break
            self.reconnects += 1
            if self.reconnects > 20:
                self._log("RTSP 连续重连失败，停止抓帧。")
                self.failed = True
                with self._cond:
                    self._cond.notify_all()
                return
            self._log(f"RTSP 连接中断，2 秒后第 {self.reconnects} 次重连…")
            for _ in range(20):
                if self._stopped:
                    return
                time.sleep(0.1)


_FFMPEG_ENV_KEY = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_FFMPEG_ENV_OLD = None
_FFMPEG_OPEN_LOCK = threading.Lock()


def _enable_ffmpeg_rtsp_opts(transport):
    """设置 FFmpeg 抓流参数（全局环境变量，OpenCV 创建 capture 时读取）。"""
    global _FFMPEG_ENV_OLD
    if _FFMPEG_ENV_OLD is None:
        _FFMPEG_ENV_OLD = os.environ.get(_FFMPEG_ENV_KEY, "")
    os.environ[_FFMPEG_ENV_KEY] = FFMPEG_RTSP_OPTS.format(transport=transport)


def _restore_ffmpeg_env():
    """恢复原有环境变量，避免影响后续打开本地文件。"""
    global _FFMPEG_ENV_OLD
    if _FFMPEG_ENV_OLD is None:
        return
    if _FFMPEG_ENV_OLD:
        os.environ[_FFMPEG_ENV_KEY] = _FFMPEG_ENV_OLD
    else:
        os.environ.pop(_FFMPEG_ENV_KEY, None)
    _FFMPEG_ENV_OLD = None


# ----------------------------- 匹配器 -----------------------------
class TrackMatcher:
    """逐帧检测 + 特征比对 + 跨帧轨迹 top-k 投票（文件 / 实时流共用）。

    人物库按标签分组（同一人物的多个角度/表情照片共用一个标签），
    匹配时对同一标签的所有图片取最大相似度（多镜头匹配），
    可显著提升对姿态、视角、表情变化的鲁棒性。
    """
    TRACK_HIST = 20    # 每条轨迹保留最近 20 帧的相似度历史
    TRACK_TOPK = 5     # 轨迹得分取历史中最高的 5 帧平均（抗遮挡/姿态突变）
    MIN_FRAMES = 3     # 轨迹至少出现 3 帧才显示匹配标签（抑制偶发误检）
    TRACK_TTL = 60     # 轨迹连续 60 帧未出现则清除

    def __init__(self, persons, detector, extractor, threshold, display_labels=None):
        self.detector = detector
        self.extractor = extractor
        self.threshold = threshold
        self.display_labels = display_labels or {}

        label_names, label_feats = [], []
        for p in persons:
            lab = p["label"].strip() or "未命名"
            if lab in label_names:
                label_feats[label_names.index(lab)].append(p["feature"])
            else:
                label_names.append(lab)
                label_feats.append([p["feature"]])
        self.label_names = label_names
        self.gallery_all = np.stack([f for fs in label_feats for f in fs])  # (F, D)
        label_cols, start = [], 0
        for fs in label_feats:
            label_cols.append(list(range(start, start + len(fs))))
            start += len(fs)
        self.label_cols = label_cols
        self.n_labels = len(label_names)

        # 跨帧跟踪状态：track_id -> {"hist": 各帧对各标签的相似度, "seen": 最近帧号}
        self.tracks = {}
        self.idx = 0
        self.matched = 0  # 命中（达到阈值）的检测框总数
        self.last_detections = []  # 最近一帧的匹配结果（画布坐标），供 ROI 上报

    def annotate(self, frame, canvas=None, offset=(0, 0)):
        """对一帧做检测+匹配，返回标注后的新图（不修改原图）。

        canvas / offset（ROI 模式）：frame 是 ROI 裁剪图，检测在裁剪图上做，
        框与文字画在整帧画布 canvas 上（坐标平移 offset=(x, y)），
        匹配结果写入 self.last_detections（画布坐标系）供 ROI 上报使用。
        """
        annotated = frame.copy() if canvas is None else canvas.copy()
        ox, oy = int(offset[0]), int(offset[1])
        xyxy, confs, track_ids = self.detector.detect_track(frame)

        crops, valid_idx, valid_tids = [], [], []
        for i, b in enumerate(xyxy):
            c = crop_from_frame(frame, b)
            if valid_crop(c):
                crops.append(c)
                valid_idx.append(i)
                valid_tids.append(track_ids[i] if i < len(track_ids) else None)

        # 每个检测框对每个标签的相似度（同标签多张图取最大），
        # 并写入轨迹历史，用 top-k 平滑后的轨迹得分做最终判定，
        # 避免逐帧匹配抖动与偶发误配（如外观相似的路人）。
        lab_sims = None
        if crops:
            feats = self.extractor.extract(crops, tta=True)   # (N, D)
            s = feats @ self.gallery_all.T                    # (N, F)
            lab_sims = np.zeros((len(crops), self.n_labels))
            for li, cols in enumerate(self.label_cols):
                lab_sims[:, li] = s[:, cols].max(axis=1)
            for j, tid in enumerate(valid_tids):
                if tid is None:
                    continue
                st = self.tracks.get(tid)
                if st is None:
                    st = self.tracks[tid] = {
                        "hist": deque(maxlen=self.TRACK_HIST), "seen": self.idx}
                st["hist"].append(lab_sims[j])
                st["seen"] = self.idx
            # 清理长时间未出现的轨迹
            stale = [t for t, v in self.tracks.items()
                     if self.idx - v["seen"] > self.TRACK_TTL]
            for t in stale:
                del self.tracks[t]

        detections = []
        for j, i in enumerate(valid_idx):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            tid = valid_tids[j]
            st = self.tracks.get(tid) if tid is not None else None
            if st is not None and len(st["hist"]) > 0:
                h = np.stack(st["hist"])                      # (T, L)
                k = min(self.TRACK_TOPK, len(h))
                vec = np.sort(h, axis=0)[-k:].mean(axis=0)    # (L,) 轨迹得分
                mature = len(h) >= self.MIN_FRAMES
            else:
                vec = lab_sims[j] if lab_sims is not None else np.zeros(self.n_labels)
                mature = True
            sim = float(vec.max())
            # 低于匹配阈值（或轨迹未成熟）的结果直接丢弃：
            # 不画框、不显示标签/置信度，也不进入 detections/MQTT 上报。
            if sim >= self.threshold and mature:
                label = self.label_names[int(vec.argmax())]
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                shown = self.display_labels.get(label, label)
                draw_text(annotated, f"{shown} {sim:.2f}", (x1, max(0, y1 - 12)),
                          color, 0.8, 2)
                self.matched += 1
                detections.append({"label": label, "score": round(sim, 4),
                                   "bbox_xyxy": [x1, y1, x2, y2], "tid": tid})

        if canvas is None:
            draw_text(annotated, f"frame {self.idx}  matched={self.matched}", (10, 25),
                      (255, 255, 255), 0.6, 2)
        self.idx += 1
        self.last_detections = detections
        return annotated


# ----------------------------- 后台线程 -----------------------------
class SearchWorker(QThread):
    """检索线程：支持本地视频文件与 RTSP 实时流，实时预览标注画面。

    文件模式：按顺序读完全片（可选按原速预览），有总帧数可显示百分比；
    实时流模式：由 StreamGrabber 后台抓帧，本线程只消费最新帧，流断开或
    用户停止后退出线程（实时流无总帧数，进度以已处理帧数汇报）。
    只预览和上报 MQTT，不保存 MP4；finished_ok 携带本次处理的帧数。
    """
    frame_ready = pyqtSignal(np.ndarray)
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)   # (已处理帧数, 总帧数；0 表示未知)
    finished_ok = pyqtSignal(int)     # 检索结束时的已处理帧数（0 = 未产生画面）
    roi_message = pyqtSignal(str)     # ROI 检测结果 JSON（发往 MQTT topic reidtest）

    # 预览最多 25fps 推送到界面，避免高速读文件时刷爆 Qt 事件队列
    PREVIEW_INTERVAL = 0.04
    # 每个 ROI 的 MQTT 上报间隔（秒）
    ROI_SEND_INTERVAL = 0.3
    # ROI 上报滞回阈值：置信度 > ROI_ON_CONF 进入上报，< ROI_OFF_CONF 退出
    ROI_ON_CONF = 0.65
    ROI_OFF_CONF = 0.60

    def __init__(self, source, persons, detector, extractor, threshold,
                 realtime_preview=False, rotate180=False, rois=None):
        super().__init__()
        self.source = source  # 本地视频路径 或 RTSP/HTTP 地址
        self.persons = persons  # [{"label", "feature", ...}]
        self.detector = detector
        self.extractor = extractor
        self.threshold = threshold
        self.realtime_preview = realtime_preview  # 文件模式：按原速播放预览
        self.rotate180 = rotate180   # 文件模式：画面倒装时先整帧转 180° 再检测
        self.is_stream = is_stream_source(source)
        self._stop = False
        self._grabber = None  # 实时流模式下由 _run_stream 创建

        # ROI 状态（rect 为归一化 (x1, y1, x2, y2)，None = 未启用；
        # 运行中可通过 set_rois 动态启用/调整/停用，下一帧即生效）
        self._roi_lock = threading.Lock()
        self._roi_rects = {"A": None, "B": None}
        for r in (rois or []):
            rid = (r or {}).get("roi_id")
            if rid in self._roi_rects and r.get("rect"):
                self._roi_rects[rid] = tuple(r["rect"])
        self._roi_matchers = {}   # roi_id -> TrackMatcher（各配独立检测器/轨迹）
        self._roi_active = {}     # roi_id -> {track_key: 是否处于上报状态}（滞回）
        self._roi_current = {}    # roi_id -> 当前帧的上报检测列表
        self._roi_last_send = {}  # roi_id -> 上次上报时刻
        self._roi_frame_idx = 0   # ROI 模式的帧计数
        self._detected = 0        # 真正做过检测/识别的帧数（无 ROI 时恒为 0）

    def set_rois(self, rois):
        """运行中更新 ROI（GUI 线程调用）。

        rois: [{"roi_id": "A", "rect": (x1,y1,x2,y2) | None}, ...]
        rect 传 None 表示停用该 ROI。
        """
        with self._roi_lock:
            for r in (rois or []):
                rid = (r or {}).get("roi_id")
                if rid in self._roi_rects:
                    self._roi_rects[rid] = tuple(r["rect"]) if r.get("rect") else None

    def stop(self):
        self._stop = True
        if self._grabber is not None:
            self._grabber.interrupt()  # 立即唤醒阻塞中的抓帧等待

    def _orient(self, frame):
        """按需把整帧旋转 180°。

        必须在检测/跟踪之前做：旋转后人体才是正立的，检测框、轨迹 id
        与画面上叠的标签才对得上；预览画面同样是旋转后的正向画面。
        """
        if self.rotate180 and frame is not None:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    # ----- 线程入口 -----
    def run(self):
        if self.is_stream:
            self._run_stream()
        else:
            self._run_file()

    def _matched_count(self):
        """当前命中总数（各 ROI 匹配器之和；未启用 ROI 时恒为 0）。"""
        return sum(m.matched for m in self._roi_matchers.values())

    def _roi_enabled(self):
        with self._roi_lock:
            return any(rc for rc in self._roi_rects.values())

    # ----- ROI 模式 -----
    def _annotate_frame(self, frame):
        """单帧处理入口：只在各 ROI 框内检测；未启用任何 ROI 时不识别。

        没有 ROI 时直接返回原画面（不跑检测、不跑匹配，画面上给出提示）；
        “ROI 区域不用识别”= 检测器只看 ROI 裁剪图，框外内容不参与；
        框内检测结果平移回整帧坐标后画框/上报。
        """
        with self._roi_lock:
            rects = {rid: rc for rid, rc in self._roi_rects.items() if rc}
        if not rects:
            # 未启用 ROI：不进行任何识别，只在画面上提示（不写框、不发消息）
            canvas = frame.copy()
            draw_text(canvas, "NO ROI: REID DISABLED (draw ROI to start)",
                      (10, 25), (0, 200, 255), 0.6, 2)
            return canvas

        H, W = frame.shape[:2]
        canvas = frame.copy()
        self._detected += 1  # 该帧真正跑了 ROI 内检测
        pixel = {}
        for rid, rc in rects.items():
            x1, y1 = int(rc[0] * W), int(rc[1] * H)
            x2, y2 = int(rc[2] * W), int(rc[3] * H)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 16 or y2 - y1 < 16:  # 框太小，视为无效
                self._roi_current[rid] = []
                continue
            pixel[rid] = (x1, y1, x2, y2)
            crop = frame[y1:y2, x1:x2]
            m = self._roi_matchers.get(rid)
            if m is None:
                # 每个 ROI 配独立检测器：ByteTrack 的 persist 跟踪状态按
                # “虚拟相机”隔离，否则 A/B 两路裁剪图交替喂同一个跟踪器，
                # 轨迹 id 会互相串导致匹配抖动。
                m = TrackMatcher(self.persons, PersonDetector(),
                                 self.extractor, self.threshold)
                self._roi_matchers[rid] = m
                self.log.emit(f"ROI {rid} 已启用（{x2 - x1}x{y2 - y1} 像素，仅框内检测）")
            # 注意必须接收返回值：annotate 画在 canvas 的副本上并返回，
            # 丢弃返回值会导致框内检测/标签全部丢失（画面上只剩 ROI 橙框）。
            canvas = m.annotate(crop, canvas=canvas, offset=(x1, y1))
            self._update_roi_state(rid, m.last_detections, W, H)

        # 停用的 ROI：清掉残留状态（不再上报）
        for rid in list(self._roi_current):
            if rid not in rects:
                self._roi_current.pop(rid, None)
                self._roi_active.pop(rid, None)

        # ROI 框可视化
        for rid, (x1, y1, x2, y2) in pixel.items():
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
            draw_text(canvas, f"ROI {rid}", (x1 + 6, y1 + 24), (0, 200, 255), 0.7, 2)
        draw_text(canvas, f"frame {self._roi_frame_idx}  matched={self._matched_count()}",
                  (10, 25), (255, 255, 255), 0.6, 2)
        self._roi_frame_idx += 1
        return canvas

    def _update_roi_state(self, rid, detections, W, H):
        """按滞回阈值维护每个 ROI 的“当前上报检测”列表。

        置信度 > 0.65 进入上报；已上报的目标滑落到 [0.6, 0.65) 区间时保持
        （防止在阈值边界反复进出抖动）；< 0.6 或目标离开 ROI / 轨迹消失
        则移除。无任何上报目标时列表为空（仍按 300ms 节拍发空列表）。
        """
        active = self._roi_active.setdefault(rid, {})
        new_active, reported = {}, []
        for d in detections or []:
            # 双保险：低于 Step2 匹配阈值的结果绝不进入 MQTT 上报
            if float(d["score"]) < self.threshold:
                continue
            key = d["tid"] if d["tid"] is not None else f"{d['label']}@{tuple(d['bbox_xyxy'])}"
            s = float(d["score"])
            prev = bool(active.get(key, False))
            on = (s > self.ROI_ON_CONF) or (prev and s >= self.ROI_OFF_CONF)
            new_active[key] = on
            if on:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                reported.append({
                    "passenger_id": d["label"],
                    "bbox_xyxy": [int(max(0, min(W - 1, x1))),
                                  int(max(0, min(H - 1, y1))),
                                  int(max(0, min(W, x2))),
                                  int(max(0, min(H, y2)))],
                })
        self._roi_active[rid] = new_active
        self._roi_current[rid] = reported

    def _pump_roi_messages(self):
        """把各 ROI 当前状态按 300ms 节拍上报（无目标也发空 detections）。"""
        with self._roi_lock:
            rects = {rid for rid, rc in self._roi_rects.items() if rc}
        if not rects:
            return
        now = time.perf_counter()
        for rid in sorted(rects):
            if rid not in self._roi_current:
                continue  # 该 ROI 尚未处理过任何帧
            if now - self._roi_last_send.get(rid, 0.0) < self.ROI_SEND_INTERVAL:
                continue
            self._roi_last_send[rid] = now
            msg = {
                "roi_id": rid,
                "captured_at_ms": int(time.time() * 1000),
                "detections": self._roi_current.get(rid, []),
            }
            self.roi_message.emit(json.dumps(msg, ensure_ascii=False))

    # ----- 本地文件检索 -----
    def _run_file(self):
        tmp_paths = []

        # 1) 先单独探测帧率：探测需要 seek 到尾帧，而 TS/M2TS 这类容器没有索引，
        #    seek 之后往往读不出帧，所以探测不能和正式读取共用一个 capture。
        fps = 25.0
        probe, tmp = open_file_capture(self.source)
        if tmp:
            tmp_paths.append(tmp)
        try:
            if not probe.isOpened():
                self.log.emit(f"无法打开视频文件：{self.source}")
                self.finished_ok.emit(0)
                return
            # TS/M2TS 等无索引容器不做尾帧 seek 校验（大文件上会明显拖慢启动）
            ext = os.path.splitext(self.source)[1].lower()
            fps = probe_real_fps(probe, allow_seek=ext not in (".ts", ".m2ts", ".mts"))
        finally:
            probe.release()

        # 2) 重新从头正式读取
        cap, tmp = open_file_capture(self.source)
        if tmp:
            tmp_paths.append(tmp)
        try:
            if not cap.isOpened():
                self.log.emit(f"无法打开视频文件：{self.source}")
                self.finished_ok.emit(0)
                return
            ret, frame0 = cap.read()
            if not ret or frame0 is None:
                self.log.emit("视频文件为空或无法解码第一帧，请确认编码是否受支持。")
                self.finished_ok.emit(0)
                return
            frame0 = self._orient(frame0)  # 首帧也按同一方向处理，尺寸/画面保持一致
            h, w = frame0.shape[:2]
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            self.log.emit(f"开始检索文件「{os.path.basename(self.source)}」"
                          f"（{w}x{h} @ {fps:.1f}fps，约 {total} 帧"
                          f"{'，画面已旋转 180°' if self.rotate180 else ''}）")
            if self.rotate180:
                self.log.emit("提示：按倒装画面处理，先整帧旋转 180° 再检测，预览画面为正向。")
            if not self._roi_enabled():
                self.log.emit("未启用任何 ROI：本次检索只预览画面、不做人物识别。"
                              "点左侧「框选」拖出 ROI 后，下一帧立即开始框内识别。")

            # 首帧先处理（预览立即有画面，也用于确定预览尺寸），idx 从 1 起算
            annotated0 = self._annotate_frame(frame0)
            self.frame_ready.emit(annotated0)
            idx = 1            # 已读到的帧号（时间轴位置，含跳帧）
            # 原速模式的播放时钟：从首帧画面出现那一刻起算
            start_t = time.perf_counter()
            last_preview = time.perf_counter()
            last_report = time.perf_counter()
            eof = False
            while not self._stop:
                # ---- 取帧 ----
                # 原速播放：以墙钟为准 —— 处理跟不上就跳帧追赶时间轴（与实时流
                # “只消费最新帧”的行为一致），处理比时间轴快就等待。
                # 旧实现只做“超前时 sleep”，一旦检测慢于帧率就永远不 sleep，
                # 实际按检测速度播放（远慢于原速），这就是“勾选了也不按原速
                # 播放”的原因。跳过的帧不检测。
                if self.realtime_preview and fps > 0:
                    t = time.perf_counter() - start_t
                    want = int(t * fps)          # 当前时刻应显示的帧号
                    if want < idx:               # 已处理到时间轴前方 -> 等待
                        time.sleep(min(idx / fps - t, 0.2))
                        self._pump_roi_messages()
                        continue
                    while idx < want:            # 落后 -> 跳帧（只读不检测）
                        ok, _f = cap.read()
                        if not ok or _f is None:
                            eof = True
                            break
                        idx += 1
                    if eof:
                        break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = self._orient(frame)  # 检测前转正（旋转 180° 时）
                idx += 1

                # ---- 检测 / 匹配 / 预览 ----
                annotated = self._annotate_frame(frame)
                if annotated.shape[1] != w or annotated.shape[0] != h:
                    annotated = cv2.resize(annotated, (w, h))  # 花屏/分辨率突变保护
                self._pump_roi_messages()

                now = time.perf_counter()
                if now - last_preview >= self.PREVIEW_INTERVAL:
                    last_preview = now
                    self.frame_ready.emit(annotated)
                if now - last_report >= 1.0:
                    last_report = now
                    self.progress.emit(idx, total)
                    if self._roi_matchers:
                        skip = idx - self._detected
                        extra = f"（检测 {self._detected}，跳帧 {skip}）" if skip > 0 \
                            else f"（检测 {self._detected}）"
                    else:
                        extra = "（未启用 ROI，未识别）"
                    self.log.emit(f"已处理 {idx}/{total or '?'} 帧{extra}，"
                                  f"命中 {self._matched_count()} 处")

            self.progress.emit(idx, total)
            reason = "已停止" if self._stop else "文件检索完毕"
            if self._roi_matchers:
                skip = idx - self._detected
                extra = f"，检测 {self._detected} 帧，跳帧 {skip} 帧" if skip > 0 \
                    else f"，检测 {self._detected} 帧"
            else:
                extra = "（未启用 ROI，只预览未识别）"
            self.log.emit(f"检索结束（{reason}）：共 {idx} 帧{extra}，"
                          f"命中 {self._matched_count()} 处。")
            self.finished_ok.emit(idx)
        finally:
            cap.release()
            for p in tmp_paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # ----- RTSP 实时流检索 -----
    def _run_stream(self):
        grabber = StreamGrabber(self.source, log=self.log.emit)
        self._grabber = grabber
        grabber.start()
        try:
            frame, seq = grabber.wait_frame(-1, timeout=20.0)
            if frame is None:
                self.log.emit(f"无法打开视频流：{self.source}")
                self.finished_ok.emit(0)
                return
            h, w = frame.shape[:2]
            fps = grabber.fps

            self.log.emit(f"开始检索实时流（{w}x{h} @ {fps:.1f}fps）：{self.source}")
            if not self._roi_enabled():
                self.log.emit("未启用任何 ROI：本次检索只预览画面、不做人物识别。"
                              "点左侧「框选」拖出 ROI 后，下一帧立即开始框内识别。")
            idx = 0
            last_seq = seq
            last_preview = 0.0
            last_report = time.perf_counter()
            last_frame_t = time.perf_counter()
            while not self._stop:
                frame, seq = grabber.wait_frame(last_seq, timeout=5.0)
                if frame is None:
                    # 彻底连不上 / 用户停止 / 30 秒没有新帧 -> 结束
                    if grabber.failed or self._stop:
                        break
                    if time.perf_counter() - last_frame_t > 30:
                        self.log.emit("超过 30 秒未收到新画面，结束检索。")
                        break
                    self._pump_roi_messages()  # 等帧期间也维持 300ms 上报节拍
                    continue
                last_seq = seq
                last_frame_t = time.perf_counter()

                annotated = self._annotate_frame(frame)
                if annotated.shape[1] != w or annotated.shape[0] != h:
                    annotated = cv2.resize(annotated, (w, h))
                self._pump_roi_messages()

                now = time.perf_counter()
                if now - last_preview >= self.PREVIEW_INTERVAL:
                    last_preview = now
                    self.frame_ready.emit(annotated)
                if now - last_report >= 2.0:
                    last_report = now
                    self.progress.emit(idx + 1, 0)
                    self.log.emit(f"已处理 {idx + 1} 帧（抓取 {grabber.grabbed} 帧），"
                                  f"命中 {self._matched_count()} 处")
                idx += 1

            self.progress.emit(idx, 0)
            reason = "已停止" if self._stop else "流已断开"
            self.log.emit(f"检索结束（{reason}）：处理 {idx} 帧 / 抓取 "
                          f"{grabber.grabbed} 帧，命中 {self._matched_count()} 处。")
            self.finished_ok.emit(idx)
        finally:
            grabber.stop()
            self._grabber = None


# ----------------------------- MQTT 接收器 -----------------------------
class MqttReceiver(QObject):
    """通过 MQTT 接收人物消息（后台网络线程 -> Qt 信号转发到主线程）。

    连接 broker 并订阅指定 topic；每条 JSON 消息解析出 images.upper
    （base64 上半身图）后 emit 给主线程建立人物库条目。
    """

    person_received = pyqtSignal(dict)   # 解析成功的 JSON dict
    log = pyqtSignal(str)
    connected_changed = pyqtSignal(bool)  # 连接状态变化

    def __init__(self, host, port, topic):
        super().__init__()
        self.host, self.port, self.topic = host, port, topic
        # paho-mqtt 2.x 要求显式 CallbackAPIVersion；1.x 无此参数
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self._running = False

    def start(self):
        self.client.connect_async(self.host, self.port, keepalive=60)
        self.client.loop_start()
        self._running = True

    def stop(self):
        if self._running:
            self._running = False
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client.loop_stop()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self.topic)
            self.log.emit(f"MQTT 已连接 {self.host}:{self.port}，订阅 topic：{self.topic}")
            self.connected_changed.emit(True)
        else:
            self.log.emit(f"MQTT 连接失败（rc={rc}）：{self.host}:{self.port}")
            self.connected_changed.emit(False)

    def _on_disconnect(self, client, userdata, rc):
        self.log.emit("MQTT 已断开。")
        self.connected_changed.emit(False)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
            if not isinstance(payload, dict):
                raise ValueError("消息不是 JSON 对象")
        except Exception as e:
            self.log.emit(f"MQTT 消息解析失败：{e}")
            return
        self.person_received.emit(payload)


class MqttPublisher(QObject):
    """ROI 检测结果发布器：连接 broker 并向 topic「reidtest」发送 JSON。

    与 Step1 的 MqttReceiver 使用各自的客户端（broker 允许多连接），
    broker 地址取 Step1 输入框的值。连接建立前 publish 的消息会缓存，
    连上后自动补发，避免检索刚开始时的消息丢失。
    """

    log = pyqtSignal(str)

    def __init__(self, host, port, topic):
        super().__init__()
        self.host, self.port, self.topic = host, port, topic
        # paho-mqtt 2.x 要求显式 CallbackAPIVersion；1.x 无此参数
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._pending = []      # 连接建立前的缓存消息（最多保留最近 100 条）
        self._running = False

    def start(self):
        self._running = True
        self.client.connect_async(self.host, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        if self._running:
            self._running = False
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client.loop_stop()

    def publish(self, payload):
        """发布一条 JSON 字符串；未连上时先缓存，连上后自动补发。"""
        if self._running and self.client.is_connected():
            try:
                self.client.publish(self.topic, payload, qos=1)
            except Exception as e:
                self.log.emit(f"MQTT 发布失败：{e}")
        else:
            self._pending.append(payload)
            if len(self._pending) > 100:
                self._pending = self._pending[-100:]

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            n = len(self._pending)
            for p in self._pending:
                try:
                    client.publish(self.topic, p, qos=1)
                except Exception:
                    pass
            self._pending.clear()
            msg = f"ROI 上报已连接 MQTT {self.host}:{self.port}，topic：{self.topic}"
            self.log.emit(msg + (f"（补发缓存 {n} 条）" if n else ""))
        else:
            self.log.emit(f"ROI 上报 MQTT 连接失败（rc={rc}）：{self.host}:{self.port}")

    def _on_disconnect(self, client, userdata, rc):
        # 断线由 paho 自动重连，期间消息进入缓存，无需提示
        pass


def decode_base64_image(b64):
    """base64 字符串 -> BGR 图像（兼容 data:image/...;base64, 前缀）。

    解码或解码后不是有效图像时返回 None。
    """
    if not b64 or not isinstance(b64, str):
        return None
    data = re.sub(r"^data:image/\w+;base64,", "", b64.strip())
    try:
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ----------------------------- 上传人物条目 -----------------------------
class PersonItem(QFrame):
    """单张上传的人物图像条目：缩略图 + 标签输入框 + 删除按钮。"""
    remove_requested = pyqtSignal(object)

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        self.setFixedSize(170, 260)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        img = QLabel()
        img.setAlignment(Qt.AlignCenter)
        img.setPixmap(cv2_to_pixmap(data["crop"]).scaled(
            158, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(img)

        self.edit = QLineEdit(data["label"])
        self.edit.setFont(QFont("Microsoft YaHei", 9))
        self.edit.setPlaceholderText("输入标签")
        self.edit.textChanged.connect(self._on_text)
        lay.addWidget(self.edit)

        btn = QPushButton("删除")
        btn.clicked.connect(lambda: self.remove_requested.emit(self))
        lay.addWidget(btn)

        self.setStyleSheet("PersonItem{border:2px solid #bbb; background:#ffffff;}")

    def _on_text(self, text):
        self.data["label"] = text


# ----------------------------- 手动添加对话框 -----------------------------
class ManualAddDialog(QDialog):
    """手动添加人物：从本地选择图片 + 手动输入 ID 号。

    支持一次选多张图片（同一人物的不同角度/姿态共用一个 ID，
    匹配时对这些图取最大相似度），也可把图片直接拖进对话框。
    """

    IMG_FILTER = ("图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;"
                  "所有文件 (*)")

    def __init__(self, parent=None, last_dir=""):
        super().__init__(parent)
        self.setWindowTitle("手动添加人物")
        self.setMinimumWidth(640)
        self.setAcceptDrops(True)
        self._dir = last_dir or ""
        self.paths = []

        lay = QVBoxLayout(self)

        body = QHBoxLayout()
        self.preview = QLabel("（未选择图片）")
        self.preview.setFixedSize(220, 170)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            "background:#f5f5f5; border:1px solid #ccc; color:#888;")
        body.addWidget(self.preview, 0, Qt.AlignTop)

        form = QVBoxLayout()
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("图片："))
        self.edit_path = QLineEdit()
        self.edit_path.setReadOnly(True)
        self.edit_path.setPlaceholderText("点击「选择图片…」或把图片拖到这里")
        r1.addWidget(self.edit_path, 1)
        btn_img = QPushButton("选择图片…")
        btn_img.clicked.connect(self._pick)
        r1.addWidget(btn_img)
        form.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("ID 号："))
        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("例如 P048（自定义，用于区分人物）")
        self.edit_label.textChanged.connect(self._refresh_ok)
        r2.addWidget(self.edit_label, 1)
        form.addLayout(r2)

        tip = QLabel("同一 ID 可选多张图片（同一人物的不同角度/姿态），"
                     "入库时会自动裁剪出图中最大的行人并提取特征。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#666;")
        form.addWidget(tip)
        form.addStretch()
        body.addLayout(form, 1)
        lay.addLayout(body)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_cancel = QPushButton("取消")
        self.btn_ok = QPushButton("确定")
        self.btn_ok.setDefault(True)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        lay.addLayout(btns)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._on_ok)
        self._refresh_ok()

    # ----- 内部 -----
    def _pick(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人物图片", self._dir, self.IMG_FILTER)
        if paths:
            self._set_paths(paths)

    def _set_paths(self, paths):
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            return
        self.paths = paths
        self._dir = os.path.dirname(paths[0])
        if len(paths) == 1:
            self.edit_path.setText(os.path.basename(paths[0]))
        else:
            self.edit_path.setText(
                f"{len(paths)} 张：" + "，".join(os.path.basename(p) for p in paths))
        img = imread_unicode(paths[0])
        if img is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("（无法预览该图片）")
        else:
            self.preview.setText("")
            self.preview.setPixmap(cv2_to_pixmap(img).scaled(
                self.preview.width(), self.preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._refresh_ok()

    def _refresh_ok(self):
        self.btn_ok.setEnabled(bool(self.paths) and bool(self.edit_label.text().strip()))

    def label(self):
        return self.edit_label.text().strip()

    def _on_ok(self):
        if not self.paths:
            QMessageBox.information(self, "提示", "请先选择图片。")
            return
        if not self.label():
            QMessageBox.information(self, "提示", "请填写 ID 号（如 P048）。")
            self.edit_label.setFocus()
            return
        self.accept()

    # ----- 拖拽图片 -----
    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._set_paths(paths)
            e.acceptProposedAction()


# ----------------------------- 视频播放器 -----------------------------
class VideoPlayer(QWidget):
    """检索结果回放器，兼作检索预览画面与 ROI 框选画布。

    roi_drawn(roi_id, rect)：框选完成时发出，rect 为归一化 (x1,y1,x2,y2)。
    """
    roi_drawn = pyqtSignal(str, tuple)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next)
        self.timer.setInterval(40)  # 默认 25fps，load() 时按视频实际 FPS 调整
        self.playing = False
        self._last_frame = None
        self._tmp = None

        # ROI 框选状态
        self._draw_roi = None     # 正在框选的 roi_id（None = 非框选模式）
        self._draw_anchor = None  # 按下点（帧坐标）
        self._draw_rect = None    # 当前矩形（帧坐标）

        v = QVBoxLayout(self)
        self.view = QLabel("（检索后在此播放结果）")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(560, 320)
        self.view.setStyleSheet("background:#000; color:#888;")
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)
        v.addWidget(self.view)

        bar = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_pause = QPushButton("暂停")
        self.btn_stop = QPushButton("停止")
        self.lbl_pos = QLabel("0 / 0")
        bar.addWidget(self.btn_play)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_stop)
        bar.addStretch()
        bar.addWidget(self.lbl_pos)
        v.addLayout(bar)

        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)

    def load(self, path):
        self.stop()
        self.release()
        self.cap, self._tmp = open_file_capture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开结果视频：\n{path}")
            return
        # 按视频实际 FPS 设置播放间隔，保证与现实时间同步
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps:
            fps = 25.0
        self.timer.setInterval(int(round(1000.0 / fps)))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.lbl_pos.setText(f"0 / {total}")
        self._show_next_frame()
        self.play()

    def play(self):
        if self.cap is None:
            return
        self.playing = True
        self.timer.start()

    def pause(self):
        self.playing = False
        self.timer.stop()

    def stop(self):
        self.playing = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._show_next_frame()

    def _next(self):
        self._show_next_frame()

    def _show_next_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.lbl_pos.setText(f"{pos} / {total}")
        self._display(frame)

    def _display(self, frame, store=True):
        if store:
            self._last_frame = frame
        pm = cv2_to_pixmap(frame)
        sc = pm.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(sc)

    def show_preview(self, frame):
        """检索预览帧入口：记录原始帧（框选坐标换算的基准）并显示。

        框选进行中会叠加当前橡皮筋矩形。
        """
        self._last_frame = frame
        if self._draw_rect is not None:
            disp = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in self._draw_rect]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 255), 2)
            self._display(disp, store=False)
        else:
            self._display(frame)

    # ----- ROI 框选 -----
    def begin_roi_draw(self, roi_id):
        """进入框选模式：之后在画面上按住左键拖拽、松开即确定矩形。"""
        self._draw_roi = roi_id
        self._draw_anchor = None
        self._draw_rect = None
        self.view.setCursor(Qt.CrossCursor)

    def cancel_roi_draw(self):
        self._draw_roi = None
        self._draw_anchor = None
        self._draw_rect = None
        self.view.setCursor(Qt.ArrowCursor)
        if self._last_frame is not None:
            self._display(self._last_frame, store=False)

    def eventFilter(self, obj, ev):
        if obj is self.view and self._draw_roi is not None:
            t = ev.type()
            if t == QEvent.MouseButtonPress:
                if ev.button() == Qt.RightButton:
                    self.cancel_roi_draw()   # 右键取消框选
                    return True
                if ev.button() == Qt.LeftButton:
                    p = self._label_to_frame(ev.pos())
                    if p is not None:
                        self._draw_anchor = p
                        self._draw_rect = (*p, *p)
                        return True
            elif t == QEvent.MouseMove and self._draw_anchor is not None:
                p = self._label_to_frame(ev.pos())
                if p is not None:
                    x1, y1 = self._draw_anchor
                    x2, y2 = p
                    self._draw_rect = (min(x1, x2), min(y1, y2),
                                       max(x1, x2), max(y1, y2))
                    self._refresh_overlay()
                    return True
            elif t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton \
                    and self._draw_anchor is not None:
                rid, rect = self._draw_roi, self._norm_rect(self._draw_rect)
                self.cancel_roi_draw()
                if rect is not None:
                    self.roi_drawn.emit(rid, rect)
                return True
        return super().eventFilter(obj, ev)

    def _label_to_frame(self, pos):
        """把控件坐标换算成帧坐标（按当前显示的 pixmap 与原始帧尺寸）。"""
        pm = self.view.pixmap()
        f = self._last_frame
        if pm is None or pm.isNull() or f is None:
            return None
        pw, ph = max(1, pm.width()), max(1, pm.height())
        # QLabel AlignCenter：pixmap 在控件内居中
        ox = (self.view.width() - pm.width()) // 2
        oy = (self.view.height() - pm.height()) // 2
        H, W = f.shape[:2]
        x = (pos.x() - ox) * W / pw
        y = (pos.y() - oy) * H / ph
        return x, y

    def _norm_rect(self, r):
        """帧坐标矩形 -> 归一化矩形；太小（<16px）或无画面时返回 None。"""
        f = self._last_frame
        if f is None or not r:
            return None
        H, W = f.shape[:2]
        x1, y1, x2, y2 = r
        x1, x2 = max(0.0, min(W, x1)), max(0.0, min(W, x2))
        y1, y2 = max(0.0, min(H, y1)), max(0.0, min(H, y2))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return (x1 / W, y1 / H, x2 / W, y2 / H)

    def _refresh_overlay(self):
        if self._last_frame is None:
            return
        disp = self._last_frame.copy()
        if self._draw_rect is not None:
            x1, y1, x2, y2 = [int(v) for v in self._draw_rect]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 255), 2)
        self._display(disp, store=False)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._last_frame is not None:
            self._display(self._last_frame)

    def release(self):
        """关闭当前视频并清理为兼容中文路径而建的临时硬链接。"""
        self.timer.stop()
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._tmp and os.path.exists(self._tmp):
            try:
                os.remove(self._tmp)
            except OSError:
                pass
        self._tmp = None


# ----------------------------- 主窗口 -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人物重识别（Person Re-ID）")
        self.resize(1920, 1080)
        self.setAcceptDrops(True)  # 支持把视频文件直接拖进窗口

        self.detector = PersonDetector()
        self.extractor = ReidExtractor()
        self.persons = []  # [{"label", "crop", "feature"}]
        self.search_worker = None
        self.mqtt = None  # 当前的 MqttReceiver
        self._last_dir = ""  # 文件选择框的上次目录

        # ROI 感兴趣区域（A/B 各一份：enabled + 归一化矩形）
        self.rois = {"A": {"enabled": False, "rect": None},
                     "B": {"enabled": False, "rect": None}}
        self.roi_publisher = None   # ROI 上报用 MQTT 客户端（检索时按需创建）
        self._roi_sent_count = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ===== 左侧面板（贴边）：Step1 在上 + Step2 在下 =====
        left = QWidget()
        left.setFixedWidth(620)
        left_lay = QVBoxLayout(left)

        # ----- Step 1 · MQTT 接收 / 手动添加人物 -----
        gb1 = QGroupBox("Step 1 · 接收或手动添加人物图像并设置标签（最多 10 张）")
        g1 = QVBoxLayout(gb1)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Broker:"))
        self.edit_broker = QLineEdit("localhost:1883")
        self.edit_broker.setFixedWidth(180)
        self.edit_broker.setToolTip("MQTT broker 地址，格式 host:port")
        row1.addWidget(self.edit_broker)
        self.btn_mqtt = QPushButton("连接 MQTT")
        self.btn_mqtt.clicked.connect(self._toggle_mqtt)
        row1.addWidget(self.btn_mqtt)
        self.btn_manual = QPushButton("手动添加")
        self.btn_manual.setMinimumWidth(88)
        self.btn_manual.setToolTip("从本地选择图片，手动输入 ID 号加入人物库")
        self.btn_manual.clicked.connect(self._manual_add)
        row1.addWidget(self.btn_manual)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._clear_persons)
        row1.addWidget(self.btn_clear)
        row1.addStretch()
        self.lbl_count = QLabel(f"0 / {MAX_PERSONS}")
        self.lbl_count.setStyleSheet("font-weight:bold;")
        row1.addWidget(self.lbl_count)
        g1.addLayout(row1)

        hint = QLabel("topic: check_gate/person/collected\n"
                      "已添加人物：MQTT 取 images.upper 并以 local_id 为标签；"
                      "手动添加用自填 ID；同一人物的多张图请用同一标签。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        g1.addWidget(hint)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(240)
        self.gallery_widget = QWidget()
        self.gallery_layout = QHBoxLayout(self.gallery_widget)
        self.gallery_layout.addStretch()
        self.scroll.setWidget(self.gallery_widget)
        g1.addWidget(self.scroll, 1)
        left_lay.addWidget(gb1, 4)

        # ----- Step 2 · 匹配阈值（左侧底部） -----
        gb2 = QGroupBox("Step 2 · 匹配阈值")
        g2 = QHBoxLayout(gb2)
        g2.addWidget(QLabel("相似度阈值:"))
        self.sld = QSlider(Qt.Horizontal)
        self.sld.setRange(20, 95)
        self.sld.setValue(48)
        self.sld.valueChanged.connect(self._on_thresh)
        g2.addWidget(self.sld, 1)
        self.lbl_thresh = QLabel("0.48")
        self.lbl_thresh.setMinimumWidth(40)
        g2.addWidget(self.lbl_thresh)
        left_lay.addWidget(gb2)

        # ----- ROI · 感兴趣区域（Step2 下方，最多 2 个） -----
        gb_roi = QGroupBox("ROI 感兴趣区域（最多 2 个）")
        g_roi = QVBoxLayout(gb_roi)
        self.roi_rows = {}
        for rid in ("A", "B"):
            row = QHBoxLayout()
            chk = QCheckBox(f"ROI {rid}")
            chk.setToolTip("识别只在已启用并框选的 ROI 内进行；未启用任何 ROI 时不识别。\n"
                           "命中结果按 300ms 间隔发送到 MQTT topic「reidtest」")
            row.addWidget(chk)
            btn = QPushButton("框选")
            btn.setFixedWidth(64)
            btn.setToolTip("点击后在右侧视频画面上按住鼠标拖拽出矩形（右键取消）")
            row.addWidget(btn)
            lbl = QLabel("未框选")
            lbl.setStyleSheet("color:#555;")
            row.addWidget(lbl, 1)
            btn_clear = QPushButton("清除")
            btn_clear.setFixedWidth(64)
            row.addWidget(btn_clear)
            chk.toggled.connect(lambda on, r=rid: self._on_roi_toggled(r, on))
            btn.clicked.connect(lambda _, r=rid: self._begin_roi_draw(r))
            btn_clear.clicked.connect(lambda _, r=rid: self._clear_roi(r))
            g_roi.addLayout(row)
            self.roi_rows[rid] = (chk, btn, lbl)
        roi_hint = QLabel(
            "必须启用并「框选」ROI 才会识别人物；未框选时只预览、不识别。"
            "检索中可随时框选/调整，下一帧生效。框内相似度低于匹配阈值的"
            "人物不画框、不显示标签/置信度，也不上报；达到阈值的显示绿色框+"
            "标签+置信度。MQTT 上报规则：置信度 > 0.65 发送 detections，"
            "< 0.6 发空列表。")
        roi_hint.setWordWrap(True)
        roi_hint.setStyleSheet("color:#555;")
        g_roi.addWidget(roi_hint)
        left_lay.addWidget(gb_roi)
        root.addWidget(left)

        # ===== 右侧全部区域：Step 3 · 视频文件 / RTSP 实时流检索 =====
        gb3 = QGroupBox("Step 3 · 读取视频文件（TS/MP4）或 RTSP 实时流识别人物并显示标签")
        g3 = QVBoxLayout(gb3)

        row2 = QHBoxLayout()
        self.combo_src = QComboBox()
        self.combo_src.addItems(["本地文件", "RTSP 实时流"])
        self.combo_src.setFixedWidth(120)
        self.combo_src.currentIndexChanged.connect(self._on_src_mode)
        row2.addWidget(QLabel("来源:"))
        row2.addWidget(self.combo_src)

        self.edit_v2 = QLineEdit()
        self.edit_v2.setPlaceholderText("选择或拖入 .ts / .mp4 等视频文件")
        self.edit_v2.returnPressed.connect(self._toggle_search)
        row2.addWidget(self.edit_v2, 1)

        self.btn_browse = QPushButton("浏览…")
        self.btn_browse.setMinimumWidth(80)
        self.btn_browse.clicked.connect(self._browse_file)
        row2.addWidget(self.btn_browse)

        self.chk_realtime = QCheckBox("按原速预览")
        self.chk_realtime.setToolTip("文件检索时按视频原速逐帧预览；不勾选则全速跑完")
        row2.addWidget(self.chk_realtime)

        self.chk_rotate = QCheckBox("旋转180°")
        self.chk_rotate.setToolTip(
            "相机倒装/画面上下颠倒时勾选：先整帧旋转 180° 再做检测，"
            "预览画面为正向（仅本地文件有效）")
        row2.addWidget(self.chk_rotate)

        self.btn_search = QPushButton("开始检索")
        self.btn_search.setMinimumWidth(120)
        self.btn_search.clicked.connect(self._toggle_search)
        row2.addWidget(self.btn_search)
        g3.addLayout(row2)

        row3 = QHBoxLayout()
        self.pb = QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)
        self.pb.setTextVisible(True)
        self.pb.setFormat("%p%")
        row3.addWidget(self.pb, 1)
        self.lbl_src = QLabel("未开始")
        self.lbl_src.setMinimumWidth(220)
        self.lbl_src.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row3.addWidget(self.lbl_src)
        g3.addLayout(row3)

        self.player = VideoPlayer()
        self.player.roi_drawn.connect(self._on_roi_drawn)
        g3.addWidget(self.player, 1)
        root.addWidget(gb3, 1)

        # 状态栏
        device_desc = "CUDA (GPU)" if str(self.extractor.device) == "cuda" else "CPU"
        self.statusBar().showMessage(
            f"推理设备：{device_desc}。特征模型：FastReID SBS R50(MSMT17)。"
            "请连接 MQTT 接收人物，或点「手动添加」选图片，"
            "然后在 Step3 选择视频文件或 RTSP 流开始检索；"
            "如需只识别特定区域，可在左下角启用 ROI 并在画面上框选。")

        self._on_thresh(self.sld.value())
        self._on_src_mode(self.combo_src.currentIndex())  # 同步来源模式的控件状态
        self._load_roi_config()  # 恢复上次保存的 ROI 位置与启用状态

    # ----- 事件处理 -----
    def _log(self, msg):
        self.statusBar().showMessage(msg)

    def _on_thresh(self, v):
        self.lbl_thresh.setText(f"{v / 100:.2f}")

    def _threshold(self):
        return self.sld.value() / 100.0

    # ----- ROI 感兴趣区域 -----
    def _begin_roi_draw(self, rid):
        """开始框选：需要有可参照的画面（检索预览或已载入的结果视频）。"""
        if self.player._last_frame is None:
            QMessageBox.information(
                self, "提示",
                f"当前没有可参照的画面。\n请先开始检索（或播放一段结果视频），"
                f"画面出现后再点「框选」拖拽 ROI {rid}。")
            return
        self.player.begin_roi_draw(rid)
        self._log(f"请在右侧画面上拖拽出 ROI {rid} 的矩形（松开完成，右键取消）。")

    def _on_roi_toggled(self, rid, on):
        self.rois[rid]["enabled"] = on
        if on and self.rois[rid]["rect"] is None:
            self._log(f"ROI {rid} 已启用，但尚未框选：点「框选」在画面上拖拽出区域。")
        self._sync_worker_rois()
        if on and self.rois[rid]["rect"] is not None:
            self._ensure_roi_publisher()
        self._save_roi_config()

    def _on_roi_drawn(self, rid, rect):
        """VideoPlayer 框选完成回调：保存归一化矩形并同步给检索线程。"""
        self.rois[rid]["rect"] = rect
        self.rois[rid]["enabled"] = True
        self._refresh_roi_label(rid)
        if not self.roi_rows[rid][0].isChecked():
            self.roi_rows[rid][0].setChecked(True)  # 触发 _on_roi_toggled 同步
        self._log(f"ROI {rid} 已设置（x {rect[0] * 100:.0f}~{rect[2] * 100:.0f}%，"
                  f"y {rect[1] * 100:.0f}~{rect[3] * 100:.0f}%），只在框内做重识别。")
        self._sync_worker_rois()
        self._ensure_roi_publisher()
        self._save_roi_config()

    def _clear_roi(self, rid):
        self.rois[rid]["rect"] = None
        self.rois[rid]["enabled"] = False
        self.roi_rows[rid][0].setChecked(False)
        self._refresh_roi_label(rid)
        self._sync_worker_rois()
        self._log(f"ROI {rid} 已清除。")
        self._save_roi_config()

    def _refresh_roi_label(self, rid):
        """把 ROI 的归一化矩形刷新到该行的状态标签上。"""
        rect = self.rois[rid]["rect"]
        if rect is None:
            self.roi_rows[rid][2].setText("未框选")
        else:
            self.roi_rows[rid][2].setText(
                f"x {rect[0] * 100:.0f}~{rect[2] * 100:.0f}%  "
                f"y {rect[1] * 100:.0f}~{rect[3] * 100:.0f}%")

    # ----- ROI 配置持久化（roi_config.json） -----
    def _save_roi_config(self):
        """把 ROI A/B 的位置与启用状态写入 roi_config.json，下次启动恢复。"""
        try:
            cfg = {"rois": {rid: {"enabled": bool(v["enabled"]),
                                  "rect": list(v["rect"]) if v["rect"] else None}
                            for rid, v in self.rois.items()}}
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"ROI 配置保存失败：{e}")

    def _load_roi_config(self):
        """启动时从 roi_config.json 恢复 ROI 位置与启用状态。"""
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            self._log(f"ROI 配置读取失败（已忽略）：{e}")
            return
        restored = []
        for rid in ("A", "B"):
            v = (cfg.get("rois") or {}).get(rid) or {}
            rect = v.get("rect")
            if (isinstance(rect, (list, tuple)) and len(rect) == 4
                    and all(isinstance(x, (int, float)) for x in rect)):
                x1, y1, x2, y2 = [float(x) for x in rect]
                if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                    self.rois[rid]["rect"] = (x1, y1, x2, y2)
            if self.rois[rid]["rect"] is not None and v.get("enabled"):
                self.rois[rid]["enabled"] = True
                chk = self.roi_rows[rid][0]
                # 屏蔽信号：启动恢复时不走 _on_roi_toggled，
                # 避免一启动就自动创建 MQTT 上报连接
                chk.blockSignals(True)
                chk.setChecked(True)
                chk.blockSignals(False)
            if self.rois[rid]["rect"] is not None:
                self._refresh_roi_label(rid)
                restored.append(f"{rid}{'（启用）' if self.rois[rid]['enabled'] else ''}")
        if restored:
            self._log(f"已从 {os.path.basename(CONFIG_PATH)} 恢复 ROI："
                      f"{'、'.join(restored)}，下次检索自动生效。")

    def _active_rois(self):
        """已启用且已框选的 ROI 列表。"""
        return [{"roi_id": r, "rect": v["rect"]}
                for r, v in self.rois.items()
                if v["enabled"] and v["rect"] is not None]

    def _sync_worker_rois(self):
        """把 ROI 状态同步给正在运行的检索线程（下一帧生效）。"""
        if self.search_worker is not None and self.search_worker.isRunning():
            self.search_worker.set_rois(
                [{"roi_id": r, "rect": v["rect"] if v["enabled"] else None}
                 for r, v in self.rois.items()])

    def _ensure_roi_publisher(self):
        """按需创建并连接 ROI 上报客户端（broker 地址同 Step1）。"""
        if self.roi_publisher is not None:
            return
        addr = self.edit_broker.text().strip() or "localhost:1883"
        host, port = addr, 1883
        if ":" in addr:
            host, _, p = addr.rpartition(":")
            try:
                port = int(p)
            except ValueError:
                self._log(f"Broker 地址格式错误：{addr}，ROI 上报改用 localhost:1883")
                host, port = "localhost", 1883
        self.roi_publisher = MqttPublisher(host, port, "reidtest")
        self.roi_publisher.log.connect(self._log)
        self.roi_publisher.start()
        self._log(f"ROI 上报连接 MQTT {host}:{port}（topic: reidtest）中…")

    def _on_roi_message(self, payload):
        """检索线程的 ROI 上报 -> MQTT。"""
        if self.roi_publisher is None:
            self._ensure_roi_publisher()
        if self.roi_publisher is None:
            return
        self.roi_publisher.publish(payload)
        self._roi_sent_count += 1
        try:
            rid = json.loads(payload).get("roi_id", "?")
        except Exception:
            rid = "?"
        if self._roi_sent_count == 1 or self._roi_sent_count % 20 == 0:
            self._log(f"ROI {rid} 检测结果已上报 {self._roi_sent_count} 条到 topic reidtest")

    # ----- 来源模式（本地文件 / RTSP） -----
    def _on_src_mode(self, index):
        is_file = (index == 0)
        self.btn_browse.setVisible(is_file)
        self.chk_realtime.setEnabled(is_file)
        self.chk_rotate.setEnabled(is_file)   # 画面旋转只对本地文件开放
        self.edit_v2.setPlaceholderText(
            "选择或拖入 .ts / .mp4 等视频文件" if is_file
            else "rtsp://user:pass@ip:554/stream")
        # 切换模式时清掉与当前模式不符的输入，避免拿错地址去检索
        txt = self.edit_v2.text().strip()
        if txt:
            looks_stream = is_stream_source(txt)
            if looks_stream == is_file:
                self.edit_v2.clear()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择要检索的视频文件", self._last_dir or "",
            "视频文件 (*.ts *.mp4 *.mkv *.avi *.mov *.m4v *.flv *.m2ts *.mts "
            "*.h264 *.264);;所有文件 (*)")
        if path:
            self._set_file_path(path)

    def _set_file_path(self, path):
        self.combo_src.setCurrentIndex(0)  # 先切模式（内部会清空不符的内容）
        self.edit_v2.setText(os.path.normpath(path))
        self._last_dir = os.path.dirname(path)
        self._log(f"已选择视频文件：{self.edit_v2.text()}")

    # 拖拽视频文件到窗口即可检索
    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self._set_file_path(u.toLocalFile())
                e.acceptProposedAction()
                return

    # ----- MQTT 接收人物 -----
    def _update_count(self):
        self.lbl_count.setText(f"{len(self.persons)} / {MAX_PERSONS}")

    def _toggle_mqtt(self):
        """连接 / 断开 MQTT broker。"""
        if self.mqtt is not None:
            self.mqtt.stop()
            self.mqtt = None
            self.btn_mqtt.setText("连接 MQTT")
            self.edit_broker.setEnabled(True)
            return

        addr = self.edit_broker.text().strip() or "localhost:1883"
        host, port = addr, 1883
        if ":" in addr:
            host, _, p = addr.rpartition(":")
            try:
                port = int(p)
            except ValueError:
                self._log(f"Broker 地址格式错误：{addr}（应为 host:port）")
                return
        self.mqtt = MqttReceiver(host, port, "check_gate/person/collected")
        self.mqtt.person_received.connect(self._on_person_received)
        self.mqtt.log.connect(self._log)
        self.mqtt.connected_changed.connect(self._on_mqtt_state)
        self.mqtt.start()
        self.btn_mqtt.setText("断开 MQTT")
        self.edit_broker.setEnabled(False)
        self._log(f"正在连接 MQTT broker {host}:{port} ...")

    def _on_mqtt_state(self, ok):
        if not ok and self.mqtt is not None:
            self.mqtt.stop()
            self.mqtt = None
            self.btn_mqtt.setText("连接 MQTT")
            self.edit_broker.setEnabled(True)

    def _add_person_from_image(self, img, label):
        """把一张 BGR 图像加入人物库（自动裁剪图中最大行人并提取特征）。

        返回 True 表示入库成功；图像无效或人物库已满时返回 False。
        MQTT 接收与手动添加共用这段逻辑。
        """
        if img is None or getattr(img, "size", 0) == 0:
            return False
        label = (label or "").strip() or "未命名"
        if len(self.persons) >= MAX_PERSONS:
            self._log(f"人物库已满（{MAX_PERSONS} 张），忽略：{label}")
            return False

        # 自动检测并裁剪图中最大的行人；未检出则使用整图
        xyxy, confs = self.detector.detect(img)
        crop = None
        if len(xyxy):
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            c = crop_from_frame(img, xyxy[int(np.argmax(areas))])
            if valid_crop(c):
                crop = c
        if crop is None:
            crop = img

        data = {
            "crop": crop,
            "feature": self.extractor.extract([crop], tta=True)[0],
            "label": label,
        }
        self.persons.append(data)
        item = PersonItem(data)
        item.remove_requested.connect(self._remove_person)
        self.gallery_layout.insertWidget(self.gallery_layout.count() - 1, item)
        self._update_count()
        return True

    def _on_person_received(self, payload):
        """处理一条 MQTT 人物消息：解码 images.upper 并建立人物库条目。"""
        label = str(payload.get("local_id") or payload.get("passenger_id") or "").strip()
        images = payload.get("images") or {}
        img = decode_base64_image(images.get("upper"))
        if img is None:
            self._log(f"消息缺少有效的 images.upper（base64），已忽略：{label or '?'}")
            return
        if len(self.persons) >= MAX_PERSONS:
            self._log(f"人物库已满（{MAX_PERSONS} 张），忽略消息：{label or '?'}")
            return

        if self._add_person_from_image(img, label):
            self._log(f"接收人物 [{label or '未命名'}]（来源："
                      f"{payload.get('source_video', '?')}，"
                      f"{payload.get('capture_time', '?')}）")

    # ----- 手动添加人物（本地图片 + 手动输入 ID） -----
    def _manual_add(self):
        dlg = ManualAddDialog(self, self._last_dir)
        if dlg.exec_() != QDialog.Accepted:
            return
        label, paths = dlg.label(), list(dlg.paths)
        self._last_dir = os.path.dirname(paths[0])

        added = 0
        for p in paths:
            img = imread_unicode(p)
            if img is None:
                self._log(f"无法读取图片（格式不支持或文件损坏）：{os.path.basename(p)}")
                continue
            if not self._add_person_from_image(img, label):
                break  # 人物库已满（_add_person_from_image 内已记录日志）
            added += 1
        if added:
            self._log(f"手动添加人物 [{label}]：{added} 张图片已入库"
                      f"（当前 {len(self.persons)}/{MAX_PERSONS}）。")

    def _remove_person(self, item):
        if item.data in self.persons:
            self.persons.remove(item.data)
        item.deleteLater()
        self._update_count()

    def _clear_persons(self):
        while self.gallery_layout.count() > 1:
            it = self.gallery_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.persons = []
        self._update_count()

    # ----- 检索（本地文件 / RTSP） -----
    def _resolve_source(self):
        """从输入框取出检索来源，并让来源下拉框与实际内容保持一致。"""
        text = self.edit_v2.text().strip().strip('"').strip("'")
        if not text:
            return None, None
        is_stream = is_stream_source(text)
        self.combo_src.setCurrentIndex(1 if is_stream else 0)
        return text, is_stream

    def _toggle_search(self):
        """开始 / 停止检索。"""
        # 正在检索 -> 停止
        if self.search_worker is not None and self.search_worker.isRunning():
            self.search_worker.stop()
            self.btn_search.setEnabled(False)  # 防重复点击，结束后在回调恢复
            self._log("正在停止检索...")
            return

        if not self.persons:
            QMessageBox.information(
                self, "提示", "人物库为空，请先连接 MQTT 接收人物图像，"
                              "或用 Step1 的「手动添加」选择图片。")
            return

        source, is_stream = self._resolve_source()
        if not source:
            QMessageBox.information(
                self, "提示", "请先选择本地视频文件，或填写 RTSP 流地址。")
            return
        if is_stream:
            pass  # RTSP 地址本身即来源，无需检查本地文件
        else:
            if not os.path.isfile(source):
                QMessageBox.warning(self, "提示", f"视频文件不存在：\n{source}")
                return

        self.player.pause()  # 停掉旧结果回放，避免和实时预览抢画面
        self.pb.setRange(0, 0)  # 忙碌态
        self.pb.setFormat("准备中…")
        self.lbl_src.setText(f"{'实时流' if is_stream else '文件'}：{os.path.basename(source)}")

        self.btn_search.setText("停止检索")
        rois = self._active_rois()
        for r in self.rois:
            if self.rois[r]["enabled"] and self.rois[r]["rect"] is None:
                self._log(f"提示：ROI {r} 已勾选但未框选，本次检索忽略该 ROI。")
        if not rois:
            self._log("未启用任何 ROI：本次只预览画面、不识别；"
                      "点「框选」拖出 ROI 后立即开始框内识别。")
        self._roi_sent_count = 0
        if rois:
            self._ensure_roi_publisher()
        self.search_worker = SearchWorker(
            source, self.persons, self.detector, self.extractor,
            self._threshold(),
            realtime_preview=self.chk_realtime.isChecked(),
            rotate180=(not is_stream) and self.chk_rotate.isChecked(),
            rois=rois)
        self.search_worker.frame_ready.connect(self._on_preview)
        self.search_worker.log.connect(self._log)
        self.search_worker.progress.connect(self._on_progress)
        self.search_worker.finished_ok.connect(self._on_search_done)
        self.search_worker.roi_message.connect(self._on_roi_message)
        self.search_worker.start()

    def _on_progress(self, idx, total):
        if total > 0:
            if self.pb.maximum() != total:
                self.pb.setRange(0, total)
            self.pb.setValue(min(idx, total))
            self.pb.setFormat(f"{idx}/{total} 帧 ({idx * 100 // total}%)")
        else:
            if self.pb.maximum() != 0:
                self.pb.setRange(0, 0)
                self.pb.setFormat(f"已处理 {idx} 帧")

    def _on_preview(self, frame):
        # 经由播放器显示：同时记录原始帧，供 ROI 框选做坐标换算
        self.player.show_preview(frame)

    def _on_search_done(self, frame_count):
        """检索结束（不再保存 mp4）：复位按钮，仅保留实时预览结果。"""
        self.btn_search.setEnabled(True)
        self.btn_search.setText("开始检索")
        self.pb.setRange(0, 100)
        if self.roi_publisher is not None:
            sent = self._roi_sent_count
            if self.roi_publisher is not None:
                self.roi_publisher.stop()
                self.roi_publisher = None
            if sent:
                self._log(f"ROI 上报结束，本次检索共发送 {sent} 条到 topic reidtest。")
        if frame_count and frame_count > 0:
            self.pb.setValue(100)
            self.pb.setFormat("完成")
            self.lbl_src.setText(f"检索完成（{frame_count} 帧，未保存视频）")
            self._log(f"检索完成：共处理 {frame_count} 帧（结果仅实时预览，不保存视频）。")
        else:
            self.pb.setFormat("未产生结果")
            self.lbl_src.setText("未产生结果")

    def closeEvent(self, e):
        self._save_roi_config()  # 兜底：退出前再保存一次 ROI 设置
        if self.search_worker:
            self.search_worker.stop()
            self.search_worker.wait(3000)
        if self.mqtt is not None:
            self.mqtt.stop()
            self.mqtt = None
        if self.roi_publisher is not None:
            self.roi_publisher.stop()
            self.roi_publisher = None
        self.player.release()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
