#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 MP4 人像特征提取工具（数据全程留在本机，不上传任何服务器）
================================================================
功能：
  1. 画面与烧录时间戳倒置 → 统一顺时针旋转 180° 校正（可选不旋转）
  2. 在画面上拖出一个矩形 ROI 抓拍区域，只有位于 ROI 内、且站在
     最前面（离镜头最近、目标框最大）的那个人才会被提取特征
  3. 人像裁剪以检测到的人脸为锚点，统一裁成"头 + 上半身"整体，
     不拍单独的手部/腰部等局部；检测不到人脸（或人脸太小）不抓拍
  4. 必须检测到明显的"朝摄像头移动"动作（人脸框随跟踪逐渐放大）
     才抓拍；人脸大小稳定的人在缩小的人不会被拍摄
  5. 对该人员进行跨帧跟踪并逐帧打分，只保留一张最优人像照片：
       - 上半身特征图（头+上半身整体，底部带时间戳水印）
       - 脸部特征图（底部带时间戳水印）
       - 脸部五官描线图（MediaPipe 468 点网格；未装 mediapipe 时用
         YuNet 五官关键点：双眼/鼻尖/双嘴角描线）
       - 全景标注图（画框 + 时间戳，核对用）
  6. 时间戳自动从文件名解析（如 BC5496691_1_20260923T180406Z_...mp4
     → 2026-09-23 18:04:06 起），按帧号/fps 逐帧精确计算；也可手动覆盖。
  7. 跨视频片段外观合并（HSV 直方图相似度）：同一人多次出现 /
     跨片段出现，只保留一张最优照片。
  8. 输出：人像图片 + 人员汇总.csv

依赖：pip install PyQt5 opencv-python numpy
      推荐（可选）安装 mediapipe：提供 468 点五官网格描线图。
检测模型（全部离线本地推理，不联网）：
  - models/yolo11n-pose.onnx  人体检测 + 17 关键点（鼻/双眼/双肩等），
    cv2.dnn 推理。人员候选直接来自 CNN 人体框，"人脸"必须落在人体
    框头部区域且与头部关键点吻合才算真人，从根上消除把海报/衣物
    纹理/图案误当人脸的问题；双肩关键点还用于保证上半身完整入镜。
    下载：https://github.com/ultralytics/assets/releases/download/
          v8.4.0/yolo11n-pose.onnx（放 models 目录下即自动启用）
  - models/face_detection_yunet_2023mar.onnx  人脸框 + 五官关键点
运行：python person_extractor.py
"""

import os
import re
import sys
import csv
import json
import math
import time
import traceback
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import cv2

# OpenCV 5 的新推理引擎在 readNetFromONNX 内部会打印
# "Targets are not supported by the new graph engine" 的无害 WARN，
# 每加载一个模型刷一次。把 OpenCV 日志级别压到 ERROR（真正的错误仍显示）。
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# MediaPipe（可选依赖，装了自动启用，没装自动降级到 YuNet/Haar）：
# - FaceDetection：CNN 人脸检测，几乎不会把手/衣物纹理误判成脸
# - FaceMesh：468 点人脸网格 → 五官描线图
# - Pose：人体姿态关键点 → 用双肩实际位置保证上半身完整入镜
try:
    import mediapipe as _mp_module
    HAS_MP = True
except Exception:
    _mp_module = None
    HAS_MP = False

# MQTT 实时推送（paho 缺失 / broker 不可达时由 mqtt_publisher 自动降级，不影响主流程）
try:
    if __package__:
        from .mqtt_publisher import MqttPublisher
    else:
        from mqtt_publisher import MqttPublisher
except Exception:
    MqttPublisher = None

# ============================================================
# 第一部分：核心检测 / 跟踪 / 人员登记逻辑（不依赖 PyQt5）
# ============================================================

TS_RE = re.compile(r"(\d{8})T(\d{6})Z")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
# 人脸模型文件目录：脚本同目录 / models 子目录（离线可用，不联网）
MODEL_DIRS = (MODEL_DIR, SCRIPT_DIR)
# OpenCV 5.x 起官方包不再自带 Haar 级联数据，YuNet 模型（onnx）为首选：
# https://github.com/opencv/opencv_zoo/raw/main/models/
#   face_detection_yunet/face_detection_yunet_2023mar.onnx
YUNET_NAMES = ("face_detection_yunet_2023mar.onnx",
               "face_detection_yunet_2022mar.onnx", "yunet.onnx")
# 人体检测 + 姿态关键点模型（YOLO11n-pose，COCO 80 类中只取 person；
# 输出 17 个 COCO 关键点：0 鼻尖、1/2 双眼、5/6 双肩 …）
# https://github.com/ultralytics/assets/releases/download/v8.4.0/
#   yolo11n-pose.onnx
YOLO_POSE_NAMES = ("yolo11n-pose.onnx", "yolov8n-pose.onnx",
                   "yolo11s-pose.onnx")
# 人脸识别特征模型（SFace，OpenCV zoo）：
# https://github.com/opencv/opencv_zoo/raw/main/models/
#   face_recognition_sface/face_recognition_sface_2021dec.onnx
# 输出 128 维特征，余弦相似度 ≥ 0.363 判为同一人（LFW 99.5%+）。
# 用于身份合并，远胜上半身 HSV 直方图（同款制服下极易误并）。
SFACE_NAMES = ("face_recognition_sface_2021dec.onnx",)
SFACE_MATCH_THR = 0.363
# COCO 关键点索引（yolo11-pose 输出的 17 点）
KP_NOSE, KP_LEYE, KP_REYE = 0, 1, 2
KP_LSHOULDER, KP_RSHOULDER = 5, 6


def parse_start_time(path):
    """从文件名解析视频起始时间（如 ..._20260923T180406Z_...mp4），
    解析失败则退回文件修改时间。"""
    m = TS_RE.search(os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(path))


def iou(a, b):
    """两个 (x,y,w,h) 框的交并比。"""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def nms(boxes, thr=0.35):
    """按面积从大到小的简单非极大值抑制。"""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for b in boxes:
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def candidate_score(upper_bgr, upper_box, face_box, roi_rect, frame_shape,
                    static_bonus=0.0, face_sharp=None):
    """"头+上半身"整体人像候选照片质量打分：越清晰、人脸越大
    （离镜头越近）、越靠近抓拍区域中心、越不贴画面边缘，分数越高。
    upper_box 为人像框像素坐标 (x,y,w,h)，roi_rect 同为像素坐标。
    face_sharp 为脸部裁剪图单独的清晰度（衣服纹理再丰富也代偿不了糊脸）。"""
    x, y, w, h = upper_box
    H, W = frame_shape[:2]
    if upper_bgr is None or upper_bgr.size == 0:
        return -1.0
    gray = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 清晰度对数标度（Laplacian 方差跨数量级），权重最高；
    # 明显模糊（sharp<220）额外扣分。人越走近脸越大但运动模糊
    # 越重——不能让"脸大但糊"的帧反超"清晰的中等脸"
    s = 3.0 * math.log(max(sharp, 1.0) / 50.0 + 1.0) + static_bonus
    if sharp < 220.0:
        s -= (220.0 - sharp) / 220.0 * 1.5
    if face_sharp is not None and face_sharp > 0:
        # 脸部清晰度单独加权：脸部区域小、纹理少，打分线比整图低
        s += 1.5 * math.log(max(face_sharp, 1.0) / 50.0 + 1.0)
        if face_sharp < 150.0:
            s -= (150.0 - face_sharp) / 150.0 * 1.0
    if face_box is not None:
        # 有脸：脸越大加分（封顶提前到 10% 屏宽，已足够识别），
        # 权重刻意低于清晰度
        s += 2.0
        s += 1.8 * min(1.0, face_box[2] / (0.10 * W))
        s += 1.0 * min(1.0, h / (0.55 * H))
    else:
        s += 1.5 * min(1.0, h / (0.55 * H))
    rx, ry, rw, rh = roi_rect                       # 抓拍区域中心奖励（次要）
    cx, cy = x + w / 2.0, y + h / 2.0
    dx = abs(cx - (rx + rw / 2.0)) / max(rw / 2.0, 1.0)
    dy = abs(cy - (ry + rh / 2.0)) / max(rh / 2.0, 1.0)
    s += 0.5 * (1.0 - min(1.0, max(dx, dy)))
    if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
        # 轻微贴边惩罚：完整性已由 _make_candidate 的可见比例检查把关，
        # 这里不能太重，否则会让人选到更远但更小的那张
        s -= 0.5
    return s


def draw_timestamp_strip(img, text):
    """在图片底部加一条黑色时间戳水印条。"""
    h, w = img.shape[:2]
    sh = max(24, int(w * 0.055))
    strip = np.zeros((sh, w, 3), np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    scale = min((w - 12) / max(tw, 1), (sh * 0.68) / max(th, 1), 1.0)
    cv2.putText(strip, text, (6, int(sh * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    return np.vstack([img, strip])


def hist_of(img):
    """上半身外观直方图（用于跨片段同人合并）。"""
    r = cv2.resize(img, (48, 96))
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                     [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h)
    return h


def imwrite_safe(path, img):
    """cv2.imwrite 在 Windows 中文路径下会静默失败，
    改用 imencode + numpy tofile 写文件（对任意 Unicode 路径可靠）。"""
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
        return False
    except Exception:
        return False


def save_person(out_dir, p, face_lm=None):
    """把一个人员的最优照片写入输出目录，并记录文件路径。
    face_lm：脸部五官描线图（MediaPipe 468 点网格，可为 None）。"""
    ts = p['ts']
    stamp_file = ts.strftime("%Y%m%d_%H%M%S") + "_%03d" % (ts.microsecond // 1000)
    text = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    base = os.path.join(out_dir, "%s_%s" % (p['id'], stamp_file))
    p['ts_str'] = text
    p['upper_path'] = base + "_上半身.jpg"
    imwrite_safe(p['upper_path'], draw_timestamp_strip(p['upper'], text))
    if p.get('face') is not None:
        p['face_path'] = base + "_脸部.jpg"
        imwrite_safe(p['face_path'], draw_timestamp_strip(p['face'], text))
    else:
        p['face_path'] = ""
    if face_lm is not None:
        p['face_lm_path'] = base + "_脸部描线.jpg"
        imwrite_safe(p['face_lm_path'], draw_timestamp_strip(face_lm, text))
    else:
        p['face_lm_path'] = ""
    p['full_path'] = base + "_全景.jpg"
    imwrite_safe(p['full_path'], p['full'])


def save_csv(out_dir, persons):
    path = os.path.join(out_dir, "人员汇总.csv")
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wtr = csv.writer(f)
            wtr.writerow(["人员ID", "最优出现时间", "质量分", "含人脸",
                          "来源视频", "上半身图", "脸部图", "脸部描线图",
                          "全景图"])
            for p in persons:
                wtr.writerow([p['id'], p.get('ts_str', ''),
                              "%.2f" % p['score'],
                              "是" if p.get('face') is not None else "否",
                              p.get('video', ''),
                              p.get('upper_path', ''),
                              p.get('face_path', ''),
                              p.get('face_lm_path', ''),
                              p.get('full_path', '')])
    except Exception:
        pass


# (文件名, 是否需要水平镜像重试) —— 侧脸分类器只认一个方向
HAAR_SPECS = (
    ("haarcascade_frontalface_default.xml", False),
    ("haarcascade_frontalface_alt2.xml", False),
    ("haarcascade_profileface.xml", True),
    ("haarcascade_frontalface_alt.xml", False),
    ("haarcascade_frontalface_alt_tree.xml", False),
)


def find_haar_dirs():
    """Haar 级联文件可能的存放目录（不同安装方式位置不同）。

    脚本自带的 models/ 目录优先：OpenCV 5.x 官方包已不再附带这些 xml，
    且部分安装的 cv2/data 目录为空。"""
    out = []
    seen = set()

    def add(p):
        if p and os.path.isdir(p) and p not in seen:
            seen.add(p)
            out.append(p)

    for p in MODEL_DIRS:               # 离线自带，优先
        add(p)
    try:
        add(getattr(cv2.data, "haarcascades", None))
    except Exception:
        pass
    try:
        base = os.path.dirname(cv2.__file__ or "")
        add(os.path.join(base, "data"))
        add(os.path.join(base, "data", "haarcascades"))
    except Exception:
        pass
    return out


def find_yunet_model():
    """查找本地 YuNet onnx 模型（脚本 models/ 目录或脚本同目录）。"""
    for d in MODEL_DIRS:
        if not os.path.isdir(d):
            continue
        for n in YUNET_NAMES:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
        try:
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".onnx"):
                    return os.path.join(d, f)
        except OSError:
            pass
    return None


def load_yunet(path):
    """创建 YuNet 检测器。OpenCV 在中文/非 ASCII 路径下读不了 onnx，
    失败时复制到临时 ASCII 目录再加载。"""
    for p in (path, None):
        try:
            return cv2.FaceDetectorYN.create(p or "", "", (320, 240), 0.6)
        except Exception:
            continue
    try:
        import shutil
        import tempfile
        tmpd = tempfile.mkdtemp(prefix="px_face_model_")
        dst = os.path.join(tmpd, "model.onnx")
        shutil.copyfile(path, dst)
        return cv2.FaceDetectorYN.create(dst, "", (320, 240), 0.6)
    except Exception:
        return None


# ============================================================
# MediaPipe：人脸网格（五官描线图）与上半身姿态关键点
# ============================================================

_mp_mesh = None
_mp_pose = None


def get_face_mesh():
    """惰性创建 MediaPipe 人脸网格（468 点，含瞳孔关键点）。"""
    global _mp_mesh
    if _mp_mesh is None:
        _mp_mesh = _mp_module.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.4)
    return _mp_mesh


def get_pose():
    """惰性创建 MediaPipe 姿态关键点检测器（轻量模型）。"""
    global _mp_pose
    if _mp_pose is None:
        _mp_pose = _mp_module.solutions.pose.Pose(
            model_complexity=0, smooth_landmarks=False,
            min_detection_confidence=0.5)
    return _mp_pose


def face_landmark_overlay(bgr):
    """在人脸照片上绘制 468 点网格 + 五官轮廓描线（眉/眼/鼻唇/脸型）。
    检测不到人脸（脸太小/太糊/本来就是误检）时返回 None。"""
    if not HAS_MP or bgr is None or bgr.size == 0:
        return None
    try:
        mp = _mp_module
        mesh = get_face_mesh()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        ann = rgb.copy()
        du = mp.solutions.drawing_utils
        # 细网格（底图描线）
        du.draw_landmarks(
            ann, lm,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=du.DrawingSpec(
                color=(0, 200, 0), thickness=1))
        # 五官轮廓加粗：脸型 / 眉 / 眼 / 唇
        for conn, color in (
                (mp.solutions.face_mesh.FACEMESH_FACE_OVAL, (0, 255, 255)),
                (mp.solutions.face_mesh.FACEMESH_LEFT_EYE, (255, 90, 90)),
                (mp.solutions.face_mesh.FACEMESH_RIGHT_EYE, (255, 90, 90)),
                (mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW, (255, 200, 0)),
                (mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW, (255, 200, 0)),
                (mp.solutions.face_mesh.FACEMESH_LIPS, (255, 255, 0))):
            du.draw_landmarks(
                ann, lm, connections=conn,
                landmark_drawing_spec=du.DrawingSpec(
                    color=color, thickness=1, circle_radius=1),
                connection_drawing_spec=du.DrawingSpec(
                    color=color, thickness=2))
        return cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def pose_upper(bgr):
    """全画面检测人体姿态，取上半身信息（鼻尖 + 双肩，像素坐标）。
    返回 {'nose': (x,y)|None, 'shoulder_l': (x,y)|None,
    'shoulder_r': (x,y)|None}；失败/无人时返回 None。"""
    if not HAS_MP or bgr is None or bgr.size == 0:
        return None
    try:
        res = get_pose().process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        H, W = bgr.shape[:2]
        lms = res.pose_landmarks.landmark

        def pt(i, thr=0.4):
            l = lms[i]
            if l.visibility is not None and l.visibility < thr:
                return None
            return (l.x * W, l.y * H)

        return {'nose': pt(0), 'shoulder_l': pt(11), 'shoulder_r': pt(12)}
    except Exception:
        return None


class FaceEngine(object):
    """人脸检测引擎（自动选择可用的最优模型）：
    - MediaPipe FaceDetection（CNN，首选，误检率远低于 Haar/YuNet）
    - YuNet onnx（需要把 onnx 模型文件放到脚本同目录）
    - Haar 级联分类器（保底方案，容易把手/衣物误判成脸）

    级联文件逐个尝试加载，只要有一个可用就启用（某些安装会缺个别 xml，
    例如常见的 haarcascade_profileface.xml 缺失，不应导致整个人脸检测失效）。
    """

    def __init__(self, prefer=None):
        """prefer：'mp' | 'yunet' | 'haar' | None（自动：MediaPipe > YuNet > Haar）"""
        self.mode = None
        self.cascades = []      # [(CascadeClassifier, need_mirror)]
        self.yunet = None
        self.mp_face = None     # MediaPipe FaceDetection（CNN）
        self.loaded = []        # 已加载的模型说明
        self.missing = []       # 缺失的文件名（用于提示）
        self.model_path = None
        mp_face = self._load_mp()
        haar = self._load_haar()
        yunet = self._load_yunet()
        order = ('mp', 'yunet', 'haar')
        if prefer == 'haar':
            order = ('haar', 'mp', 'yunet')
        elif prefer == 'yunet':
            order = ('yunet', 'mp', 'haar')
        for m in order:
            if m == 'mp' and mp_face is not None:
                self.mp_face = mp_face
                self.mode = 'mp'
                self.loaded.append("MediaPipe FaceDetection")
                return
            if m == 'yunet' and yunet is not None:
                self.yunet = yunet
                self.mode = 'yunet'
                self.loaded.append("YuNet %s" % os.path.basename(
                    self.model_path or ""))
                return
            if m == 'haar' and haar:
                self.cascades = haar
                self.mode = 'haar'
                return

    def _load_mp(self):
        """创建 MediaPipe FaceDetection（CNN，误检率远低于 Haar）。"""
        if not HAS_MP:
            return None
        try:
            return _mp_module.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        except Exception:
            return None

    def _load_haar(self):
        if not hasattr(cv2, "CascadeClassifier"):
            return []
        out = []
        for name, mirror in HAAR_SPECS:
            path = None
            for d in find_haar_dirs():
                p = os.path.join(d, name)
                if os.path.exists(p):
                    path = p
                    break
            if path is None:
                self.missing.append(name)
                continue
            try:
                c = cv2.CascadeClassifier(path)
                if c is not None and not c.empty():
                    out.append((c, mirror))
                    self.loaded.append(name)
            except Exception:
                self.missing.append(name)
        return out

    def _load_yunet(self):
        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        p = find_yunet_model()
        if p is None:
            return None
        d = load_yunet(p)
        if d is None:
            return None
        self.model_path = p
        return d

    @property
    def available(self):
        return self.mode is not None

    def status_text(self):
        """给界面/日志用的引擎状态说明。"""
        if self.mode == 'mp':
            return "MediaPipe FaceDetection（CNN 模型，几乎不会把手/身体误判成脸）"
        if self.mode == 'yunet':
            base = os.path.basename(self.model_path or "")
            return "YuNet（本地模型 %s）" % base
        if self.mode == 'haar':
            return "Haar 级联（%d/%d 个可用%s）" % (
                len(self.cascades), len(HAAR_SPECS),
                ("，缺：" + "、".join(self.missing)) if self.missing else "")
        return "不可用（未找到任何人脸检测模型）"

    @staticmethod
    def _rescale(faces, scale):
        if abs(scale - 1.0) < 1e-6:
            return faces
        out = []
        for f in faces:
            lms = f[4]
            if lms:
                lms = [(kx / scale, ky / scale) for (kx, ky) in lms]
            out.append((int(round(f[0] / scale)), int(round(f[1] / scale)),
                        max(1, int(round(f[2] / scale))),
                        max(1, int(round(f[3] / scale))), lms))
        return out

    @staticmethod
    def _sort_faces(faces):
        faces = list(faces)
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        return faces

    def find_in_roi(self, bgr_roi, min_neighbors=3, upscale_small=True):
        """在 BGR 图像内检测人脸。返回 [(x, y, w, h, lms)]：
        人脸框（图像内坐标，按面积降序）+ 五官关键点 lms。
        lms 为 [(x, y)]×5，顺序：右眼、左眼、鼻尖、右嘴角、左嘴角
        （YuNet / MediaPipe 输出；Haar 无关键点，lms 为 None）。"""
        if bgr_roi is None or bgr_roi.size == 0:
            return []
        h, w = bgr_roi.shape[:2]
        scale = 1.0
        img = bgr_roi
        if upscale_small and w < 300 and self.mode == 'haar':
            scale = 2.0
            img = cv2.resize(bgr_roi, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_LINEAR)
        if self.mode == 'haar':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            minf = (max(20, int(0.02 * img.shape[1])),) * 2
            for casc, mirror in self.cascades:
                faces = casc.detectMultiScale(gray, scaleFactor=1.05,
                                              minNeighbors=min_neighbors,
                                              minSize=minf)
                if len(faces):
                    haar = [(int(f[0]), int(f[1]), int(f[2]), int(f[3]),
                             None) for f in faces]
                    return self._rescale(self._sort_faces(haar), scale)
                if not mirror:
                    continue
                # 侧脸分类器只认一个朝向，水平翻转后再试一次
                flipped = cv2.flip(gray, 1)
                faces = casc.detectMultiScale(flipped, scaleFactor=1.05,
                                              minNeighbors=min_neighbors,
                                              minSize=minf)
                if len(faces):
                    rw = gray.shape[1]
                    out = [(rw - int(f[0]) - int(f[2]), int(f[1]),
                            int(f[2]), int(f[3]), None) for f in faces]
                    return self._rescale(self._sort_faces(out), scale)
            return []
        if self.mode == 'mp':
            if w < 24 or h < 24:
                return []
            try:
                rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
                res = self.mp_face.process(rgb)
            except Exception:
                return []
            out = []
            for d in (res.detections or []):
                r = d.location_data.relative_bounding_box
                score = float(d.score[0]) if d.score else 0.0
                x, fw = int(r.xmin * w), int(r.width * w)
                y, fh = int(r.ymin * h), int(r.height * h)
                if score < 0.55 or fw <= 0 or fh <= 0:
                    continue
                x = max(0, x)
                y = max(0, y)
                fw = min(fw, w - x)
                fh = min(fh, h - y)
                if fw > 0 and fh > 0:
                    # BlazeFace 6 点：0 右眼 1 左眼 2 鼻尖 3 嘴中心
                    # 4/5 左右耳 —— 嘴角用嘴中心近似（几何校验够用）
                    kps = d.location_data.relative_keypoints
                    lms = None
                    if len(kps) >= 4:
                        lms = [(kps[i].x * w, kps[i].y * h)
                               for i in (0, 1, 2, 3, 3)]
                    out.append((x, y, fw, fh, lms))
            return self._sort_faces(out)
        if self.mode == 'yunet':
            if w < 20 or h < 20:
                return []
            # 大 ROI 限幅降采样：YuNet 耗时随面积增长，640 已远超
            # 正常人脸尺寸（≥10px 即可检出），不影响小脸召回
            scale = 1.0
            img = bgr_roi
            if max(w, h) > 640:
                scale = 640.0 / max(w, h)
                img = cv2.resize(bgr_roi, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            try:
                self.yunet.setInputSize((img.shape[1], img.shape[0]))
                _, faces = self.yunet.detect(img)
            except Exception:
                return []
            if faces is None:
                return []
            out = []
            for f in faces:
                x, y, fw, fh = (f[0] / scale, f[1] / scale,
                                f[2] / scale, f[3] / scale)
                x, y = int(x), int(y)
                fw, fh = max(1, int(round(fw))), max(1, int(round(fh)))
                score = float(f[14]) if len(f) > 14 else 1.0
                if score < 0.6 or fw <= 0 or fh <= 0:
                    continue
                x = max(0, x)
                y = max(0, y)
                fw = min(fw, w - x)
                fh = min(fh, h - y)
                if fw > 0 and fh > 0:
                    # YuNet 输出 f[4:14] 为 5 个五官关键点 (x, y)×5
                    lms = None
                    if len(f) >= 14:
                        pairs = [(float(f[i]) / scale,
                                  float(f[i + 1]) / scale)
                                 for i in range(4, 14, 2)]
                        if all(px > 0.0 or py > 0.0 for px, py in pairs):
                            lms = pairs
                    out.append((x, y, fw, fh, lms))
            return self._sort_faces(out)
        return []


def landmarks_valid(lms, face_box):
    """五官关键点几何校验：lms 为 [(x,y)]×5（右眼、左眼、鼻尖、
    右嘴角、左嘴角），face_box 为同图人脸框 (x,y,w,h)。

    检查双眼间距/脸宽比例、双眼倾斜角、鼻在双眼下方且横向落在
    双眼之间（侧脸留余量）、嘴在鼻下方、嘴宽合理——用于拒绝
    "看起来像脸"的衣物纹理/图案误检（YuNet 只看框内相似度，
    不保证五官位置符合人脸解剖结构）。返回 True 表示像真脸。"""
    if not lms or len(lms) != 5:
        return False
    (rx, ry), (lx, ly), (nx, ny), (mx, my), (m2x, m2y) = lms
    fx, fy, fw, fh = face_box
    if fw <= 0 or fh <= 0:
        return False
    # 关键点应大致落在人脸框内（±30% 余量）
    for (px, py) in lms:
        if not (fx - 0.3 * fw <= px <= fx + 1.3 * fw
                and fy - 0.3 * fh <= py <= fy + 1.3 * fh):
            return False
    eye_dx = abs(lx - rx)
    eye_dy = abs(ly - ry)
    eye_dist = (eye_dx ** 2 + eye_dy ** 2) ** 0.5
    if eye_dist < 0.5 * fh:
        return False                    # 双眼距离应约为人脸高的一半以上
    # 双眼连线倾斜角 ≤ 40°（歪头/俯仰留余量）
    if eye_dx < 1e-3 or eye_dy / max(eye_dx, 1e-3) > 0.85:
        return False
    ratio = eye_dist / float(fw)
    if not (0.25 <= ratio <= 0.95):
        return False                    # 眼距/脸宽 正常约 0.4~0.7
    eye_y = max(ry, ly)
    if ny < eye_y + 0.05 * fh:
        return False                    # 鼻尖必须在双眼下方
    eye_min_x, eye_max_x = min(rx, lx), max(rx, lx)
    if not (eye_min_x - 0.6 * eye_dist <= nx <= eye_max_x + 0.6 * eye_dist):
        return False                    # 鼻尖横向落在双眼之间（侧脸余量）
    mouth_cx = (mx + m2x) / 2.0
    mouth_cy = (my + m2y) / 2.0
    if mouth_cy < ny + 0.05 * fh:
        return False                    # 嘴必须在鼻下方
    mw = abs(m2x - mx)
    if not (0.15 * fw <= mw <= 1.2 * fw):
        return False                    # 嘴宽/脸宽 合理范围
    return True


def face_matches_person(face_box, person_box):
    """人脸框与 CNN 人体框的几何一致性：真实的人脸必然位于人体框
    顶部、水平落在人体框内、宽度与人体成合理比例。
    海报/图案/衣物纹理骗过人脸检测器时，通常没有位置与比例都
    吻合的 CNN 人体框，在这里被拒绝。"""
    fx, fy, fw, fh = face_box[:4]
    bx, by, bw, bh = person_box
    if bw <= 0 or bh <= 0:
        return False
    fcx = fx + fw / 2.0
    if not (bx - 0.15 * bw <= fcx <= bx + 1.15 * bw):
        return False                    # 人脸水平位置应在人体框内
    if fy + fh > by + 0.60 * bh:
        return False                    # 人脸整体应在人体框上部 60%
    if fy < by - 1.2 * fh:
        return False                    # 人脸不应高悬在人体框上方
    r = fw / float(bw)
    if not (0.08 <= r <= 0.85):
        return False                    # 脸宽/人体宽 正常约 0.15~0.5
    return True


def kps_head_consistent(face_box, kps, conf_thr=0.30):
    """YuNet 人脸框与 YOLO 姿态头部关键点（鼻尖/双眼）交叉验证：
    两个独立模型在同一位置都找到"头部"，误检概率极低。
    kps 为 17 点 [(x, y, conf)]，None 表示无关键点。"""
    if not kps:
        return False
    fx, fy, fw, fh = face_box[:4]
    n_pt = kps[KP_NOSE]
    le = kps[KP_LEYE]
    re = kps[KP_REYE]
    good = 0
    for (kx, ky, kc) in (n_pt, le, re):
        if kc < conf_thr:
            continue
        if (fx - 0.5 * fw <= kx <= fx + 1.5 * fw
                and fy - 0.6 * fh <= ky <= fy + 1.3 * fh):
            good += 1
    return good >= 2                    # 鼻+任一眼，或双眼


def head_refine(face_box, kps, conf_thr=0.30):
    """用 YOLO 头部关键点校正人脸框。

    侧脸/低头时 YuNet 可能只框住半张脸（如仅鼻/嘴/下颌），若直接
    以该框为锚点裁剪"头+上半身"，会切掉大半个头。鼻/眼关键点由
    人体检测模型给出，描述的是整个头部位置，不受此影响：
    双眼（或鼻+单眼）足够置信时，按眼距估计完整头部框并与原框
    取并集，保证头顶、后脑不被裁掉。"""
    if not kps:
        return face_box
    fx, fy, fw, fh = face_box[:4]
    le, re, no = kps[KP_LEYE], kps[KP_REYE], kps[KP_NOSE]
    if le[2] >= conf_thr and re[2] >= conf_thr:
        ex = (le[0] + re[0]) / 2.0
        ey = (le[1] + re[1]) / 2.0
        ed = ((le[0] - re[0]) ** 2 + (le[1] - re[1]) ** 2) ** 0.5
    elif no[2] >= conf_thr and (le[2] >= conf_thr or re[2] >= conf_thr):
        e = le if le[2] >= conf_thr else re
        ex = (no[0] + e[0]) / 2.0
        ey = (no[1] + e[1]) / 2.0
        ed = ((no[0] - e[0]) ** 2 + (no[1] - e[1]) ** 2) ** 0.5 * 1.6
    else:
        return face_box
    if ed < 4:
        return face_box
    # 头部框：以双眼（或鼻-眼）中点为中心，宽 2.6 倍、高 3.4 倍眼距，
    # 上缘预留 1.2 倍眼距的头发空间
    gw, gh = 2.6 * ed, 3.4 * ed
    gx = ex - gw / 2.0
    gy = ey - 1.2 * ed
    ux0 = min(fx, gx)
    uy0 = min(fy, gy)
    ux1 = max(fx + fw, gx + gw)
    uy1 = max(fy + fh, gy + gh)
    return (int(ux0), int(uy0), int(ux1 - ux0), int(uy1 - uy0))


class YoloPoseDetector(object):
    """YOLO11n-pose 人体检测器（cv2.dnn 本地推理，onnx 离线运行）。

    单模型同时输出：整个人体框（含上半身与全身）+ 17 个 COCO
    姿态关键点（鼻尖、双眼、双肩、双肘…）。相比 MOG2 运动块：
    - 静止的人也能检出；
    - "人"由 CNN 确认，衣物/包裹/图案不会再被当作人员候选；
    - 鼻/眼关键点用于与人脸检测框交叉验证（拒绝海报、纹理误检）；
    - 双肩关键点用于把上半身裁剪锚定到真实肩宽上。"""

    def __init__(self, opts=None):
        opts = opts or {}
        self.model_path = self._find_model()
        self.net = None
        if self.model_path is None or not hasattr(cv2, "dnn"):
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
        except Exception:
            # OpenCV 在中文/非 ASCII 路径下读不了 onnx：复制到临时目录
            try:
                import shutil
                import tempfile
                tmpd = tempfile.mkdtemp(prefix="px_yolo_model_")
                dst = os.path.join(tmpd, "model.onnx")
                shutil.copyfile(self.model_path, dst)
                self.net = cv2.dnn.readNetFromONNX(dst)
            except Exception:
                self.net = None
                return
        self.inp = int(opts.get('yolo_input', 640))
        if self.inp not in (320, 416, 480, 640):
            self.inp = 640
        # 探测 onnx 模型实际支持的输入尺寸（很多导出模型固定 640×640，
        # 其他尺寸 forward 直接报错）：用户选择的尺寸不可用时自动回退
        if not self._input_ok(self.net, self.inp):
            for alt in (640, 480, 416, 320):
                if alt != self.inp and self._input_ok(self.net, alt):
                    self.inp = alt
                    break
        self.conf_thr = float(opts.get('yolo_conf', 0.35))
        self.nms_thr = 0.5
        # 纯 CPU 推理约 20~50ms/帧（输入 416~640）。
        # 不调用 setPreferableBackend/Target：默认即为 OPENCV 后端 + CPU，
        # 且 OpenCV 5 新推理引擎不支持该设置，调用只会产生 WARN。

    @staticmethod
    def _find_model():
        for d in MODEL_DIRS:
            if not os.path.isdir(d):
                continue
            for n in YOLO_POSE_NAMES:
                p = os.path.join(d, n)
                if os.path.exists(p):
                    return p
        return None

    @staticmethod
    def _input_ok(net, size):
        """试探 onnx 模型是否支持该输入尺寸（固定尺寸的导出模型
        用其他尺寸 forward 会抛异常）。"""
        try:
            blob = cv2.dnn.blobFromImage(
                np.zeros((size, size, 3), np.uint8), 1.0 / 255.0,
                (size, size), swapRB=True, crop=False)
            net.setInput(blob)
            net.forward('output0')
            return True
        except Exception:
            return False

    @property
    def available(self):
        return self.net is not None

    def status_text(self):
        if self.available:
            return "YOLO11n-pose（本地模型 %s，人体框 + 鼻/眼/肩关键点）" \
                % os.path.basename(self.model_path)
        return "不可用（models\\ 下无 %s）" % YOLO_POSE_NAMES[0]

    def detect(self, frame):
        """返回 [(person_box(x,y,w,h), score, kps)]。
        kps 为 17 点 [(x, y, conf)]（原图绝对坐标）。"""
        if self.net is None:
            return []
        H, W = frame.shape[:2]
        inp = self.inp
        s = inp / float(max(H, W))
        rw, rh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        canvas = np.full((inp, inp, 3), 114, np.uint8)
        canvas[(inp - rh) // 2:(inp - rh) // 2 + rh,
               (inp - rw) // 2:(inp - rw) // 2 + rw] = \
            cv2.resize(frame, (rw, rh))
        try:
            blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (inp, inp),
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            out = self.net.forward('output0')[0].T   # (8400, 56)
        except Exception:
            return []
        pad_x, pad_y = (inp - rw) // 2, (inp - rh) // 2
        dets = []
        for row in out:
            score = float(row[4])
            if score < self.conf_thr:
                continue
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x = (cx - w / 2.0 - pad_x) / s
            y = (cy - h / 2.0 - pad_y) / s
            bw, bh = w / s, h / s
            kps = []
            for i in range(17):
                kx, ky, kc = row[5 + i * 3], row[6 + i * 3], row[7 + i * 3]
                kps.append(((kx - pad_x) / s, (ky - pad_y) / s, float(kc)))
            dets.append(([x, y, bw, bh, score], kps))
        dets.sort(key=lambda d: -d[0][4])
        keep = []
        for d in dets:
            if all(iou_box_5(d[0], k[0]) < self.nms_thr for k in keep):
                keep.append(d)
        return [([int(v) for v in d[0][:4]], d[0][4], d[1]) for d in keep]


def iou_box_5(a, b):
    """两个 (x, y, w, h[, score]) 框的交并比。"""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


class PersonDetector:
    """固定摄像机场景的人员候选检测器：
    - 首选：YOLO11n-pose（CNN）全画面人体检测 → 人体框 + 头部关键点；
      人体框头部区域做人脸检测，人脸需通过"五官几何校验 / 姿态关键点
      交叉验证 + 脸-体位置一致性"三重确认才算真人（防海报/纹理误检）
    - 降级（models\\ 下无 yolo onnx）：MOG2 运动块 + 人脸检测的旧逻辑
    """

    def __init__(self, opts):
        self.work_w = 800
        self.min_h_ratio = max(2, opts.get('min_h', 15)) / 100.0
        self.face_scan = bool(opts.get('face_scan', True))
        self.scan_iv = 8
        self.frame_no = 0
        self.warnings = []
        # 人体检测模型（可关：use_yolo=False 时走旧运动块逻辑）
        self.use_yolo = bool(opts.get('use_yolo', True))
        self.yolo = YoloPoseDetector(opts) if self.use_yolo else None
        if self.use_yolo and (self.yolo is None or not self.yolo.available):
            self.yolo = None
            self.warnings.append(
                "未启用 YOLO 人体检测（models\\ 下没有 %s），"
                "回退到运动块 + 人脸检测的旧逻辑，误检率较高。可从 "
                "https://github.com/ultralytics/assets/releases/download/"
                "v8.4.0/yolo11n-pose.onnx 下载模型放到 models\\ 下。"
                % YOLO_POSE_NAMES[0])
        # MOG2 仅降级路径使用
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=2000, varThreshold=32, detectShadows=False)
        self.k3 = np.ones((3, 3), np.uint8)
        self.k5 = np.ones((5, 5), np.uint8)
        self.k7 = np.ones((7, 7), np.uint8)
        self.face_engine = FaceEngine()
        if not self.face_engine.available:
            self.warnings.append(
                "未找到可用的人脸检测模型（OpenCV %s，%s），将只提取上半身"
                "特征、无法判断人脸放大。请把 face_detection_yunet_2023mar.onnx "
                "或 haarcascade_*.xml 放到脚本同目录的 models\\ 下。"
                % (cv2.__version__, self.face_engine.status_text()))
        elif self.face_engine.missing:
            self.warnings.append(
                "部分人脸级联文件缺失（%s），已用可用的 %d 个继续工作。"
                % ("、".join(self.face_engine.missing),
                   len(self.face_engine.cascades)))
        self.hog_on = bool(opts.get('hog_confirm', False))
        self.hog = None
        if self.hog_on:
            if hasattr(cv2, "HOGDescriptor"):
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(
                    cv2.HOGDescriptor_getDefaultPeopleDetector())
            else:
                self.hog_on = False
                self.warnings.append(
                    "当前 OpenCV(%s) 不含 HOGDescriptor，已自动关闭 "
                    "HOG 行人确认（YOLO 人体检测不受影响）。" % cv2.__version__)

    def detect(self, frame):
        """返回 [(person_box, face_box|None, kps|None), ...]
        face_box 为 (x, y, w, h, 五官关键点|None)；kps 为 17 点姿态
        关键点 [(x, y, conf)]（仅 YOLO 路径有）。"""
        self.frame_no += 1
        if self.yolo is not None:
            return self._detect_yolo(frame)
        return self._detect_legacy(frame)

    # ---------- 首选：YOLO 人体检测 ----------

    def _detect_yolo(self, frame):
        H, W = frame.shape[:2]
        min_h = self.min_h_ratio * H
        out = []
        for (box, _score, kps) in self.yolo.detect(frame):
            x, y, w, h = box
            if h < min_h or w < 0.04 * W:
                continue
            if w * h > 0.95 * W * H:
                continue
            face = self.find_face(frame, box, kps)
            out.append((box, face, kps))
        return out

    def find_face(self, frame, box, kps=None):
        """在人员框头部区域检测人脸并做多级校验，返回绝对坐标
        (x, y, w, h, 五官关键点) 或 None。

        校验（任一不通过即拒绝，专治"非人照片被识别人脸"）：
        1) 五官几何：YuNet 关键点须符合人脸结构（眼上鼻下、比例合理）
           —— 与 YOLO 头部关键点（鼻/眼）吻合也可通过（交叉验证）；
        2) 脸-体一致：人脸必须位于该人体框顶部、水平居中、宽度成比例。
        """
        if not self.face_engine.available:
            return None
        H, W = frame.shape[:2]
        x, y, w, h = box
        # 头部区域：人体框上部（人脸可能略高于框顶，向上留余量）
        x0 = max(0, int(x - 0.10 * w))
        y0 = max(0, int(y - 0.25 * h))
        x1 = min(W, int(x + 1.10 * w))
        y1 = min(H, int(y + 0.50 * h))
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        # minNeighbors 逐级放宽重试只对 Haar 有意义；YuNet/MediaPipe
        # 忽略该参数，重复调用只是白白多跑几遍（严重拖慢每帧耗时）
        tries = (3, 2, 1) if self.face_engine.mode == 'haar' else (3,)
        for nb in tries:
            faces = self.face_engine.find_in_roi(roi, min_neighbors=nb)
            for (fx, fy, fw, fh, lms) in faces:
                # 1) 五官几何校验（ROI 坐标系）或跨模型头部关键点互证
                if not (landmarks_valid(lms, (fx, fy, fw, fh))
                        or kps_head_consistent(
                            (fx + x0, fy + y0, fw, fh), kps)):
                    continue
                abs_lms = [(px + x0, py + y0) for px, py in lms] \
                    if lms else None
                # 2) 人脸必须位于真实人体框的头部位置
                if not face_matches_person((fx + x0, fy + y0, fw, fh), box):
                    continue
                # 3) 用姿态关键点校正侧脸时 YuNet 的半脸框，
                #    避免裁剪锚点偏移切掉头顶
                rbx, rby, rbw, rbh = head_refine(
                    (int(fx) + x0, int(fy) + y0, int(fw), int(fh)), kps)
                return (rbx, rby, rbw, rbh, abs_lms)
        return None

    # ---------- 降级：运动块 + 人脸（无 YOLO 模型时） ----------

    def _detect_legacy(self, frame):
        H, W = frame.shape[:2]
        work = frame
        if W > self.work_w:
            work = cv2.resize(
                frame, (self.work_w, int(round(H * self.work_w / W))))
        s = W / float(work.shape[1])
        mask = self.bg.apply(work)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k7)
        mask = cv2.dilate(mask, self.k5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        min_h = self.min_h_ratio * H
        min_w = 0.04 * W
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            x, y, w, h = int(x * s), int(y * s), int(w * s), int(h * s)
            if h < min_h or w < min_w:
                continue
            if w * h > 0.95 * W * H:
                continue
            boxes.append((x, y, w, h))
        boxes = nms(boxes, 0.35)
        out = []
        for b in boxes:
            face = self.find_face_legacy(frame, b)
            if face is None and self.hog_on and not self.hog_confirm(frame, b):
                continue
            out.append((b, face, None))
        # 全画面人脸扫描：弥补运动检测漏掉的静止人员
        if self.face_scan and self.face_engine.available \
                and self.frame_no % self.scan_iv == 0:
            for fb, face in self.scan_faces(frame):
                if all(iou(fb, b) < 0.2 for b, _, _ in out):
                    out.append((fb, face, None))
        return out

    def find_face_legacy(self, frame, box):
        """旧路径：运动框内检测人脸（含五官校验，仍优于原版），
        返回 (x, y, w, h, lms) 或 None。"""
        if not self.face_engine.available:
            return None
        H, W = frame.shape[:2]
        x, y, w, h = box
        x0 = max(0, int(x - 0.25 * w))
        y0 = max(0, int(y - 0.20 * h))
        x1 = min(W, int(x + 1.25 * w))
        y1 = min(H, int(y + 1.0 * h))
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        # 同 find_face：逐级放宽重试只对 Haar 有意义
        tries = (3, 2, 1) if self.face_engine.mode == 'haar' else (3,)
        for nb in tries:
            faces = self.face_engine.find_in_roi(roi, min_neighbors=nb)
            first_no_lms = None
            for det in faces:
                fx, fy, fw, fh, lms = det[:5]
                if landmarks_valid(lms, (fx, fy, fw, fh)):
                    return (int(fx) + x0, int(fy) + y0, int(fw), int(fh),
                            [(px + x0, py + y0) for px, py in lms])
                if first_no_lms is None and lms is None:
                    first_no_lms = det
            # Haar 无关键点、无法做五官校验：直接采用最大框
            if first_no_lms is not None:
                fx, fy, fw, fh = first_no_lms[:4]
                return (int(fx) + x0, int(fy) + y0, int(fw), int(fh), None)
        return None

    def scan_faces(self, frame):
        """全画面人脸扫描（降级路径用），由人脸框推算人员框。
        返回 [(person_box, face_box)]"""
        if not self.face_engine.available:
            return []
        H, W = frame.shape[:2]
        if W > 640:
            sw = 640
            sh = int(round(H * sw / W))
            small = cv2.resize(frame, (sw, sh))
            f = W / float(sw)
        else:
            small = frame
            f = 1.0
        faces = self.face_engine.find_in_roi(small)
        res = []
        for det in faces[:3]:
            sfx, sfy, sfw, sfh = det[:4]
            lms = det[4]
            fx, fy, fw, fh = sfx * f, sfy * f, sfw * f, sfh * f
            px = max(0, int(fx - 1.1 * fw))
            py = max(0, int(fy - 0.3 * fh))
            pw = min(int(3.2 * fw), W - px)
            ph = min(int(6.5 * fh), H - py)
            if pw < 0.04 * W or ph < 0.1 * H:
                continue
            res.append(((px, py, pw, ph),
                        (int(fx), int(fy), int(fw), int(fh),
                         [(px2 * f, py2 * f) for px2, py2 in lms]
                         if lms else None)))
        return res

    def hog_confirm(self, frame, box):
        """HOG 行人确认（降级路径可选，较慢，仅 OpenCV 4.x 可用）。"""
        if self.hog is None:
            return False
        H, W = frame.shape[:2]
        x, y, w, h = box
        x0 = max(0, x - 30)
        y0 = max(0, y - 30)
        x1 = min(W, x + w + 30)
        y1 = min(H, y + h + 30)
        patch = frame[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        if ph < 128 or pw < 64:
            return False
        found, _ = self.hog.detectMultiScale(patch, winStride=(8, 8),
                                             padding=(8, 8), scale=1.03)
        return len(found) > 0


# 框仍"新鲜"的最大丢失次数：超过后轨迹冻结（不显示、不抓拍、
# 也不再参与新检测的匹配），保留到 max_missed 仅为定稿已有候选。
FRESH_MISSED = 12


class Track(object):
    __slots__ = ("tid", "box", "face", "missed", "hits", "best",
                 "face_hist", "size_hist", "approached", "away_run",
                 "growth", "last_face", "last_box", "kps", "max_depth")

    def __init__(self, tid, box, face, fi=-1, kps=None):
        self.tid = tid
        self.box = box
        self.face = face
        self.kps = kps       # 17 点姿态关键点（YOLO 路径，可为 None）
        self.missed = 0
        self.hits = 1
        self.best = None
        self.last_face = None      # 最近一次确实检测到的人脸框
        self.last_box = None       # 与之对应的人员框（用于位置推算）
        # 人脸框历史 (fi, face_w, face_h)，用于判断"朝镜头靠近"
        self.face_hist = deque(maxlen=64)
        # 人员框高度历史 (fi, box_h)，人脸检测不稳定时的降级依据
        self.size_hist = deque(maxlen=64)
        self.size_hist.append((fi, box[3]))
        # 一旦确认过"朝镜头靠近"就锁定，之后即使人脸停止放大也继续抓拍
        # （走到最近处时人脸最大最清晰，正是最佳抓拍时机）
        self.approached = False
        self.away_run = 0        # 连续"明显远离"的计数
        self.growth = 0.0        # 最近一次增长率（界面诊断用）
        # 历史最大深度：人体框底部（脚部）到达过的最深画面 y 坐标。
        # 闸机口在画面中位置固定——"实际进入闸机"的人脚部必须到达过
        # 闸机口那条线以下，折返旅客与在闸机旁维持秩序的安检员都到不了
        self.max_depth = float(box[1] + box[3])
        if face is not None:
            self.face_hist.append((fi, face[2], face[3]))

    def observe_face(self, fi, face, box=None):
        if face is not None:
            self.face_hist.append((fi, face[2], face[3]))
            self.last_face = tuple(face)
            self.last_box = tuple(box) if box is not None else tuple(self.box)

    def observe_size(self, fi, box):
        self.size_hist.append((fi, box[3]))
        self.max_depth = max(self.max_depth, float(box[1] + box[3]))

    @staticmethod
    def _trend(vals):
        """对序列做 3 点平滑后计算趋势指标。

        返回 (前段均值, 后段均值, 端到端增长, 线性趋势增长)。
        序列为整数像素时容易原地抖动，故先平滑再比较。"""
        a = np.asarray(vals, dtype=float)
        n = len(a)
        if n >= 5:
            k = np.ones(3) / 3.0
            s = np.convolve(a, k, mode='same')
            s[0], s[-1] = a[0], a[-1]
        else:
            s = a
        m = max(1, n // 3)
        early = float(np.mean(s[:m]))
        late = float(np.mean(s[-m:]))
        if early <= 1.0:
            return early, late, 0.0, 0.0
        end2end = (float(s[-1]) - float(s[0])) / float(s[0])
        slope = float(np.polyfit(np.arange(n, dtype=float), s, 1)[0]) \
            if n >= 3 else 0.0
        trend = slope * (n - 1) / early
        return early, late, end2end, trend

    def _grown(self, hist, growth_thr, min_samples, window):
        """判断某个尺寸序列是否呈明显放大趋势。返回 (是否放大, 增长率)。"""
        vals = [h[1] for h in list(hist)[-max(window, min_samples):]]
        n = len(vals)
        if n < min_samples:
            return False, 0.0
        early, late, end2end, trend = self._trend(vals)
        if early <= 1.0:
            return False, 0.0
        growth = (late - early) / early
        # 三条判据取其一：分段均值增长 / 端到端增长 / 线性回归趋势
        best = max(growth, end2end, trend)
        if best < growth_thr:
            return False, growth
        # 绝对增量太小（可能只是噪声抖动）
        if late - early < 2.0 and (vals[-1] - vals[0]) < 3:
            return False, growth
        # 轻微放宽：要求后段至少 60% 的样本不低于前段均值
        m = max(1, n // 3)
        smooth = np.asarray(vals, dtype=float)
        if n >= 5:
            k = np.ones(3) / 3.0
            smooth = np.convolve(smooth, k, mode='same')
            smooth[0], smooth[-1] = vals[0], vals[-1]
        above = sum(1 for v in smooth[-m:] if v >= early * 0.98)
        if above < 0.6 * m and vals[-1] < early * (1.0 + growth_thr):
            return False, growth
        return True, growth

    def approaching(self, growth_thr=0.10, min_samples=4, window=30):
        """是否"朝镜头靠近"（人脸/人体逐渐放大）。

        优先用人脸宽度序列；人脸样本不足时降级使用人员框高度序列
        （阈值更严，避免把横向移动误判成靠近）。一旦历史上确认过
        靠近，则保持 True（走到最近处人脸最大，仍应继续抓拍）。
        返回 (是否接近, 增长率)。"""
        ok, g = self._grown(self.face_hist, growth_thr, min_samples, window)
        if not ok:
            # 降级：用人员框高度判定（人脸检测不稳定的真实画面）
            ok2, g2 = self._grown(self.size_hist, growth_thr * 1.5,
                                  max(min_samples, 8), window)
            if ok2:
                ok, g = True, g2
        self.growth = g
        if ok:
            self.approached = True
            self.away_run = 0
        elif self.approached:
            # 已确认靠近过：只有明显持续远离才解锁，否则继续抓拍
            # （用较短窗口看最近的脸宽变化，反应更快）
            vals = [h[1] for h in
                    list(self.face_hist)[-(max(10, min_samples * 2)):]]
            if len(vals) >= max(6, min_samples):
                early, late, _, _ = self._trend(vals)
                if early > 1.0 and (late - early) / early < -growth_thr:
                    self.away_run += 1
                else:
                    self.away_run = 0
            if self.away_run < max(10, min_samples * 2):
                ok = True
            else:
                self.approached = False
                self.away_run = 0
        return self.approached, g


def box_affinity(box_a, box_b):
    """两个框的匹配度：IoU 与"中心距离 + 尺寸比"取较大者。

    人员走近时框会快速放大，纯 IoU 容易跟丢，故补充：
    新框中心落在旧框内（或很近）且尺寸比合理时同样算匹配。"""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    # 尺寸骤缩否决（先于 IoU 与保底匹配）：同一人的框高度在相邻两次
    # 检测间缓变（走近放大/走远缩小，单次间隔内变化通常 <20%）。
    # 前方旅客出画面后其轨迹大框冻结在镜头前，身后较远的旅客框
    # 要小得多——若放行匹配，轨迹会跳到后人身上，用后人的照片
    # 顶替前人，导致前人漏采。高度骤缩过半即判不同人。
    if ah > 0 and bh < 0.5 * ah:
        return 0.0
    v = iou(box_a, box_b)
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    diag = max(1.0, (max(aw, bw) ** 2 + max(ah, bh) ** 2) ** 0.5)
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    if dist > 0.6 * diag:
        return v
    rw = bw / max(aw, 1.0)
    rh = bh / max(ah, 1.0)
    if 0.4 <= rw <= 2.8 and 0.4 <= rh <= 2.8:
        v = max(v, 0.30)
    return v


class SimpleTracker(object):
    """基于框匹配的多目标跟踪器。"""

    def __init__(self, iou_thr=0.22, max_missed=40):
        """iou_thr：匹配阈值；max_missed：检测轮次内允许的最大丢失次数
        （检测间隔为 N 帧时，实际帧数 = max_missed × N），用于在人员
        静止不动、被背景模型吸收后仍能继续抓拍。"""
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.tracks = []
        self._next = 1

    def update(self, dets, fi=-1):
        """dets: [(box, face|None, kps|None)]，fi 为当前帧号
        （记入人脸历史）。返回本回合因消失而定稿的轨迹列表。

        匹配采用全局贪心一对一：所有（轨迹, 检测）组合按匹配度
        降序依次占用，每条轨迹、每个检测至多匹配一次。否则同一个人
        会被多条轨迹同时锁定（重复轨迹），白白重复做人脸校验。"""
        # 兼容二元组输入（旧调用方式）
        dets = [(d[0], d[1], d[2] if len(d) > 2 else None) for d in dets]
        pairs = []
        for ti, t in enumerate(self.tracks):
            if t.missed > FRESH_MISSED:
                continue    # 冻结的旧轨迹不再抢走新检测，避免匹配抢占
            for j, (box, _face, _kps) in enumerate(dets):
                v = box_affinity(t.box, box)
                if v > self.iou_thr:
                    pairs.append((v, ti, j))
        pairs.sort(key=lambda p: -p[0])
        used_t, used_d = set(), set()
        for _v, ti, j in pairs:
            if ti in used_t or j in used_d:
                continue
            t = self.tracks[ti]
            t.box = dets[j][0]
            t.face = dets[j][1]
            t.kps = dets[j][2]
            t.missed = 0
            t.hits += 1
            t.observe_face(fi, t.face, t.box)
            t.observe_size(fi, t.box)
            used_t.add(ti)
            used_d.add(j)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                t.face = None
        for j, (box, face, kps) in enumerate(dets):
            if j not in used_d:
                self.tracks.append(
                    Track(self._next, box, face, fi, kps=kps))
                self._next += 1
        done = [t for t in self.tracks
                if t.missed > self.max_missed and t.best is not None]
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return done

    def finalize_all(self):
        done = [t for t in self.tracks if t.best is not None]
        self.tracks = []
        return done


class PersonRegistry(object):
    """人员登记表：跨片段外观合并，同一人只保留一张最优照片。

    身份判定优先用 SFace 人脸特征（128 维，余弦相似度），能从
    根本上区分着装相近（同款制服/深色外套）的不同人；当候选或已存
    人员缺少人脸特征时，回退到上半身 HSV 直方图相关性。"""

    def __init__(self, merge_thr=0.5, time_window_min=2.0):
        self.persons = []
        self.merge_thr = merge_thr
        # 同人合并时间窗（秒）：候选抓拍时间与已登记时间相差超过
        # 该值时不再做身份比对，直接登记为新人员——间隔很久的两人
        # 即使特征相似也大概率不是同一次过闸，宁可重复登记也不冒
        # 跨时段误并风险（低质量人脸特征不可靠）。0=不限制
        self.time_window = float(time_window_min or 0) * 60.0
        self.fr = None
        # 懒加载 SFace 人脸识别器（OpenCV 5 自带 FaceRecognizerSF）
        if hasattr(cv2, 'FaceRecognizerSF'):
            path = None
            for d in MODEL_DIRS:
                if not os.path.isdir(d):
                    continue
                for n in SFACE_NAMES:
                    p = os.path.join(d, n)
                    if os.path.exists(p):
                        path = p
                        break
                if path:
                    break
            if path:
                try:
                    self.fr = cv2.FaceRecognizerSF.create(path, "")
                except Exception:
                    self.fr = None

    def _embedding(self, cand):
        """从候选的脸部图 + 关键点计算 SFace 特征；不可用返回 None。"""
        if self.fr is None:
            return None
        img = cand.get('face')
        box = cand.get('face_box')      # 脸框在脸部裁剪图坐标系 (x,y,w,h)
        lm5 = cand.get('face_lm5')      # 5 个五官关键点在裁剪图坐标系
        if img is None or box is None or lm5 is None:
            return None
        h, w = img.shape[:2]
        if h < 16 or w < 16:
            return None
        try:
            x, y, fw, fh = box
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            det = np.array([x, y, fw, fh,
                            lm5[0][0], lm5[0][1], lm5[1][0], lm5[1][1],
                            lm5[2][0], lm5[2][1], lm5[3][0], lm5[3][1],
                            lm5[4][0], lm5[4][1], 1.0], dtype=np.float32)
            aligned = self.fr.alignCrop(img, det)
            return self.fr.feature(aligned)
        except Exception:
            return None

    def _compare(self, emb, hist, p, has_face):
        """返回 (是否同一人, 相似度)。两者都有脸特征时用 SFace 余弦相似度。

        有人脸却拿不到脸特征时不再退回直方图相关性：上半身颜色
        直方图会把穿相似衣服的不同人误判成同一人，后者覆盖前者的
        照片（错人入库比重复入库更糟）。仅当双方都没有人脸图时
        才用直方图判断。"""
        p_emb = p.get('emb')
        if emb is not None and p_emb is not None:
            try:
                c = self.fr.match(
                    emb, p_emb, cv2.FaceRecognizerSF_FR_COSINE)
                if c >= SFACE_MATCH_THR and self._hist_conflict(hist, p):
                    # SFace 判同但上半身外观强烈冲突：严重模糊/偏色/
                    # 侧脸的低质量脸部图会产生失真特征，与完全不同的
                    # 人也能算出 0.8+ 的假高相似度（实测绿光糊脸与
                    # 白衣路人 0.829）。同一视频里同一人衣着不变，
                    # 颜色直方图强冲突时宁可另立新人，不误并。
                    return False, c
                return (c >= SFACE_MATCH_THR), c
            except Exception:
                pass
        if self.fr is not None and (has_face or p.get('face') is not None):
            return False, -1.0     # 有脸可比对却缺特征：宁可另立新人
        c = cv2.compareHist(hist, p['hist'], cv2.HISTCMP_CORREL)
        return (c >= self.merge_thr), c

    @staticmethod
    def _hist_conflict(hist, p):
        """上半身 HSV 直方图是否与已存人员强烈冲突（可信的"不是同一人"证据）。
        只否决极端情形（如黑衣 vs 白衣），正常光线波动下的同人不会误伤。"""
        ph = p.get('hist')
        if hist is None or ph is None:
            return False
        try:
            rel = cv2.compareHist(hist, ph, cv2.HISTCMP_CORREL)
        except Exception:
            return False
        return rel < 0.15

    def submit(self, cand):
        """提交一个人员的最优候选。返回 (person, 是否新增或更新, 被替换的旧文件路径)。"""
        cand['hist'] = hist_of(cand['upper'])
        # 注意：这里必须用下标定位，不能用 list.index(best_p)——
        # 人员字典里含 numpy 图像数组，== 比较会触发 numpy 广播，
        # 两张人像尺寸不同时会抛 ValueError（cv2 的 numpy 数组之间
        # 逐元素比较），导致整个处理线程中断、界面进度卡死。
        cand['emb'] = self._embedding(cand)
        best_i, best_c = -1, -2.0
        cand_ts = cand.get('ts')
        for i, p in enumerate(self.persons):
            # 登记时间隔太久：不再做身份比对，直接视为新人员
            if (self.time_window > 0 and cand_ts is not None
                    and p.get('ts') is not None
                    and abs((cand_ts - p['ts']).total_seconds())
                    > self.time_window):
                continue
            same, c = self._compare(cand['emb'], cand['hist'], p,
                                    cand.get('face') is not None)
            if same and c > best_c:
                best_c, best_i = c, i
        if best_i >= 0:
            best_p = self.persons[best_i]
            if cand['score'] > best_p['score']:
                cand['id'] = best_p['id']
                old_paths = [best_p.get(k) for k in
                             ('upper_path', 'face_path', 'face_lm_path',
                              'full_path')]
                self.persons[best_i] = cand
                return cand, True, old_paths
            return best_p, False, []
        cand['id'] = "P%03d" % (len(self.persons) + 1)
        self.persons.append(cand)
        return cand, True, []


# ============================================================
# 第二部分：PyQt5 图形界面与处理线程
# ============================================================

HAS_QT = True
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIcon
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
        QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QPlainTextEdit,
        QListWidget, QListWidgetItem, QProgressBar, QSplitter, QVBoxLayout,
        QHBoxLayout, QGroupBox, QFileDialog, QMessageBox, QFormLayout,
        QScrollArea, QAbstractItemView, QSlider
    )
except ImportError:
    HAS_QT = False

if HAS_QT:

    class ProcessorThread(QThread):
        """后台批量处理线程：读视频 → 旋转 → 检测 → 跟踪 →
        只抓拍 ROI 内最前面（目标框最大）且正朝镜头靠近
        （人脸逐渐放大）的人 → 打分 → 保存最优。"""
        sig_frame = pyqtSignal(object, object)      # bgr 帧, 检测列表
        sig_person = pyqtSignal(object)             # 新增/更新的人员信息
        sig_progress = pyqtSignal(float, str)       # 总进度 0~1, 当前视频名
        sig_pos = pyqtSignal(int, int, float)       # 当前帧号, 总帧数, fps
        sig_log = pyqtSignal(str)
        sig_finished = pyqtSignal()

        def __init__(self, files, roi, opts, out_dir, start_frames=None,
                     publisher=None, mqtt_enabled=True, stream_factory=None):
            super().__init__()
            self.files = list(files)
            self.roi = dict(roi)
            self.opts = dict(opts)
            self.out_dir = out_dir
            self.stream_factory = stream_factory
            # 各视频的起始帧（界面进度条指定的快进起点）
            self.start_frames = list(start_frames) if start_frames \
                else [0] * len(self.files)
            self._stop = False
            self._seek_target = -1   # 处理中拖动进度条 → 快进到该帧
            self.reg = None
            self._frame_h = 0
            self._diag_last = {}
            self._no_face_hinted = False
            # MQTT 实时推送：broker 不可达会自动禁用，不影响主提取流程
            self.mqtt = publisher
            if self.mqtt is None and mqtt_enabled and MqttPublisher is not None:
                try:
                    self.mqtt = MqttPublisher(out_dir=self.out_dir)
                except Exception as e:
                    self.sig_log.emit("MQTT 初始化失败（将不推送）：%s" % e)

        def stop(self):
            self._stop = True

        def request_seek(self, frame):
            """处理中把当前视频快进到指定帧（线程安全：仅写一个 int）。"""
            self._seek_target = max(0, int(frame))

        # ---------- 工具 ----------

        def _roi_rect(self, shape):
            """把 ROI 归一化矩形映射为像素矩形 (x, y, w, h)。"""
            H, W = shape[:2]
            r = self.roi
            return (r['x'] * W, r['y'] * H, r['w'] * W, r['h'] * H)

        def _in_roi(self, box, rect):
            """人员框中心是否落在 ROI 矩形内。"""
            x, y, w, h = box
            cx, cy = x + w / 2.0, y + h / 2.0
            rx, ry, rw, rh = rect
            return rx <= cx <= rx + rw and ry <= cy <= ry + rh

        def _make_candidate(self, frame, track, ts, video_name,
                            detector=None):
            """构造"头 + 上半身"整体人像候选。

            以检测到的人脸为锚点统一裁剪（头顶上方留头发、下方裁到
            腰线、两侧含双肩），保证不出现单独的手部/腰部等局部；
            检测不到人脸、人脸太小、人像被画面裁掉太多或框内没有
            对应运动能量时返回 None（不抓拍）。"""
            H, W = frame.shape[:2]
            min_sharp = float(self.opts.get('min_sharp', 30) or 0)
            prefer_static = bool(self.opts.get('prefer_static', True))
            face = track.face
            require_face = self.opts.get('require_face', True)
            min_face_w = self.opts.get('min_face_w', 30)
            if face is not None and face[2] < min_face_w:
                return None                    # 人脸太小：人还远，不拍
            if face is None:
                if require_face:
                    return None                # 必须看到脸，否则不算完整人像
                # 降级模式：按运动/人体框上部裁剪（不保证含头部）
                x, y, w, h = track.box
                ux0 = max(0, int(x - 0.05 * w))
                uy0 = max(0, int(y - 0.03 * h))
                ux1 = min(W, int(x + 1.05 * w))
                uy1 = min(H, int(y + 0.68 * h))
                upper_box = (ux0, uy0, ux1 - ux0, uy1 - uy0)
            else:
                fx, fy, fw, fh = face[:4]
                # 人贴镜头导致脸被画面边缘切掉时，无法得到完整人脸，
                # 这一帧不拍（其余帧会更完整）。判定从严：脸框必须
                # 完整落在画面内，且五官关键点（若有）也不得越界，
                # 否则保存的会是"半张脸"。
                if (fx < 0 or fy < 0
                        or fx + fw > W or fy + fh > H):
                    return None
                lm5 = face[4] if len(face) > 4 else None
                if lm5 and any(px < 0 or py < 0 or px >= W or py >= H
                               for px, py in lm5):
                    return None
                cx = fx + fw / 2.0
                ux0 = int(cx - 1.25 * fw)       # 两侧含双肩（约 2.5 倍脸宽）
                ux1 = int(cx + 1.25 * fw)
                uy0 = int(fy - 0.55 * fh)       # 头顶（含头发）
                uy1 = int(fy + fh + 3.4 * fh)   # 上半身至腰线
                # 头部姿态锚点（修正裁剪宽度/高度，保证完整上半身入镜）：
                # 首选 YOLO 姿态关键点（鼻尖 + 双肩），MediaPipe 为备用。
                # 鼻尖必须落在本脸框内，避免用错旁边人的关键点。
                anchors = []
                kps = getattr(track, 'kps', None)
                if kps is not None:

                    def _kp(i):
                        kx, ky, kc = kps[i]
                        return (kx, ky) if kc > 0.3 else None

                    anchors.append((_kp(KP_NOSE), _kp(KP_LSHOULDER),
                                    _kp(KP_RSHOULDER)))
                if HAS_MP:
                    pinfo = pose_upper(frame)
                    if pinfo:
                        anchors.append((pinfo['nose'], pinfo['shoulder_l'],
                                        pinfo['shoulder_r']))
                for nose, shl, shr in anchors:
                    if nose is None:
                        continue
                    if not ((fx - 0.5 * fw) <= nose[0] <= (fx + 1.5 * fw)
                            and (fy - 0.5 * fh) <= nose[1]
                            <= (fy + 1.5 * fh)):
                        continue
                    m = 0.30 * fh
                    if shl:
                        ux0 = min(ux0, int(shl[0] - m))
                    if shr:
                        ux1 = max(ux1, int(shr[0] + m))
                    break
                # YOLO 人体框约束上下界：人很近、框只到腰部/画面边时，
                # 不再向下裁出下半身。上界钳制不允许切到脸——人体框顶
                # 偏低（低头/头部被遮挡导致关键点框从颈部开始）时，
                # 放弃钳制，保住头顶
                pbx, pby, pbw, pbh = track.box
                uy0 = max(uy0, min(int(pby - 0.10 * fh), int(fy - 0.20 * fh)))
                if pby + pbh < uy1:
                    uy1 = int(pby + pbh + 0.10 * fh)
                vx0, vy0 = max(0, ux0), max(0, uy0)
                vx1, vy1 = min(W, ux1), min(H, uy1)
                if vx1 - vx0 < 16 or vy1 - vy0 < 16:
                    return None
                # 裁剪框被画面边界/人体框钳制后不得切到脸，
                # 确保保存的上半身照里脸是完整的
                if (vx0 > fx or vy0 > fy
                        or vx1 < fx + fw or vy1 < fy + fh):
                    return None
                exp = max(1, (ux1 - ux0) * (uy1 - uy0))
                vis = (vx1 - vx0) * (vy1 - vy0)
                if vis < 0.72 * exp:
                    return None                 # 人被画面裁掉太多，不完整
                # 运动能量应主要来自这个人（排除只框到局部肢体的情形）
                bx, by, bw, bh = track.box
                ix0, iy0 = max(vx0, bx), max(vy0, by)
                ix1, iy1 = min(vx1, bx + bw), min(vy1, by + bh)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                if inter < 0.25 * vis:
                    return None
                upper_box = (vx0, vy0, vx1 - vx0, vy1 - vy0)
            ux0, uy0, uw, uh = upper_box
            if uw < 16 or uh < 16:
                return None
            upper = frame[uy0:uy0 + uh, ux0:ux0 + uw].copy()
            # 清晰度（Laplacian 方差）：低于下限直接不抓拍，避免保存模糊图
            _gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(_gray, cv2.CV_64F).var())
            if min_sharp > 0 and sharp < min_sharp:
                return None
            # 一图两脸：裁剪出的上半身照里若还有第二张不重叠的人脸
            # （旁边/身后的旅客），保存出来会带上别人的脸，这张不要
            eng = getattr(detector, 'face_engine', None) \
                if detector is not None else None
            if eng is not None and eng.available:
                kept = []
                for f in eng.find_in_roi(upper, min_neighbors=2):
                    if all(iou(f[:4], k[:4]) < 0.3 for k in kept):
                        kept.append(f)
                if len(kept) >= 2:
                    return None
            # 静止优先：用人员框高度近期变化率近似运动强度，越静止加分越多
            static_bonus = 0.0
            if prefer_static:
                hs = [hh[1] for hh in list(track.size_hist)[-8:]]
                if len(hs) >= 4:
                    m = max(1, len(hs) // 3)
                    early = float(np.mean(hs[:m]))
                    late = float(np.mean(hs[-m:]))
                    if early > 1.0:
                        motion = abs(late - early) / early
                        static_bonus = 1.0 * max(0.0, 1.0 - motion / 0.10)
            face_img = None
            face_lm5 = None      # 脸部图坐标系下的五官关键点（描线图用）
            face_roi = None
            face_box = None      # 脸框在脸部裁剪图坐标系（SFace 特征用）
            if face is not None:
                fx, fy, fw, fh = face[:4]
                fx0 = max(0, int(fx - 0.35 * fw))
                fy0 = max(0, int(fy - 0.35 * fh))
                fx1 = min(W, int(fx + 1.35 * fw))
                fy1 = min(H, int(fy + 1.35 * fh))
                if fx1 - fx0 >= 16 and fy1 - fy0 >= 16:
                    face_img = frame[fy0:fy1, fx0:fx1].copy()
                    lm5 = face[4] if len(face) > 4 else None
                    if lm5:
                        face_lm5 = [(px - fx0, py - fy0) for px, py in lm5]
                        face_roi = (fx0, fy0)
                        # 人脸框在裁剪图坐标系内（SFace 生成特征用）
                        face_box = (fx - fx0, fy - fy0, fw, fh)
            roi_rect = self._roi_rect(frame.shape)
            # 脸部裁剪图单独测清晰度：整图 sharp 会被衣服纹理抬高，
            # 脸糊不糊只有看脸本身才知道；低于 min_sharp 直接放弃
            face_sharp = None
            if face_img is not None:
                _fgray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_sharp = float(cv2.Laplacian(_fgray, cv2.CV_64F).var())
                if min_sharp > 0 and face_sharp < min_sharp:
                    return None
            score = candidate_score(upper, upper_box, face, roi_rect,
                                    frame.shape, static_bonus, face_sharp)
            full = frame.copy()
            cv2.rectangle(full, (ux0, uy0), (ux0 + uw, uy0 + uh),
                          (0, 220, 0), 2)
            if face is not None:
                fx, fy, fw, fh = face[:4]
                cv2.rectangle(full, (fx, fy), (fx + fw, fy + fh),
                              (255, 180, 0), 2)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(full, ts_str, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            return {'upper': upper, 'face': face_img, 'full': full,
                    'score': score, 'sharp': sharp, 'ts': ts, 'ts_str': ts_str,
                    'video': video_name, 'face_lm5': face_lm5,
                    'face_roi': face_roi, 'face_box': face_box,
                    'face_sharp': face_sharp}

        def _face_overlay(self, p):
            """为保存的人脸图生成五官描线图。
            有 MediaPipe：468 点网格 + 五官轮廓（本身也是二次校验——
            画不出网格的"人脸"多半是误检）。
            无 MediaPipe：用检测时的五官关键点（双眼/鼻尖/双嘴角）
            描点连线，同样能核对五官是否齐全。返回 None 表示无法生成。"""
            img = p.get('face')
            if img is None:
                return None
            if HAS_MP:
                ov = face_landmark_overlay(img)
                if ov is not None:
                    return ov
            lm5 = p.get('face_lm5')
            if not lm5:
                return None
            ann = img.copy()
            pts = [(int(round(px)), int(round(py))) for px, py in lm5]
            # 双眼连线 / 双嘴角连线，五个关键点画圈标记
            cv2.line(ann, pts[0], pts[1], (255, 200, 0), 2)
            cv2.line(ann, pts[3], pts[4], (255, 255, 0), 2)
            for (px, py) in pts:
                cv2.circle(ann, (px, py), 6, (0, 255, 255), 2)
                cv2.circle(ann, (px, py), 2, (0, 0, 255), -1)
            return ann

        def _passed_gate(self, t):
            """是否实际进入安检闸机（轨迹定稿提交时把关）。

            唯一判定条件——闸机口虚线：以"距画面底部的百分比"表示
            （界面画面上会画出这条橙色虚线，可拖动调整，默认 10%）。
            闸机口在画面中位置固定，真正进入闸机的人，脚部（人体框
            底部）必须到达过该线以下；折返回去的旅客、以及在闸机旁
            维持秩序却没进去的安检员，脚部都到不了这里。
            设为 0 表示不启用该判定。"""
            gate_bottom = float(self.opts.get('gate_line_bottom_pct', 0) or 0)
            if gate_bottom <= 0:
                return True
            fh = getattr(self, '_frame_h', 0) or 0
            if fh <= 0:
                return True
            line_y = (100.0 - gate_bottom) / 100.0 * fh
            return t.max_depth >= line_y

        def _submit(self, track):
            if track.best is None or self.reg is None:
                return
            # 未实际进入闸机（折返旅客、驻留安检员等）一律不入库
            if not self._passed_gate(track):
                return
            p, changed, old_paths = self.reg.submit(track.best)
            if changed:
                for op in old_paths:      # 更优照片覆盖时删除旧文件
                    if op and os.path.exists(op):
                        try:
                            os.remove(op)
                        except OSError:
                            pass
                save_person(self.out_dir, p, face_lm=self._face_overlay(p))
                # 零延迟实时推送：保存那一刻立即把人脸 + 人员ID + 时间戳发出去
                if self.mqtt is not None:
                    try:
                        self.mqtt.publish_person(
                            local_id=p['id'], capture_time=p['ts_str'],
                            face_img=p.get('face'),        # 内存 numpy 图，免落盘再读
                            upper_img=p.get('upper'),
                            source_video=p.get('video', ''),
                            score=p['score'],
                            has_face=p.get('face') is not None)
                    except Exception as e:
                        self.sig_log.emit("MQTT 发布失败：%s" % e)
                thumb_src = p.get('face')
                if thumb_src is None:
                    thumb_src = p['upper']
                th, tw = thumb_src.shape[:2]
                sc = 96.0 / max(tw, 1)
                thumb = cv2.resize(thumb_src, (96, max(1, int(th * sc))))
                self.sig_person.emit(dict(
                    id=p['id'], ts_str=p['ts_str'], score=p['score'],
                    face=p.get('face') is not None, thumb=thumb,
                    upper_path=p.get('upper_path'),
                    face_path=p.get('face_path'),
                    face_lm_path=p.get('face_lm_path'),
                    full_path=p.get('full_path')))
                self.sig_log.emit("%s 已保存/更新：%s（分数 %.1f）"
                                  % (p['id'], p['ts_str'], p['score']))

        # ---------- 主流程 ----------

        def run(self):
            try:
                self._run()
            except Exception:
                self.sig_log.emit("处理异常：\n" + traceback.format_exc())
            finally:
                if self.reg is not None:
                    save_csv(self.out_dir, self.reg.persons)
                if getattr(self, 'mqtt', None) is not None:
                    try:
                        self.mqtt.close()
                    except Exception:
                        pass
                self.sig_finished.emit()

        def _run(self):
            os.makedirs(self.out_dir, exist_ok=True)
            detector = PersonDetector(self.opts)
            for wmsg in detector.warnings:
                self.sig_log.emit("提示：" + wmsg)
            self.reg = PersonRegistry(
                self.opts.get('merge_thr', 0.5),
                time_window_min=float(
                    self.opts.get('merge_time_window', 2) or 0))
            if self.opts.get('require_approach', True):
                self.sig_log.emit(
                    "抓拍条件：ROI 内站在最前面的人，且人脸宽度增长 ≥ "
                    "%.0f%%（朝镜头移动）；判定为接近后持续抓拍，"
                    "走到最近处仍会保存最清晰的一张"
                    % (self.opts.get('face_growth', 0.08) * 100))
            else:
                self.sig_log.emit(
                    "抓拍条件：ROI 内站在最前面的人（未启用接近判定）")
            starts = ([datetime.now() for _ in self.files] if self.stream_factory
                      else [parse_start_time(f) for f in self.files])
            # 手动时间覆盖：以第一个视频的偏移量平移所有片段
            ov = (self.opts.get('time_override') or "").strip()
            if ov and self.stream_factory:
                self.sig_log.emit("实时流使用收到画面的时间，忽略文件时间覆盖")
            elif ov:
                try:
                    base = datetime.strptime(ov, "%Y%m%dT%H%M%S")
                    delta = base - starts[0]
                    starts = [s + delta for s in starts]
                    self.sig_log.emit("时间已手动覆盖为 %s 起" % ov)
                except ValueError:
                    self.sig_log.emit("时间覆盖格式无效（应为 20260923T180406），"
                                      "已改用文件名解析")
            n_files = len(self.files)
            for i, path in enumerate(self.files):
                if self._stop:
                    break
                try:
                    self._process_video(
                        i, path, starts[i], detector,
                        self.start_frames[i] if i < len(self.start_frames)
                        else 0)
                except Exception:
                    if self.stream_factory:
                        raise
                    # 单个视频出错不再中断整批任务：记录完整堆栈后继续下一个
                    self.sig_log.emit(
                        "处理 %s 时出错（已跳过该视频，继续后面的）：\n%s"
                        % (os.path.basename(path), traceback.format_exc()))
            # 收尾：把还在跟踪中的轨迹定稿
            if self._stop and self.reg is not None:
                save_csv(self.out_dir, self.reg.persons)
            self.sig_log.emit("处理结束，共登记 %d 人。输出目录：%s"
                              % (len(self.reg.persons), self.out_dir))

        def _recover_face(self, detector, frame, t):
            """上一轮检到过脸、这一轮丢失时，按人员框变化推算人脸位置，
            在预测点周围降低阈值重新锁定。真实画面里人脸检测是断续的，
            这一步能大幅提高人脸命中率与"接近"判定的样本连续性。"""
            eng = getattr(detector, 'face_engine', None)
            if eng is None or not eng.available:
                return None
            if t.last_face is None or t.last_box is None:
                return None
            H, W = frame.shape[:2]
            obx, oby, obw, obh = t.last_box
            nbx, nby, nbw, nbh = t.box
            if obw <= 0 or obh <= 0 or nbw <= 0 or nbh <= 0:
                return None
            lfx, lfy, lfw, lfh = t.last_face[:4]
            rx = (lfx + lfw / 2.0 - obx) / float(obw)     # 脸心相对位置
            ry = (lfy + lfh / 2.0 - oby) / float(obh)
            rw = lfw / float(obw)
            rh = lfh / float(obh)
            cx = nbx + rx * nbw
            cy = nby + ry * nbh
            fw = rw * nbw
            fh = rh * nbh
            if fw < 8 or fh < 8:
                return None
            x0 = max(0, int(cx - 1.15 * fw))
            x1 = min(W, int(cx + 1.15 * fw))
            y0 = max(0, int(cy - 1.15 * fh))
            y1 = min(H, int(cy + 1.15 * fh))
            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                return None
            # 同 find_face：YuNet 忽略 min_neighbors，只跑一次
            tries = (2, 1) if eng.mode == 'haar' else (2,)
            for nb in tries:
                faces = eng.find_in_roi(roi, min_neighbors=nb)
                for det in faces:
                    fx, fy, fw2, fh2 = det[:4]
                    if fw2 < 0.55 * fw or fw2 > 1.9 * fw:
                        continue               # 尺寸应接近预测值
                    lms = det[4] if len(det) > 4 else None
                    return (int(fx) + x0, int(fy) + y0, int(fw2), int(fh2),
                            [(px + x0, py + y0) for px, py in lms]
                            if lms else None)
            return None

        def _diag(self, t, name, ts, growth_thr):
            """限频输出"未接近"原因，便于调参（每个轨迹约 3 秒一次）。"""
            key = (name, t.tid)
            now = time.monotonic()
            if now - self._diag_last.get(key, -9.9) < 3.0:
                return
            self._diag_last[key] = now
            nf = len(t.face_hist)
            nb = len(t.size_hist)
            fw = t.face_hist[-1][1] if nf else 0
            bh0 = t.size_hist[0][1] if nb else 0
            bh1 = t.size_hist[-1][1] if nb else 0
            self.sig_log.emit(
                "%s 未判定接近：人脸样本=%d 当前脸宽=%dpx 增长率=%.0f%%"
                "（需≥%.0f%%）；人体框高 %d→%dpx%s"
                % (ts.strftime("%H:%M:%S"), nf, fw, t.growth * 100,
                   growth_thr * 100, bh0, bh1,
                   "（无人脸时按人体框判定，阈值×1.5）" if nf < 4 else ""))
            if nf == 0 and nb >= 12 and not self._no_face_hinted:
                self._no_face_hinted = True
                self.sig_log.emit(
                    "提示：一直检不到人脸（人脸样本=0），可能是拍摄角度/"
                    "光线/分辨率导致人脸检测器漏检。可尝试：① 取消勾选"
                    "「必须检测到人脸」改为按人体框抓拍；② 缩小「最小"
                    "有效脸宽」；③ 改用 YuNet 模型（把 onnx 放到脚本目录）")

        def _process_video(self, idx, path, start, detector, start_frame=0):
            cap = (self.stream_factory(path) if self.stream_factory
                   else cv2.VideoCapture(path))
            try:
                self._process_capture(idx, path, start, detector, cap, start_frame)
            finally:
                cap.release()

        def _process_capture(self, idx, path, start, detector, cap, start_frame=0):
            name = os.path.basename(path)
            if self.stream_factory:
                name = "闸机实时流"
            if not cap.isOpened():
                self.sig_log.emit("无法打开视频：%s（跳过）" % name)
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 1:
                fps = 25.0
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.sig_log.emit("[%d/%d] %s  起始时间 %s  fps=%.1f"
                              % (idx + 1, len(self.files), name,
                                 start.strftime("%Y-%m-%d %H:%M:%S"), fps))
            tracker = SimpleTracker()
            rotate180 = (self.opts.get('rotate', 180) == 180)
            det_iv = max(1, self.opts.get('det_interval', 1))
            fi = 0
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                fi = start_frame
                self.sig_log.emit(
                    "快进：从第 %d 帧（约 %s）开始处理"
                    % (fi, (start + timedelta(seconds=fi / fps))
                       .strftime("%H:%M:%S")))
            last_emit = 0.0
            last_progress = 0.0
            last_frame_at = time.monotonic()
            n_files = len(self.files)
            while not self._stop:
                # 处理中拖动进度条：跳到指定帧继续（时间戳按帧号换算，
                # 不受快进影响）
                if self._seek_target >= 0 and self._seek_target != fi:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_target)
                    fi = self._seek_target
                self._seek_target = -1
                ok, frame = cap.read()
                read_wait = time.monotonic() - last_frame_at
                last_frame_at = time.monotonic()
                if read_wait >= 5.0:
                    self.sig_log.emit("读取视频帧耗时 %.1f 秒，正在继续…" % read_wait)
                if not ok:
                    break
                if rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)  # 顺时针 180°
                if self._frame_h != frame.shape[0]:
                    self._frame_h = frame.shape[0]   # 闸机口虚线换算用
                ts = (datetime.fromtimestamp(cap.captured_at_ms / 1000)
                      if self.stream_factory else start + timedelta(seconds=fi / fps))
                do_det = (fi % det_iv == 0)
                if do_det:
                    detect_started = time.monotonic()
                    dets = detector.detect(frame)
                    detect_wait = time.monotonic() - detect_started
                    if detect_wait >= 2.0:
                        self.sig_log.emit("第 %d 帧检测耗时 %.1f 秒，正在继续…"
                                          % (fi, detect_wait))
                    for t in tracker.update(dets, fi):
                        self._submit(t)
                    # 人脸断续丢失时在预测位置重新锁定
                    for t in tracker.tracks:
                        if t.face is None and t.missed == 0:
                            f = self._recover_face(detector, frame, t)
                            if f is not None:
                                t.face = f
                                t.observe_face(fi, f, t.box)
                # 抓拍判定：ROI 内所有人中只取"最前面"（目标框最高、最近）的
                # 一个，且必须呈"朝镜头靠近"趋势（人脸逐渐放大）才抓拍
                rect = self._roi_rect(frame.shape)
                in_roi_tracks = [t for t in tracker.tracks
                                 if t.missed <= FRESH_MISSED
                                 and self._in_roi(t.box, rect)]
                front = None
                if in_roi_tracks:
                    front = max(in_roi_tracks,
                                key=lambda t: (t.box[3], t.box[2] * t.box[3]))
                require_approach = self.opts.get('require_approach', True)
                growth_thr = self.opts.get('face_growth', 0.08)
                min_samples = int(self.opts.get('approach_min_frames', 4))
                min_hits = int(self.opts.get('min_hits', 2))
                view = []
                for t in tracker.tracks:
                    in_roi = self._in_roi(t.box, rect)
                    is_front = (front is not None and t is front)
                    appr, growth = t.approaching(growth_thr, min_samples)
                    if is_front and not appr and require_approach:
                        # 诊断输出：让用户知道差多少，便于调阈值
                        nf = len(t.face_hist)
                        nb = len(t.size_hist)
                        if nf or nb:
                            self._diag(t, name, ts, growth_thr)
                    if t.missed <= FRESH_MISSED:
                        view.append((t.box, t.face, in_roi, t.tid, is_front,
                                     appr, growth))
                    # missed<=FRESH_MISSED：框仍新鲜（人员可能静止被背景吸收，
                    # 容忍一小段时间后停止抓拍，避免拍空场景）。
                    # 不再要求"只抓最前方"：多人紧跟通过时，前方的人
                    # 会一直占据最前方，后面的人将完全无法积累候选而
                    # 被整条漏掉；未进闸机的人由 _passed_gate 统一过滤。
                    ok = (do_det and in_roi and t.hits >= min_hits
                          and t.missed <= FRESH_MISSED)
                    if ok and require_approach and not appr:
                        ok = False    # 没有"朝镜头移动"的动作，不拍
                    if ok:
                        cand = self._make_candidate(frame, t, ts, name,
                                                    detector)
                        if cand is not None:
                            if t.best is None:
                                t.best = cand
                            elif cand['score'] > t.best.get('score', -1):
                                t.best = cand
                            else:
                                # 分数接近时保留"脸更清楚"的那张：
                                # 优先比脸部清晰度，无人脸再比整图
                                cs = cand.get('face_sharp') \
                                    or cand.get('sharp', 0)
                                bs = t.best.get('face_sharp') \
                                    or t.best.get('sharp', 0)
                                if (cs > bs and cand['score'] >
                                        t.best.get('score', -1) - 0.8):
                                    t.best = cand
                # 限频推送界面；低频更新避免 GUI 事件队列积压而显示旧画面。
                now = time.monotonic()
                if now - last_emit > 0.15:
                    self.sig_frame.emit(frame, view)
                    last_emit = now
                # 不依赖每 20 帧：检测较慢时至少每 0.5 秒反馈一次进度。
                if fi % 20 == 0 or now - last_progress >= 0.5:
                    if n_total > 0:
                        prog = (idx + fi / float(n_total)) / n_files
                        status = name
                    else:
                        prog = (idx + 0.9) / n_files
                        status = "%s（已处理 %d 帧）" % (name, fi)
                    self.sig_progress.emit(prog, status)
                    self.sig_pos.emit(fi, n_total, fps)
                    last_progress = now
                fi += 1
            # 停止时不再把尚未处理完的轨迹写出为最终结果。
            if not self._stop:
                for t in tracker.finalize_all():
                    self._submit(t)
            if not self._stop:
                self.sig_progress.emit((idx + 1) / float(n_files), name)

    class VideoLabel(QLabel):
        """视频画面显示控件：绘制矩形抓拍区域/检测框，
        支持拖动移动、拖角缩放、空白处重新框选。"""
        roiChanged = pyqtSignal(dict)

        def __init__(self):
            super().__init__()
            self.setAlignment(Qt.AlignCenter)
            self.setMinimumSize(520, 380)
            self.setStyleSheet("background:#181818;")
            self._pix = None
            self._roi = {'x': 0.2, 'y': 0.1, 'w': 0.6, 'h': 0.8}
            self._dets = []
            self._fw = 0
            self._fh = 0
            self._mode = None      # 'move' | 'nw' | 'ne' | 'sw' | 'se'
            self._anchor = (0.0, 0.0)   # 按下时的帧坐标
            self._orig = None      # 按下时的 roi 备份
            self._editable = True
            self._gate_line = 0.0    # 闸机口虚线（距画面底部百分比）

        def set_roi(self, roi):
            self._roi = dict(roi)
            self.update()

        def set_gate_line(self, pct):
            """闸机口虚线（距画面底部百分比，0=隐藏）。"""
            self._gate_line = float(pct or 0)
            self.update()

        def set_editable(self, e):
            self._editable = bool(e)
            if not e:
                self._mode = None

        def set_frame(self, bgr, dets):
            if bgr is None:
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self._fw, self._fh = w, h
            qimg = QImage(rgb.data, w, h, 3 * w,
                          QImage.Format_RGB888).copy()
            self._pix = QPixmap.fromImage(qimg)
            self._dets = dets or []
            self.update()

        def _geom(self):
            lw, lh = self.width(), self.height()
            pw, ph = self._pix.width(), self._pix.height()
            s = min(lw / float(pw), lh / float(ph))
            w, h = int(pw * s), int(ph * s)
            return (lw - w) // 2, (lh - h) // 2, s, w, h

        # ---------- 绘制 ----------

        def paintEvent(self, ev):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(24, 24, 24))
            if self._pix is None:
                p.setPen(QColor(150, 150, 150))
                p.drawText(self.rect(), Qt.AlignCenter,
                           "① 选择文件 → ② 框选抓拍区域 → ③ 开始处理")
                return
            ox, oy, s, w, h = self._geom()
            p.drawPixmap(ox, oy, w, h, self._pix)
            if self._fw <= 0 or self._fh <= 0:
                return
            # 抓拍区域矩形（半透明 + 边框 + 角点）
            roi = self._roi
            rx = ox + roi['x'] * self._fw * s
            ry = oy + roi['y'] * self._fh * s
            rw = roi['w'] * self._fw * s
            rh = roi['h'] * self._fh * s
            p.fillRect(int(rx), int(ry), int(rw), int(rh),
                       QColor(0, 200, 0, 26))
            p.setPen(QPen(QColor(0, 230, 100), 2, Qt.DashLine))
            p.drawRect(int(rx), int(ry), int(rw), int(rh))
            p.setPen(QPen(QColor(0, 230, 100), 3))
            hs = 5
            for cx, cy in ((rx, ry), (rx + rw, ry),
                           (rx, ry + rh), (rx + rw, ry + rh)):
                p.drawPoint(int(cx), int(cy))
                p.drawRect(int(cx) - hs, int(cy) - hs, hs * 2, hs * 2)
            p.setPen(QColor(0, 230, 100))
            p.drawText(int(rx) + 4, int(ry) + 14,
                       "抓拍区域（只拍区域内最前面 + 正朝镜头靠近的人）")
            # 闸机口虚线：人的脚部到达过该线以下才算实际进入闸机
            if self._gate_line > 0:
                gy = oy + (1.0 - self._gate_line / 100.0) * self._fh * s
                p.setPen(QPen(QColor(255, 140, 0), 2, Qt.DashLine))
                p.drawLine(int(rx), int(gy), int(rx + rw), int(gy))
                p.setPen(QColor(255, 170, 60))
                p.drawText(int(rx) + 4, int(gy) - 5,
                           "闸机口（脚部须到达此线以下才算进入）")
            # 检测框
            for item in self._dets:
                try:
                    self._draw_det(p, item, ox, oy, s)
                except Exception:
                    continue          # 单个目标绘制异常不影响整个界面

        def _draw_det(self, p, item, ox, oy, s):
            box, face, in_roi, tid = item[0], item[1], item[2], item[3]
            is_front, appr = item[4], item[5]
            growth = item[6] if len(item) > 6 else 0.0
            bx, by = ox + box[0] * s, oy + box[1] * s
            bw, bh = box[2] * s, box[3] * s
            gtxt = " %+d%%" % int(round(growth * 100))
            if is_front and appr:
                color = QColor(255, 60, 60)
                tag = "ID%d [抓拍对象·接近中%s]" % (tid, gtxt)
            elif is_front:
                color = QColor(255, 200, 0)
                tag = "ID%d [最前·未接近%s]" % (tid, gtxt)
            elif in_roi:
                color, tag = QColor(0, 255, 120), "ID%d [区域内%s]" % (
                    tid, gtxt)
            else:
                color, tag = QColor(160, 160, 160), "ID%d" % tid
            p.setPen(QPen(color, 3 if is_front else 2))
            p.drawRect(int(bx), int(by), int(bw), int(bh))
            p.setPen(color)
            p.drawText(int(bx) + 3, int(by) - 4, tag)
            if face is not None:
                fx, fy = ox + face[0] * s, oy + face[1] * s
                p.setPen(QPen(QColor(255, 120, 0), 2))
                p.drawRect(int(fx), int(fy),
                           int(face[2] * s), int(face[3] * s))
            else:
                p.setPen(QColor(255, 120, 0))
                p.drawText(int(bx) + 3, int(by + bh) - 6, "未检到人脸")

        # ---------- 鼠标交互 ----------

        def _frame_pos(self, pos):
            ox, oy, s, w, h = self._geom()
            fx = (pos.x() - ox) / s
            fy = (pos.y() - oy) / s
            return (max(0.0, min(float(self._fw), fx)),
                    max(0.0, min(float(self._fh), fy)))

        def _hit_test(self, fx, fy):
            roi = self._roi
            x = roi['x'] * self._fw
            y = roi['y'] * self._fh
            w = roi['w'] * self._fw
            h = roi['h'] * self._fh
            thr = 16.0 / max(self._geom()[2], 0.001)
            corners = {'nw': (x, y), 'ne': (x + w, y),
                       'sw': (x, y + h), 'se': (x + w, y + h)}
            for k, (cx, cy) in corners.items():
                if abs(fx - cx) <= thr and abs(fy - cy) <= thr:
                    return k
            if x <= fx <= x + w and y <= fy <= y + h:
                return 'move'
            return 'se'   # 空白处：以按下点为固定角重新框选

        def mousePressEvent(self, e):
            if not self._editable or self._pix is None:
                return
            fx, fy = self._frame_pos(e.pos())
            if not (0 <= fx <= self._fw and 0 <= fy <= self._fh):
                return
            self._mode = self._hit_test(fx, fy)
            self._anchor = (fx, fy)
            if self._mode == 'se' and not self._inside_current(fx, fy):
                # 空白处重新框选：从按下点开始
                self._orig = {'x': fx / self._fw, 'y': fy / self._fh,
                              'w': 0.0, 'h': 0.0}
            else:
                self._orig = dict(self._roi)
            self._apply(fx, fy)

        def _inside_current(self, fx, fy):
            roi = self._roi
            x, y = roi['x'] * self._fw, roi['y'] * self._fh
            w, h = roi['w'] * self._fw, roi['h'] * self._fh
            return x <= fx <= x + w and y <= fy <= y + h

        def mouseMoveEvent(self, e):
            if self._mode:
                fx, fy = self._frame_pos(e.pos())
                self._apply(fx, fy)

        def mouseReleaseEvent(self, e):
            self._mode = None

        def _apply(self, fx, fy):
            if not self._mode or self._orig is None:
                return
            fw = max(float(self._fw), 1.0)
            fh = max(float(self._fh), 1.0)
            o = self._orig
            if self._mode == 'move':
                dx = (fx - self._anchor[0]) / fw
                dy = (fy - self._anchor[1]) / fh
                nx = min(max(0.0, o['x'] + dx), 1.0 - o['w'])
                ny = min(max(0.0, o['y'] + dy), 1.0 - o['h'])
                self._roi['x'] = nx
                self._roi['y'] = ny
            else:
                # 拖角 / 新框选：固定对角，另一角跟随鼠标
                x0 = o['x'] * fw
                y0 = o['y'] * fh
                x1 = (o['x'] + o['w']) * fw
                y1 = (o['y'] + o['h']) * fh
                fixed = {'nw': (x1, y1), 'ne': (x0, y1),
                         'sw': (x1, y0), 'se': (x0, y0)}[self._mode]
                nx0 = min(fixed[0], fx)
                ny0 = min(fixed[1], fy)
                nx1 = max(fixed[0], fx)
                ny1 = max(fixed[1], fy)
                self._roi['x'] = nx0 / fw
                self._roi['y'] = ny0 / fh
                self._roi['w'] = max(0.03, (nx1 - nx0) / fw)
                self._roi['h'] = max(0.03, (ny1 - ny0) / fh)
            self.roiChanged.emit(dict(self._roi))
            self.update()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("MP4 人像特征提取工具（纯本地处理）")
            self.resize(1320, 820)
            self.folder = ""
            self.out_dir = ""
            self.files = []
            self.preview_cap = None
            self.preview_path = None
            self.preview_fi = 0
            self.preview_fps = 25.0
            self.proc = None
            self.roi = {'x': 0.2, 'y': 0.1, 'w': 0.6, 'h': 0.8}
            self.item_map = {}
            self.config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "extractor_config.json")
            self._build_ui()
            self._load_config()
            # 检测引擎状态提示
            fe = FaceEngine()
            if fe.available:
                self._log("人脸检测引擎就绪：%s（OpenCV %s）"
                          % (fe.status_text(), cv2.__version__))
                if fe.missing:
                    self._log("提示：以下级联文件缺失但不影响使用：%s"
                              % "、".join(fe.missing))
            else:
                self._log("警告：未找到人脸检测模型（OpenCV %s），将只提取"
                          "上半身特征、无法用“人脸放大”判断接近。请把 "
                          "face_detection_yunet_2023mar.onnx 或 "
                          "haarcascade_*.xml 放到脚本同目录的 models\\ 下。"
                          % cv2.__version__)
            if HAS_MP:
                self._log("MediaPipe 可用：人脸检测精度更高（几乎不会把手/"
                          "身体部位误判成脸），保存时自动附带脸部五官描线图，"
                          "并以双肩关键点保证上半身完整入镜。")
            elif fe.available:
                self._log("提示：未安装 mediapipe（pip install mediapipe），"
                          "当前使用 %s。安装后可显著减少把手/衣物误判成人脸"
                          "的情况，并支持五官描线图与上半身姿态校正。"
                          % fe.status_text())
            if not hasattr(cv2, "HOGDescriptor"):
                self.cb_hog.setEnabled(False)
                self.cb_hog.setToolTip("当前 OpenCV 版本不含 HOGDescriptor，不可用")

        # ---------- 界面 ----------

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)

            # 顶部工具栏
            tb = QHBoxLayout()
            self.btn_files = QPushButton("① 选择文件（可多选）")
            self.btn_folder = QPushButton("① 选择文件夹")
            self.btn_start = QPushButton("② 开始处理")
            self.btn_stop = QPushButton("停止")
            self.btn_frame = QPushButton("预览下一帧")
            self.btn_open = QPushButton("打开输出目录")
            self.btn_stop.setEnabled(False)
            self.btn_start.setEnabled(False)
            for b in (self.btn_files, self.btn_folder, self.btn_start,
                      self.btn_stop, self.btn_frame, self.btn_open):
                tb.addWidget(b)
            self.lbl_folder = QLabel("未选择文件")
            tb.addWidget(self.lbl_folder, 1)
            root.addLayout(tb)

            split = QSplitter(Qt.Horizontal)

            # 左：视频画面 + 进度
            left = QWidget()
            lv = QVBoxLayout(left)
            self.lbl_video = QLabel("等待选择文件夹…")
            self.lbl_video.setStyleSheet("color:#666;")
            lv.addWidget(self.lbl_video)
            self.video = VideoLabel()
            lv.addWidget(self.video, 1)
            # 快进进度条：预览态拖动可跳到任意位置（松手位置即该视频
            # 的处理起点）；处理中拖动可实时快进跳过前面内容
            sh = QHBoxLayout()
            self.seek = QSlider(Qt.Horizontal)
            self.seek.setRange(0, 0)
            self.lbl_seek = QLabel("00:00:00")
            self.lbl_seek.setStyleSheet("color:#666;")
            sh.addWidget(self.seek, 1)
            sh.addWidget(self.lbl_seek)
            lv.addLayout(sh)
            self.progress = QProgressBar()
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            self.lbl_status = QLabel("就绪")
            hv = QHBoxLayout()
            hv.addWidget(self.progress, 1)
            hv.addWidget(self.lbl_status)
            lv.addLayout(hv)
            split.addWidget(left)

            # 右：ROI / 参数 / 结果 / 日志
            right = QScrollArea()
            right.setWidgetResizable(True)
            panel = QWidget()
            pv = QVBoxLayout(panel)

            g_roi = QGroupBox("抓拍区域（单个 ROI 矩形）")
            f1 = QFormLayout(g_roi)
            self.sp_rx = QSpinBox()
            self.sp_rx.setRange(0, 95)
            self.sp_rx.setValue(20)
            self.sp_rx.setSuffix(" %")
            self.sp_ry = QSpinBox()
            self.sp_ry.setRange(0, 95)
            self.sp_ry.setValue(10)
            self.sp_ry.setSuffix(" %")
            self.sp_rw = QSpinBox()
            self.sp_rw.setRange(3, 100)
            self.sp_rw.setValue(60)
            self.sp_rw.setSuffix(" %")
            self.sp_rh = QSpinBox()
            self.sp_rh.setRange(3, 100)
            self.sp_rh.setValue(80)
            self.sp_rh.setSuffix(" %")
            f1.addRow("区域左边 X：", self.sp_rx)
            f1.addRow("区域顶边 Y：", self.sp_ry)
            f1.addRow("区域宽度 W：", self.sp_rw)
            f1.addRow("区域高度 H：", self.sp_rh)
            lab = QLabel("在左侧画面中框选绿色矩形：拖内部移动、拖四角缩放、"
                         "空白处重新画。\n只抓拍矩形内站在最前面（离镜头最近、"
                         "目标最大）的人，且仅当他的人脸逐渐放大\n（朝镜头移动）"
                         "时才会保存；统一裁取“头+上半身”整体，不拍单独的"
                         "手/腰。")
            lab.setStyleSheet("color:#666;")
            f1.addRow(lab)
            pv.addWidget(g_roi)

            g_opt = QGroupBox("检测与输出参数")
            f2 = QFormLayout(g_opt)
            self.sp_interval = QSpinBox()
            self.sp_interval.setRange(1, 10)
            self.sp_interval.setValue(1)
            self.sp_interval.setToolTip(
                "每隔多少帧做一次检测。默认逐帧检测，人脸样本更连续，"
                "「接近」判定更准；调大可提速但可能漏判")
            self.sp_minh = QSpinBox()
            self.sp_minh.setRange(5, 80)
            self.sp_minh.setValue(15)
            self.sp_minh.setSuffix(" %")
            self.sp_minh.setToolTip("小于该画面高度占比的目标忽略")
            self.ds_merge = QDoubleSpinBox()
            self.ds_merge.setRange(0.20, 0.90)
            self.ds_merge.setSingleStep(0.05)
            self.ds_merge.setValue(0.50)
            self.ds_merge.setToolTip(
                "无人脸特征时，上半身外观相似度超过该值判定为同一人。"
                "正常情况下身份合并由 SFace 人脸识别（余弦相似度≥0.363）完成，"
                "能真正区分同款制服/深色外套的不同人")
            self.sp_mergetw = QSpinBox()
            self.sp_mergetw.setRange(0, 720)
            self.sp_mergetw.setValue(2)
            self.sp_mergetw.setSuffix(" 分钟")
            self.sp_mergetw.setToolTip(
                "同人合并时间窗：候选抓拍时间与已登记时间相差超过该值时"
                "不再做身份比对，直接登记为新人员（0=不限制）。"
                "可避免间隔很久的低质量人脸误合并")
            self.cb_face_req = QCheckBox(
                "必须检测到人脸（只拍“头+上半身”整体）")
            self.cb_face_req.setChecked(True)
            self.cb_face_req.setToolTip(
                "勾选后：检测不到人脸、或人脸太小，一律不抓拍，"
                "避免拍出单独的手部/腰部等局部")
            self.cb_approach = QCheckBox("必须检测到接近动作（人脸逐渐放大）")
            self.cb_approach.setChecked(True)
            self.cb_approach.setToolTip(
                "勾选后：只有人脸框随跟踪逐渐放大（人在朝镜头走）才抓拍；"
                "站着不动或正在远离的人不拍。\n一旦判定为接近过就持续抓拍，"
                "直到人脸明显持续变小才停止（走到最近处最清晰，会正常抓拍）")
            self.ds_growth = QDoubleSpinBox()
            self.ds_growth.setRange(2, 60)
            self.ds_growth.setValue(8)
            self.ds_growth.setSuffix(" %")
            self.ds_growth.setToolTip(
                "人脸宽度在跟踪窗口内的增长幅度下限，调小更容易判定为接近")
            self.sp_minface = QSpinBox()
            self.sp_minface.setRange(10, 200)
            self.sp_minface.setValue(30)
            self.sp_minface.setSuffix(" px")
            self.sp_minface.setToolTip("人脸小于该宽度视为离镜头太远，不抓拍")
            self.sp_appmin = QSpinBox()
            self.sp_appmin.setRange(3, 20)
            self.sp_appmin.setValue(4)
            self.sp_appmin.setSuffix(" 帧")
            self.sp_appmin.setToolTip("至少积累这么多帧人脸样本才做接近趋势判定")
            # 闸机口虚线：判定"是否实际进入安检闸机"的主条件
            self.sp_gateline = QSpinBox()
            self.sp_gateline.setRange(0, 50)
            self.sp_gateline.setValue(10)
            self.sp_gateline.setSuffix(" %")
            self.sp_gateline.setToolTip(
                "闸机口虚线位置：距画面底部百分之多少。画面上会画出这条"
                "橙色虚线，改数值即可看到它移动。人的脚部必须到达过该线"
                "以下，才算实际进入安检闸机——折返回去的旅客、以及在闸机"
                "旁维持秩序却没进去的安检员，脚部都到不了这里。\n"
                "数值越大，虚线越靠上，判定越宽松（越不容易漏人）；"
                "数值越小，虚线越靠下，判定越严格。0 表示不启用该判定。")
            self.sp_gateline.valueChanged.connect(self._sync_gate_line)
            self.sp_minsharp = QSpinBox()
            self.sp_minsharp.setRange(0, 500)
            self.sp_minsharp.setValue(30)
            self.sp_minsharp.setToolTip(
                "清晰度下限（Laplacian 方差）：低于该值视为过模糊，不抓拍；"
                "0 表示不过滤。数值越大越严格。")
            self.sp_minhits = QSpinBox()
            self.sp_minhits.setRange(1, 10)
            self.sp_minhits.setValue(2)
            self.sp_minhits.setSuffix(" 次")
            self.sp_minhits.setToolTip(
                "轨迹至少被检测到这么多次才抓拍；想更少漏可调成 1。")
            self.cb_static = QCheckBox(
                "静止优先（低速帧加分，减少运动模糊）")
            self.cb_static.setChecked(True)
            self.cb_static.setToolTip(
                "勾选后：近期尺寸变化越小（人越静止）的候选得分越高，"
                "从而优先保存不模糊的那张；只是加分，不会因此不拍。")
            self.cmb_rotate = QComboBox()
            self.cmb_rotate.addItems(["180°（画面倒置时使用）", "0°（不旋转）"])
            self.cb_hog = QCheckBox(
                "HOG 行人确认（仅 YOLO 不可用时生效，较慢）")
            self.cb_scan = QCheckBox(
                "全画面人脸扫描（仅 YOLO 不可用时生效的兜底）")
            self.cb_scan.setChecked(True)
            self.cb_yolo = QCheckBox(
                "YOLO 人体检测（CNN 确认真人：人体框+鼻/眼/肩关键点）")
            self.cb_yolo.setChecked(True)
            self.cb_yolo.setToolTip(
                "启用后：人员候选来自 YOLO11n-pose 人体框（静止的人也能"
                "检出），且人脸必须通过\n①五官关键点几何校验（眼上鼻下、"
                "比例合理）或与姿态头部关键点交叉吻合\n②位于人体框顶部"
                "且比例合理\n两道验证才会抓拍——海报/衣物纹理/图案不会再"
                "被误当人脸。需要 models\\yolo11n-pose.onnx。")
            self.cmb_yolo_in = QComboBox()
            self.cmb_yolo_in.addItems(
                ["320（最快，人较近时够用）", "416",
                 "480", "640（默认，推荐；模型不支持的尺寸会自动回退）"])
            self.cmb_yolo_in.setCurrentIndex(3)
            self.ds_yolo_conf = QDoubleSpinBox()
            self.ds_yolo_conf.setRange(0.10, 0.90)
            self.ds_yolo_conf.setSingleStep(0.05)
            self.ds_yolo_conf.setValue(0.35)
            self.ds_yolo_conf.setToolTip(
                "YOLO 人体框置信度下限。误检多时调高（如 0.5）；"
                "有人漏检时调低（如 0.25）")
            self.ed_time = QLineEdit()
            self.ed_time.setPlaceholderText(
                "如 20260923T180406，留空则自动从文件名解析")
            f2.addRow("画面旋转：", self.cmb_rotate)
            f2.addRow("检测间隔（帧）：", self.sp_interval)
            f2.addRow("最小目标高度：", self.sp_minh)
            f2.addRow("同人合并阈值：", self.ds_merge)
            f2.addRow("同人合并时间窗：", self.sp_mergetw)
            f2.addRow(self.cb_yolo)
            f2.addRow("YOLO 输入尺寸：", self.cmb_yolo_in)
            f2.addRow("YOLO 置信度阈值：", self.ds_yolo_conf)
            f2.addRow(self.cb_face_req)
            f2.addRow(self.cb_approach)
            f2.addRow("人脸放大阈值：", self.ds_growth)
            f2.addRow("最小有效脸宽：", self.sp_minface)
            f2.addRow("接近判定样本数：", self.sp_appmin)
            f2.addRow("闸机口虚线(距底部)：", self.sp_gateline)
            f2.addRow("清晰度下限：", self.sp_minsharp)
            f2.addRow("最小检测次数：", self.sp_minhits)
            f2.addRow(self.cb_static)
            f2.addRow(self.cb_hog)
            f2.addRow(self.cb_scan)
            f2.addRow("起始时间覆盖：", self.ed_time)
            pv.addWidget(g_opt)

            g_res = QGroupBox("提取结果（双击打开图片）")
            rv = QVBoxLayout(g_res)
            self.lst = QListWidget()
            self.lst.setIconSize(QSize(96, 96))
            self.lst.setSelectionMode(QAbstractItemView.SingleSelection)
            rv.addWidget(self.lst)
            pv.addWidget(g_res, 1)

            g_log = QGroupBox("运行日志")
            lv2 = QVBoxLayout(g_log)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(150)
            lv2.addWidget(self.log)
            pv.addWidget(g_log)

            right.setWidget(panel)
            split.addWidget(right)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 2)
            root.addWidget(split, 1)

            # 信号连接
            self.btn_files.clicked.connect(self._select_files)
            self.btn_folder.clicked.connect(self._select_folder)
            self.btn_start.clicked.connect(self._start)
            self.btn_stop.clicked.connect(self._stop)
            self.btn_frame.clicked.connect(self._preview_next)
            self.btn_open.clicked.connect(self._open_out)
            self.seek.sliderMoved.connect(self._on_seek_moved)
            self.seek.sliderReleased.connect(self._on_seek_released)
            self.video.roiChanged.connect(self._on_roi_changed)
            self.sp_rx.valueChanged.connect(self._on_spins)
            self.sp_ry.valueChanged.connect(self._on_spins)
            self.sp_rw.valueChanged.connect(self._on_spins)
            self.sp_rh.valueChanged.connect(self._on_spins)
            self.lst.itemDoubleClicked.connect(self._open_item)
            self._sync_gate_line()   # 首次启动（无配置文件）也画出闸机线

        # ---------- ROI / 闸机线同步 ----------

        def _sync_gate_line(self):
            """闸机口虚线数值变化 → 画面上的橙色虚线实时更新。"""
            self.video.set_gate_line(self.sp_gateline.value())


        def _on_spins(self):
            self.roi['x'] = self.sp_rx.value() / 100.0
            self.roi['y'] = self.sp_ry.value() / 100.0
            self.roi['w'] = self.sp_rw.value() / 100.0
            self.roi['h'] = self.sp_rh.value() / 100.0
            self.video.set_roi(self.roi)

        def _on_roi_changed(self, roi):
            self.roi = dict(roi)
            for sp, key in ((self.sp_rx, 'x'), (self.sp_ry, 'y'),
                            (self.sp_rw, 'w'), (self.sp_rh, 'h')):
                sp.blockSignals(True)
                sp.setValue(int(round(roi[key] * 100)))
                sp.blockSignals(False)

        # ---------- 文件夹与预览 ----------

        def _set_file_list(self, files, source_label, folder):
            """统一设置待处理文件列表并预览第一个文件。"""
            self.files = [f for f in files if f.lower().endswith(VIDEO_EXTS)]
            self.folder = folder
            self.lbl_folder.setText("%s（%d 个视频）"
                                    % (source_label, len(self.files)))
            if not self.files:
                self.lbl_video.setText("没有可处理的视频")
                self.btn_start.setEnabled(False)
                self._log("没有选择到视频文件（支持 mp4/mov/avi/mkv）")
                return
            self.btn_start.setEnabled(True)
            self.lbl_video.setText("共 %d 个视频" % len(self.files))
            self._log("已选择：%s（视频 %d 个）"
                      % (source_label, len(self.files)))
            self._open_preview(self.files[0])

        def _select_files(self):
            """自行选择一个或多个视频文件。"""
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择视频文件（可多选）", "",
                "视频文件 (*.mp4 *.mov *.avi *.mkv);;所有文件 (*.*)")
            if not files:
                return
            self._set_file_list(files, "%d 个文件" % len(files),
                                os.path.dirname(files[0]))

        def _select_folder(self):
            d = QFileDialog.getExistingDirectory(self, "选择包含 MP4 的文件夹")
            if not d:
                return
            files = [os.path.join(d, f) for f in os.listdir(d)
                     if f.lower().endswith(VIDEO_EXTS)]
            self._set_file_list(files, d, d)

        def _open_preview(self, path):
            self._close_preview()
            self.preview_cap = cv2.VideoCapture(path)
            ok, frame = self.preview_cap.read()
            if ok:
                if self._rotate_on():
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.preview_path = path
                self.preview_fi = 0
                self.preview_fps = self.preview_cap.get(cv2.CAP_PROP_FPS) \
                    or 25.0
                n = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.seek.blockSignals(True)
                self.seek.setRange(0, max(0, n - 1))
                self.seek.setValue(0)
                self.seek.blockSignals(False)
                self.lbl_seek.setText("00:00:00")
                self.video.set_frame(frame, [])
                self.lbl_video.setText(
                    "预览：%s（第 1 帧 / 共 %d 帧，拖动下方进度条可快进，"
                    "松手位置即处理起点）" % (os.path.basename(path), n))

        def _preview_next(self):
            if self.preview_cap is None:
                return
            for _ in range(5):  # 跳几帧便于观察
                ok, frame = self.preview_cap.read()
                if not ok:
                    self._close_preview()
                    self._log("预览结束")
                    return
            self.preview_fi += 5
            self.seek.blockSignals(True)
            self.seek.setValue(min(self.preview_fi, self.seek.maximum()))
            self.seek.blockSignals(False)
            self.lbl_seek.setText(self._fmt_fi(self.preview_fi))
            if self._rotate_on():
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            self.video.set_frame(frame, [])

        # ---------- 快进进度条 ----------

        def _fmt_fi(self, fi):
            """帧号 → HH:MM:SS 时间文本。"""
            t = timedelta(seconds=fi / max(1.0, self.preview_fps))
            return str(t).split('.')[0]

        def _on_seek_moved(self, v):
            """处理中拖动：实时快进。预览态等松手再跳（seek 较慢）。"""
            if self.proc is not None:
                self.proc.request_seek(v)

        def _on_seek_released(self):
            if self.proc is not None:
                self.proc.request_seek(self.seek.value())
            elif self.preview_cap is not None:
                self._preview_seek(self.seek.value())

        def _preview_seek(self, fi):
            """预览态：跳到指定帧并显示。"""
            if self.preview_cap is None or self.preview_path is None:
                return
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = self.preview_cap.read()
            if not ok:
                return
            if self._rotate_on():
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            self.preview_fi = fi
            self.lbl_seek.setText(self._fmt_fi(fi))
            self.video.set_frame(frame, [])
            self.lbl_video.setText("预览：%s（第 %d 帧）"
                                   % (os.path.basename(self.preview_path),
                                      fi + 1))

        def _on_pos(self, fi, n_total, fps):
            """处理中：同步进度条到当前视频内的播放位置。"""
            self.seek.blockSignals(True)
            if self.seek.maximum() != max(0, n_total - 1):
                self.seek.setRange(0, max(0, n_total - 1))
            self.seek.setValue(fi)
            self.seek.blockSignals(False)
            self.preview_fps = fps or 25.0
            self.lbl_seek.setText(self._fmt_fi(fi))

        def _close_preview(self):
            if self.preview_cap is not None:
                self.preview_cap.release()
                self.preview_cap = None

        def _rotate_on(self):
            return self.cmb_rotate.currentIndex() == 0

        # ---------- 处理 ----------

        def _collect_opts(self):
            return {
                'rotate': 180 if self.cmb_rotate.currentIndex() == 0 else 0,
                'det_interval': self.sp_interval.value(),
                'min_h': self.sp_minh.value(),
                'merge_thr': self.ds_merge.value(),
                'merge_time_window': self.sp_mergetw.value(),
                'use_yolo': self.cb_yolo.isChecked(),
                'yolo_input': (320, 416, 480, 640)[
                    self.cmb_yolo_in.currentIndex()],
                'yolo_conf': self.ds_yolo_conf.value(),
                'require_face': self.cb_face_req.isChecked(),
                'require_approach': self.cb_approach.isChecked(),
                'face_growth': self.ds_growth.value() / 100.0,
                'min_face_w': self.sp_minface.value(),
                'approach_min_frames': self.sp_appmin.value(),
                'gate_line_bottom_pct': self.sp_gateline.value(),
                'min_sharp': self.sp_minsharp.value(),
                'min_hits': self.sp_minhits.value(),
                'prefer_static': self.cb_static.isChecked(),
                'hog_confirm': self.cb_hog.isChecked(),
                'face_scan': self.cb_scan.isChecked(),
                'time_override': self.ed_time.text().strip(),
            }

        def _start(self):
            if not self.files or self.proc is not None:
                return
            self.out_dir = os.path.join(self.folder, "人像提取结果")
            os.makedirs(self.out_dir, exist_ok=True)
            opts = self._collect_opts()
            self._save_config()
            # 进度条指定的快进起点只对当前预览的视频生效（其余从 0 开始）
            start_frames = [0] * len(self.files)
            if (self.preview_cap is not None and self.preview_path
                    and self.preview_path == self.files[0]
                    and self.preview_fi > 0):
                start_frames[0] = self.preview_fi
                self._log("快进：%s 将从 %s（第 %d 帧）开始处理"
                          % (os.path.basename(self.preview_path),
                             self._fmt_fi(self.preview_fi),
                             self.preview_fi))
            self._close_preview()
            self.item_map = {}
            self.lst.clear()
            self.video.set_editable(False)
            self.btn_start.setEnabled(False)
            self.btn_files.setEnabled(False)
            self.btn_folder.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_frame.setEnabled(False)
            self._set_params_enabled(False)
            self.progress.setValue(0)
            self._log("=" * 50)
            self._log("开始处理 %d 个视频，输出目录：%s"
                      % (len(self.files), self.out_dir))
            self.proc = ProcessorThread(self.files, dict(self.roi), opts,
                                        self.out_dir, start_frames)
            self.proc.sig_frame.connect(self.video.set_frame)
            self.proc.sig_person.connect(self._on_person)
            self.proc.sig_progress.connect(self._on_progress)
            self.proc.sig_pos.connect(self._on_pos)
            self.proc.sig_log.connect(self._log)
            self.proc.sig_finished.connect(self._on_finished)
            self.proc.start()

        def _stop(self):
            if self.proc is not None:
                self.proc.stop()
                self.btn_stop.setEnabled(False)
                self._log("已请求停止，正在完成当前帧并保存已确认的结果…")

        def _set_params_enabled(self, e):
            for w in (self.sp_rx, self.sp_ry, self.sp_rw, self.sp_rh,
                      self.cmb_rotate, self.sp_interval, self.sp_minh,
                      self.ds_merge, self.sp_mergetw,
                      self.cb_face_req, self.cb_approach,
                      self.ds_growth, self.sp_minface, self.sp_appmin,
                      self.sp_gateline,
                      self.cb_hog, self.cb_scan, self.ed_time,
                      self.sp_minsharp, self.sp_minhits, self.cb_static):
                w.setEnabled(e)

        def _on_person(self, info):
            icon = None
            if info.get('thumb') is not None:
                rgb = cv2.cvtColor(info['thumb'], cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, 3 * w,
                              QImage.Format_RGB888).copy()
                icon = QIcon(QPixmap.fromImage(qimg))
            text = "%s  %s  分数 %.1f  %s" % (
                info['id'], info['ts_str'], info['score'],
                "含人脸" if info['face'] else "仅上半身")
            paths = {'upper': info.get('upper_path'),
                     'face': info.get('face_path'),
                     'face_lm': info.get('face_lm_path'),
                     'full': info.get('full_path')}
            if info['id'] in self.item_map:
                item = self.item_map[info['id']]
                item.setText(text)
                if icon is not None:
                    item.setIcon(icon)
                item.setData(Qt.UserRole, paths)
            else:
                item = QListWidgetItem(icon, text)
                item.setData(Qt.UserRole, paths)
                self.lst.addItem(item)
                self.item_map[info['id']] = item

        def _open_item(self, item):
            d = item.data(Qt.UserRole) or {}
            for k in ('upper', 'face', 'face_lm', 'full'):
                p = d.get(k)
                if p and os.path.exists(p):
                    os.startfile(p)
                    return

        def _on_progress(self, v, name):
            self.progress.setValue(int(max(0.0, min(1.0, v)) * 1000))
            self.lbl_status.setText(name)

        def _on_finished(self):
            self.video.set_editable(True)
            self.btn_start.setEnabled(bool(self.files))
            self.btn_files.setEnabled(True)
            self.btn_folder.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_frame.setEnabled(True)
            self._set_params_enabled(True)
            self.proc = None
            # 处理结束后重新打开预览，进度条恢复可用于选起点
            if self.files:
                self._open_preview(self.files[0])
            self._log("全部完成。")

        def _open_out(self):
            if self.out_dir and os.path.isdir(self.out_dir):
                os.startfile(self.out_dir)
            else:
                self._log("还没有输出目录，请先开始处理。")

        def _log(self, msg):
            self.log.appendPlainText("[%s] %s"
                                     % (datetime.now().strftime("%H:%M:%S"),
                                        msg))

        # ---------- 配置 ----------

        def _save_config(self):
            try:
                cfg = {'roi': self.roi, 'opts': self._collect_opts()}
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _load_config(self):
            if not os.path.exists(self.config_path):
                return
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                roi = cfg.get('roi') or {}
                if all(k in roi for k in ('x', 'y', 'w', 'h')):
                    self.roi = dict(roi)
                opts = cfg.get('opts') or {}
                self.sp_rx.setValue(int(round(self.roi['x'] * 100)))
                self.sp_ry.setValue(int(round(self.roi['y'] * 100)))
                self.sp_rw.setValue(int(round(self.roi['w'] * 100)))
                self.sp_rh.setValue(int(round(self.roi['h'] * 100)))
                self.sp_interval.setValue(opts.get('det_interval', 1))
                self.sp_minh.setValue(opts.get('min_h', 15))
                self.ds_merge.setValue(opts.get('merge_thr', 0.5))
                self.sp_mergetw.setValue(opts.get('merge_time_window', 2))
                self.cb_face_req.setChecked(opts.get('require_face', True))
                self.cb_approach.setChecked(opts.get('require_approach', True))
                self.ds_growth.setValue(
                    round(float(opts.get('face_growth', 0.08)) * 100))
                self.sp_minface.setValue(opts.get('min_face_w', 30))
                self.sp_appmin.setValue(opts.get('approach_min_frames', 4))
                self.sp_gateline.setValue(opts.get('gate_line_bottom_pct', 10))
                self.sp_minsharp.setValue(opts.get('min_sharp', 30))
                self.sp_minhits.setValue(opts.get('min_hits', 2))
                self.cb_static.setChecked(opts.get('prefer_static', True))
                self.cmb_rotate.setCurrentIndex(0 if opts.get('rotate', 180) == 180 else 1)
                self.cb_yolo.setChecked(opts.get('use_yolo', True))
                self.cmb_yolo_in.setCurrentIndex(
                    {320: 0, 416: 1, 480: 2, 640: 3}.get(
                        int(opts.get('yolo_input', 640)), 3))
                self.ds_yolo_conf.setValue(float(opts.get('yolo_conf', 0.35)))
                self.cb_hog.setChecked(opts.get('hog_confirm', False))
                self.cb_scan.setChecked(opts.get('face_scan', True))
                self.ed_time.setText(opts.get('time_override', ''))
                self.video.set_roi(self.roi)
                self._sync_gate_line()
            except Exception:
                pass

        def closeEvent(self, ev):
            self._save_config()
            if self.proc is not None:
                self.proc.stop()
                self.proc.wait(5000)
            self._close_preview()
            ev.accept()


def main():
    if not HAS_QT:
        print("缺少 PyQt5，请先安装依赖：")
        print("    pip install PyQt5 opencv-python numpy")
        input("按回车键退出…")
        sys.exit(1)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
