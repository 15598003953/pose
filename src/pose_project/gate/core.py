#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闸机抓拍核心，不依赖图形界面。

FaceEngine / PersonDetector 检测人脸和人体；SimpleTracker 维护短时轨迹；
PersonRegistry 合并同一旅客的候选照片；保存函数输出图片和人员汇总。
抓拍条件和处理顺序见 processor.py，独立界面入口为 pose gate。
本地模型统一读取 models/gate，当前模块不负责 MQTT 发送。
"""

import os
import re
import csv
import math
from collections import deque
from datetime import datetime

import numpy as np
import cv2

from pose_project.common.project_paths import GATE_MODEL_DIR

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

# ============================================================
# 第一部分：核心检测 / 跟踪 / 人员登记逻辑（不依赖 PyQt5）
# ============================================================

TS_RE = re.compile(r"(\d{8})T(\d{6})Z")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
# 模型位置由公共路径模块决定，不依赖启动目录，也不再与源码混放。
MODEL_DIRS = (str(GATE_MODEL_DIR),)
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
