"""闸机独立调试界面。算法和后台线程位于 gate 包。"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import cv2
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QPlainTextEdit,
    QListWidget, QListWidgetItem, QProgressBar, QSplitter, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFileDialog, QFormLayout, QScrollArea,
    QAbstractItemView, QSlider,
)

from pose_project.common.project_paths import CONFIG_DIR
from pose_project.gate.core import FaceEngine, HAS_MP, VIDEO_EXTS
from pose_project.gate.processor import ProcessorThread


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
        self.config_path = str(CONFIG_DIR / "gate_config.json")
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
                      "haarcascade_*.xml 放到 models/gate/ 下。"
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


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
