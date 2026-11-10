"""重识别独立调试界面：旅客库、区域设置和实时预览。"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QScrollArea, QSlider, QGroupBox, QFrame, QMessageBox, QSizePolicy,
    QLineEdit, QComboBox, QFileDialog, QProgressBar, QCheckBox, QDialog,
)

from pose_project.common.project_paths import CONFIG_DIR
from pose_project.reid.core import PersonDetector, ReidExtractor, crop_from_frame, valid_crop
from pose_project.reid.messaging import MqttPublisher, MqttReceiver, decode_base64_image
from pose_project.reid.search_worker import SearchWorker
from pose_project.video.stream_capture import is_stream_source, open_file_capture

MAX_PERSONS = 10
CONFIG_PATH = str(CONFIG_DIR / "reid_roi_config.json")


def cv2_to_pixmap(frame):
    """BGR numpy -> QPixmap（安全拷贝）。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def imread_unicode(path):
    """支持中文路径的图像读取。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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

    # 独立界面使用单独配置，避免覆盖联合工作台的区域。
    def _save_roi_config(self):
        """保存 A/B 区域，下次启动从 configs/reid_roi_config.json 恢复。"""
        try:
            cfg = {"rois": {rid: {"enabled": bool(v["enabled"]),
                                  "rect": list(v["rect"]) if v["rect"] else None}
                            for rid, v in self.rois.items()}}
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"ROI 配置保存失败：{e}")

    def _load_roi_config(self):
        """启动时恢复区域；屏蔽勾选信号，避免自动建立 MQTT 连接。"""
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


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
