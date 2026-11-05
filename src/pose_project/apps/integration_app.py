"""Integrated passenger capture, re-identification and tray-binding workbench."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
import uuid

from PyQt5.QtCore import QCoreApplication, QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QFontDatabase, QIcon
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QStyle, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar, QToolButton,
    QVBoxLayout, QWidget,
)

from pose_project.integration.configuration import (
    DEFAULT_CONFIG, environment_report, is_stream_source, load_settings,
    save_settings, validate, validate_video_source,
)
from pose_project.integration.controller import IntegrationController
from pose_project.integration.widgets import VideoCanvas
from pose_project.integration.workers import PreviewWorker


def scroll_panel(widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setWidget(widget)
    return scroll


def configure_application(app):
    if not QFontDatabase().families():
        import os
        QFontDatabase.addApplicationFont(
            str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"))
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().hide()
    widget.setEditTriggers(QTableWidget.NoEditTriggers)
    widget.setSelectionBehavior(QTableWidget.SelectRows)
    widget.setAlternatingRowColors(True)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    widget.horizontalHeader().setStretchLastSection(True)
    return widget


def add_row(widget, values, payload=None, limit=500):
    if widget.rowCount() >= limit:
        widget.removeRow(0)
    row = widget.rowCount()
    widget.insertRow(row)
    for column, value in enumerate(values):
        item = QTableWidgetItem(str(value))
        item.setToolTip(str(value))
        if payload is not None:
            item.setData(Qt.UserRole, payload)
        widget.setItem(row, column, item)
    widget.scrollToBottom()
    return row


class IntegrationWindow(QMainWindow):
    def __init__(self, settings, config_path=DEFAULT_CONFIG, controller=None):
        super().__init__()
        self.settings = deepcopy(settings)
        self.config_path = Path(config_path)
        self.controller = controller or IntegrationController(self)
        self.previews = {}
        self._closing = False
        self._person_rows = {}
        self.setWindowTitle("旅客与行李框 · 联调工作台")
        self.resize(1440, 940)
        self.setMinimumSize(1000, 720)
        self._build()
        self._apply()
        self.controller.frame.connect(self._frame)
        self.controller.person.connect(self._person)
        self.controller.message.connect(self._message)
        self.controller.decision.connect(self._decision)
        self.controller.log.connect(self.log_view.appendPlainText)
        self.controller.state.connect(self._state)
        self.controller.running_changed.connect(self._running)
        self.controller.stopped.connect(self._stopped)
        self._running(False)

    def icon_button(self, icon, tooltip, callback):
        button = QToolButton()
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.clicked.connect(callback)
        return button

    def _build(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f7; color: #263137; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background: white; border: 1px solid #cbd3d7; border-radius: 3px;
                padding: 5px; min-height: 21px;
            }
            QPushButton, QToolButton { background: #ffffff; border: 1px solid #cbd3d7;
                border-radius: 3px; padding: 6px 10px; min-height: 22px; }
            QPushButton:hover, QToolButton:hover { background: #e6edef; }
            QPushButton:disabled, QToolButton:disabled { color: #9ba5ab; }
            QPushButton#start { background: #19735b; color: white; border-color: #19735b; }
            QTabWidget::pane { border: 1px solid #d5dcdf; background: white; }
            QTabBar::tab { padding: 8px 12px; background: #e9eef0; }
            QTabBar::tab:selected { background: white; color: #176c55; }
            QTableWidget { background: white; alternate-background-color: #f7f9fa;
                gridline-color: #e4e9eb; border: none; }
            QHeaderView::section { background: #edf1f3; border: none; padding: 7px; }
            QSplitter::handle { background: #dce2e5; }
            QToolBar { border: none; spacing: 8px; padding: 8px; }
        """)
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(toolbar)
        title = QLabel("旅客与行李框")
        title.setFont(QFont("Microsoft YaHei UI", 16, QFont.DemiBold))
        toolbar.addWidget(title)
        toolbar.addSeparator()
        self.start_button = QPushButton("启动联调")
        self.start_button.setObjectName("start")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self.start_session)
        self.stop_button = QPushButton("停止")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_session)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(self.icon_button(QStyle.SP_DialogSaveButton, "保存联调配置",
                                           self.save_config))
        toolbar.addWidget(self.icon_button(QStyle.SP_DialogApplyButton, "检查环境",
                                           self.check_environment))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self.session_label = QLabel("未启动")
        toolbar.addWidget(self.session_label)
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 0, 12, 10)
        status = QHBoxLayout()
        self.status_labels = {}
        for name, title in (("gate", "闸机采集"), ("reid", "旅客重识别"),
                            ("binding", "行李框绑定"), ("transport", "消息连接")):
            label = QLabel(title + "：未启动")
            label.setMinimumWidth(155)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self.status_labels[name] = (title, label)
            status.addWidget(label, 1)
        root.addLayout(status)
        horizontal = QSplitter(Qt.Horizontal)
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setMinimumWidth(300)
        self._build_sources()
        self._build_regions()
        self._build_connection()
        self._build_parameters()
        horizontal.addWidget(self.settings_tabs)
        right = QSplitter(Qt.Vertical)
        preview_panel = QWidget()
        previews = QHBoxLayout(preview_panel)
        previews.setContentsMargins(0, 0, 0, 0)
        self.canvases = {}
        for stage, title in (("gate", "闸机采集"), ("reid", "供框台重识别")):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(title)
            label.setFont(QFont("Microsoft YaHei UI", 11, QFont.DemiBold))
            layout.addWidget(label)
            canvas = VideoCanvas(title)
            canvas.rectangle_drawn.connect(self._rectangle)
            self.canvases[stage] = canvas
            layout.addWidget(canvas, 1)
            previews.addWidget(panel, 1)
        right.addWidget(preview_panel)
        self.event_tabs = QTabWidget()
        self.results_table = table(["区域", "旅客编号", "行李框 RFID", "结果", "判定依据"])
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.event_tabs.addTab(self.results_table, "绑定结果")
        messages = QSplitter(Qt.Horizontal)
        self.messages_table = table(["接收时间", "主题", "内容"])
        self.messages_table.itemSelectionChanged.connect(self._select_message)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(2000)
        messages.addWidget(self.messages_table)
        messages.addWidget(self.details)
        messages.setSizes([520, 400])
        self.event_tabs.addTab(messages, "消息")
        self.people_table = table(["图片", "本地编号", "完整旅客编号", "采集时间"])
        self.people_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.people_table.setIconSize(QSize(44, 62))
        self.event_tabs.addTab(self.people_table, "旅客库")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.event_tabs.addTab(self.log_view, "运行日志")
        right.addWidget(self.event_tabs)
        right.setSizes([480, 280])
        horizontal.addWidget(right)
        horizontal.setSizes([330, 1080])
        root.addWidget(horizontal, 1)
        rfid_bar = QHBoxLayout()
        rfid_bar.addWidget(QLabel("RFID 测试注入"))
        self.rfid_region = QComboBox()
        self.rfid_region.addItems(["A", "B"])
        rfid_bar.addWidget(self.rfid_region)
        self.rfid_text = QLineEdit()
        self.rfid_text.setPlaceholderText("行李框编号")
        self.rfid_text.returnPressed.connect(self.send_rfid)
        rfid_bar.addWidget(self.rfid_text, 1)
        self.rfid_send = QPushButton("发送测试事件")
        self.rfid_send.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        self.rfid_send.clicked.connect(self.send_rfid)
        rfid_bar.addWidget(self.rfid_send)
        self.import_button = QPushButton("导入上半身图")
        self.import_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.import_button.clicked.connect(self.import_image)
        rfid_bar.addWidget(self.import_button)
        root.addLayout(rfid_bar)
        self.setCentralWidget(central)

    def _tab_form(self, title):
        widget = QWidget()
        form = QFormLayout(widget)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setVerticalSpacing(10)
        self.settings_tabs.addTab(scroll_panel(widget), title)
        return form

    def _build_sources(self):
        form = self._tab_form("视频")
        self.sources, self.rotations = {}, {}
        self.source_modes, self.source_browse = {}, {}
        for stage, name in (("gate", "闸机视频"), ("reid", "供框台视频")):
            mode = QComboBox()
            mode.addItem("本地文件", "file")
            mode.addItem("RTSP / 网络流", "stream")
            self.source_modes[stage] = mode
            form.addRow(name, mode)
            source = QLineEdit()
            source.setMinimumWidth(100)
            source.setPlaceholderText("选择本地视频文件")
            source.setAccessibleName(name + "地址")
            self.sources[stage] = source
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(source, 1)
            browse = self.icon_button(QStyle.SP_DirOpenIcon, "选择" + name,
                                      lambda checked=False, key=stage: self.browse(key))
            self.source_browse[stage] = browse
            layout.addWidget(browse)
            layout.addWidget(self.icon_button(QStyle.SP_FileDialogContentsView, "读取首帧",
                                              lambda checked=False, key=stage: self.preview(key)))
            form.addRow("", row)
            mode.currentIndexChanged.connect(
                lambda index, key=stage: self._source_mode_changed(key))
            source.textChanged.connect(
                lambda text, key=stage: self._source_changed(key, text))
            rotate = QCheckBox("旋转 180°")
            self.rotations[stage] = rotate
            form.addRow("", rotate)
        self.detector_path = QLineEdit()
        self.reid_path = QLineEdit()
        self.output_path = QLineEdit()
        form.addRow("人员检测模型", self.detector_path)
        form.addRow("重识别模型", self.reid_path)
        form.addRow("输出目录", self.output_path)

    def _build_regions(self):
        form = self._tab_form("区域")
        self.region_enabled = {}
        self.draw_buttons = []
        gate = QPushButton("框选闸机抓拍区")
        gate.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        gate.clicked.connect(lambda: self.canvases["gate"].begin_draw("gate"))
        self.draw_buttons.append(gate)
        form.addRow(gate)
        self.size_label = QLabel()
        form.addRow("供框台画面", self.size_label)
        for region in ("A", "B"):
            check = QCheckBox("启用 " + region)
            self.region_enabled[region] = check
            form.addRow(check)
            for kind, name in (("roi_xyxy", "识别区"), ("operation_roi_xyxy", "操作区")):
                target = region + "." + kind
                button = QPushButton("框选 " + region + " " + name)
                button.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
                button.clicked.connect(
                    lambda checked=False, value=target: self.canvases["reid"].begin_draw(value))
                self.draw_buttons.append(button)
                form.addRow(button)
        self.region_json = QPlainTextEdit()
        self.region_json.setReadOnly(True)
        self.region_json.setMinimumHeight(210)
        form.addRow(self.region_json)

    def _build_connection(self):
        form = self._tab_form("连接")
        self.transport = QComboBox()
        self.transport.addItem("本机直连", "local")
        self.transport.addItem("MQTT 联调", "mqtt")
        self.host, self.username, self.password = QLineEdit(), QLineEdit(), QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.tls, self.output_enabled = QCheckBox("TLS"), QCheckBox("发送绑定结果")
        form.addRow("消息方式", self.transport)
        for label, widget in (("服务器", self.host), ("端口", self.port),
                              ("用户名", self.username), ("密码", self.password),
                              ("", self.tls)):
            form.addRow(label, widget)
        self.topics = {}
        for key, name in (("person_topic", "旅客图片主题"), ("detections", "人员位置主题"),
                           ("rfid", "RFID 主题"), ("output_topic", "绑定结果主题")):
            self.topics[key] = QLineEdit()
            form.addRow(name, self.topics[key])
        form.addRow(self.output_enabled)

    def _build_parameters(self):
        form = self._tab_form("算法")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.01)
        form.addRow("重识别阈值", self.threshold)
        self.parameters = {}
        labels = {"lookback_ms": "历史范围", "min_dwell_ms": "最少有效停留",
                  "binding_wait_ms": "绑定等待", "max_sample_gap_ms": "最大采样间隔",
                  "max_position_age_ms": "位置有效期", "rfid_time_offset_ms": "RFID 时间补偿"}
        for name, label in labels.items():
            control = QSpinBox()
            control.setRange(0 if name in ("binding_wait_ms", "rfid_time_offset_ms") else 1,
                             3600000)
            control.setSuffix(" ms")
            self.parameters[name] = control
            form.addRow(label, control)
        self.gate_approach = QCheckBox("要求朝镜头接近")
        self.gate_face = QCheckBox("要求有效人脸")
        self.gate_line = QDoubleSpinBox()
        self.gate_line.setRange(0, 100)
        self.gate_line.setSuffix(" %")
        form.addRow(self.gate_approach)
        form.addRow(self.gate_face)
        form.addRow("闸机线距底部", self.gate_line)

    def _apply(self):
        settings = self.settings
        for stage in ("gate", "reid"):
            self.sources[stage].setText(settings[stage]["source"])
        self.rotations["gate"].setChecked(settings["gate"]["opts"].get("rotate") == 180)
        self.rotations["reid"].setChecked(settings["reid"]["rotate180"])
        self.detector_path.setText(settings["reid"]["detector_weights"])
        self.reid_path.setText(settings["reid"]["reid_weights"])
        self.output_path.setText(settings["output_dir"])
        self.threshold.setValue(settings["reid"]["threshold"])
        self.transport.setCurrentIndex(self.transport.findData(settings["transport"]))
        mqtt = settings["mqtt"]
        self.host.setText(mqtt["host"])
        self.port.setValue(mqtt["port"])
        self.username.setText(mqtt.get("username", ""))
        self.password.setText(mqtt.get("password", ""))
        self.tls.setChecked(mqtt.get("tls", False))
        self.output_enabled.setChecked(mqtt["output_enabled"])
        for key, widget in self.topics.items():
            widget.setText(settings["binding"]["topics"][key] if key in ("detections", "rfid")
                           else mqtt[key])
        for name, widget in self.parameters.items():
            widget.setValue(settings["binding"]["binding_parameters"][name])
        for region, widget in self.region_enabled.items():
            widget.setChecked(region in settings["reid"]["enabled_regions"])
        self.gate_approach.setChecked(settings["gate"]["opts"]["require_approach"])
        self.gate_face.setChecked(settings["gate"]["opts"]["require_face"])
        self.gate_line.setValue(settings["gate"]["opts"]["gate_line_bottom_pct"])
        self._update_regions()

    def collect(self):
        settings = deepcopy(self.settings)
        for stage, source in self.sources.items():
            settings[stage]["source"] = source.text().strip().strip('"')
        settings["gate"]["opts"].update(
            rotate=180 if self.rotations["gate"].isChecked() else 0,
            require_approach=self.gate_approach.isChecked(),
            require_face=self.gate_face.isChecked(),
            gate_line_bottom_pct=self.gate_line.value())
        settings["reid"].update(
            rotate180=self.rotations["reid"].isChecked(), threshold=self.threshold.value(),
            detector_weights=self.detector_path.text().strip(),
            reid_weights=self.reid_path.text().strip(),
            enabled_regions=[key for key, box in self.region_enabled.items() if box.isChecked()])
        settings["binding"]["binding_parameters"] = {
            key: widget.value() for key, widget in self.parameters.items()}
        settings["transport"] = self.transport.currentData()
        settings["output_dir"] = self.output_path.text().strip()
        settings["mqtt"].update(host=self.host.text().strip(), port=self.port.value(),
            username=self.username.text(), password=self.password.text(),
            tls=self.tls.isChecked(), output_enabled=self.output_enabled.isChecked())
        for key, widget in self.topics.items():
            target = settings["binding"]["topics"] if key in ("detections", "rfid") else settings["mqtt"]
            target[key] = widget.text().strip()
        return settings

    def browse(self, stage):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.ts *.mkv *.avi *.mov);;全部文件 (*)")
        if path:
            self.sources[stage].setText(path)
            self.preview(stage)

    def _source_mode_changed(self, stage):
        stream = self.source_modes[stage].currentData() == "stream"
        source = self.sources[stage]
        if source.text() and is_stream_source(source.text()) != stream:
            source.clear()
        source.setPlaceholderText(
            "rtsp://user:password@192.168.1.10:554/stream" if stream
            else "选择本地视频文件")
        self.source_browse[stage].setEnabled(not stream)

    def _source_changed(self, stage, text):
        if is_stream_source(text):
            self.source_modes[stage].setCurrentIndex(1)
        elif Path(text.strip().strip('"')).is_file():
            self.source_modes[stage].setCurrentIndex(0)

    def preview(self, stage):
        if stage in self.previews or self.controller.running:
            return
        source = self.sources[stage].text().strip().strip('"')
        try:
            validate_video_source(source)
        except ValueError as exc:
            self.error(str(exc))
            return
        worker = PreviewWorker(source, self.rotations[stage].isChecked())
        self.previews[stage] = worker
        worker.frame.connect(lambda image, key=stage: self._preview_frame(key, image))
        worker.failed.connect(lambda text: self.controller.write_log(text))
        worker.finished.connect(lambda key=stage: self._preview_done(key))
        worker.start()

    def _preview_done(self, stage):
        worker = self.previews.pop(stage)
        worker.deleteLater()
        if self._closing and not self.previews and not self.controller.running:
            self.close()

    def _preview_frame(self, stage, image):
        if stage == "reid":
            height, width = image.shape[:2]
            old_width, old_height = self.settings["reid"]["image_size"]
            if [width, height] != [old_width, old_height]:
                for region in self.settings["binding"]["regions"].values():
                    for kind in ("roi_xyxy", "operation_roi_xyxy"):
                        region[kind] = [
                            round(v * (width / old_width if i % 2 == 0 else height / old_height), 2)
                            for i, v in enumerate(region[kind])]
                self.settings["reid"]["image_size"] = [width, height]
                self.controller.write_log("供框台画面尺寸更新为 {}x{}，区域按比例换算".format(width, height))
        self._frame(stage, image)
        self._update_regions()

    def _frame(self, stage, image):
        self.canvases[stage].set_frame(image)
        self._update_regions()

    def _rectangle(self, target, box):
        if self.controller.running:
            return
        if target == "gate":
            canvas = self.canvases["gate"]
            width, height = canvas.image.width(), canvas.image.height()
            self.settings["gate"]["roi"] = {
                "x": box[0] / width, "y": box[1] / height,
                "w": (box[2] - box[0]) / width, "h": (box[3] - box[1]) / height}
        else:
            region, kind = target.split(".")
            self.settings["binding"]["regions"][region][kind] = box
        self._update_regions()

    def _update_regions(self):
        width, height = self.settings["reid"]["image_size"]
        self.size_label.setText("{} × {}".format(width, height))
        self.region_json.setPlainText(json.dumps(self.settings["binding"]["regions"],
                                                ensure_ascii=False, indent=2))
        self.canvases["reid"].overlays = {
            region + (" 识别区" if kind == "roi_xyxy" else " 操作区"):
                (box, "#e4b642" if kind == "roi_xyxy" else "#29cda2")
            for region, geometry in self.settings["binding"]["regions"].items()
            for kind, box in geometry.items()}
        gate = self.canvases["gate"]
        roi = self.settings["gate"]["roi"]
        w, h = gate.image.width(), gate.image.height()
        gate.overlays = {"抓拍区": ([roi["x"] * w, roi["y"] * h,
                                     (roi["x"] + roi["w"]) * w,
                                     (roi["y"] + roi["h"]) * h], "#e4b642")}
        for canvas in self.canvases.values():
            canvas.update()

    def start_session(self):
        try:
            if self.previews:
                raise ValueError("正在读取视频首帧，请稍候")
            settings = self.collect()
            validate(settings, sources=True)
            report = environment_report(settings)
            if not report["ok"]:
                raise ValueError("环境或模型不完整，请点击工具栏的环境检查")
            self.controller.start(settings)
            self.settings = settings
            self.results_table.setRowCount(0)
            self.messages_table.setRowCount(0)
            self.people_table.setRowCount(0)
            self._person_rows.clear()
            self.event_tabs.setTabText(2, "旅客库")
            self.details.clear()
            self.session_label.setText("运行中")
            self.session_label.setToolTip(str(self.controller.run_dir))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.error(str(exc))

    def stop_session(self):
        self.stop_button.setEnabled(False)
        self.session_label.setText("停止中")
        self.controller.stop()

    def _running(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.settings_tabs.setEnabled(not running)
        self.rfid_send.setEnabled(running)
        self.import_button.setEnabled(running)
        if not running:
            self.session_label.setText("未启动" if self.controller.run_dir is None else "已停止")

    def _stopped(self):
        if self._closing and not self.previews:
            self.close()

    def _state(self, stage, text):
        title, label = self.status_labels[stage]
        full_text = title + "：" + text
        label.setText(label.fontMetrics().elidedText(full_text, Qt.ElideRight,
                                                    max(150, label.width())))
        label.setToolTip(full_text)
        label.setStyleSheet("color: {};".format(
            "#ad3f38" if "错误" in text or "失败" in text else "#216d57"))

    def _person(self, payload):
        from PyQt5.QtGui import QImage, QPixmap
        passenger = payload["passenger_id"]
        row = self._person_rows.get(passenger)
        if row is None:
            row = add_row(self.people_table, ["", payload["local_id"],
                                             passenger, payload["capture_time"]], limit=256)
            self._person_rows[passenger] = row
        image = payload["image"][:, :, ::-1].copy()
        h, w = image.shape[:2]
        qimage = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888).copy()
        self.people_table.item(row, 0).setIcon(QIcon(QPixmap.fromImage(qimage)))
        self.people_table.setRowHeight(row, 68)
        self.event_tabs.setTabText(2, "旅客库 ({})".format(len(self._person_rows)))

    def _message(self, record):
        payload = record["payload"]
        summary = payload.get("passenger_id") or (
            "{} / {} 人".format(payload.get("roi_id", ""), len(payload["detections"]))
            if "detections" in payload else payload.get("rfid", ""))
        received = datetime.fromtimestamp(record["received_at_ms"] / 1000).strftime("%H:%M:%S.%f")[:-3]
        add_row(self.messages_table, [received, record["topic"], summary], record)

    def _select_message(self):
        items = self.messages_table.selectedItems()
        if items:
            self.details.setPlainText(json.dumps(items[0].data(Qt.UserRole),
                                                 ensure_ascii=False, indent=2))

    def _decision(self, result):
        statuses = {"bound": "已绑定", "unbound": "未绑定", "deferred": "多人符合"}
        reasons = {
            "unique_candidate": "只有一名旅客符合条件",
            "no_position_data": "尚无人员位置记录",
            "stale_position_data": "人员位置已过期",
            "no_eligible_candidate": "没有符合条件的旅客",
            "multiple_candidates": "多名旅客符合条件",
        }
        add_row(self.results_table, [result["roi_id"], result["passenger_id"] or "",
            result["rfid"], statuses.get(result["status"], result["status"]),
            reasons.get(result["reason"], result["reason"])], result)

    def send_rfid(self):
        try:
            self.controller.inject_rfid(self.rfid_region.currentText(), self.rfid_text.text())
            self.rfid_text.clear()
        except ValueError as exc:
            self.error(str(exc))

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择上半身图片", "", "图片 (*.jpg *.jpeg *.png)")
        if path:
            try:
                self.controller.import_image(path)
            except (ValueError, OSError) as exc:
                self.error(str(exc))

    def save_config(self):
        try:
            save_settings(self.collect(), self.config_path)
            self.controller.write_log("联调配置已保存：" + str(self.config_path))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.error(str(exc))

    def check_environment(self):
        report = environment_report(self.collect())
        self.details.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        self.event_tabs.setCurrentIndex(1)
        self.controller.write_log("环境检查：" + ("通过" if report["ok"] else "缺少依赖或模型"))

    def error(self, text):
        self.controller.write_log(text)
        QMessageBox.warning(self, "联调工作台", text)

    def closeEvent(self, event):
        if self.controller.running or self.previews:
            self._closing = True
            for preview in self.previews.values():
                preview.stop()
            self.controller.stop()
            self.session_label.setText("正在停止后台任务")
            event.ignore()
        else:
            event.accept()


def self_test(settings):
    controller = IntegrationController()
    controller.start(dict(deepcopy(settings), transport="local"), start_workers=False)
    base = controller.inbox.now() - 6200
    passenger = uuid.uuid4().hex + "-" + datetime.now().strftime("%Y-%m-%d")
    region = settings["reid"]["enabled_regions"][0]
    operation = settings["binding"]["regions"][region]["operation_roi_xyxy"]
    cx, cy = (operation[0] + operation[2]) / 2, (operation[1] + operation[3]) / 2
    topic = settings["binding"]["topics"]["detections"]
    results = []
    controller.decision.connect(results.append)
    try:
        for offset in range(0, 6001, 200):
            now = base + offset
            controller._accept(topic, {"roi_id": region, "captured_at_ms": now,
                "detections": [{"passenger_id": passenger,
                                "bbox_xyxy": [cx - 10, cy - 10, cx + 10, cy + 10]}]}, now)
        controller._accept(settings["binding"]["topics"]["rfid"],
            {"roi_id": region, "rfid": "SELF-TEST-TRAY", "event_id": uuid.uuid4().hex}, base + 6010)
        controller._record_decisions(controller.binder.finish())
        ok = bool(results and results[0]["binding"]
                  and results[0]["binding"]["passenger_id"] == passenger)
        print(json.dumps({"ok": ok, "kind": "synthetic-binding-only",
                          "run_dir": str(controller.run_dir), "decisions": results},
                         ensure_ascii=False), flush=True)
        return 0 if ok else 1
    finally:
        controller.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="Check dependencies and local models")
    parser.add_argument("--check-models", action="store_true", help="Run local model inference checks")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic binding without cameras")
    parser.add_argument("--headless", action="store_true", help="Run workers with terminal logs only")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.check:
            report = environment_report(settings)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        if args.check_models:
            from pose_project.integration.diagnostics import check_models
            report = check_models(settings)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        if args.duration is not None and args.duration <= 0:
            parser.error("--duration must be positive")
        if args.self_test or args.headless:
            app = QCoreApplication.instance() or QCoreApplication([])
            if args.self_test:
                return self_test(settings)
            controller = IntegrationController()
            controller.stopped.connect(app.quit)
            controller.start(settings)
            signal.signal(signal.SIGINT, lambda *_: controller.stop())
            if args.duration is not None:
                QTimer.singleShot(int(args.duration * 1000), controller.stop)
            app.exec_()
            return 1 if controller.counts["gate_errors"] or controller.counts["reid_errors"] else 0
        app = QApplication.instance() or QApplication(sys.argv[:1])
        configure_application(app)
        window = IntegrationWindow(settings, args.config)
        window.show()
        if args.duration is not None:
            QTimer.singleShot(int(args.duration * 1000), window.close)
        signal.signal(signal.SIGINT, lambda *_: window.close())
        heartbeat = QTimer(window)
        heartbeat.start(200)
        heartbeat.timeout.connect(lambda: None)
        return app.exec_()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("启动失败：" + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
