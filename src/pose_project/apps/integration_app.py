"""联合工作台界面：编辑配置，展示两路画面、旅客证据、识别和绑定结果。"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import signal
import string
import sys
import time
import uuid

from PyQt5.QtCore import QCoreApplication, QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QFontDatabase, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QStyle, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar, QToolButton,
    QVBoxLayout, QWidget,
)

from pose_project.integration.configuration import (
    DEFAULT_CONFIG, DEFAULT_FRONT_MODE, FRONT_MODE_ITEMS, FRONT_MODE_VALUES, REID_RUNTIME_DEFAULTS,
    environment_report, is_stream_source, load_settings, region_error,
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
    widget.setWordWrap(False)
    widget.verticalHeader().setDefaultSectionSize(28)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    widget.horizontalHeader().setResizeContentsPrecision(32)
    widget.horizontalHeader().setStretchLastSection(True)
    return widget


def add_row(widget, values, payload=None, limit=500, autoscroll=True):
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
    if autoscroll:
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
        self._evidence = {}
        self._latest_recognition = None
        self._recognition_arrived = 0.0
        self.setWindowTitle("旅客与行李框 · 联调工作台")
        self.resize(1440, 940)
        self.setMinimumSize(1000, 720)
        self._build()
        self._apply()
        self.controller.frame.connect(self._frame)
        self.controller.recognition.connect(self._recognition)
        self.controller.person.connect(self._person)
        self.controller.message.connect(self._message)
        self.controller.decision.connect(self._decision)
        self.controller.log.connect(self.log_view.appendPlainText)
        self.controller.state.connect(self._state)
        self.controller.running_changed.connect(self._running)
        self.controller.stopped.connect(self._stopped)
        self.result_timer = QTimer(self)
        self.result_timer.setInterval(200)
        self.result_timer.timeout.connect(self._refresh_recognition_age)
        self.result_timer.start()
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
        recognition_panel = QWidget()
        recognition_layout = QVBoxLayout(recognition_panel)
        recognition_controls = QHBoxLayout()
        self.recognition_age = QLabel("尚无检测结果")
        self.show_detection_boxes = QCheckBox("显示检测框")
        self.show_detection_boxes.setChecked(True)
        self.show_detection_boxes.toggled.connect(self._refresh_recognition_age)
        recognition_controls.addWidget(self.recognition_age, 1)
        recognition_controls.addWidget(self.show_detection_boxes)
        recognition_layout.addLayout(recognition_controls)
        self.recognition_table = table(["区域", "轨迹", "候选旅客", "相似度", "状态", "坐标"])
        self.recognition_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        recognition_layout.addWidget(self.recognition_table)
        self.event_tabs.addTab(recognition_panel, "重识别结果")
        performance_panel = QWidget()
        performance_layout = QVBoxLayout(performance_panel)
        self.debug_recording = QCheckBox("记录性能日志")
        debug_settings = self.settings.get("debug", {})
        self.debug_recording.setChecked(
            debug_settings.get("enabled", False) or
            debug_settings.get("field_test", {}).get("enabled", False))
        self.debug_recording.toggled.connect(self.controller.set_debug)
        performance_layout.addWidget(self.debug_recording)
        self.performance_table = table(["指标", "当前值"])
        performance_layout.addWidget(self.performance_table)
        self.event_tabs.addTab(performance_panel, "性能调试")
        evidence_panel = QWidget()
        evidence_layout = QVBoxLayout(evidence_panel)
        self.evidence_table = table([
            "本地编号", "完整旅客编号", "闸机抓拍", "重识别状态",
            "区域", "轨迹", "相似度", "绑定结果"])
        self.evidence_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.evidence_table.itemSelectionChanged.connect(self._select_evidence)
        evidence_layout.addWidget(self.evidence_table)
        evidence_splitter = QSplitter(Qt.Horizontal)
        self.evidence_gate_panel, self.evidence_gate_image = self._image_panel("闸机抓拍")
        self.evidence_reid_panel, self.evidence_reid_image = self._image_panel(
            "供框台当前目标")
        evidence_splitter.addWidget(self.evidence_gate_panel)
        evidence_splitter.addWidget(self.evidence_reid_panel)
        evidence_splitter.setSizes([360, 360])
        evidence_layout.addWidget(evidence_splitter, 1)
        self.evidence_details = QPlainTextEdit()
        self.evidence_details.setReadOnly(True)
        self.evidence_details.setMaximumBlockCount(200)
        self.evidence_details.setMaximumHeight(96)
        evidence_layout.addWidget(self.evidence_details)
        self.event_tabs.addTab(evidence_panel, "旅客证据")
        field_test_panel = QWidget()
        field_test_layout = QVBoxLayout(field_test_panel)
        field_test_form = QFormLayout()
        self.field_test_enabled = QCheckBox("启用现场测试记录")
        self.field_test_enabled.setToolTip(
            "保存每帧识别结果、绑定判定和人工核对记录")
        field_test_form.addRow(self.field_test_enabled)
        self.field_test_id = QLineEdit()
        self.field_test_operator = QLineEdit()
        self.field_test_site = QLineEdit()
        self.field_test_scenario = QLineEdit()
        self.field_test_notes = QPlainTextEdit()
        self.field_test_notes.setMaximumHeight(72)
        self.field_test_id.setPlaceholderText("例如 FT-20261110-01")
        self.field_test_operator.setPlaceholderText("记录人员")
        self.field_test_site.setPlaceholderText("机场、通道或设备位置")
        self.field_test_scenario.setPlaceholderText("例如 单人连续通行、多人排队、遮挡返回")
        self.field_test_notes.setPlaceholderText("本次测试的目标、设备和补充说明")
        field_test_form.addRow("测试编号", self.field_test_id)
        field_test_form.addRow("记录人员", self.field_test_operator)
        field_test_form.addRow("测试地点", self.field_test_site)
        field_test_form.addRow("测试场景", self.field_test_scenario)
        field_test_form.addRow("测试说明", self.field_test_notes)
        field_test_layout.addLayout(field_test_form)
        field_test_layout.addWidget(QLabel(
            "启动后自动保存 field_test.json 和 field_test_events.jsonl；"
            "现场图片、原始视频和设备截图仍需单独归档。"))
        observation_form = QFormLayout()
        self.field_test_category = QComboBox()
        for label, value in (
                ("闸机抓拍", "gate_capture"), ("旅客重识别", "reid"),
                ("行李框绑定", "binding"), ("视频取流", "stream"),
                ("MQTT 消息", "mqtt"), ("性能", "performance"),
                ("其他", "other")):
            self.field_test_category.addItem(label, value)
        self.field_test_result = QComboBox()
        for label, value in (("通过", "pass"), ("失败", "fail"),
                             ("待核对", "uncertain")):
            self.field_test_result.addItem(label, value)
        self.field_test_expected = QLineEdit()
        self.field_test_actual = QLineEdit()
        self.field_test_note = QLineEdit()
        self.field_test_expected.setPlaceholderText("现场应出现的结果")
        self.field_test_actual.setPlaceholderText("现场实际观察到的结果")
        self.field_test_note.setPlaceholderText("补充现象、截图编号或视频时间点")
        observation_form.addRow("核对类别", self.field_test_category)
        observation_form.addRow("核对结论", self.field_test_result)
        observation_form.addRow("预期结果", self.field_test_expected)
        observation_form.addRow("实际结果", self.field_test_actual)
        observation_form.addRow("现场备注", self.field_test_note)
        field_test_layout.addLayout(observation_form)
        self.field_test_record = QPushButton("记录现场观察")
        self.field_test_record.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.field_test_record.clicked.connect(self.record_field_observation)
        field_test_layout.addWidget(self.field_test_record)
        self.field_test_path = QLabel("尚未启动现场测试记录")
        self.field_test_path.setWordWrap(True)
        field_test_layout.addWidget(self.field_test_path)
        self.field_test_table = table(
            ["时间", "类别", "结论", "预期", "实际", "备注"])
        self.field_test_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.field_test_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.field_test_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        field_test_layout.addWidget(self.field_test_table, 1)
        self.event_tabs.addTab(field_test_panel, "现场测试")
        right.addWidget(self.event_tabs)
        right.setSizes([360, 400])
        horizontal.addWidget(right)
        horizontal.setSizes([330, 1080])
        root.addWidget(horizontal, 1)
        rfid_bar = QHBoxLayout()
        rfid_bar.addWidget(QLabel("RFID 测试注入"))
        self.rfid_region = QComboBox()
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

    def _image_panel(self, title):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel(title)
        heading.setFont(QFont("Microsoft YaHei UI", 10, QFont.DemiBold))
        layout.addWidget(heading)
        image = QLabel("暂无图片")
        image.setAlignment(Qt.AlignCenter)
        image.setMinimumSize(280, 240)
        image.setStyleSheet("background: #171c20; color: #a8b3ba;")
        image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(image, 1)
        return panel, image

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
        self.draw_buttons, self.delete_buttons = {}, {}
        self.region_status = {}
        form.addRow(self._region_editor("gate", "闸机抓拍区"))
        self.gate_region_status = QLabel()
        form.addRow(self.gate_region_status)
        self.size_label = QLabel()
        form.addRow("供框台画面", self.size_label)
        self.add_region_button = QPushButton("新增供框台区域")
        self.add_region_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        self.add_region_button.clicked.connect(self.add_region)
        form.addRow(self.add_region_button)
        self.region_controls = QWidget()
        self.region_form = QFormLayout(self.region_controls)
        self.region_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.region_controls)
        self._rebuild_region_controls()
        self.region_json = QPlainTextEdit()
        self.region_json.setReadOnly(True)
        self.region_json.setMinimumHeight(150)
        form.addRow(self.region_json)

    def _region_editor(self, target, title):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        draw = QPushButton("框选" + title)
        draw.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        draw.clicked.connect(lambda checked=False: self.begin_region_draw(target))
        remove = self.icon_button(QStyle.SP_TrashIcon, "删除" + title,
                                  lambda checked=False: self.delete_region(target))
        self.draw_buttons[target], self.delete_buttons[target] = draw, remove
        layout.addWidget(draw, 1)
        layout.addWidget(remove)
        return row

    def _rebuild_region_controls(self):
        while self.region_form.rowCount():
            self.region_form.removeRow(0)
        self.region_enabled, self.region_status = {}, {}
        self.region_front_modes = {}
        self.draw_buttons = {"gate": self.draw_buttons["gate"]}
        self.delete_buttons = {"gate": self.delete_buttons["gate"]}
        for region in self.settings["binding"]["regions"]:
            header = QWidget()
            layout = QHBoxLayout(header)
            layout.setContentsMargins(0, 0, 0, 0)
            check = QCheckBox("启用 " + region)
            check.setChecked(region in self.settings["reid"]["enabled_regions"])
            check.toggled.connect(self._region_toggled)
            self.region_enabled[region] = check
            remove = self.icon_button(QStyle.SP_TrashIcon, "删除 " + region + " 整组区域",
                lambda checked=False, name=region: self.delete_region(name))
            self.delete_buttons[region] = remove
            layout.addWidget(check, 1)
            layout.addWidget(remove)
            self.region_form.addRow(header)
            self.region_form.addRow(self._region_editor(
                region + ".roi_xyxy", " " + region + " 区域"))
            front_mode = QComboBox()
            for label, value in zip(FRONT_MODE_ITEMS, FRONT_MODE_VALUES):
                front_mode.addItem(label, value)
            current = self.settings["binding"]["regions"][region].get(
                "front_mode", DEFAULT_FRONT_MODE)
            front_mode.setCurrentIndex(
                FRONT_MODE_VALUES.index(current)
                if current in FRONT_MODE_VALUES else 0)
            front_mode.currentIndexChanged.connect(
                lambda index, name=region: self._front_mode_changed(name, index))
            self.region_front_modes[region] = front_mode
            self.region_form.addRow("多人排队", front_mode)
            status = QLabel()
            status.setWordWrap(True)
            self.region_status[region] = status
            self.region_form.addRow(status)

    def _front_mode_changed(self, region, index):
        if region not in self.settings["binding"]["regions"]:
            return
        mode = FRONT_MODE_VALUES[index] if 0 <= index < len(FRONT_MODE_VALUES) else None
        self.settings["binding"]["regions"][region]["front_mode"] = mode
        self._update_regions()
        self._save_region_config()

    def _region_toggled(self):
        self.settings["reid"]["enabled_regions"] = [
            key for key, check in self.region_enabled.items() if check.isChecked()]
        self._update_regions()

    def add_region(self):
        if self.controller.running:
            return
        regions = self.settings["binding"]["regions"]
        name = next((letter for letter in string.ascii_uppercase if letter not in regions), None)
        if name is None:
            self.error("供框台区域编号 A 到 Z 已用完")
            return
        regions[name] = {}
        self.settings["reid"]["enabled_regions"].append(name)
        self._rebuild_region_controls()
        self._update_regions()

    def delete_region(self, target):
        if self.controller.running:
            return
        for canvas in self.canvases.values():
            canvas.cancel_draw()
        if target == "gate":
            self.settings["gate"]["roi"] = None
        elif "." not in target:
            self.settings["binding"]["regions"].pop(target, None)
            self.settings["reid"]["enabled_regions"] = [
                key for key in self.settings["reid"]["enabled_regions"] if key != target]
            self._rebuild_region_controls()
        else:
            name, kind = target.split(".")
            region = self.settings["binding"]["regions"][name]
            region.pop(kind, None)
            if kind == "roi_xyxy":
                region.pop("operation_roi_xyxy", None)
        self._update_regions()

    def begin_region_draw(self, target):
        if self.controller.running:
            return
        stage = "gate" if target == "gate" else "reid"
        canvas = self.canvases[stage]
        if canvas.image.isNull():
            self.error("请先读取{}视频首帧".format("闸机" if stage == "gate" else "供框台"))
            return
        for view in self.canvases.values():
            view.cancel_draw()
        canvas.begin_draw(target)

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
        self.reid_controls = {}
        for name, lower, upper, step, suffix, label in (
                ("sample_interval_ms", 50, 1000, 50, " ms", "识别间隔"),
                ("detector_image_size", 320, 1280, 32, " px", "检测输入尺寸"),
                ("preview_fps", 1, 30, 1, " fps", "预览帧率")):
            control = QSpinBox()
            control.setRange(lower, upper)
            control.setSingleStep(step)
            control.setSuffix(suffix)
            self.reid_controls[name] = control
            form.addRow(label, control)
        self.flip_tta = QCheckBox("重识别翻转增强")
        form.addRow(self.flip_tta)
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
        field_test = settings.get("debug", {}).get("field_test", {})
        self.field_test_enabled.setChecked(field_test.get("enabled", False))
        self.field_test_id.setText(field_test.get("test_id", ""))
        self.field_test_operator.setText(field_test.get("operator", ""))
        self.field_test_site.setText(field_test.get("site", ""))
        self.field_test_scenario.setText(field_test.get("scenario", ""))
        self.field_test_notes.setPlainText(field_test.get("notes", ""))
        self.threshold.setValue(settings["reid"]["threshold"])
        for name, control in self.reid_controls.items():
            control.setValue(settings["reid"].get(name, REID_RUNTIME_DEFAULTS[name]))
        self.flip_tta.setChecked(settings["reid"].get("flip_tta", True))
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
        settings["reid"].update({name: control.value() for name, control in self.reid_controls.items()})
        settings["reid"]["flip_tta"] = self.flip_tta.isChecked()
        settings["debug"] = {
            "enabled": self.debug_recording.isChecked() or self.field_test_enabled.isChecked(),
            "field_test": {
                "enabled": self.field_test_enabled.isChecked(),
                "test_id": self.field_test_id.text().strip(),
                "operator": self.field_test_operator.text().strip(),
                "site": self.field_test_site.text().strip(),
                "scenario": self.field_test_scenario.text().strip(),
                "notes": self.field_test_notes.toPlainText().strip(),
            },
        }
        for name, control in self.region_front_modes.items():
            settings["binding"]["regions"][name]["front_mode"] = control.currentData()
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
                    for kind in ("roi_xyxy",):
                        if region.get(kind) is None:
                            continue
                        region[kind] = [
                            round(v * (width / old_width if i % 2 == 0 else height / old_height), 2)
                            for i, v in enumerate(region[kind])]
                self.settings["reid"]["image_size"] = [width, height]
                self.controller.write_log("供框台画面尺寸更新为 {}x{}，区域按比例换算".format(width, height))
        self._frame(stage, image)
        self._update_regions()

    def _frame(self, stage, image):
        previous_size = (self.canvases[stage].image.width(), self.canvases[stage].image.height())
        self.canvases[stage].set_frame(image)
        if previous_size != (image.shape[1], image.shape[0]):
            self._update_regions()

    def _recognition(self, report):
        self._latest_recognition = report
        self._recognition_arrived = time.monotonic()
        reid_frame = report.get("_frame")
        if reid_frame is None:
            reid_frame = self.canvases["reid"].image.copy()
        result_scroll = self.recognition_table.verticalScrollBar().value()
        statuses = {"matched": "已匹配", "below_threshold": "未达阈值",
                    "warming_track": "轨迹确认中", "no_gallery": "等待旅客入库",
                    "waiting_track": "等待稳定轨迹",
                    "ambiguous_match": "候选接近，暂不锁定",
                    "passenger_locked": "旅客已完成匹配，不再分配",
                    "duplicate": "身份重复，未采用"}
        self.recognition_table.setUpdatesEnabled(False)
        self.recognition_table.setRowCount(0)
        for region, result in report["regions"].items():
            for candidate in result["candidates"]:
                passenger = candidate.get("label")
                if passenger and candidate.get("status") not in (
                        "passenger_locked", "duplicate", "waiting_track"):
                    evidence = self._evidence.setdefault(passenger, {
                        "passenger_id": passenger,
                        "local_id": candidate.get("display_label", ""),
                    })
                    evidence["recognition"] = {
                        "region": region,
                        "tid": candidate.get("tid", ""),
                        "score": candidate.get("score"),
                        "status": candidate.get("status", ""),
                        "identity_locked": candidate.get("identity_locked", False),
                        "bbox_xyxy": candidate.get("bbox_xyxy"),
                        "captured_at_ms": report.get("captured_at_ms"),
                        "completed_at_ms": report.get("completed_at_ms"),
                        "frame": reid_frame,
                    }
                score = candidate.get("score")
                add_row(self.recognition_table, [
                    region, candidate.get("tid", ""), candidate.get("label") or "未识别",
                    "{:.3f}".format(score) if score is not None else "",
                    statuses.get(candidate["status"], candidate["status"]),
                    str(candidate["bbox_xyxy"])], limit=1000, autoscroll=False)
        self.recognition_table.verticalScrollBar().setValue(result_scroll)
        self.recognition_table.setUpdatesEnabled(True)
        metrics = report["metrics"]
        labels = {
            "device": "推理设备", "capture_fps": "采集帧率", "inference_fps": "识别帧率",
            "processing_ms": "整帧处理 ms", "detection_ms": "人体检测 ms",
            "reid_ms": "特征提取 ms", "matching_ms": "身份比对及绘制 ms",
            "input_age_ms": "取帧时图像年龄 ms", "worker_age_ms": "识别完成时位置延迟 ms",
            "ui_delivery_ms": "界面接收等待 ms", "gallery_ms": "本次旅客入库 ms",
            "gallery_count": "旅客库人数", "gallery_pending": "等待入库条数",
            "processed": "已识别帧数", "skipped": "累计跳过旧帧数",
            "reconnects": "取流重连次数", "mqtt_pending": "MQTT 等待发送条数",
            "message_pending": "等待处理消息数", "position_age_ms": "各区域接收位置延迟 ms",
        }
        self.performance_table.setUpdatesEnabled(False)
        performance_scroll = self.performance_table.verticalScrollBar().value()
        self.performance_table.setRowCount(0)
        for key, label in labels.items():
            if key in metrics:
                value = metrics[key]
                add_row(self.performance_table, [label,
                    "{:.2f}".format(value) if isinstance(value, float) else value],
                        autoscroll=False)
        self.performance_table.verticalScrollBar().setValue(performance_scroll)
        self.performance_table.setUpdatesEnabled(True)
        self._refresh_evidence_table()
        self._refresh_recognition_age()

    def _refresh_recognition_age(self):
        report = self._latest_recognition
        if report is None:
            return
        # Use monotonic elapsed time; the capture epoch belongs to the session clock.
        age = (max(0, report["completed_at_ms"] - report["captured_at_ms"])
               + report["metrics"].get("ui_delivery_ms", 0)
               + int((time.monotonic() - self._recognition_arrived) * 1000))
        valid = age <= self.parameters["max_position_age_ms"].value()
        self.recognition_age.setText("结果年龄 {} ms / {}".format(
            age, "有效" if valid and self.controller.running else "已过期"))
        self.recognition_age.setStyleSheet("color: {};".format("#216d57" if valid else "#ad3f38"))
        canvas = self.canvases["reid"]
        canvas.detections = [
            candidate for region in report["regions"].values()
            for candidate in region["candidates"]
        ] if valid and self.controller.running and self.show_detection_boxes.isChecked() else []
        canvas.update()

    def _rectangle(self, target, box):
        if self.controller.running:
            return
        if target == "gate":
            canvas = self.canvases["gate"]
            width, height = canvas.image.width(), canvas.image.height()
            if not width or not height:
                return
            self.settings["gate"]["roi"] = {
                "x": box[0] / width, "y": box[1] / height,
                "w": (box[2] - box[0]) / width, "h": (box[3] - box[1]) / height}
        else:
            name, kind = target.split(".")
            region = self.settings["binding"]["regions"].get(name)
            if region is None:
                return
            region[kind] = box
        self._update_regions()
        self._save_region_config()

    def _save_region_config(self):
        try:
            save_settings(self.collect(), self.config_path)
            self.controller.write_log("区域配置已自动保存：" + str(self.config_path))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.error("区域配置自动保存失败：{}".format(exc))

    def _update_regions(self):
        width, height = self.settings["reid"]["image_size"]
        self.size_label.setText("{} × {}".format(width, height))
        regions = self.settings["binding"]["regions"]
        text = json.dumps(regions, ensure_ascii=False, indent=2)
        if self.region_json.toPlainText() != text:
            self.region_json.setPlainText(text)
        active = []
        for name, geometry in regions.items():
            problem = region_error(name, geometry, (width, height))
            enabled = self.region_enabled[name].isChecked()
            self.region_status[name].setText(problem or ("已启用" if enabled else "未启用"))
            self.region_status[name].setStyleSheet(
                "color: {};".format("#ad3f38" if problem else "#216d57"))
            if enabled and not problem:
                active.append(name)
            kind = "roi_xyxy"
            exists = geometry.get(kind) is not None
            target = name + "." + kind
            self.draw_buttons[target].setText(
                "{} {} 区域".format("重画" if exists else "新增", name))
            self.delete_buttons[target].setEnabled(exists)
        current = self.rfid_region.currentText()
        if [self.rfid_region.itemText(i) for i in range(self.rfid_region.count())] != active:
            self.rfid_region.clear()
            self.rfid_region.addItems(active)
            if current in active:
                self.rfid_region.setCurrentText(current)
        self.add_region_button.setEnabled(len(regions) < 26)
        self.canvases["reid"].overlays = {
            region + " 区域": (box, "#e4b642")
            for region, geometry in regions.items()
            for kind, box in geometry.items()
            if kind == "roi_xyxy" and box is not None}
        gate = self.canvases["gate"]
        roi = self.settings["gate"].get("roi")
        w, h = gate.image.width(), gate.image.height()
        gate.overlays = {} if roi is None else {"抓拍区": ([roi["x"] * w, roi["y"] * h,
                                     (roi["x"] + roi["w"]) * w,
                                     (roi["y"] + roi["h"]) * h], "#e4b642")}
        self.draw_buttons["gate"].setText("新增闸机抓拍区" if roi is None else "重画闸机抓拍区")
        self.delete_buttons["gate"].setEnabled(roi is not None)
        self.gate_region_status.setText("缺少闸机抓拍区" if roi is None else "")
        self.gate_region_status.setStyleSheet("color: #ad3f38;")
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
            self.recognition_table.setRowCount(0)
            self.performance_table.setRowCount(0)
            self._latest_recognition = None
            self.recognition_age.setText("尚无检测结果")
            self.canvases["reid"].detections = []
            self._person_rows.clear()
            self._evidence.clear()
            self.evidence_table.setRowCount(0)
            self.evidence_details.clear()
            self._set_evidence_image(self.evidence_gate_image, None, "暂无闸机抓拍")
            self._set_evidence_image(self.evidence_reid_image, None, "暂无供框台目标")
            self.event_tabs.setTabText(2, "旅客库")
            self.field_test_table.setRowCount(0)
            self.field_test_path.setText(
                "现场测试记录目录：" + str(self.controller.run_dir)
                if self.controller.field_test_enabled
                else "本次未启用现场测试记录")
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
        if running:
            for canvas in self.canvases.values():
                canvas.cancel_draw()
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.settings_tabs.setEnabled(not running)
        metadata_controls = (
            self.field_test_enabled, self.field_test_id, self.field_test_operator,
            self.field_test_site, self.field_test_scenario, self.field_test_notes)
        for control in metadata_controls:
            control.setEnabled(not running)
        self.field_test_record.setEnabled(running and self.controller.field_test_enabled)
        self.rfid_send.setEnabled(running)
        self.import_button.setEnabled(running)
        if not running:
            self.session_label.setText("未启动" if self.controller.run_dir is None else "已停止")

    def _stopped(self):
        if self.controller.run_dir is not None and self.controller.field_test_enabled:
            self.field_test_path.setText(
                "现场测试记录目录：" + str(self.controller.run_dir))
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
        passenger = payload["passenger_id"]
        evidence = self._evidence.setdefault(passenger, {"passenger_id": passenger})
        evidence.update({
            "local_id": payload.get("local_id", ""),
            "capture_time": payload.get("capture_time", ""),
            "gate_image": payload.get("image"),
        })
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
        self._refresh_evidence_table()

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
        passenger = result.get("passenger_id")
        if passenger:
            evidence = self._evidence.setdefault(passenger, {"passenger_id": passenger})
            evidence["binding"] = result
            self._refresh_evidence_table()

    def _refresh_evidence_table(self):
        selected = self._selected_evidence_id()
        self.evidence_table.blockSignals(True)
        self.evidence_table.setRowCount(0)
        statuses = {
            "matched": "已匹配",
            "below_threshold": "未达阈值",
            "warming_track": "轨迹确认中",
            "ambiguous_match": "候选接近，暂不锁定",
            "duplicate": "身份重复，未采用",
        }
        binding_statuses = {"bound": "已绑定", "unbound": "未绑定",
                            "deferred": "多人符合"}
        for passenger, evidence in self._evidence.items():
            recognition = evidence.get("recognition") or {}
            binding = evidence.get("binding") or {}
            score = recognition.get("score")
            add_row(self.evidence_table, [
                evidence.get("local_id", ""),
                passenger,
                "已采集" if evidence.get("gate_image") is not None else "无",
                ("已匹配，任务锁定" if recognition.get("identity_locked") and
                 recognition.get("status") == "matched" else
                 statuses.get(recognition.get("status"), "等待重识别")),
                recognition.get("region", ""),
                recognition.get("tid", ""),
                "{:.3f}".format(score) if score is not None else "",
                binding_statuses.get(binding.get("status"), "未触发"),
            ], passenger, limit=256, autoscroll=False)
        self.evidence_table.blockSignals(False)
        if selected:
            for row in range(self.evidence_table.rowCount()):
                item = self.evidence_table.item(row, 0)
                if item and item.data(Qt.UserRole) == selected:
                    self.evidence_table.selectRow(row)
                    break
        if self.evidence_table.currentRow() < 0 and self.evidence_table.rowCount():
            self.evidence_table.selectRow(self.evidence_table.rowCount() - 1)

    def _selected_evidence_id(self):
        items = self.evidence_table.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def _set_evidence_image(self, label, image, message):
        if image is None:
            label.setPixmap(QPixmap())
            label.setText(message)
            return
        if hasattr(image, "isNull"):
            qimage = image
        else:
            rgb = image[:, :, ::-1].copy()
            height, width = rgb.shape[:2]
            qimage = QImage(rgb.data, width, height, rgb.strides[0],
                            QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        label.setText("")
        label.setPixmap(pixmap.scaled(
            max(1, label.width() - 8), max(1, label.height() - 8),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _select_evidence(self):
        passenger = self._selected_evidence_id()
        if not passenger:
            self.evidence_details.clear()
            self._set_evidence_image(self.evidence_gate_image, None, "暂无闸机抓拍")
            self._set_evidence_image(self.evidence_reid_image, None, "暂无供框台目标")
            return
        evidence = self._evidence.get(passenger, {})
        recognition = evidence.get("recognition") or {}
        binding = evidence.get("binding") or {}
        self._set_evidence_image(
            self.evidence_gate_image, evidence.get("gate_image"), "暂无闸机抓拍")
        reid_image = recognition.get("frame")
        box = recognition.get("bbox_xyxy")
        if reid_image is not None and box:
            x1, y1, x2, y2 = [int(value) for value in box]
            if hasattr(reid_image, "width"):
                x1 = max(0, min(reid_image.width(), x1))
                y1 = max(0, min(reid_image.height(), y1))
                x2 = max(x1, min(reid_image.width(), x2))
                y2 = max(y1, min(reid_image.height(), y2))
                reid_image = reid_image.copy(x1, y1, x2 - x1, y2 - y1)
            else:
                height, width = reid_image.shape[:2]
                x1 = max(0, min(width, x1))
                y1 = max(0, min(height, y1))
                x2 = max(x1, min(width, x2))
                y2 = max(y1, min(height, y2))
                reid_image = reid_image[y1:y2, x1:x2]
        else:
            reid_image = None
        self._set_evidence_image(
            self.evidence_reid_image, reid_image, "暂无供框台目标")
        details = {
            "passenger_id": passenger,
            "local_id": evidence.get("local_id", ""),
            "gate_capture_time": evidence.get("capture_time", ""),
            "recognition": {
                "region": recognition.get("region", ""),
                "track_id": recognition.get("tid", ""),
                "score": recognition.get("score"),
                "status": recognition.get("status", ""),
                "captured_at_ms": recognition.get("captured_at_ms"),
                "completed_at_ms": recognition.get("completed_at_ms"),
                "bbox_xyxy": recognition.get("bbox_xyxy"),
            },
            "binding": {
                "status": binding.get("status", ""),
                "roi_id": binding.get("roi_id", ""),
                "rfid": binding.get("rfid", ""),
                "reason": binding.get("reason", ""),
            },
        }
        self.evidence_details.setPlainText(json.dumps(details, ensure_ascii=False, indent=2))

    def record_field_observation(self):
        latest = self._latest_recognition or {}
        context = {"selected_passenger_id": self._selected_evidence_id()}
        if latest:
            context["latest_recognition"] = {
                key: value for key, value in latest.items() if key != "_frame"}
            context["latest_recognition"].pop("metrics", None)
            context["latest_recognition"]["metrics"] = dict(
                latest.get("metrics") or {})
        try:
            values = self.controller.record_field_observation(
                self.field_test_category.currentData(),
                self.field_test_result.currentData(),
                self.field_test_expected.text(),
                self.field_test_actual.text(),
                self.field_test_note.text(),
                context,
            )
        except ValueError as exc:
            self.error(str(exc))
            return
        result_labels = {"pass": "通过", "fail": "失败", "uncertain": "待核对"}
        category_labels = {
            self.field_test_category.itemData(index): self.field_test_category.itemText(index)
            for index in range(self.field_test_category.count())}
        add_row(self.field_test_table, [
            datetime.now().strftime("%H:%M:%S"),
            category_labels.get(values["category"], values["category"]),
            result_labels.get(values["result"], values["result"]),
            values["expected"], values["actual"], values["note"],
        ], limit=500)
        self.field_test_expected.clear()
        self.field_test_actual.clear()
        self.field_test_note.clear()

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
    operation = settings["binding"]["regions"][region]["roi_xyxy"]
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
