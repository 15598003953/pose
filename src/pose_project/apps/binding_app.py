"""Qt workbench for passenger/tray binding."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QBrush, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QStackedWidget, QStyle, QTableWidget, QTableWidgetItem,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from pose_project.binding.binding_runtime import BindingSession, DEFAULT_FIELDS
from pose_project.apps.binding_simulation_dialog import SimulationDialog
from pose_project.binding.passenger_tray_binding import binding_passenger_id, validate_config
from pose_project.binding.replay_binding import DEFAULT_CONFIG, DEFAULT_INPUT


STATUS_TEXT = {"bound": "已绑定", "unbound": "未绑定", "deferred": "多人封存"}
STATUS_COLOR = {"bound": "#167d68", "unbound": "#b15b17", "deferred": "#b53b50"}
REASON_TEXT = {
    "unique_candidate": "唯一候选符合条件",
    "no_position_data": "缺少位置数据",
    "stale_position_data": "位置数据过期",
    "no_eligible_candidate": "没有合格候选",
    "multiple_candidates": "多名候选同时合格",
}
PARAMETER_LABELS = {
    "lookback_ms": "回看范围（毫秒）",
    "min_dwell_ms": "有效停留（毫秒）",
    "binding_wait_ms": "等待坐标（毫秒）",
    "max_sample_gap_ms": "采集最大间隔（毫秒）",
    "max_position_age_ms": "位置最大时差（毫秒）",
    "rfid_time_offset_ms": "RFID 延迟补偿（毫秒）",
}


def button(parent, text, icon, callback, tooltip=None):
    result = QPushButton(parent.style().standardIcon(icon), text)
    result.setToolTip(tooltip or text)
    result.clicked.connect(callback)
    return result


def scroll_form(widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setWidget(widget)
    return scroll


class BrokerForm(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        self.host = QLineEdit("127.0.0.1")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(1883)
        self.username, self.password = QLineEdit(), QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.tls = QCheckBox("TLS 加密")
        for label, widget in (
            ("服务器", self.host), ("端口", self.port), ("用户名", self.username),
            ("密码", self.password), ("", self.tls),
        ):
            layout.addRow(label, widget)

    def values(self):
        return {
            "host": self.host.text().strip(), "port": self.port.value(),
            "username": self.username.text(), "password": self.password.text(),
            "tls": self.tls.isChecked(),
        }


class RegionView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#f7f8fa"))
        self.setMinimumHeight(210)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.config = {"regions": {}}
        self.snapshots = {}
        self.names = {}
        self.selected = None

    def reset(self, config):
        self.config = config
        self.snapshots, self.names, self.selected = {}, {}, None
        self.redraw()

    def update_snapshot(self, payload):
        self.snapshots[payload["roi_id"]] = payload
        self.redraw()

    def highlight(self, passenger_id):
        self.selected = passenger_id
        self.redraw()

    def redraw(self):
        scene = self.scene()
        scene.clear()
        for roi, geometry in self.config["regions"].items():
            outer, operation = geometry["roi_xyxy"], geometry["operation_roi_xyxy"]
            x1, y1, x2, y2 = outer
            scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1),
                          QPen(QColor("#b8c2cc"), 3), QBrush(QColor("#ffffff")))
            title = scene.addText("供框台 {}".format(roi), QFont("Microsoft YaHei", 11))
            title.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            title.setDefaultTextColor(QColor("#34434e"))
            title.setPos(x1 + 22, y1 + 15)
            a, b, c, d = operation
            pen = QPen(QColor("#168675"), 3, Qt.DashLine)
            scene.addRect(QRectF(a, b, c - a, d - b), pen, QBrush(QColor("#e0f1ec")))
            label = scene.addText("操作区", QFont("Microsoft YaHei", 9))
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            label.setDefaultTextColor(QColor("#167d68"))
            label.setPos(a, d + 8)
            for person in self.snapshots.get(roi, {}).get("detections", []):
                passenger_id = person["passenger_id"]
                if passenger_id not in self.names:
                    self.names[passenger_id] = "P{:02d}".format(len(self.names) + 1)
                box = person["bbox_xyxy"]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                selected = passenger_id == self.selected
                inside = a <= cx < c and b <= cy < d
                color = "#d2475d" if selected else "#168675" if inside else "#5076ad"
                item = scene.addRect(
                    QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1]),
                    QPen(QColor(color), 6 if selected else 3),
                    QBrush(Qt.NoBrush),
                )
                item.setToolTip(passenger_id)
                point = scene.addEllipse(cx - 7, cy - 7, 14, 14,
                                         QPen(Qt.NoPen), QBrush(QColor(color)))
                point.setToolTip(passenger_id)
                text = scene.addText(self.names[passenger_id], QFont("Microsoft YaHei", 9))
                text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                text.setDefaultTextColor(QColor(color))
                text.setPos(box[0], box[1] - 43)
                text.setToolTip(passenger_id)
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)


class BindingWindow(QMainWindow):
    def __init__(self, config_path=DEFAULT_CONFIG):
        super().__init__()
        self.setWindowTitle("人包绑定工作台")
        self.resize(1320, 900)
        self.setMinimumSize(980, 700)
        with Path(config_path).open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        validate_config(self.config)
        self.session = BindingSession(self)
        self.simulation_dialog = None
        self.result_rows, self.results = {}, {}
        self.counts = {"bound": 0, "unbound": 0, "deferred": 0}
        self._build()
        self._apply_config()
        self.session.snapshot.connect(self.region_view.update_snapshot)
        self.session.decision.connect(self.add_decision)
        self.session.delivery.connect(self.update_delivery)
        self.session.state_changed.connect(self.update_state)
        self.session.connection_changed.connect(self.update_connection)
        self.session.progress.connect(self.update_progress)
        self.session.log.connect(self.append_log)
        self.session.failed.connect(self.show_failure)
        self.update_state("stopped")

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 10)
        toolbar = QHBoxLayout()
        title = QLabel("人包绑定工作台")
        title.setObjectName("title")
        toolbar.addWidget(title)
        toolbar.addStretch()
        self.start_button = button(self, "开始", QStyle.SP_MediaPlay, self.start)
        self.pause_button = button(self, "暂停", QStyle.SP_MediaPause, self.pause)
        self.stop_button = button(self, "停止", QStyle.SP_MediaStop, self.stop)
        self.export_button = button(self, "导出结果", QStyle.SP_DialogSaveButton, self.export)
        for control in (
            self.start_button, self.pause_button, self.stop_button, self.export_button
        ):
            toolbar.addWidget(control)
        root.addLayout(toolbar)
        status = QHBoxLayout()
        self.input_status, self.output_status = QLabel("输入：未启动"), QLabel("输出：关闭")
        self.summary = QLabel("消息 0　已绑定 0　未绑定 0　多人封存 0")
        status.addWidget(self.input_status)
        status.addWidget(self.output_status)
        status.addStretch()
        status.addWidget(self.summary)
        root.addLayout(status)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setMinimumWidth(320)
        self.settings_tabs.setMaximumWidth(440)
        self._build_input()
        self._build_parameters()
        self._build_output()
        splitter.addWidget(self.settings_tabs)
        self._build_results(splitter)
        splitter.setSizes([350, 950])
        self.statusBar().showMessage("就绪")
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #ffffff; color: #27343d; }
            QLabel#title { font-size: 20px; font-weight: 600; }
            QTabWidget::pane { border: 1px solid #dce2e7; }
            QTabBar::tab { padding: 9px 12px; background: #f3f5f7; }
            QTabBar::tab:selected { background: #e3f1ed; color: #126e5b; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                min-height: 27px; border: 1px solid #cbd4dc;
                border-radius: 3px; padding: 1px 5px; background: #ffffff;
            }
            QLineEdit:disabled { color: #8a949e; background: #f3f5f6; }
            QPushButton { padding: 6px 12px; border: 1px solid #cbd4dc;
                border-radius: 4px; background: #f7f9fa; }
            QPushButton:hover { background: #e7efed; }
            QPushButton:disabled { color: #a2acb5; background: #f6f7f8; }
            QHeaderView::section { background: #f1f4f6; border: none;
                border-bottom: 1px solid #d9e0e5; padding: 7px; }
            QTableWidget { gridline-color: #e8edf0; selection-background-color: #e1eee9;
                selection-color: #203d33; border: 1px solid #dce2e7; }
            QPlainTextEdit { border: 1px solid #dce2e7; background: #f8fafb; }
            QProgressBar { border: none; background: #edf1f3; text-align: center; }
            QProgressBar::chunk { background: #64af9a; }
        """)
        self.change_source(self.source.currentIndex())

    def _build_input(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.source = QComboBox()
        for text, value in (
            ("本地文件回放", "file"), ("MQTT 订阅", "mqtt"), ("HTTP 接收", "http")
        ):
            self.source.addItem(text, value)
        form.addRow("数据接口", self.source)
        layout.addLayout(form)
        self.source_pages = QStackedWidget()
        file_panel = QWidget()
        file_form = QFormLayout(file_panel)
        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_path = QLineEdit(str(DEFAULT_INPUT))
        self.file_path.setToolTip(str(DEFAULT_INPUT))
        browse = QToolButton()
        browse.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse.setToolTip("选择回放文件")
        browse.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(browse)
        file_form.addRow("消息文件", file_row)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.25, 2000)
        self.speed.setValue(10)
        self.speed.setSuffix(" 倍")
        file_form.addRow("回放速度", self.speed)
        self.simulate_button = button(
            self, "生成模拟数据", QStyle.SP_FileIcon, self.open_simulation
        )
        file_form.addRow(self.simulate_button)
        self.source_pages.addWidget(file_panel)
        self.input_broker = BrokerForm()
        self.source_pages.addWidget(self.input_broker)
        http_panel = QWidget()
        http_form = QFormLayout(http_panel)
        self.http_host = QLineEdit("127.0.0.1")
        self.http_port = QSpinBox()
        self.http_port.setRange(1, 65535)
        self.http_port.setValue(8765)
        http_form.addRow("监听地址", self.http_host)
        http_form.addRow("监听端口", self.http_port)
        http_form.addRow("检测路径", QLabel("/detections"))
        http_form.addRow("RFID 路径", QLabel("/rfid"))
        self.source_pages.addWidget(http_panel)
        layout.addWidget(self.source_pages)
        topics = QFormLayout()
        self.detection_topic, self.rfid_topic = QLineEdit(), QLineEdit()
        topics.addRow("检测主题", self.detection_topic)
        topics.addRow("RFID 主题", self.rfid_topic)
        layout.addLayout(topics)
        layout.addWidget(QLabel("消息字段"))
        fields_form = QFormLayout()
        self.fields = {}
        for key, label in (
            ("roi_id", "供框台区域"), ("captured_at_ms", "采集时间"),
            ("detections", "人员列表"), ("passenger_id", "旅客 ID"),
            ("bbox_xyxy", "检测框"), ("rfid", "RFID"),
            ("event_id", "RFID 事件编号"),
        ):
            field = QLineEdit(DEFAULT_FIELDS[key])
            self.fields[key] = field
            field.setEnabled(False)
            fields_form.addRow(label, field)
        self.source.currentIndexChanged.connect(self.change_source)
        self.change_source(0)
        layout.addLayout(fields_form)
        layout.addStretch()
        self.settings_tabs.addTab(scroll_form(panel), "数据接口")

    def change_source(self, index):
        self.source_pages.setCurrentIndex(index)
        self.source_pages.setFixedHeight(
            self.source_pages.currentWidget().sizeHint().height()
        )
        for field in self.fields.values():
            field.setEnabled(index != 0)

    def _build_parameters(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        controls = QHBoxLayout()
        controls.addWidget(button(self, "加载", QStyle.SP_DialogOpenButton, self.load_config))
        controls.addWidget(button(
            self, "另存", QStyle.SP_DialogSaveButton, self.save_config,
            "另存算法参数和区域配置"
        ))
        layout.addLayout(controls)
        form = QFormLayout()
        self.parameters = {}
        for key, label in PARAMETER_LABELS.items():
            spin = QSpinBox()
            spin.setRange(0 if key in ("binding_wait_ms", "rfid_time_offset_ms") else 1,
                          3600000)
            form.addRow(label, spin)
            self.parameters[key] = spin
        layout.addLayout(form)
        layout.addWidget(QLabel("供框台区域"))
        self.regions = QTableWidget(0, 3)
        self.regions.setHorizontalHeaderLabels(["区域", "供框台坐标", "操作区坐标"])
        self.regions.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.regions.setMinimumHeight(180)
        self.regions.setToolTip("坐标：[左, 上, 右, 下]")
        layout.addWidget(self.regions)
        row = QHBoxLayout()
        row.addWidget(button(self, "增加区域", QStyle.SP_FileDialogNewFolder, self.add_region))
        row.addWidget(button(self, "移除区域", QStyle.SP_TrashIcon, self.remove_region))
        layout.addLayout(row)
        layout.addStretch()
        self.settings_tabs.addTab(scroll_form(panel), "绑定参数")

    def _build_output(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.output_enabled = QCheckBox("通过 MQTT 发送成功绑定")
        layout.addWidget(self.output_enabled)
        self.output_broker = BrokerForm()
        layout.addWidget(self.output_broker)
        form = QFormLayout()
        self.output_topic = QLineEdit("pose/bindings")
        self.output_qos = QComboBox()
        for value in (0, 1, 2):
            self.output_qos.addItem(str(value), value)
        self.output_qos.setCurrentIndex(1)
        form.addRow("结果主题", self.output_topic)
        form.addRow("QoS", self.output_qos)
        layout.addLayout(form)
        layout.addStretch()
        self.settings_tabs.addTab(scroll_form(panel), "MQTT 输出")

    def _build_results(self, splitter):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 0, 0, 0)
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("区域与旅客位置"))
        title_row.addStretch()
        for text, color in (("操作区内", "#168675"), ("操作区外", "#5076ad"),
                            ("选中旅客", "#d2475d")):
            label = QLabel("■ " + text)
            label.setStyleSheet("color: {};".format(color))
            title_row.addWidget(label)
        layout.addLayout(title_row)
        vertical = QSplitter(Qt.Vertical)
        self.region_view = RegionView()
        vertical.addWidget(self.region_view)
        tabs = QTabWidget()
        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        results_layout.setContentsMargins(0, 6, 0, 0)
        filter_row = QHBoxLayout()
        self.result_filter = QComboBox()
        for text, value in (("全部结果", ""), ("已绑定", "bound"),
                            ("未绑定", "unbound"), ("多人封存", "deferred")):
            self.result_filter.addItem(text, value)
        self.result_filter.currentIndexChanged.connect(self.filter_results)
        filter_row.addWidget(self.result_filter)
        filter_row.addStretch()
        self.reason_label = QLabel("判定依据")
        filter_row.addWidget(self.reason_label)
        results_layout.addLayout(filter_row)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["接收时间", "区域", "旅客 ID", "行李框 RFID", "绑定结果", "MQTT 输出"]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.results_table.itemSelectionChanged.connect(self.show_decision)
        result_splitter = QSplitter(Qt.Vertical)
        result_splitter.addWidget(self.results_table)
        details = QTabWidget()
        self.candidates_table = QTableWidget(0, 4)
        self.candidates_table.setHorizontalHeaderLabels(
            ["旅客 ID", "有效停留 / 秒", "操作区位置", "符合条件"]
        )
        self.candidates_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.candidates_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.candidates_table.verticalHeader().setVisible(False)
        self.payload_view = QPlainTextEdit()
        self.payload_view.setReadOnly(True)
        details.addTab(self.candidates_table, "候选人依据")
        details.addTab(self.payload_view, "绑定返回值")
        result_splitter.addWidget(details)
        result_splitter.setSizes([210, 130])
        results_layout.addWidget(result_splitter)
        tabs.addTab(results_page, "绑定结果")
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(1000)
        tabs.addTab(self.logs, "运行记录")
        vertical.addWidget(tabs)
        vertical.setSizes([300, 400])
        layout.addWidget(vertical)
        splitter.addWidget(panel)

    def _apply_config(self):
        for key, control in self.parameters.items():
            control.setValue(self.config["binding_parameters"][key])
        self.detection_topic.setText(self.config["topics"]["detections"])
        self.rfid_topic.setText(self.config["topics"]["rfid"])
        self.regions.setRowCount(0)
        for roi, geometry in self.config["regions"].items():
            row = self.regions.rowCount()
            self.regions.insertRow(row)
            for column, text in enumerate((
                roi, json.dumps(geometry["roi_xyxy"]),
                json.dumps(geometry["operation_roi_xyxy"])
            )):
                self.regions.setItem(row, column, QTableWidgetItem(text))
        self.region_view.reset(self.config)

    def current_config(self):
        regions = {}
        for row in range(self.regions.rowCount()):
            cells = [self.regions.item(row, column) for column in range(3)]
            if any(cell is None for cell in cells):
                raise ValueError("请填写完整的区域坐标")
            roi = cells[0].text().strip()
            if roi in regions:
                raise ValueError("供框台区域不能重复")
            regions[roi] = {
                "roi_xyxy": json.loads(cells[1].text()),
                "operation_roi_xyxy": json.loads(cells[2].text()),
            }
        config = {
            "topics": {"detections": self.detection_topic.text().strip(),
                       "rfid": self.rfid_topic.text().strip()},
            "binding_parameters": {
                key: control.value() for key, control in self.parameters.items()
            },
            "regions": regions,
        }
        validate_config(config)
        return config

    def current_options(self):
        return {
            "source": self.source.currentData(),
            "file_path": self.file_path.text().strip(), "speed": self.speed.value(),
            "input_mqtt": self.input_broker.values(),
            "http_host": self.http_host.text().strip(), "http_port": self.http_port.value(),
            "fields": {key: field.text().strip() for key, field in self.fields.items()},
            "output_enabled": self.output_enabled.isChecked(),
            "output_mqtt": self.output_broker.values(),
            "output_topic": self.output_topic.text().strip(),
            "output_qos": self.output_qos.currentData(),
        }

    def start(self):
        try:
            config, options = self.current_config(), self.current_options()
            self.result_rows, self.results = {}, {}
            self.counts = {"bound": 0, "unbound": 0, "deferred": 0}
            self.results_table.setRowCount(0)
            self.candidates_table.setRowCount(0)
            self.payload_view.clear()
            self.region_view.reset(config)
            self.reason_label.setText("判定依据")
            self.input_status.setText("输入：连接中")
            self.output_status.setText(
                "输出：连接中" if options["output_enabled"] else "输出：关闭"
            )
            self.session.start(config, options)
            self.config = config
            self.append_log("会话开始：{}".format(self.source.currentText()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_failure(str(exc))

    def pause(self):
        self.session.set_paused(not self.session.paused)

    def stop(self):
        self.session.stop()
        self.input_status.setText("输入：已停止")
        self.output_status.setText("输出：已停止")
        self.append_log("会话已停止")

    def update_state(self, state):
        active = state in ("running", "paused")
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active or state == "finished")
        self.pause_button.setEnabled(active and self.source.currentData() == "file")
        self.pause_button.setText("继续" if state == "paused" else "暂停")
        self.pause_button.setIcon(self.style().standardIcon(
            QStyle.SP_MediaPlay if state == "paused" else QStyle.SP_MediaPause
        ))
        self.settings_tabs.setEnabled(not active)
        self.export_button.setEnabled(bool(self.results))
        self.statusBar().showMessage({
            "running": "运行中", "paused": "回放已暂停", "stopped": "已停止",
            "finished": "回放完成",
        }[state])
        if state == "finished":
            self.append_log("回放完成")

    def update_connection(self, direction, connected, text):
        label = self.input_status if direction == "input" else self.output_status
        label.setText(("输入：" if direction == "input" else "输出：") + text)
        label.setStyleSheet("color: {};".format("#167d68" if connected else "#a46425"))
        self.append_log(label.text())

    def update_progress(self, messages, decisions, invalid):
        self.summary.setText(
            "消息 {}　已绑定 {}　未绑定 {}　多人封存 {}　无效 {}".format(
                messages, self.counts["bound"], self.counts["unbound"],
                self.counts["deferred"], invalid
            )
        )

    def add_decision(self, index, decision):
        self.results[index] = decision
        self.counts[decision["status"]] += 1
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.result_rows[index] = row
        passenger_id = (
            decision["binding"]["passenger_id"] if decision["binding"] is not None else ""
        )
        values = (
            datetime.fromtimestamp(decision["rfid_received_at_ms"] / 1000)
            .strftime("%H:%M:%S.%f")[:-3],
            decision["roi_id"], passenger_id or "—", decision["rfid"],
            STATUS_TEXT[decision["status"]],
            "未启用" if decision["binding"] else "不发送",
        )
        for column, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            item.setData(Qt.UserRole, index)
            if column == 4:
                item.setForeground(QColor(STATUS_COLOR[decision["status"]]))
            self.results_table.setItem(row, column, item)
        self.filter_results()
        self.results_table.selectRow(row)
        self.export_button.setEnabled(True)

    def update_delivery(self, index, status):
        if index in self.result_rows:
            self.results_table.item(self.result_rows[index], 5).setText(status)

    def filter_results(self):
        selected = self.result_filter.currentData()
        for index, row in self.result_rows.items():
            self.results_table.setRowHidden(
                row, bool(selected) and self.results[index]["status"] != selected
            )

    def show_decision(self):
        selected = self.results_table.selectedItems()
        if not selected:
            return
        decision = self.results[selected[0].data(Qt.UserRole)]
        self.region_view.highlight(decision["passenger_id"])
        self.reason_label.setText(REASON_TEXT[decision["reason"]])
        self.reason_label.setToolTip(
            "位置时差：{} 毫秒".format(decision["position_age_ms"])
        )
        self.payload_view.setPlainText(json.dumps(
            decision["binding"], ensure_ascii=False, indent=2
        ))
        self.candidates_table.setRowCount(0)
        for candidate in decision["candidates"]:
            row = self.candidates_table.rowCount()
            self.candidates_table.insertRow(row)
            item = QTableWidgetItem(binding_passenger_id(candidate["passenger_id"]))
            item.setToolTip(candidate["passenger_id"])
            self.candidates_table.setItem(row, 0, item)
            dwell = QProgressBar()
            dwell.setRange(0, self.config["binding_parameters"]["lookback_ms"])
            dwell.setValue(candidate["dwell_ms"])
            dwell.setFormat("{:.2f}".format(candidate["dwell_ms"] / 1000))
            self.candidates_table.setCellWidget(row, 1, dwell)
            self.candidates_table.setItem(row, 2, QTableWidgetItem(
                "在操作区" if candidate["present_in_latest_snapshot"] else "已离开"
            ))
            self.candidates_table.setItem(row, 3, QTableWidgetItem(
                "是" if candidate["eligible"] else "否"
            ))

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择消息文件", "", "JSONL (*.jsonl)")
        if path:
            self.file_path.setText(path)
            self.file_path.setToolTip(path)

    def open_simulation(self):
        try:
            self.current_config()
            if self.simulation_dialog is not None:
                if self.simulation_dialog.isVisible():
                    self.simulation_dialog.raise_()
                    self.simulation_dialog.activateWindow()
                    return
                self.simulation_dialog.deleteLater()
            self.simulation_dialog = SimulationDialog(self)
            self.simulation_dialog.generated.connect(self.use_simulation)
            self.simulation_dialog.open()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_failure(str(exc))

    def use_simulation(self, output, counts, simulation_config):
        config = self.current_config()
        config["topics"] = simulation_config["topics"]
        config["regions"] = simulation_config["regions"]
        self.config = config
        self._apply_config()
        self.source.setCurrentIndex(0)
        replay_path = str(output / "replay.jsonl")
        self.file_path.setText(replay_path)
        self.file_path.setToolTip(replay_path)
        message = "模拟数据已生成：检测 {} 条，RFID {} 条，共 {} 条。目录：{}".format(
            counts["detections.jsonl"], counts["rfid.jsonl"], counts["replay.jsonl"], output
        )
        self.append_log(message)
        self.statusBar().showMessage(message)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载算法配置", "", "JSON (*.json)")
        if path:
            try:
                with Path(path).open(encoding="utf-8") as handle:
                    config = json.load(handle)
                validate_config(config)
                self.config = config
                self._apply_config()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.show_failure(str(exc))

    def save_config(self):
        try:
            config = self.current_config()
            path, _ = QFileDialog.getSaveFileName(self, "另存算法配置", "", "JSON (*.json)")
            if path:
                destination = Path(path)
                if destination.exists():
                    backup = destination.with_name(
                        destination.name + ".{}.bak".format(
                            datetime.now().strftime("%Y%m%d%H%M%S_%f")
                        )
                    )
                    with backup.open("xb") as handle:
                        handle.write(destination.read_bytes())
                with destination.open("w", encoding="utf-8", newline="\n") as handle:
                    json.dump(config, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                self.append_log("算法配置已保存：{}".format(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_failure(str(exc))

    def add_region(self):
        used = {self.regions.item(row, 0).text() for row in range(self.regions.rowCount())
                if self.regions.item(row, 0)}
        roi = next((letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in used), None)
        if roi is None:
            return
        row = self.regions.rowCount()
        self.regions.insertRow(row)
        for column, text in enumerate((roi, "[0, 0, 960, 1080]", "[300, 250, 500, 500]")):
            self.regions.setItem(row, column, QTableWidgetItem(text))

    def remove_region(self):
        row = self.regions.currentRow()
        if row >= 0:
            self.regions.removeRow(row)

    def export(self):
        parent = QFileDialog.getExistingDirectory(self, "选择结果保存位置")
        if parent:
            try:
                folder = Path(parent) / datetime.now().strftime("binding_%Y%m%d_%H%M%S_%f")
                self.session.export(folder)
                self.append_log("结果已保存：{}".format(folder))
                self.statusBar().showMessage("结果已保存：{}".format(folder))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.show_failure(str(exc))

    def append_log(self, text):
        self.logs.appendPlainText("{}  {}".format(datetime.now().strftime("%H:%M:%S"), text))

    def show_failure(self, text):
        self.append_log("错误：" + text)
        self.statusBar().showMessage("错误：" + text)
        QMessageBox.critical(self, "运行失败", text)

    def closeEvent(self, event):
        if (self.simulation_dialog is not None
                and self.simulation_dialog.worker is not None
                and self.simulation_dialog.worker.isRunning()):
            event.ignore()
            return
        self.session.stop()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 9))
    window = BindingWindow(args.config)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
