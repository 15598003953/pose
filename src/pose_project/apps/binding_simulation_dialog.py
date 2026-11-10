"""模拟数据生成对话框：编辑场景模板并在后台生成可回放消息。"""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QStyle, QToolButton, QVBoxLayout, QWidget,
)

from pose_project.binding.generate_binding_simulation import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, generate_dataset, validate_config, write_dataset,
)


class SimulationWorker(QThread):
    def __init__(self, config, output, parent):
        super().__init__(parent)
        self.config, self.output = config, output
        self.result, self.error = None, None

    def run(self):
        try:
            messages, expected = generate_dataset(self.config)
            self.result = write_dataset(self.output, self.config, messages, expected)
        except Exception as exc:
            # 后台异常必须返回界面，避免一直显示正在生成。
            self.error = str(exc) or type(exc).__name__


class SimulationDialog(QDialog):
    generated = pyqtSignal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("生成模拟数据")
        self.resize(680, 460)
        self.setMinimumWidth(580)
        self.setModal(True)
        self.worker = None
        self.config = None
        layout = QVBoxLayout(self)
        self.form_panel = QWidget()
        form = QFormLayout(self.form_panel)
        self.template_path = QLineEdit(str(DEFAULT_CONFIG))
        self.template_path.setReadOnly(True)
        form.addRow("场景模板", self.path_row(self.template_path, self.browse_template))
        destination = DEFAULT_OUTPUT.with_name(
            "binding_simulation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        self.output_path = QLineEdit(str(destination))
        form.addRow("保存目录", self.path_row(self.output_path, self.browse_output))
        self.seed = QLineEdit()
        form.addRow("随机种子", self.seed)
        self.start_time = QLineEdit()
        form.addRow("起始时间（含时区）", self.start_time)
        self.interval = self.spin(1)
        form.addRow("更新间隔（毫秒）", self.interval)
        self.jitter = self.spin(0)
        form.addRow("坐标扰动（像素）", self.jitter)
        self.delays = {}
        for key, label in (
            ("detection_delay_ms", "检测传输延迟（毫秒）"),
            ("photoelectric_delay_ms", "光电接收延迟（毫秒）"),
        ):
            row = QWidget()
            pair = QHBoxLayout(row)
            pair.setContentsMargins(0, 0, 0, 0)
            controls = [self.spin(0), self.spin(0)]
            pair.addWidget(controls[0])
            pair.addWidget(QLabel("至"))
            pair.addWidget(controls[1])
            self.delays[key] = controls
            form.addRow(label, row)
        self.scenario_info = QLabel()
        form.addRow("场景规模", self.scenario_info)
        layout.addWidget(self.form_panel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        controls = QHBoxLayout()
        controls.addStretch()
        self.generate_button = QPushButton(
            self.style().standardIcon(QStyle.SP_FileIcon), "生成"
        )
        self.generate_button.clicked.connect(self.generate)
        self.cancel_button = QPushButton(
            self.style().standardIcon(QStyle.SP_DialogCancelButton), "取消"
        )
        self.cancel_button.clicked.connect(self.reject)
        controls.addWidget(self.generate_button)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)
        self.load_template(DEFAULT_CONFIG)

    @staticmethod
    def spin(minimum):
        control = QSpinBox()
        control.setRange(minimum, 2147483647)
        return control

    def path_row(self, field, callback):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QToolButton()
        browse.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse.setToolTip("选择文件" if field is self.template_path else "选择保存目录")
        browse.clicked.connect(callback)
        field.textChanged.connect(field.setToolTip)
        field.setToolTip(field.text())
        layout.addWidget(field)
        layout.addWidget(browse)
        return row

    def load_template(self, path):
        try:
            with Path(path).open(encoding="utf-8") as handle:
                config = json.load(handle)
            validate_config(config)
            for value in (config["update_interval_ms"], config["bbox_jitter_px"],
                          *config["detection_delay_ms"], *config["photoelectric_delay_ms"]):
                if value > 2147483647:
                    raise ValueError("模拟参数超出界面可填写的整数范围")
            self.config = config
            self.template_path.setText(str(Path(path).resolve()))
            self.seed.setText(str(config["seed"]))
            self.start_time.setText(config["start_time"])
            self.interval.setValue(config["update_interval_ms"])
            self.jitter.setValue(config["bbox_jitter_px"])
            for key, controls in self.delays.items():
                for control, value in zip(controls, config[key]):
                    control.setValue(value)
            self.scenario_info.setText("{} 秒 · {} 个区域 · {} 条轨迹 · {} 次推框".format(
                config["duration_ms"] / 1000, len(config["regions"]),
                len(config["tracks"]), len(config["push_events"])
            ))
            self.status.clear()
            self.generate_button.setEnabled(True)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            self.generate_button.setEnabled(self.config is not None)
            self.show_failure(str(exc))

    def browse_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模拟场景模板", str(DEFAULT_CONFIG.parent), "JSON (*.json)"
        )
        if path:
            self.load_template(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择模拟数据目录")
        if path:
            self.output_path.setText(path)

    def generate(self):
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            if self.config is None:
                raise ValueError("请先选择有效的场景模板")
            config = deepcopy(self.config)
            config.update(
                seed=int(self.seed.text().strip()),
                start_time=self.start_time.text().strip(),
                update_interval_ms=self.interval.value(),
                bbox_jitter_px=self.jitter.value(),
            )
            for key, controls in self.delays.items():
                config[key] = [control.value() for control in controls]
            validate_config(config)
            if not self.output_path.text().strip():
                raise ValueError("请填写模拟数据保存目录")
            output = Path(self.output_path.text().strip()).resolve()
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("保存目录必须是新目录或空目录：{}".format(output))
            self.worker = SimulationWorker(config, output, self)
            self.worker.finished.connect(self.generation_finished)
            self.form_panel.setEnabled(False)
            self.generate_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.progress.show()
            self.status.setText("正在生成模拟数据…")
            self.worker.start()
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            self.show_failure(str(exc))

    def generation_finished(self):
        self.progress.hide()
        self.form_panel.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        if self.worker.error is not None:
            self.show_failure(self.worker.error)
            return
        output, counts = self.worker.result
        self.generated.emit(output, counts, self.worker.config)
        self.accept()

    def show_failure(self, text):
        self.status.setText("生成失败：" + text)
        QMessageBox.critical(self, "模拟数据生成失败", text)

    def reject(self):
        if self.worker is None or not self.worker.isRunning():
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)
