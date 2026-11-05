# -*- coding: utf-8 -*-
"""验证 1920x1080 左右分栏布局 + RTSP 检索链路（offscreen）。

1. 布局断言：窗口尺寸、左栏固定宽、edit_v2 为 QLineEdit、进度条已移除。
2. Step1 MQTT 入库回归（模拟消息）。
3. RTSP 失败路径：假地址 -> worker 启动 -> 打开失败 -> 按钮恢复"开始检索"。
"""
import base64
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLineEdit
from PyQt5.QtCore import QEventLoop, QTimer

from main import MainWindow, MAX_PERSONS

app = QApplication([])
w = MainWindow()

# --- 1. 布局断言 ---
assert w.width() == 1920 and w.height() == 1080, f"窗口尺寸错误: {w.width()}x{w.height()}"
left = w.scroll.parentWidget().parentWidget()  # scroll -> gb1 -> left
assert left.maximumWidth() == 620 and left.minimumWidth() == 620, "左栏未贴边固定宽度"
assert isinstance(w.edit_v2, QLineEdit), "RTSP 输入框应为 QLineEdit"
assert w.combo_src.count() == 2, "缺少来源下拉框（本地文件/RTSP）"
assert not hasattr(w, "pb2"), "进度条应已移除"
assert w.btn_search.text() == "开始检索"
print(f"[1] 布局断言通过：{w.width()}x{w.height()}，左栏固定 {left.maximumWidth()}px")

# --- 2. MQTT 入库回归 ---
cap = cv2.VideoCapture("2bee809c50e7ce07de5a3a0f2651dcc8.mp4")
ret, frame = cap.read()
cap.release()
assert ret
b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()
w._on_person_received({"local_id": "P048",
                       "images": {"face": b64, "upper": b64}})
assert len(w.persons) == 1 and w.persons[0]["label"] == "P048"
print("[2] MQTT 入库回归通过")

# --- 3. RTSP 失败路径 ---
# 空地址应被拒绝
w.edit_v2.setText("")
n_err = [0]
def catch_msg(*a):
    n_err[0] += 1
w.show = lambda: None  # offscreen 下避免弹窗阻塞：QMessageBox 仍会 exec... 需要规避
# 用 monkeypatch 规避 QMessageBox 弹窗
import main as m
class FakeBox:
    @staticmethod
    def information(*a, **k):
        n_err[0] += 1
        return None
m.QMessageBox = FakeBox
w._toggle_search()
assert w.search_worker is None, "空 RTSP 地址不应启动 worker"
print("[3a] 空 RTSP 地址被正确拒绝")

# 假地址 -> worker 启动 -> 打开失败 -> finished_ok("") -> 按钮恢复
w.edit_v2.setText("rtsp://127.0.0.1:5599/not_exist")
w._toggle_search()
assert w.search_worker is not None and w.search_worker.isRunning(), "worker 未启动"
assert w.btn_search.text() == "停止检索", "按钮未切换为停止"
loop = QEventLoop()
w.search_worker.finished.connect(lambda: loop.quit())
QTimer.singleShot(60000, loop.quit)  # 兜底超时（应远早于此，open 超时 8s）
loop.exec_()
app.processEvents()
assert w.btn_search.text() == "开始检索", f"按钮未恢复: {w.btn_search.text()}"
assert w.btn_search.isEnabled()
print("[3b] RTSP 打开失败路径通过：worker 退出，按钮恢复")
print("全部测试通过")
