# -*- coding: utf-8 -*-
"""Step3 界面级回归（offscreen）：
- 来源下拉框 / 浏览按钮 / 进度条 布局与联动；
- 本地文件一键检索：进度到 100%、按钮复位、实时预览正常且不保存 mp4；
- 粘贴 rtsp 地址时来源自动切到“实时流”。
"""
import base64
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

import main as m
import _test_media

_test_media.ensure(short_only=True)  # 准备 20 帧短片

app = QApplication([])
w = m.MainWindow()
w.show()
app.processEvents()

# --- 1. 布局与默认态 ---
assert w.combo_src.count() == 2
assert w.combo_src.currentIndex() == 0, "默认应为本地文件模式"
assert w.btn_browse.isVisible(), "文件模式下应显示浏览按钮"
assert "ts" in w.edit_v2.placeholderText().lower()
assert w.pb.value() == 0
assert w.chk_rotate.isEnabled() and not w.chk_rotate.isChecked(), "旋转开关默认关闭"
print("[1] 默认布局通过：文件模式 / 浏览按钮可见 / 进度条 0 / 旋转开关可用且默认关")

# --- 2. 切到 RTSP 模式 ---
w.combo_src.setCurrentIndex(1)
app.processEvents()
assert not w.btn_browse.isVisible(), "实时流模式不应显示浏览按钮"
assert "rtsp" in w.edit_v2.placeholderText()
assert not w.chk_realtime.isEnabled()
assert not w.chk_rotate.isEnabled(), "实时流模式应禁用画面旋转"
print("[2] RTSP 模式联动通过")

# 切回文件模式，粘贴 rtsp 地址后应自动切到实时流
w.edit_v2.setText("rtsp://user:pass@10.0.0.1:554/stream")
src, is_stream = w._resolve_source()
assert is_stream and w.combo_src.currentIndex() == 1, "输入 rtsp 地址后来源应自动切换"
print("[3] 地址自动识别通过")

# --- 3. 文件检索全流程 ---
class FakeBox:
    @staticmethod
    def information(*a, **k):
        return None

    @staticmethod
    def warning(*a, **k):
        return None

m.QMessageBox = FakeBox

# 用测试视频首帧造一个人物
cap, _ = m.open_file_capture("_test_short.mp4")
ret, frame = cap.read()
cap.release()
assert ret
b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()
w._on_person_received({"local_id": "P048", "images": {"face": b64, "upper": b64}})
assert len(w.persons) == 1

w._set_file_path("_test_short.mp4")
app.processEvents()
assert w.combo_src.currentIndex() == 0
assert w.edit_v2.text().endswith("_test_short.mp4")
print("[4] 选择文件通过：", w.edit_v2.text())

w._toggle_search()
assert w.search_worker is not None and w.search_worker.isRunning()
assert w.btn_search.text() == "停止检索"
assert w.search_worker.rotate180 is False, "默认不应旋转"
loop = QEventLoop()
w.search_worker.finished.connect(loop.quit)
QTimer.singleShot(120000, loop.quit)
loop.exec_()
for _ in range(20):
    app.processEvents()
assert w.btn_search.text() == "开始检索" and w.btn_search.isEnabled()
assert w.pb.value() == 100, f"进度未到 100：{w.pb.value()}"
assert w.player._last_frame is not None, "预览画面为空"
assert not hasattr(w.search_worker, "out_path"), "不应再有输出视频路径"
assert not os.path.exists("_out_step3_ui.mp4"), "不应保存 mp4"
print(f"[5] 文件检索全流程通过：进度 {w.pb.value()}%，实时预览正常且不保存 mp4")

# --- 5b. 勾选「旋转180°」后检索：开关透传 + 输出画面确实翻转 ---
def first_frame(p):
    c, _ = m.open_file_capture(p)
    ok, f = c.read()
    c.release()
    assert ok, f"读取 {p} 失败"
    return f


w.chk_rotate.setChecked(True)
w._set_file_path("_test_short.mp4")
app.processEvents()
# 记录预览帧（不再保存 mp4，改为直接校验实时预览画面）
prev_frames = []
w._on_preview = lambda f: prev_frames.append(f.copy())
w._toggle_search()
assert w.search_worker.rotate180 is True, "勾选后未把旋转传给检索线程"
loop = QEventLoop()
w.search_worker.finished.connect(loop.quit)
QTimer.singleShot(120000, loop.quit)
loop.exec_()
for _ in range(20):
    app.processEvents()
assert prev_frames, "旋转模式没有预览帧"
assert not os.path.exists("_out_rot_ui.mp4"), "不应保存 mp4"
src_f = first_frame("_test_short.mp4")
out_f = prev_frames[0]
d_rot = float(np.abs(out_f.astype(np.int16)
                     - cv2.rotate(src_f, cv2.ROTATE_180).astype(np.int16)).mean())
d_src = float(np.abs(out_f.astype(np.int16) - src_f.astype(np.int16)).mean())
assert 0 < d_rot < d_src / 3, f"预览画面未翻转（d_rot={d_rot:.1f} d_src={d_src:.1f}）"
print(f"[5b] 旋转 180° 界面链路通过：预览 {len(prev_frames)} 帧已翻转，"
      f"差异 rot={d_rot:.1f} vs 原={d_src:.1f}")
w.chk_rotate.setChecked(False)

# --- 4. 不存在的文件应被拒绝 ---
w._set_file_path("_not_exist_file.ts")
w.search_worker = None
w._toggle_search()
assert w.search_worker is None, "不存在的文件不应启动检索"
print("[6] 非法文件路径被拒绝")

w.player.release()
print("全部界面测试通过")
