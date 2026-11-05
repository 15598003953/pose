# -*- coding: utf-8 -*-
"""Step1「手动添加」回归（offscreen）：
- 对话框：选图预览、ID 必填校验、确定/取消结果；
- 主流程：读图 -> 检测裁剪 -> 提特征 -> 入库 -> 计数/-画廊更新；
- 多选同 ID、非法图片、人物库已满。
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
from PyQt5.QtWidgets import QApplication, QDialog

import main as m
import _test_media

_test_media.ensure(short_only=True)


class FakeBox:
    """避免 offscreen 下弹出模态 QMessageBox 阻塞/崩溃。"""

    calls = []

    @classmethod
    def information(cls, *a, **k):
        cls.calls.append(a[1] if len(a) > 1 else "")
        return None

    @classmethod
    def warning(cls, *a, **k):
        cls.calls.append(a[1] if len(a) > 1 else "")
        return None


m.QMessageBox = FakeBox  # 对话框与主窗口都用 m 模块内的名字

app = QApplication([])
w = m.MainWindow()
w.show()
app.processEvents()

# 造两张测试图片（有行人的帧）
cap, _ = m.open_file_capture("_test_short.mp4")
ret, frame = cap.read()
cap.release()
assert ret
cv2.imwrite("_m1.jpg", frame)
cv2.imwrite("_m2.jpg", frame)
with open("_bad.jpg", "wb") as f:
    f.write(b"not an image")

# --- 1. 按钮存在且可点 ---
assert w.btn_manual.text() == "手动添加"
assert w.btn_manual.isEnabled()
print("[1] Step1 存在「手动添加」按钮")

# --- 2. 对话框：无图/无 ID 时确定按钮禁用 ---
dlg = m.ManualAddDialog(w, os.getcwd())
assert not dlg.btn_ok.isEnabled(), "未选图、无 ID 时确定应禁用"
dlg._set_paths(["_m1.jpg"])
assert dlg.paths == ["_m1.jpg"]
assert "_m1.jpg" in dlg.edit_path.text()
assert dlg.preview.pixmap() is not None and not dlg.preview.pixmap().isNull(), "预览为空"
assert not dlg.btn_ok.isEnabled(), "只选图未填 ID 时确定应禁用"
dlg.edit_label.setText("P100")
assert dlg.btn_ok.isEnabled()
print("[2] 对话框：预览正常，ID 为空时确定禁用，填写后启用")

# --- 3. 多选图片共用同一个 ID ---
dlg._set_paths(["_m1.jpg", "_m2.jpg"])
assert len(dlg.paths) == 2
assert dlg.edit_path.text().startswith("2 张")
assert dlg.result() != QDialog.Accepted
dlg._on_ok()
assert dlg.result() == QDialog.Accepted, "校验通过后应 accept"
print(f"[3] 多选通过：{dlg.edit_path.text()}")

# --- 4. 校验分支（ID 为空 -> 不 accept） ---
dlg2 = m.ManualAddDialog(w, os.getcwd())
dlg2._set_paths(["_m1.jpg"])
dlg2.edit_label.setText("   ")
n_box = len(FakeBox.calls)
dlg2._on_ok()
assert dlg2.result() != QDialog.Accepted, "ID 为空白不应 accept"
assert len(FakeBox.calls) > n_box, "应提示填写 ID"
print("[4] ID 空白被拒绝（已提示）")
dlg.deleteLater()
dlg2.deleteLater()

# --- 5. 主流程入库（用 FakeDlg 绕过模态 exec_） ---
class FakeDlg:
    def __init__(self, parent=None, last_dir=""):
        self.paths = ["_m1.jpg", "_m2.jpg"]
        self._label = "P100"

    def exec_(self):
        return QDialog.Accepted

    def label(self):
        return self._label


real_dlg = m.ManualAddDialog
m.ManualAddDialog = FakeDlg
try:
    n0 = len(w.persons)
    w._manual_add()
    assert len(w.persons) == n0 + 2, f"应入库 2 张，实际 {len(w.persons) - n0}"
    assert all(p["label"] == "P100" for p in w.persons[n0:])
    assert w.lbl_count.text() == f"{len(w.persons)} / {m.MAX_PERSONS}"
    # 画廊 widget 数量应同步（末尾是 stretch）
    assert w.gallery_layout.count() == len(w.persons) + 1
    print(f"[5] 主流程入库通过：{w.lbl_count.text()}，画廊 {w.gallery_layout.count() - 1} 项")

    # --- 6. 坏图片被跳过且不影响其他 ---
    class FakeDlgBad(FakeDlg):
        def __init__(self, parent=None, last_dir=""):
            self.paths = ["_bad.jpg", "_m1.jpg"]
            self._label = "P200"

    m.ManualAddDialog = FakeDlgBad
    n1 = len(w.persons)
    w._manual_add()
    assert len(w.persons) == n1 + 1, "坏图片应被跳过，好图片仍入库"
    print("[6] 非法图片被跳过，正常图片仍入库")

    # --- 7. 库满后不再入库 ---
    class FakeDlgMany(FakeDlg):
        def __init__(self, parent=None, last_dir=""):
            self.paths = ["_m1.jpg"] * 20
            self._label = "P300"

    m.ManualAddDialog = FakeDlgMany
    w._manual_add()
    assert len(w.persons) == m.MAX_PERSONS, f"应停在 {m.MAX_PERSONS}，实际 {len(w.persons)}"
    print(f"[7] 达到上限停止入库：{w.lbl_count.text()}")
finally:
    m.ManualAddDialog = real_dlg

# --- 8. MQTT 路径仍正常（复用同一入库逻辑后回归） ---
import base64

w._clear_persons()
b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()
w._on_person_received({"local_id": "P048", "images": {"upper": b64},
                       "source_video": "cam01.mp4", "capture_time": "2026-10-23 14:40"})
assert len(w.persons) == 1 and w.persons[0]["label"] == "P048"
print("[8] MQTT 入库回归通过")

# --- 9. 入库特征可用（维度正确，可直接参与匹配） ---
feat = w.persons[0]["feature"]
assert feat.shape == (2048,), f"特征维度异常：{feat.shape}"
assert abs(float((feat ** 2).sum()) - 1.0) < 1e-3, "特征未 L2 归一化"
print("[9] 特征维度/归一化正常，可直接用于检索")

# --- 10. 真实模态流程：打开对话框并自动点「确定」 ---
from PyQt5.QtCore import QTimer

w._clear_persons()


def _auto_accept():
    for wg in app.topLevelWidgets():
        if isinstance(wg, m.ManualAddDialog) and wg.isVisible():
            wg._set_paths(["_m1.jpg"])
            wg.edit_label.setText("P777")
            wg.btn_ok.click()
            return
    QTimer.singleShot(200, _auto_accept)


QTimer.singleShot(300, _auto_accept)
w._manual_add()  # 内部 exec_() 是模态阻塞，靠上面的定时器点确定
assert len(w.persons) == 1 and w.persons[0]["label"] == "P777", "模态添加未生效"
print(f"[10] 真实模态流程通过：{w.statusBar().currentMessage()}")

for p in ("_m1.jpg", "_m2.jpg", "_bad.jpg"):
    if os.path.exists(p):
        os.remove(p)
print("全部「手动添加」测试通过")
