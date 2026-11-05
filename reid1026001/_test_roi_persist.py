# -*- coding: utf-8 -*-
"""ROI 两项修复的回归测试（offscreen）：

1. ROI 内 REID 标注修复：此前 SearchWorker._annotate_frame 丢弃了
   TrackMatcher.annotate 的返回值（画在 canvas 副本上），导致输出画面上
   只剩橙色 ROI 框，框内人员的绿框/标签/置信度全部丢失。
   修复后输出视频里应出现绿色命中框（含标签 + 置信度文字）。

2. ROI 位置持久化：_save_roi_config 写 roi_config.json；新 MainWindow
   启动时自动恢复 rect / enabled / 复选框 / 行标签，且不会自动连 MQTT。
"""
import glob
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import main as m
from reid_core import PersonDetector, ReidExtractor, crop_from_frame
import _test_media

_test_media.ensure()  # _test_src.mp4(90帧) / _test_short.mp4(20帧)

app = QApplication([])
det = PersonDetector()
ext = ReidExtractor()

cap, _ = m.open_file_capture("_test_src.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
assert ret
xyxy, _c = det.detect(frame)
areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
crop = crop_from_frame(frame, xyxy[int(np.argmax(areas))])
persons = [{"label": "P001", "crop": crop,
            "feature": ext.extract([crop], tta=True)[0]}]
print("[gallery] 已构造 1 个行人（匹配阈值 0.30）")

# ============ 0. 无 ROI：完全不识别，也不保存视频 ============
def run_worker(src, realtime=False, rois=None, threshold=0.30):
    """跑一次检索，收集预览帧/日志/消息；返回 (worker, stats)。"""
    wk = m.SearchWorker(src, persons, det, ext, threshold,
                        realtime_preview=realtime, rois=rois)
    st = {"frames": [], "msgs": [], "logs": [], "done": None}
    wk.frame_ready.connect(st["frames"].append)
    wk.roi_message.connect(st["msgs"].append)
    wk.log.connect(st["logs"].append)
    wk.finished_ok.connect(lambda n: st.__setitem__("done", n))
    wk.start()
    while wk.isRunning():
        app.processEvents()
        time.sleep(0.02)
    wk.wait()
    app.processEvents()
    return wk, st


before = set(glob.glob("*reid_result*.mp4"))
w0, st0 = run_worker("_test_short.mp4")
after = set(glob.glob("*reid_result*.mp4"))
assert after == before, "检索不应再生成 *_reid_result_*.mp4 文件"
assert w0._detected == 0, "无 ROI 不应执行行人检测"
assert w0._matched_count() == 0, "无 ROI 不应执行 REID"
assert w0._roi_matchers == {}, "无 ROI 不应创建匹配器"
assert st0["msgs"] == [], "无 ROI 不应发送 MQTT 消息"
assert len(st0["frames"]) > 0 and st0["done"] == 20, \
    f"无 ROI 也应完整预览：{len(st0['frames'])} 帧 / done={st0['done']}"
for fr in st0["frames"]:
    assert int(np.all(fr == (0, 255, 0), axis=-1).sum()) == 0, \
        "无 ROI 预览画面不应出现命中框"
print("[0] 无 ROI：检测 0 帧、REID 命中 0、MQTT 消息 0，只预览不识别，通过")

# ============ 1. ROI 内 REID 标注应出现在预览画面 ============
w, st = run_worker("_test_short.mp4",
                   rois=[{"roi_id": "A", "rect": (0.05, 0.05, 0.95, 0.95)}])
assert st["done"] == 20, f"20 帧短片应全部处理：{st['done']}"
assert any("ROI A 已启用" in l for l in st["logs"]), "缺少 ROI 启用日志"
assert getattr(w, "_full_matcher", None) is None, "不应创建整帧匹配器"
assert "A" in w._roi_matchers and w._roi_matchers["A"].matched > 0, \
    "ROI 内应完成检测匹配"

# 扫描预览帧：绿色命中框 (0,255,0) + 绿色标签文字必须出现
max_green, frame_with_hit = 0, -1
for i, fr in enumerate(st["frames"]):
    green = int(np.all(fr == (0, 255, 0), axis=-1).sum())
    if green > max_green:
        max_green, frame_with_hit = green, i
assert max_green > 100, (
    f"预览画面未出现绿色命中框（最多 {max_green} 像素）——ROI 标注丢失")
print(f"[1] ROI 标注修复通过：命中帧 #{frame_with_hit} 绿色像素 {max_green}，"
      f"matched={w._roi_matchers['A'].matched}，MQTT 消息 {len(st['msgs'])} 条")

# ============ 1b. 检索开始后再框选 ROI：下一帧开始 REID ============
wd = m.SearchWorker("_test_src.mp4", persons, det, ext, 0.30,
                    realtime_preview=True)  # 原速运行，留出运行中框选的时间窗口
wd.start()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 0.8:  # 先无 ROI 预览一段时间
    app.processEvents()
    time.sleep(0.02)
assert wd.isRunning(), "原速检索应仍在运行"
assert wd._detected == 0, "框选前不应识别"
wd.set_rois([{"roi_id": "A", "rect": (0.05, 0.05, 0.95, 0.95)}])  # 模拟框选完成
deadline = time.perf_counter() + 15
while wd.isRunning() and time.perf_counter() < deadline:
    app.processEvents()
    if "A" in wd._roi_matchers and wd._roi_matchers["A"].matched > 0:
        break
    time.sleep(0.02)
assert "A" in wd._roi_matchers, "框选后未创建 ROI 匹配器"
assert wd._roi_matchers["A"].matched > 0, "框选后 ROI 内仍未完成 REID"
wd.stop()
while wd.isRunning():
    app.processEvents()
    time.sleep(0.02)
wd.wait()
app.processEvents()
print(f"[1b] 运行中框选 ROI：框内 REID 命中 {wd._roi_matchers['A'].matched} 次，通过")

# ============ 1c. 低相似度：不显示、不上报 ============
# 不跑线程，直接单测上报过滤：阈值 0.48 时 0.40（低相似度）应被丢弃
w_low = m.SearchWorker("_test_short.mp4", persons, det, ext, 0.48)
w_low._roi_active["A"] = {1: True, 2: True}  # 模拟之前已上报（滞回保持状态）
w_low._update_roi_state("A", [
    {"label": "P001", "score": 0.40, "bbox_xyxy": [10, 10, 100, 200], "tid": 1},
    {"label": "P001", "score": 0.70, "bbox_xyxy": [20, 20, 120, 220], "tid": 2},
], 1920, 1080)
reported = [d["passenger_id"] for d in w_low._roi_current["A"]]
assert reported == ["P001"], \
    f"低于匹配阈值 0.48 的 0.40 不应上报，只应保留 0.70：{reported}"
# 两条都低于阈值时全部丢弃（即使此前处于上报状态）
w_low.threshold = 0.99
w_low._roi_active["A"] = {1: True, 2: True}
w_low._update_roi_state("A", [
    {"label": "P001", "score": 0.40, "bbox_xyxy": [10, 10, 100, 200], "tid": 1},
    {"label": "P001", "score": 0.70, "bbox_xyxy": [20, 20, 120, 220], "tid": 2},
], 1920, 1080)
assert w_low._roi_current["A"] == [], \
    "低于匹配阈值的结果（即使处于滞回保持）也不应进入 MQTT 上报"
print("[1c] 低相似度过滤：低于匹配阈值不上报（滞回也无法保持），通过")

# ============ 1d. 阈值抬高后：画面不画框、不显示标签 ============
# 阈值 0.99（本场景不可达）-> 仍会检测，但没有任何达标结果，画面不应有绿框
w_hi, st_hi = run_worker("_test_short.mp4",
                         rois=[{"roi_id": "A", "rect": (0.05, 0.05, 0.95, 0.95)}],
                         threshold=0.99)
assert w_hi._detected > 0, "ROI 内仍应执行检测（只是判定不达标）"
assert w_hi._matched_count() == 0, "阈值 0.99 时不应有任何命中"
max_green_hi = max(int(np.all(fr == (0, 255, 0), axis=-1).sum())
                   for fr in st_hi["frames"])
assert max_green_hi == 0, f"低相似度不应画框/显示标签（绿色像素 {max_green_hi}）"
assert all(json.loads(x)["detections"] == [] for x in st_hi["msgs"]), \
    "低相似度不应出现在 MQTT 消息里"
print(f"[1d] 阈值 0.99：检测 {w_hi._detected} 帧、命中 0、绿色像素 0、"
      f"MQTT 消息 {len(st_hi['msgs'])} 条全为空，通过")

# ============ 2. ROI 位置持久化 ============
if os.path.exists(m.CONFIG_PATH):
    os.remove(m.CONFIG_PATH)

w2 = m.MainWindow()
w2.rois["A"]["rect"] = (0.10, 0.20, 0.80, 0.90)
w2.rois["A"]["enabled"] = True
w2.roi_rows["A"][0].setChecked(True)  # 触发 _on_roi_toggled -> _save_roi_config
assert os.path.exists(m.CONFIG_PATH), "roi_config.json 未生成"
with open(m.CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)
assert cfg["rois"]["A"]["rect"] == [0.10, 0.20, 0.80, 0.90], cfg
assert cfg["rois"]["A"]["enabled"] is True, cfg
assert cfg["rois"]["B"]["rect"] is None and cfg["rois"]["B"]["enabled"] is False, cfg
print(f"[2a] 保存通过：{cfg}")

# 新建窗口 = 模拟下一次启动
w3 = m.MainWindow()
assert w3.rois["A"]["rect"] == (0.10, 0.20, 0.80, 0.90), "启动未恢复 ROI A 位置"
assert w3.rois["A"]["enabled"] and w3.roi_rows["A"][0].isChecked(), \
    "启动未恢复 ROI A 启用状态/复选框"
assert w3.rois["B"]["rect"] is None and not w3.roi_rows["B"][0].isChecked()
assert "x 10~80%" in w3.roi_rows["A"][2].text(), \
    f"行标签未恢复：{w3.roi_rows['A'][2].text()}"
assert w3._active_rois() == [{"roi_id": "A", "rect": (0.10, 0.20, 0.80, 0.90)}], \
    "恢复的 ROI 应参与下一次检索"
assert w3.roi_publisher is None, "启动恢复 ROI 不应自动创建 MQTT 上报连接"
print("[2b] 重启恢复通过：位置 / 启用状态 / 复选框 / 行标签 / 检索生效，"
      "且未自动连 MQTT")

# 清除 ROI 后配置同步更新（rect -> null）
w3._clear_roi("A")
with open(m.CONFIG_PATH, encoding="utf-8") as f:
    cfg2 = json.load(f)
assert cfg2["rois"]["A"]["rect"] is None and cfg2["rois"]["A"]["enabled"] is False
print("[2c] 清除 ROI 后配置同步更新")

# 清理
w2.close()
w3.close()
app.processEvents()
os.remove(m.CONFIG_PATH)
print("全部测试通过")
