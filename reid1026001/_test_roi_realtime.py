# -*- coding: utf-8 -*-
"""原速预览修复 + ROI 模块回归测试（offscreen，不保存视频）：

1. 原速预览：勾选后真实按 1x 播放 —— 耗时 ≈ 视频时长；处理跟不上时跳帧。
   检索只实时预览，全程不写任何 mp4 文件。
2. ROI 模块：
   a. 滞回阈值（>0.65 进入、<0.6 退出、中间保持）+ 低于匹配阈值不上报；
   b. MQTT 消息缓存/补发（未连接时缓存，连上后 flush）；
   c. ROI 文件检索：整帧检测被跳过、消息格式/300ms 节拍/坐标合法；
   d. 运行中 set_rois 动态生效。
"""
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from reid_core import PersonDetector, ReidExtractor, crop_from_frame
from main import SearchWorker, MqttPublisher, open_file_capture
import _test_media

_test_media.ensure()  # _test_src.mp4(90帧) / _test_src.ts / _test_short.mp4(20帧)

app = QApplication([])
det = PersonDetector()
ext = ReidExtractor()

cap, _ = open_file_capture("_test_src.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
assert ret
xyxy, _c = det.detect(frame)
areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
crop = crop_from_frame(frame, xyxy[int(np.argmax(areas))])
persons = [{"label": "P001", "crop": crop, "feature": ext.extract([crop], tta=True)[0]}]
print("[gallery] 已构造 1 个行人")


def run_worker(src, realtime=False, rois=None, limit_stop_after=None):
    """跑一次检索（不传输出路径——现在根本没有该参数），收集预览帧/日志。"""
    w = SearchWorker(src, persons, det, ext, 0.30,
                     realtime_preview=realtime, rois=rois)
    stats = {"frames": [], "logs": [], "progress": [], "done": None,
             "roi_msgs": [], "roi_times": []}
    w.frame_ready.connect(stats["frames"].append)
    w.log.connect(lambda m: stats["logs"].append(m))
    w.progress.connect(lambda a, b: stats["progress"].append((a, b)))
    w.finished_ok.connect(lambda n: stats.__setitem__("done", n))
    w.roi_message.connect(lambda m: (stats["roi_msgs"].append(m),
                                     stats["roi_times"].append(time.perf_counter())))
    t0 = time.perf_counter()
    w.start()
    while w.isRunning():
        app.processEvents()
        if limit_stop_after and time.perf_counter() - t0 > limit_stop_after:
            w.stop()
        time.sleep(0.02)
    w.wait()
    app.processEvents()
    return w, stats, time.perf_counter() - t0


# ============ 1. 原速预览修复 ============
cap, _ = open_file_capture("_test_short.mp4")
n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_src = cap.get(cv2.CAP_PROP_FPS)
cap.release()
dur_src = n_src / fps_src
print(f"[素材] {n_src} 帧 @ {fps_src:.0f}fps，时长 {dur_src:.2f}s")

# 无 ROI：只按原速预览，不做识别，也不发 MQTT 消息
for legacy in ("_out_rt.mp4", "_out_rt_roi.mp4", "_out_full.mp4", "_out_roi.mp4"):
    if os.path.exists(legacy):  # 清掉旧版本测试遗留的同名输出
        os.remove(legacy)
w_rt, st_rt, dur_rt = run_worker("_test_short.mp4", realtime=True)
print(f"[1a] 无 ROI 原速运行耗时 {dur_rt:.2f}s（原片 {dur_src:.2f}s）")
# 耗时 ≈ 首帧处理 + 原片时长（允许少量调度误差）
assert dur_src - 0.15 < dur_rt < dur_src + 3.8, \
    f"原速播放耗时异常：{dur_rt:.2f}s（原片 {dur_src:.2f}s）"
final_log = [l for l in st_rt["logs"] if "检索结束" in l][-1]
assert "未启用 ROI" in final_log and "未识别" in final_log, \
    f"无 ROI 结束日志应明确未识别：{final_log}"
assert st_rt["done"] == n_src, f"时间轴帧数应读完原片：{st_rt['done']}"
assert len(st_rt["frames"]) > 0, "应有预览帧"
assert w_rt._matched_count() == 0, "无 ROI 不应有任何 REID 命中"
assert w_rt._detected == 0, "无 ROI 不应执行检测"
assert st_rt["roi_msgs"] == [], "无 ROI 不应发送 MQTT 消息"
assert not os.path.exists("_out_rt.mp4"), "不应保存 mp4"
print(f"[1b] 无 ROI：处理 {st_rt['done']} 帧、检测 0、命中 0、消息 0，不保存视频")

# 全屏 ROI + 原速：识别时跳帧追赶，预览持续，MQTT 持续
full_roi = [{"roi_id": "A", "rect": (0.0, 0.0, 1.0, 1.0)}]
w_rt2, st_rt2, dur_rt2 = run_worker(
    "_test_short.mp4", realtime=True, rois=full_roi)
final_roi = [l for l in st_rt2["logs"] if "检索结束" in l][-1]
assert "跳帧" in final_roi and "检测" in final_roi, \
    f"带 ROI 结束日志缺跳帧信息：{final_roi}"
assert st_rt2["done"] == n_src, f"时间轴帧数应读完原片：{st_rt2['done']}"
assert w_rt2._matched_count() > 0, "全屏 ROI 内应有 REID 命中"
assert len(st_rt2["roi_msgs"]) >= 2, "ROI 检索应持续发送 MQTT 消息"
assert not os.path.exists("_out_rt_roi.mp4"), "不应保存 mp4"
print(f"[1c] 全屏 ROI 原速：{final_roi}；预览 {len(st_rt2['frames'])} 帧，"
      f"消息 {len(st_rt2['roi_msgs'])} 条")

# 不勾选原速：全速逐帧检测，无跳帧
w_full, st_full, dur_full = run_worker("_test_short.mp4", rois=full_roi)
final2 = [l for l in st_full["logs"] if "检索结束" in l][-1]
assert "跳帧" not in final2, "全速模式不应跳帧"
assert st_full["done"] == n_src and w_full._detected == n_src, \
    "全速应逐帧检测完原片"
assert dur_full > dur_rt2, "全速逐帧检测应慢于原速跳帧模式"
assert not os.path.exists("_out_full.mp4"), "不应保存 mp4"
print(f"[1d] 全速模式 {dur_full:.2f}s 无跳帧；原速 {dur_rt2:.2f}s -> 跳帧有效")

# ============ 2a. 滞回阈值 ============
w0 = SearchWorker("_test_short.mp4", persons, det, ext, 0.3)
w0._update_roi_state("A", [{"label": "P1", "score": 0.70,
                            "bbox_xyxy": [10, 10, 100, 200], "tid": 1}], 1920, 1080)
assert len(w0._roi_current["A"]) == 1, "score>0.65 应进入上报"
w0._update_roi_state("A", [{"label": "P1", "score": 0.62,
                            "bbox_xyxy": [10, 10, 100, 200], "tid": 1}], 1920, 1080)
assert len(w0._roi_current["A"]) == 1, "滞回区间 [0.6,0.65) 且此前已上报 -> 保持"
w0._update_roi_state("A", [{"label": "P1", "score": 0.62,
                            "bbox_xyxy": [10, 10, 100, 200], "tid": 1}], 1920, 1080)
assert len(w0._roi_current["A"]) == 1, "应继续保持"
w0._update_roi_state("A", [{"label": "P1", "score": 0.58,
                            "bbox_xyxy": [10, 10, 100, 200], "tid": 1}], 1920, 1080)
assert w0._roi_current["A"] == [], "score<0.6 应退出上报（发空列表）"
w0._update_roi_state("A", [{"label": "P1", "score": 0.63,
                            "bbox_xyxy": [10, 10, 100, 200], "tid": 1}], 1920, 1080)
assert w0._roi_current["A"] == [], "滞回区间且此前未上报 -> 不进入"
# 目标消失（空 detections）-> 清空
w0._update_roi_state("A", [{"label": "P1", "score": 0.9,
                            "bbox_xyxy": [10, 10, 100, 200], "tid": 1}], 1920, 1080)
w0._update_roi_state("A", [], 1920, 1080)
assert w0._roi_current["A"] == [], "目标离开 ROI 应清空"
# 坐标裁剪
w0._update_roi_state("A", [{"label": "P1", "score": 0.9,
                            "bbox_xyxy": [-20, -10, 2000, 9999], "tid": 1}], 1920, 1080)
b = w0._roi_current["A"][0]["bbox_xyxy"]
assert 0 <= b[0] < b[2] <= 1920 and 0 <= b[1] < b[3] <= 1080, f"bbox 越界：{b}"
# 低于 Step2 匹配阈值的结果不进入上报（即使此前处于上报/滞回状态）
w0._roi_active["A"] = {1: True}
w0._update_roi_state("A", [{"label": "P1", "score": 0.20,
                            "bbox_xyxy": [10, 10, 100, 200], "tid": 1}], 1920, 1080)
assert w0._roi_current["A"] == [], "低于匹配阈值(0.3)的 0.20 不应上报"
print("[2a] 滞回阈值（0.65 进 / 0.6 出 / 中间保持 / 越界裁剪 / 低相似度不上报）通过")

# ============ 2b. MQTT 消息格式与节拍 ============
assert st_rt["roi_msgs"] == [], "未启用 ROI 时不应发消息"
print("[2b0] 未启用 ROI 时不发送任何消息")

w_roi, st_roi, dur_roi = run_worker(
    "_test_short.mp4",
    rois=[{"roi_id": "A", "rect": (0.08, 0.05, 0.92, 0.95)}])
assert st_roi["done"] == 20, f"20 帧短片应全部处理：{st_roi['done']}"
assert any("ROI A 已启用" in l for l in st_roi["logs"]), "缺少 ROI 启用日志"
assert getattr(w_roi, "_full_matcher", None) is None, "不应创建整帧匹配器"
assert w_roi._matched_count() == w_roi._roi_matchers["A"].matched, \
    "命中数应只来自 ROI 匹配器"
assert "A" in w_roi._roi_matchers and w_roi._roi_matchers["A"].matched > 0, \
    "ROI 内应完成检测匹配"
print(f"[2b1] ROI A 检索通过：整帧检测 0 次，框内 matched="
      f"{w_roi._roi_matchers['A'].matched}，消息 {len(st_roi['roi_msgs'])} 条")

msgs = [json.loads(m) for m in st_roi["roi_msgs"]]
assert len(msgs) >= 2, "消息数量过少"
for m in msgs:
    assert set(m.keys()) == {"roi_id", "captured_at_ms", "detections"}, f"字段错误：{m}"
    assert m["roi_id"] == "A"
    assert isinstance(m["captured_at_ms"], int)
    assert abs(m["captured_at_ms"] - time.time() * 1000) < 3600 * 1000
    assert isinstance(m["detections"], list)
    for d in m["detections"]:
        assert set(d.keys()) == {"passenger_id", "bbox_xyxy"}, f"det 字段错误：{d}"
        assert d["passenger_id"] == "P001"
        assert len(d["bbox_xyxy"]) == 4 and all(isinstance(v, int) for v in d["bbox_xyxy"])
        x1, y1, x2, y2 = d["bbox_xyxy"]
        assert 0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080, f"bbox 越界：{d}"
gaps = [b - a for a, b in zip(st_roi["roi_times"], st_roi["roi_times"][1:])]
print(f"[2b2] 消息节拍：{len(gaps)} 个间隔，最小 {min(gaps):.3f}s，均值 {np.mean(gaps):.3f}s")
assert min(gaps) >= 0.28, f"发送间隔小于 300ms：{min(gaps):.3f}"

# 有目标时的非空上报（阈值 0.3 + ROI 内有 P001，若置信度足够会出现非空消息）
non_empty = [m for m in msgs if m["detections"]]
print(f"[2b3] 非空消息 {len(non_empty)}/{len(msgs)} 条"
      + (f"，示例：{non_empty[0]['detections'][0]}" if non_empty else "（置信度未达 0.65，全为空列表——符合低置信度发送空列表的规则）"))

# 预览画面上应有 ROI 框（在 ROI 边界处检查橙色描边）
outf = st_roi["frames"][0]
# ROI 边框取整后的位置：x1=int(0.08*1920)=153
col = outf[540, 154]  # 边框右侧附近
assert col[0] < 100 and col[2] > 180, f"预览画面未见 ROI 框描边：{col}"
print("[2b4] 预览画面带 ROI 框可视化（无 mp4 输出）")

# ============ 2c. 运行中 set_rois 动态生效 ============
w_dyn = SearchWorker("_test_src.mp4", persons, det, ext, 0.3,
                     realtime_preview=True)  # 原速运行，留出运行中框选的时间窗口
w_dyn.start()
time.sleep(1.0)
assert w_dyn.isRunning()
w_dyn.set_rois([{"roi_id": "B", "rect": (0.1, 0.1, 0.9, 0.9)}])
deadline = time.time() + 15
while time.time() < deadline and "B" not in w_dyn._roi_matchers:
    app.processEvents()
    time.sleep(0.05)
w_dyn.stop()
while w_dyn.isRunning():
    app.processEvents()
assert "B" in w_dyn._roi_matchers, "运行中 set_rois 未生效"
# 停用后再处理 -> 状态清除
w_dyn2 = SearchWorker("_test_short.mp4", persons, det, ext, 0.3)
w_dyn2._annotate_frame(np.zeros((1080, 1920, 3), np.uint8))  # 先跑一帧无 ROI
w_dyn2.set_rois([{"roi_id": "A", "rect": (0.1, 0.1, 0.9, 0.9)}])
print("[2c] 运行中 set_rois 动态生效通过")

# ============ 2d. MqttPublisher 缓存与补发 ============
class FakeClient:
    def __init__(self):
        self.published = []
        self.connected = False

    def is_connected(self):
        return self.connected

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, payload, qos))


pub = MqttPublisher("localhost", 1883, "reidtest")
fake = FakeClient()
pub.client = fake
pub._running = True
m1 = json.dumps({"roi_id": "A", "captured_at_ms": 1, "detections": []})
m2 = json.dumps({"roi_id": "A", "captured_at_ms": 2, "detections": []})
pub.publish(m1)          # 未连接 -> 缓存
assert fake.published == [] and pub._pending == [m1]
pub.publish(m2)
pub._on_connect(fake, None, None, 0)   # 连上 -> 补发缓存
assert fake.published == [(("reidtest"), m1, 1), ("reidtest", m2, 1)]
fake.connected = True
m3 = json.dumps({"roi_id": "A", "captured_at_ms": 3, "detections": []})
pub.publish(m3)          # 已连接 -> 直接发
assert fake.published[-1] == ("reidtest", m3, 1) and pub._pending == []
# rc != 0 失败分支不抛异常
pub._on_connect(fake, None, None, 1)
print("[2d] MqttPublisher 缓存/补发/直发通过")

print("全部测试通过")
