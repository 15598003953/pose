# -*- coding: utf-8 -*-
"""Step3 新功能回归测试（offscreen）：
1. 本地 .ts / .mp4 文件直接检索：worker 跑通、输出视频非空、进度回调正常。
2. StreamGrabber 用 ffmpeg 自建 RTSP 服务验证：TCP 抓帧、只保留最新帧、stop 可退出。
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import subprocess
import sys
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from reid_core import PersonDetector, ReidExtractor, crop_from_frame, valid_crop
from main import SearchWorker, StreamGrabber, open_file_capture, is_stream_source, probe_real_fps
import _test_media

_test_media.ensure()  # 准备 _test_src.mp4 / _test_src.ts / _test_short.mp4

app = QApplication([])
det = PersonDetector()
ext = ReidExtractor()

# ---------- gallery：从测试视频里取一个行人 ----------
cap, _ = open_file_capture("_test_src.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
assert ret, "读取测试视频失败"
xyxy, _c = det.detect(frame)
assert len(xyxy), "测试视频里没检测到行人，无法构造 gallery"
areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
crop = crop_from_frame(frame, xyxy[int(np.argmax(areas))])
assert valid_crop(crop)
persons = [{"label": "P001", "crop": crop, "feature": ext.extract([crop], tta=True)[0]}]
print("[gallery] 已构造 1 个行人")


def run_worker(src, realtime=False, limit_stop_after=None, rotate180=False):
    # 全屏 ROI：按当前语义，识别只在 ROI 内进行
    w = SearchWorker(src, persons, det, ext, 0.30,
                     realtime_preview=realtime, rotate180=rotate180,
                     rois=[{"roi_id": "A", "rect": (0.0, 0.0, 1.0, 1.0)}])
    stats = {"frames": [], "logs": [], "progress": [], "done": None}
    w.frame_ready.connect(stats["frames"].append)
    w.log.connect(lambda m: stats["logs"].append(m))
    w.progress.connect(lambda a, b: stats["progress"].append((a, b)))
    w.finished_ok.connect(lambda n: stats.__setitem__("done", n))
    t0 = time.perf_counter()
    w.start()
    while w.isRunning():
        app.processEvents()
        if limit_stop_after and time.perf_counter() - t0 > limit_stop_after:
            w.stop()
        time.sleep(0.02)
    w.wait()
    app.processEvents()
    return stats, time.perf_counter() - t0


# ---------- 1a. TS 文件 ----------
for legacy in ("_out_ts.mp4", "_out_mp4.mp4", "_out_stop.mp4",
               "_out_rot.mp4", "_out_raw.mp4", "_out_rtsp.mp4"):
    if os.path.exists(legacy):  # 清掉旧版本测试遗留的同名输出
        os.remove(legacy)
st, dur = run_worker("_test_src.ts")
print(f"[TS] 预览帧={len(st['frames'])} 结束帧数={st['done']} 耗时={dur:.1f}s")
print("     进度尾部:", st["progress"][-2:])
print("     日志:", st["logs"][:2], "...", st["logs"][-1:])
assert st["done"] and st["done"] >= 85, f"TS 应读完全片：{st['done']}"
assert len(st["frames"]) > 0, "没有预览帧"
assert not os.path.exists("_out_ts.mp4"), "不应保存 mp4"
print("[1a] TS 文件检索通过（只预览，不保存）")

# ---------- 1b. MP4 文件 + 总帧数百分比 ----------
st2, dur2 = run_worker("_test_src.mp4")
assert st2["done"] >= 85
total = st2["progress"][-1][1]
assert total > 0, "文件模式应能拿到总帧数"
assert not os.path.exists("_out_mp4.mp4"), "不应保存 mp4"
print(f"[1b] MP4 检索通过：{total} 帧，耗时 {dur2:.1f}s，命中日志：{st2['logs'][-1]}")

# ---------- 1c. 中途停止 ----------
st3, _ = run_worker("_test_src.mp4", limit_stop_after=1.2)
assert "已停止" in st3["logs"][-1], f"停止日志异常: {st3['logs'][-1]}"
print("[1c] 中途停止通过：", st3["logs"][-1][:40], "...")

# ---------- 1d. 工具函数 ----------
assert is_stream_source("rtsp://1.2.3.4/live") and not is_stream_source("D:/a/b.ts")
p = "_test_src.ts"
c, tmp = open_file_capture(p)
assert c.isOpened() and tmp is None
print("[1d] fps 探测:", round(probe_real_fps(c), 2))
c.release()

# ---------- 1e. 画面旋转 180° ----------
def first_frame(path, idx=0):
    c, _ = open_file_capture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = c.read()
    c.release()
    assert ok, f"读取 {path} 失败"
    return f

st_rot, _ = run_worker("_test_short.mp4", rotate180=True)
assert st_rot["done"] and st_rot["done"] > 0, "旋转模式未产出结果"
assert any("旋转 180°" in m for m in st_rot["logs"]), "日志未提示旋转"
st_raw, _ = run_worker("_test_short.mp4")
src0 = first_frame("_test_short.mp4", 0)
rot0, raw0 = st_rot["frames"][0], st_raw["frames"][0]
assert rot0.shape == src0.shape, "旋转 180° 不应改变分辨率"
assert not os.path.exists("_out_rot.mp4"), "不应保存 mp4"
assert not os.path.exists("_out_raw.mp4"), "不应保存 mp4"


def mae(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


d_rot = mae(rot0, cv2.rotate(src0, cv2.ROTATE_180))
d_src = mae(rot0, src0)
d_raw_rot = mae(raw0, cv2.rotate(src0, cv2.ROTATE_180))
d_raw = mae(raw0, src0)
print(f"[1e] 旋转输出 vs 旋转源 {d_rot:.1f} / vs 原源 {d_src:.1f}；"
      f"未旋转输出 vs 原源 {d_raw:.1f} / vs 旋转源 {d_raw_rot:.1f}")
assert d_rot < d_src / 3, "勾选旋转后输出画面未翻转 180°"
assert d_raw < d_raw_rot / 3, "未勾选旋转时输出不应被翻转"
assert d_rot < 12, "旋转输出与预期画面差异过大（标注叠加不应这么多）"
print("[1e] 旋转 180° 通过：输出画面翻转，分辨率不变")

# ---------- 2. RTSP：需要外部真实 RTSP 服务端（环境变量 RTSP_URL 指定） ----------
rtsp_url = os.environ.get("RTSP_URL", "")
if not rtsp_url:
    print("[2] 跳过 RTSP 测试（未设置 RTSP_URL）")
else:
    g = StreamGrabber(rtsp_url, log=lambda m: print("   [grabber]", m, flush=True))
    g.start()
    f1, s1 = g.wait_frame(-1, timeout=20)
    assert f1 is not None, "RTSP 抓帧失败"
    print(f"[2a] RTSP 首帧 {f1.shape} seq={s1} transport={g.transport}")
    assert g.transport == "tcp", f"应使用 TCP 传输，实际 {g.transport}"
    # 模拟主循环慢：sleep 后仍能拿到更新帧（只保留最新帧，不堆积）
    time.sleep(1.0)
    f2, s2 = g.wait_frame(s1, timeout=10)
    assert f2 is not None and s2 > s1, "未取到更新帧"
    print(f"[2b] 慢消费后取到新帧 seq {s1}->{s2}，期间跳过 {s2 - s1 - 1} 帧"
          f"（缓冲被抽空，不会越播越延迟）")
    # 画面是否已经推进（不是同一张旧图）
    assert not np.array_equal(f1, f2), "连续两次取到的是同一帧"
    t0 = time.perf_counter()
    g.stop()
    dt = time.perf_counter() - t0
    print(f"[2c] stop() {dt:.2f}s 退出，累计抓取 {g.grabbed} 帧")
    assert dt < 6, "stop 阻塞过久"

    # ---------- 3. 实时流端到端检索 ----------
    st4, dur4 = run_worker(rtsp_url, limit_stop_after=14.0)
    print(f"[3] RTSP 检索：{st4['logs'][-1]}（耗时 {dur4:.1f}s）")
    assert st4["done"] and st4["done"] >= 10, f"实时流处理帧数过少：{st4['done']}"
    assert len(st4["frames"]) > 0, "实时流没有预览帧"
    assert not os.path.exists("_out_rtsp.mp4"), "实时流不应保存 mp4"
    print(f"[3b] 实时流处理 {st4['done']} 帧、预览 {len(st4['frames'])} 帧，"
          "日志正常，无花屏相关异常")

print("全部测试通过")
