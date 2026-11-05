# -*- coding: utf-8 -*-
"""验证 MQTT 接收链路：不真连 broker，直接用模拟消息驱动主线程处理逻辑。

1. decode_base64_image：正常 base64 / data-url 前缀 / 非法输入。
2. MainWindow._on_person_received：真实图像帧编码 base64 -> 入库、标签正确、上限控制。
"""
import base64
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from main import MainWindow, decode_base64_image, MAX_PERSONS

# --- 1. decode_base64_image 单测 ---
img = np.full((120, 80, 3), 200, dtype=np.uint8)
ok_b64 = base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()
assert decode_base64_image(ok_b64) is not None, "正常 base64 解码失败"
assert decode_base64_image("data:image/jpeg;base64," + ok_b64) is not None, "data-url 前缀解码失败"
assert decode_base64_image("not-base64!!!") is None, "非法输入应返回 None"
assert decode_base64_image(None) is None, "None 应返回 None"
print("[1] decode_base64_image 单测通过")

# --- 2. 端到端：模拟 MQTT 消息入库 ---
app = QApplication([])
w = MainWindow()

cap = cv2.VideoCapture("2bee809c50e7ce07de5a3a0f2651dcc8.mp4")
ret, frame = cap.read()
cap.release()
assert ret, "测试视频读取失败"

_, buf = cv2.imencode(".jpg", frame)
b64 = base64.b64encode(buf).decode()

payload = {
    "person_id": "a40487ad9d504923b7f496465c6a4e99-2026-09-23",
    "local_id": "P048",
    "passenger_id": "a40487ad9d504923b7f496465c6a4e99-2026-09-23",
    "capture_time": "2026-09-23 18:42:29.333",
    "timestamp": "2026-10-22T11:19:32.549",
    "source_video": "BC5496691_1_20260923T182632Z_20260923T184230Z.mp4",
    "score": 7.2,
    "has_face": True,
    "images": {"face": b64, "upper": b64},
}
# 快速整图提特征（绕过 GPU 检测耗时：不调用 detector 也行，但保持真实链路）
w._on_person_received(payload)
assert len(w.persons) == 1, "入库失败"
assert w.persons[0]["label"] == "P048", f"标签错误: {w.persons[0]['label']}"
assert w.persons[0]["feature"].shape == (2048,), "特征维度错误"
print("[2] 正常消息入库通过：label=P048, feat_dim=2048")

# 缺 upper 的消息应被忽略
w._on_person_received({"local_id": "P049", "images": {"face": b64}})
assert len(w.persons) == 1, "缺 upper 的消息不应入库"
print("[3] 缺 images.upper 的消息被正确忽略")

# local_id 缺失时回退 passenger_id
w._on_person_received({**payload, "local_id": "", "passenger_id": "PG-FALLBACK"})
assert w.persons[-1]["label"] == "PG-FALLBACK", "passenger_id 回退失败"
print("[4] local_id 缺失时回退 passenger_id 通过")

# 上限控制：填满后忽略
for i in range(MAX_PERSONS + 5):
    w._on_person_received({**payload, "local_id": f"P{i:03d}"})
assert len(w.persons) == MAX_PERSONS, f"上限控制失败: {len(w.persons)}"
print(f"[5] 上限控制通过（{MAX_PERSONS} 张）")

# 状态栏日志
print("[log]", w.statusBar().currentMessage())
print("全部测试通过")
