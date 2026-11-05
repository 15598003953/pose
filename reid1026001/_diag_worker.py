# -*- coding: utf-8 -*-
"""端到端验证新匹配逻辑：用真实 SearchWorker 跑 video1 -> video3 检索。
gallery 从 video1 取两个行人（同一人物两张图共用标签，验证多镜头匹配）。
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from reid_core import PersonDetector, ReidExtractor, crop_from_frame, valid_crop
from main import SearchWorker

det = PersonDetector()
ext = ReidExtractor()

# gallery：2bee 视频中两个不同的行人，P1 用两张图（不同帧）验证多镜头
cap = cv2.VideoCapture("2bee809c50e7ce07de5a3a0f2651dcc8.mp4")
persons = []
p1_frames = 0
while len(persons) < 3 and p1_frames < 2:
    ret, frame = cap.read()
    if not ret:
        break
    xyxy, confs = det.detect(frame)
    if not len(xyxy):
        continue
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    c = crop_from_frame(frame, xyxy[int(np.argmax(areas))])
    if not valid_crop(c):
        continue
    feat = ext.extract([c], tta=True)[0]
    if p1_frames < 2:
        persons.append({"label": "P1", "feature": feat})  # 同一标签两张图
        p1_frames += 1
    else:
        persons.append({"label": "P2", "feature": feat})
cap.release()
print("gallery:", [(p["label"]) for p in persons])

app = QApplication([])
w = SearchWorker("3c5539483f8adf2025896b26974b0694.mp4", persons, det, ext, 0.48,
                 rois=[{"roi_id": "A", "rect": (0.0, 0.0, 1.0, 1.0)}])  # 全屏 ROI 诊断识别
w.log.connect(lambda m: print("[log]", m))
w.run()
print("诊断完成（不再保存 mp4）：命中", w._matched_count(),
      "，检测帧", getattr(w, "_detected", 0))
