"""重识别独立检索线程；联合工作台使用 integration.workers 调度同一匹配器。"""

import json
import os
import threading
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# 同一个算法入口供独立界面和联合工作台复用，轨迹状态仍由各自会话维护。
from pose_project.reid.runtime import DET_THREADS, EXT_THREADS, set_phase_threads
from pose_project.reid.core import PersonDetector, ReidExtractor, crop_from_frame, valid_crop
from pose_project.reid.matcher import TrackMatcher, draw_text
from pose_project.reid.messaging import decode_base64_image
from pose_project.video.stream_capture import (
    StreamGrabber, is_stream_source, open_file_capture, probe_real_fps,
)


class SearchWorker(QThread):
    """检索线程：支持本地视频文件与 RTSP 实时流，实时预览标注画面。

    文件模式：按顺序读完全片（可选按原速预览），有总帧数可显示百分比；
    实时流模式：由 StreamGrabber 后台抓帧，本线程只消费最新帧，流断开或
    用户停止后退出线程（实时流无总帧数，进度以已处理帧数汇报）。
    只预览和上报 MQTT，不保存 MP4；finished_ok 携带本次处理的帧数。
    """
    frame_ready = pyqtSignal(np.ndarray)
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)   # (已处理帧数, 总帧数；0 表示未知)
    finished_ok = pyqtSignal(int)     # 检索结束时的已处理帧数（0 = 未产生画面）
    roi_message = pyqtSignal(str)     # ROI 检测结果 JSON（发往 MQTT topic reidtest）

    # 预览最多 25fps 推送到界面，避免高速读文件时刷爆 Qt 事件队列
    PREVIEW_INTERVAL = 0.04
    # 每个 ROI 的 MQTT 上报间隔（秒）
    ROI_SEND_INTERVAL = 0.3
    # ROI 上报滞回阈值：置信度 > ROI_ON_CONF 进入上报，< ROI_OFF_CONF 退出
    ROI_ON_CONF = 0.65
    ROI_OFF_CONF = 0.60

    def __init__(self, source, persons, detector, extractor, threshold,
                 realtime_preview=False, rotate180=False, rois=None):
        super().__init__()
        self.source = source  # 本地视频路径 或 RTSP/HTTP 地址
        self.persons = persons  # [{"label", "feature", ...}]
        self.detector = detector
        self.extractor = extractor
        self.threshold = threshold
        self.realtime_preview = realtime_preview  # 文件模式：按原速播放预览
        self.rotate180 = rotate180   # 文件模式：画面倒装时先整帧转 180° 再检测
        self.is_stream = is_stream_source(source)
        self._stop = False
        self._grabber = None  # 实时流模式下由 _run_stream 创建

        # ROI 状态（rect 为归一化 (x1, y1, x2, y2)，None = 未启用；
        # 运行中可通过 set_rois 动态启用/调整/停用，下一帧即生效）
        self._roi_lock = threading.Lock()
        self._roi_rects = {"A": None, "B": None}
        for r in (rois or []):
            rid = (r or {}).get("roi_id")
            if rid in self._roi_rects and r.get("rect"):
                self._roi_rects[rid] = tuple(r["rect"])
        self._roi_matchers = {}   # roi_id -> TrackMatcher（各配独立检测器/轨迹）
        self._passenger_locks = {}
        self._roi_active = {}     # roi_id -> {track_key: 是否处于上报状态}（滞回）
        self._roi_current = {}    # roi_id -> 当前帧的上报检测列表
        self._roi_last_send = {}  # roi_id -> 上次上报时刻
        self._roi_frame_idx = 0   # ROI 模式的帧计数
        self._detected = 0        # 真正做过检测/识别的帧数（无 ROI 时恒为 0）

    def set_rois(self, rois):
        """运行中更新 ROI（GUI 线程调用）。

        rois: [{"roi_id": "A", "rect": (x1,y1,x2,y2) | None}, ...]
        rect 传 None 表示停用该 ROI。
        """
        with self._roi_lock:
            for r in (rois or []):
                rid = (r or {}).get("roi_id")
                if rid in self._roi_rects:
                    self._roi_rects[rid] = tuple(r["rect"]) if r.get("rect") else None

    def stop(self):
        self._stop = True
        if self._grabber is not None:
            self._grabber.interrupt()  # 立即唤醒阻塞中的抓帧等待

    def _orient(self, frame):
        """按需把整帧旋转 180°。

        必须在检测/跟踪之前做：旋转后人体才是正立的，检测框、轨迹 id
        与画面上叠的标签才对得上；预览画面同样是旋转后的正向画面。
        """
        if self.rotate180 and frame is not None:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    # ----- 线程入口 -----
    def run(self):
        if self.is_stream:
            self._run_stream()
        else:
            self._run_file()

    def _matched_count(self):
        """当前命中总数（各 ROI 匹配器之和；未启用 ROI 时恒为 0）。"""
        return sum(m.matched for m in self._roi_matchers.values())

    def _roi_enabled(self):
        with self._roi_lock:
            return any(rc for rc in self._roi_rects.values())

    # ----- ROI 模式 -----
    def _annotate_frame(self, frame):
        """单帧处理入口：只在各 ROI 框内检测；未启用任何 ROI 时不识别。

        没有 ROI 时直接返回原画面（不跑检测、不跑匹配，画面上给出提示）；
        “ROI 区域不用识别”= 检测器只看 ROI 裁剪图，框外内容不参与；
        框内检测结果平移回整帧坐标后画框/上报。
        """
        with self._roi_lock:
            rects = {rid: rc for rid, rc in self._roi_rects.items() if rc}
        if not rects:
            # 未启用 ROI：不进行任何识别，只在画面上提示（不写框、不发消息）
            canvas = frame.copy()
            draw_text(canvas, "NO ROI: REID DISABLED (draw ROI to start)",
                      (10, 25), (0, 200, 255), 0.6, 2)
            return canvas

        H, W = frame.shape[:2]
        canvas = frame.copy()
        self._detected += 1  # 该帧真正跑了 ROI 内检测
        pixel = {}
        for rid, rc in rects.items():
            x1, y1 = int(rc[0] * W), int(rc[1] * H)
            x2, y2 = int(rc[2] * W), int(rc[3] * H)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 16 or y2 - y1 < 16:  # 框太小，视为无效
                self._roi_current[rid] = []
                continue
            pixel[rid] = (x1, y1, x2, y2)
            crop = frame[y1:y2, x1:x2]
            m = self._roi_matchers.get(rid)
            if m is None:
                # 每个 ROI 配独立检测器：ByteTrack 的 persist 跟踪状态按
                # “虚拟相机”隔离，否则 A/B 两路裁剪图交替喂同一个跟踪器，
                # 轨迹 id 会互相串导致匹配抖动。
                m = TrackMatcher(self.persons, PersonDetector(),
                                 self.extractor, self.threshold,
                                 passenger_locks=self._passenger_locks)
                self._roi_matchers[rid] = m
                self.log.emit(f"ROI {rid} 已启用（{x2 - x1}x{y2 - y1} 像素，仅框内检测）")
            # 注意必须接收返回值：annotate 画在 canvas 的副本上并返回，
            # 丢弃返回值会导致框内检测/标签全部丢失（画面上只剩 ROI 橙框）。
            canvas = m.annotate(crop, canvas=canvas, offset=(x1, y1),
                                defer_confirmation=True)

        TrackMatcher.confirm_frame([self._roi_matchers[rid] for rid in pixel], canvas)
        for rid in pixel:
            m = self._roi_matchers[rid]
            self._update_roi_state(rid, m.last_detections, W, H)

        # 停用的 ROI：清掉残留状态（不再上报）
        for rid in list(self._roi_current):
            if rid not in rects:
                self._roi_current.pop(rid, None)
                self._roi_active.pop(rid, None)

        # ROI 框可视化
        for rid, (x1, y1, x2, y2) in pixel.items():
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
            draw_text(canvas, f"ROI {rid}", (x1 + 6, y1 + 24), (0, 200, 255), 0.7, 2)
        draw_text(canvas, f"frame {self._roi_frame_idx}  matched={self._matched_count()}",
                  (10, 25), (255, 255, 255), 0.6, 2)
        self._roi_frame_idx += 1
        return canvas

    def _update_roi_state(self, rid, detections, W, H):
        """按滞回阈值维护每个 ROI 的“当前上报检测”列表。

        置信度 > 0.65 进入上报；已上报的目标滑落到 [0.6, 0.65) 区间时保持
        （防止在阈值边界反复进出抖动）；< 0.6 或目标离开 ROI / 轨迹消失
        则移除。无任何上报目标时列表为空（仍按 300ms 节拍发空列表）。
        """
        active = self._roi_active.setdefault(rid, {})
        new_active, reported = {}, []
        for d in detections or []:
            # 双保险：低于 Step2 匹配阈值的结果绝不进入 MQTT 上报
            if float(d["score"]) < self.threshold:
                continue
            key = d["tid"] if d["tid"] is not None else f"{d['label']}@{tuple(d['bbox_xyxy'])}"
            s = float(d["score"])
            prev = bool(active.get(key, False))
            on = (s > self.ROI_ON_CONF) or (prev and s >= self.ROI_OFF_CONF)
            new_active[key] = on
            if on:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                reported.append({
                    "passenger_id": d["label"],
                    "bbox_xyxy": [int(max(0, min(W - 1, x1))),
                                  int(max(0, min(H - 1, y1))),
                                  int(max(0, min(W, x2))),
                                  int(max(0, min(H, y2)))],
                })
        self._roi_active[rid] = new_active
        self._roi_current[rid] = reported

    def _pump_roi_messages(self):
        """把各 ROI 当前状态按 300ms 节拍上报（无目标也发空 detections）。"""
        with self._roi_lock:
            rects = {rid for rid, rc in self._roi_rects.items() if rc}
        if not rects:
            return
        now = time.perf_counter()
        for rid in sorted(rects):
            if rid not in self._roi_current:
                continue  # 该 ROI 尚未处理过任何帧
            if now - self._roi_last_send.get(rid, 0.0) < self.ROI_SEND_INTERVAL:
                continue
            self._roi_last_send[rid] = now
            msg = {
                "roi_id": rid,
                "captured_at_ms": int(time.time() * 1000),
                "detections": self._roi_current.get(rid, []),
            }
            self.roi_message.emit(json.dumps(msg, ensure_ascii=False))

    # ----- 本地文件检索 -----
    def _run_file(self):
        tmp_paths = []

        # 1) 先单独探测帧率：探测需要 seek 到尾帧，而 TS/M2TS 这类容器没有索引，
        #    seek 之后往往读不出帧，所以探测不能和正式读取共用一个 capture。
        fps = 25.0
        probe, tmp = open_file_capture(self.source)
        if tmp:
            tmp_paths.append(tmp)
        try:
            if not probe.isOpened():
                self.log.emit(f"无法打开视频文件：{self.source}")
                self.finished_ok.emit(0)
                return
            # TS/M2TS 等无索引容器不做尾帧 seek 校验（大文件上会明显拖慢启动）
            ext = os.path.splitext(self.source)[1].lower()
            fps = probe_real_fps(probe, allow_seek=ext not in (".ts", ".m2ts", ".mts"))
        finally:
            probe.release()

        # 2) 重新从头正式读取
        cap, tmp = open_file_capture(self.source)
        if tmp:
            tmp_paths.append(tmp)
        try:
            if not cap.isOpened():
                self.log.emit(f"无法打开视频文件：{self.source}")
                self.finished_ok.emit(0)
                return
            ret, frame0 = cap.read()
            if not ret or frame0 is None:
                self.log.emit("视频文件为空或无法解码第一帧，请确认编码是否受支持。")
                self.finished_ok.emit(0)
                return
            frame0 = self._orient(frame0)  # 首帧也按同一方向处理，尺寸/画面保持一致
            h, w = frame0.shape[:2]
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            self.log.emit(f"开始检索文件「{os.path.basename(self.source)}」"
                          f"（{w}x{h} @ {fps:.1f}fps，约 {total} 帧"
                          f"{'，画面已旋转 180°' if self.rotate180 else ''}）")
            if self.rotate180:
                self.log.emit("提示：按倒装画面处理，先整帧旋转 180° 再检测，预览画面为正向。")
            if not self._roi_enabled():
                self.log.emit("未启用任何 ROI：本次检索只预览画面、不做人物识别。"
                              "点左侧「框选」拖出 ROI 后，下一帧立即开始框内识别。")

            # 首帧先处理（预览立即有画面，也用于确定预览尺寸），idx 从 1 起算
            annotated0 = self._annotate_frame(frame0)
            self.frame_ready.emit(annotated0)
            idx = 1            # 已读到的帧号（时间轴位置，含跳帧）
            # 原速模式的播放时钟：从首帧画面出现那一刻起算
            start_t = time.perf_counter()
            last_preview = time.perf_counter()
            last_report = time.perf_counter()
            eof = False
            while not self._stop:
                # ---- 取帧 ----
                # 原速播放：以墙钟为准 —— 处理跟不上就跳帧追赶时间轴（与实时流
                # “只消费最新帧”的行为一致），处理比时间轴快就等待。
                # 旧实现只做“超前时 sleep”，一旦检测慢于帧率就永远不 sleep，
                # 实际按检测速度播放（远慢于原速），这就是“勾选了也不按原速
                # 播放”的原因。跳过的帧不检测。
                if self.realtime_preview and fps > 0:
                    t = time.perf_counter() - start_t
                    want = int(t * fps)          # 当前时刻应显示的帧号
                    if want < idx:               # 已处理到时间轴前方 -> 等待
                        time.sleep(min(idx / fps - t, 0.2))
                        self._pump_roi_messages()
                        continue
                    while idx < want:            # 落后 -> 跳帧（只读不检测）
                        ok, _f = cap.read()
                        if not ok or _f is None:
                            eof = True
                            break
                        idx += 1
                    if eof:
                        break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = self._orient(frame)  # 检测前转正（旋转 180° 时）
                idx += 1

                # ---- 检测 / 匹配 / 预览 ----
                annotated = self._annotate_frame(frame)
                if annotated.shape[1] != w or annotated.shape[0] != h:
                    annotated = cv2.resize(annotated, (w, h))  # 花屏/分辨率突变保护
                self._pump_roi_messages()

                now = time.perf_counter()
                if now - last_preview >= self.PREVIEW_INTERVAL:
                    last_preview = now
                    self.frame_ready.emit(annotated)
                if now - last_report >= 1.0:
                    last_report = now
                    self.progress.emit(idx, total)
                    if self._roi_matchers:
                        skip = idx - self._detected
                        extra = f"（检测 {self._detected}，跳帧 {skip}）" if skip > 0 \
                            else f"（检测 {self._detected}）"
                    else:
                        extra = "（未启用 ROI，未识别）"
                    self.log.emit(f"已处理 {idx}/{total or '?'} 帧{extra}，"
                                  f"命中 {self._matched_count()} 处")

            self.progress.emit(idx, total)
            reason = "已停止" if self._stop else "文件检索完毕"
            if self._roi_matchers:
                skip = idx - self._detected
                extra = f"，检测 {self._detected} 帧，跳帧 {skip} 帧" if skip > 0 \
                    else f"，检测 {self._detected} 帧"
            else:
                extra = "（未启用 ROI，只预览未识别）"
            self.log.emit(f"检索结束（{reason}）：共 {idx} 帧{extra}，"
                          f"命中 {self._matched_count()} 处。")
            self.finished_ok.emit(idx)
        finally:
            cap.release()
            for p in tmp_paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # ----- RTSP 实时流检索 -----
    def _run_stream(self):
        grabber = StreamGrabber(self.source, log=self.log.emit)
        self._grabber = grabber
        grabber.start()
        try:
            frame, seq = grabber.wait_frame(-1, timeout=20.0)
            if frame is None:
                self.log.emit(f"无法打开视频流：{self.source}")
                self.finished_ok.emit(0)
                return
            h, w = frame.shape[:2]
            fps = grabber.fps

            self.log.emit(f"开始检索实时流（{w}x{h} @ {fps:.1f}fps）：{self.source}")
            if not self._roi_enabled():
                self.log.emit("未启用任何 ROI：本次检索只预览画面、不做人物识别。"
                              "点左侧「框选」拖出 ROI 后，下一帧立即开始框内识别。")
            idx = 0
            last_seq = seq
            last_preview = 0.0
            last_report = time.perf_counter()
            last_frame_t = time.perf_counter()
            while not self._stop:
                frame, seq = grabber.wait_frame(last_seq, timeout=5.0)
                if frame is None:
                    # 彻底连不上 / 用户停止 / 30 秒没有新帧 -> 结束
                    if grabber.failed or self._stop:
                        break
                    if time.perf_counter() - last_frame_t > 30:
                        self.log.emit("超过 30 秒未收到新画面，结束检索。")
                        break
                    self._pump_roi_messages()  # 等帧期间也维持 300ms 上报节拍
                    continue
                last_seq = seq
                last_frame_t = time.perf_counter()

                annotated = self._annotate_frame(frame)
                if annotated.shape[1] != w or annotated.shape[0] != h:
                    annotated = cv2.resize(annotated, (w, h))
                self._pump_roi_messages()

                now = time.perf_counter()
                if now - last_preview >= self.PREVIEW_INTERVAL:
                    last_preview = now
                    self.frame_ready.emit(annotated)
                if now - last_report >= 2.0:
                    last_report = now
                    self.progress.emit(idx + 1, 0)
                    self.log.emit(f"已处理 {idx + 1} 帧（抓取 {grabber.grabbed} 帧），"
                                  f"命中 {self._matched_count()} 处")
                idx += 1

            self.progress.emit(idx, 0)
            reason = "已停止" if self._stop else "流已断开"
            self.log.emit(f"检索结束（{reason}）：处理 {idx} 帧 / 抓取 "
                          f"{grabber.grabbed} 帧，命中 {self._matched_count()} 处。")
            self.finished_ok.emit(idx)
        finally:
            grabber.stop()
            self._grabber = None
