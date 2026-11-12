"""联合工作台的后台调度：复用算法，通过信号传递画面、旅客和位置。"""

from collections import OrderedDict
from importlib import import_module
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from pose_project.integration.configuration import is_stream_source


def encode_image(image):
    import base64
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("无法编码旅客图片")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def unique_detections(detections, width, height, threshold):
    """同一旅客同时出现在多条轨迹时跳过，避免任意选择目标。"""
    counts = {}
    for detection in detections:
        if detection["score"] >= threshold:
            label = detection["label"]
            counts[label] = counts.get(label, 0) + 1
    result = []
    for detection in detections:
        if detection["score"] < threshold or counts[detection["label"]] != 1:
            continue
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        box = [int(max(0, x1)), int(max(0, y1)),
               int(min(width, x2)), int(min(height, y2))]
        if box[0] < box[2] and box[1] < box[3]:
            result.append({"passenger_id": detection["label"], "bbox_xyxy": box})
    return result, [label for label, count in counts.items() if count > 1]


class StreamCapture:
    """将实时流的最新画面交给原有闸机处理线程。"""

    def __init__(self, source, log=None, timeout=30, clock=None):
        self.grabber = import_module("pose_project.video.stream_capture").StreamGrabber(
            source, log=log, clock=clock)
        self.cancelled = threading.Event()
        self.timeout = timeout
        self.sequence = -1
        self.captured_at_ms = 0
        self.grabber.start()

    def isOpened(self):
        return not self.cancelled.is_set()

    def read(self):
        deadline = time.monotonic() + self.timeout
        while not self.cancelled.is_set():
            image, sequence, captured_at = self.grabber.wait_frame(
                self.sequence, timeout=min(.5, max(0, deadline - time.monotonic())),
                with_timestamp=True)
            if self.cancelled.is_set():
                break
            if image is not None:
                self.sequence, self.captured_at_ms = sequence, captured_at
                return True, image
            if self.grabber.failed or time.monotonic() >= deadline:
                raise ValueError("视频流连接失败或已中断，请检查地址和摄像头连接")
        return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.grabber.fps
        return 0

    def interrupt(self):
        self.cancelled.set()
        self.grabber.interrupt()

    def release(self):
        self.interrupt()
        self.grabber.stop()
        if self.grabber._thread is not None:
            self.grabber._thread.join()


class PreviewWorker(QThread):
    frame = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, rotate=False):
        super().__init__()
        self.source, self.rotate = source, rotate
        self.cancelled = threading.Event()
        self.capture = None

    def stop(self):
        self.cancelled.set()
        if isinstance(self.capture, StreamCapture):
            self.capture.interrupt()

    def run(self):
        capture = None
        try:
            if self.cancelled.is_set():
                return
            if is_stream_source(self.source):
                capture = StreamCapture(self.source, timeout=20)
            else:
                capture = cv2.VideoCapture(self.source)
            self.capture = capture
            if self.cancelled.is_set():
                return
            ok, image = capture.read()
            if self.cancelled.is_set():
                return
            if not ok:
                raise ValueError("无法读取视频首帧：" + self.source)
            if self.rotate:
                image = cv2.rotate(image, cv2.ROTATE_180)
            self.frame.emit(image)
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
            self.capture = None


class GateWorker(QThread):
    frame = pyqtSignal(object)
    person = pyqtSignal(dict)
    log = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(float)

    def __init__(self, settings, output, clock=None):
        super().__init__()
        self.settings, self.output = settings, Path(output)
        self.clock = clock
        self.processor = None
        self.capture = None
        self.cancelled = threading.Event()

    def stop(self):
        self.cancelled.set()
        if self.processor is not None:
            self.processor.stop()
        if self.capture is not None:
            self.capture.interrupt()

    def _open_stream(self, source):
        self.capture = StreamCapture(source, log=self.log.emit, clock=self.clock)
        if self.cancelled.is_set():
            self.capture.interrupt()
        return self.capture

    def run(self):
        try:
            # 延迟加载模型依赖，让界面启动与环境检查无需等待模型导入。
            module = import_module("pose_project.gate.processor")
            builder = module.MqttPublisher(out_dir=str(self.output), enabled=False)
            owner = self

            class Publisher:
                def publish_person(self, **kwargs):
                    owner.person.emit(builder.build_payload(**kwargs))
                    return True

                def close(self):
                    pass

            self.processor = module.ProcessorThread(
                [self.settings["source"]], self.settings["roi"], self.settings["opts"],
                str(self.output), publisher=Publisher(), mqtt_enabled=False,
                stream_factory=self._open_stream if is_stream_source(self.settings["source"])
                else None)
            self.processor.sig_frame.connect(self._frame, type=Qt.DirectConnection)
            self.processor.sig_progress.connect(lambda value, text: self.progress.emit(value),
                                                type=Qt.DirectConnection)
            self.processor.sig_log.connect(self.log.emit, type=Qt.DirectConnection)
            if self.cancelled.is_set():
                return
            # 在当前后台线程里处理，避免额外启动一个无法统一停止的线程。
            self.processor._run()
            if self.processor.reg is not None:
                module.save_csv(str(self.output), self.processor.reg.persons)
        except Exception:
            if not self.cancelled.is_set():
                self.failed.emit(traceback.format_exc())
        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.processor = None

    def _frame(self, image, tracks):
        image = image.copy()
        for track in tracks or []:
            x, y, width, height = map(int, track[0])
            cv2.rectangle(image, (x, y), (x + width, y + height), (60, 185, 70), 2)
        self.frame.emit(image)


class RecognitionWorker(QThread):
    frame = pyqtSignal(object)
    person_added = pyqtSignal(dict)
    snapshot = pyqtSignal(dict)
    log = pyqtSignal(str)
    failed = pyqtSignal(str)
    ready = pyqtSignal()
    result = pyqtSignal(dict)

    def __init__(self, settings, regions, clock=None):
        super().__init__()
        self.settings, self.regions = settings, regions
        self.clock = clock or (lambda: time.time_ns() // 1000000)
        self.cancelled = threading.Event()
        self.incoming = Queue(maxsize=128)
        self.gallery = OrderedDict()
        self.matchers = {}
        self.passenger_locks = {}
        self.detectors = {}
        self.module = self.extractor = self.capture = None
        self.gallery_ms = 0.0
        self.processed = self.skipped = self.source_sequence = 0
        self.started_at = time.monotonic()
        self.runtime_status = "模型加载中"

    def add_person(self, payload):
        try:
            self.incoming.put_nowait(payload)
        except Full:
            raise ValueError("旅客图片积压超过 128 条")

    def stop(self):
        self.cancelled.set()
        if self.capture is not None and hasattr(self.capture, "interrupt"):
            self.capture.interrupt()

    def _drain_gallery(self):
        started = time.perf_counter()
        changed = False
        # 限制每轮入库数量，避免集中到达的闸机图片阻塞位置更新。
        for _ in range(1):
            try:
                payload = self.incoming.get_nowait()
            except Empty:
                break
            try:
                passenger = payload.get("passenger_id") or payload.get("person_id")
                if not isinstance(passenger, str) or not passenger.strip():
                    raise ValueError("旅客消息缺少完整 passenger_id")
                images = payload.get("images") or {}
                if not isinstance(images, dict):
                    raise ValueError("images 必须是 JSON 对象")
                image = self.module.decode_base64_image(images.get("upper"))
                if image is None:
                    raise ValueError("旅客消息缺少有效的 images.upper")
                if passenger not in self.gallery and len(self.gallery) >= 256:
                    raise ValueError("人物库已达到 256 人，请开始新的调试会话")
                self.module.set_phase_threads(self.module.EXT_THREADS)
                feature = self.extractor.extract([image], tta=True)[0]
                self.gallery[passenger] = {
                    "label": passenger, "feature": feature, "crop": image,
                    "display_label": payload.get("local_id") or passenger[:8]}
                self.person_added.emit({"passenger_id": passenger,
                                        "local_id": payload.get("local_id", ""),
                                        "capture_time": payload.get("capture_time", ""),
                                        "image": image})
                changed = True
            except (ValueError, TypeError, KeyError) as exc:
                self.log.emit("旅客入库失败：" + str(exc))
        if changed:
            for matcher in self.matchers.values():
                matcher.update_gallery(
                    list(self.gallery.values()),
                    {key: person["display_label"] for key, person in self.gallery.items()})
        self.gallery_ms = (time.perf_counter() - started) * 1000

    def latest_preview(self, sequence):
        capture = self.capture
        if not is_stream_source(self.settings["source"]) or capture is None:
            return None, sequence, 0
        image, sequence, captured_at = capture.wait_frame(
            sequence, timeout=0, with_timestamp=True)
        if image is not None and self.settings["rotate180"]:
            image = cv2.rotate(image, cv2.ROTATE_180)
        return image, sequence, captured_at

    def _process(self, image, captured_at):
        started = time.perf_counter()
        selected_at = self.clock()
        width, height = self.settings["image_size"]
        if image.shape[1] != width or image.shape[0] != height:
            raise ValueError("供框台画面为 {}x{}，配置为 {}x{}。请预览后重新标定。".format(
                image.shape[1], image.shape[0], width, height))
        canvas = image.copy()
        regions = {}
        timing = dict(detection_ms=0.0, reid_ms=0.0, matching_ms=0.0)
        for region_id in self.settings["enabled_regions"]:
            geometry = self.regions[region_id]
            x1, y1, x2, y2 = map(int, geometry["roi_xyxy"])
            candidates = []
            if self.gallery:
                matcher = self.matchers.get(region_id)
                if matcher is None:
                    detector = self.detectors.get(region_id)
                    if detector is None:
                        detector = self.module.PersonDetector(
                            weights=self.settings["detector_weights"], allow_download=False)
                        self.detectors[region_id] = detector
                    matcher = self.module.TrackMatcher(
                        list(self.gallery.values()), detector, self.extractor,
                        self.settings["threshold"],
                        display_labels={key: person["display_label"]
                                        for key, person in self.gallery.items()},
                        front_mode=geometry.get("front_mode"),
                        passenger_locks=self.passenger_locks)
                    self.matchers[region_id] = matcher
                canvas = matcher.annotate(image[y1:y2, x1:x2], canvas=canvas,
                                          offset=(x1, y1),
                                          imgsz=self.settings.get("detector_image_size", 960),
                                          tta=self.settings.get("flip_tta", True),
                                          defer_confirmation=True)
                candidates = matcher.last_candidates
                for key in timing:
                    timing[key] += matcher.last_timing[key]
            else:
                before_detect = time.perf_counter()
                self.module.set_phase_threads(self.module.DET_THREADS)
                boxes, scores, ids = self.detectors[region_id].detect_track(
                    image[y1:y2, x1:x2],
                    imgsz=self.settings.get("detector_image_size", 960))
                timing["detection_ms"] += (time.perf_counter() - before_detect) * 1000
                candidates = [
                    {"label": "", "score": None, "detector_score": float(score),
                     "tid": ids[i] if i < len(ids) else None, "status": "no_gallery",
                     "bbox_xyxy": [int(box[0]) + x1, int(box[1]) + y1,
                                   int(box[2]) + x1, int(box[3]) + y1]}
                    for i, (box, score) in enumerate(zip(boxes, scores))]
            regions[region_id] = {"candidates": candidates}

        confirmed_at = time.perf_counter()
        if self.gallery:
            self.module.TrackMatcher.confirm_frame(
                [self.matchers[r] for r in self.settings["enabled_regions"]], canvas)
        timing["matching_ms"] += (time.perf_counter() - confirmed_at) * 1000
        for region_id in self.settings["enabled_regions"]:
            candidates = regions[region_id]["candidates"]
            detections = self.matchers[region_id].last_detections if self.gallery else []
            people, conflicts = unique_detections(
                detections, width, height, self.settings["threshold"])
            conflicts = sorted(set(conflicts) | {
                c["label"] for c in candidates if c["status"] == "duplicate"})
            if conflicts:
                self.log.emit("同一身份命中多个目标，本帧不采用：" + ", ".join(conflicts))
            candidates = [dict(candidate, status="duplicate")
                          if candidate["label"] in conflicts and candidate["status"] == "matched"
                          else candidate for candidate in candidates]
            candidates = [
                dict(candidate, display_label=self.gallery.get(
                    candidate["label"], {}).get("display_label", ""),
                     ranked_candidates=[
                         dict(row, display_label=self.gallery.get(
                             row["label"], {}).get("display_label", ""))
                         for row in candidate.get("ranked_candidates", [])])
                for candidate in candidates]
            regions[region_id] = {"candidates": candidates, "detections": people,
                                  "conflicts": conflicts}
            self.snapshot.emit({"roi_id": region_id, "captured_at_ms": captured_at,
                                "detections": people})
        self.frame.emit(canvas)
        self.processed += 1
        elapsed = max(.001, time.monotonic() - self.started_at)
        completed = self.clock()
        self.result.emit({
            "captured_at_ms": captured_at, "completed_at_ms": completed, "regions": regions,
            "_frame": canvas, "_source_frame": image,
            "metrics": dict(timing,
                processing_ms=round((time.perf_counter() - started) * 1000, 2),
                input_age_ms=max(0, selected_at - captured_at),
                worker_age_ms=completed - captured_at, gallery_ms=round(self.gallery_ms, 2),
                device=str(self.extractor.device), gallery_count=len(self.gallery),
                gallery_pending=self.incoming.qsize(), processed=self.processed,
                skipped=self.skipped, capture_fps=round(self.source_sequence / elapsed, 2),
                inference_fps=round(self.processed / elapsed, 2),
                reconnects=getattr(self.capture, "reconnects", 0))})

    def run(self):
        try:
            self.log.emit("正在加载重识别模型")
            self.module = import_module("pose_project.reid.search_worker")
            self.extractor = self.module.ReidExtractor(weights=self.settings["reid_weights"])
            device = str(self.extractor.device)
            self.runtime_status = (
                "识别中（CPU 20 线程并行）"
                if device.lower() == "cpu" else "识别中（{}）".format(device))
            self.log.emit("正在预热跟踪和重识别，设备：" + device)
            self.module.set_phase_threads(self.module.EXT_THREADS)
            self.extractor.extract([np.zeros((384, 128, 3), dtype=np.uint8)],
                                   tta=self.settings.get("flip_tta", True))
            for region_id in self.settings["enabled_regions"]:
                if self.cancelled.is_set():
                    return
                detector = self.module.PersonDetector(
                    weights=self.settings["detector_weights"], allow_download=False)
                box = self.regions[region_id]["roi_xyxy"]
                self.module.set_phase_threads(self.module.DET_THREADS)
                detector.detect_track(
                    np.zeros((int(box[3] - box[1]), int(box[2] - box[0]), 3), dtype=np.uint8),
                    imgsz=self.settings.get("detector_image_size", 960))
                detector.reset_tracker()
                self.detectors[region_id] = detector
            if self.cancelled.is_set():
                return
            self.ready.emit()
            self.started_at = time.monotonic()
            source = self.settings["source"]
            if self.module.is_stream_source(source):
                self._run_stream(source)
            else:
                self._run_file(source)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.capture = None
            self.matchers.clear()
            self.detectors.clear()
            self.extractor = None

    def _run_file(self, source):
        capture, temporary = self.module.open_file_capture(source)
        self.capture = capture
        try:
            if not capture.isOpened():
                raise ValueError("无法打开供框台视频")
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and 1 <= fps <= 120 else 25
            started = time.monotonic()
            epoch = self.clock()
            index, next_sample = 0, 0.0
            while not self.cancelled.is_set():
                self._drain_gallery()
                desired = int((time.monotonic() - started) * fps)
                while index < desired:
                    if not capture.grab():
                        return
                    index += 1
                    self.skipped += 1
                    if self.cancelled.is_set():
                        return
                ok, image = capture.read()
                if not ok:
                    break
                position = index / fps
                index += 1
                self.source_sequence = index
                if self.cancelled.wait(max(0, started + position - time.monotonic())):
                    break
                # 使用新采集的画面，保留真实采集时间，避免旧画面误报为当前位置。
                if position < next_sample:
                    continue
                next_sample = position + self.settings.get("sample_interval_ms", 200) / 1000
                if self.settings["rotate180"]:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                self._process(image, min(self.clock(), epoch + int(position * 1000)))
            self.log.emit("供框台视频处理结束")
        finally:
            capture.release()
            if temporary:
                Path(temporary).unlink(missing_ok=True)

    def _run_stream(self, source):
        capture = self.module.StreamGrabber(source, log=self.log.emit, clock=self.clock)
        self.capture = capture
        capture.start()
        sequence = -1
        last_frame = time.monotonic()
        next_sample = 0.0
        try:
            while not self.cancelled.is_set():
                self._drain_gallery()
                if self.cancelled.wait(max(0, next_sample - time.monotonic())):
                    break
                previous_sequence = sequence
                image, sequence, captured_at = capture.wait_frame(
                    sequence, timeout=0.5, with_timestamp=True)
                if image is None:
                    if capture.failed or time.monotonic() - last_frame > 30:
                        raise ValueError("供框台视频流中断")
                    continue
                last_frame = time.monotonic()
                next_sample = last_frame + self.settings.get("sample_interval_ms", 200) / 1000
                self.skipped += max(0, sequence - max(0, previous_sequence) - 1)
                self.source_sequence = sequence
                if self.settings["rotate180"]:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                self._process(image, captured_at)
        finally:
            capture.stop()
            if capture._thread is not None:
                capture._thread.join()
