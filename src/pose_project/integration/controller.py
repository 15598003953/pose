"""联合会话控制：连接抓拍、重识别、绑定和界面，并统一保存运行记录。"""

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime
import json
import logging
import platform
from pathlib import Path
import re
import sys
import threading
import time
import uuid

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal

from pose_project.binding.binding_runtime import (
    DEFAULT_FIELDS, LiveInbox, MqttLink, normalize_message,
)
from pose_project.binding.passenger_tray_binding import PassengerTrayBinder
from pose_project.integration.configuration import (
    active_binding, is_stream_source, new_run_directory, validate, save_settings,
)
from pose_project.integration.workers import GateWorker, RecognitionWorker, encode_image


def message_preview(payload):
    preview = deepcopy(payload)
    if "images" in preview:
        if not isinstance(preview["images"], dict):
            raise ValueError("images 必须是 JSON 对象")
        preview["images"] = {
            key: "<base64: {} chars>".format(len(value)) if value else None
            for key, value in preview["images"].items()
        }
    return preview


def _timestamp_text(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000).astimezone().isoformat()


def _redact_source(source):
    return re.sub(r"(rtsps?://)[^/@\s]+:[^/@\s]+@", r"\1***@", str(source or ""))


class IntegrationController(QObject):
    frame = pyqtSignal(str, object)
    recognition = pyqtSignal(dict)
    person = pyqtSignal(dict)
    message = pyqtSignal(dict)
    decision = pyqtSignal(dict)
    log = pyqtSignal(str)
    state = pyqtSignal(str, str)
    running_changed = pyqtSignal(bool)
    stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self.tick)
        self.running = self.stopping = False
        self.workers = {}
        self.bridge = None
        self.logger = logging.getLogger("pose.integration." + uuid.uuid4().hex)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.records = deque(maxlen=500)
        self.counts = Counter()
        self.output_index = 0
        self.pending_sends = deque()
        self.run_dir = None
        self.settings = None
        self._handles = {}
        self._display_lock = threading.Lock()
        self._preview_frames = {}
        self._latest_result = None
        self._preview_sequences = {}
        self._next_preview = 0.0
        self._next_debug = 0.0
        self._position_ages = {}
        self.debug_enabled = False
        self.field_test_enabled = False
        self._field_test_meta = None
        self._field_test_counts = Counter()
        self._field_test_image_index = 0
        self._field_test_lock = threading.RLock()
        self._field_test_gallery = {}

    def _store_frame(self, stage, image):
        # 每路只保留最新画面，后台推理较慢时也不积压界面事件。
        with self._display_lock:
            self._preview_frames[stage] = image

    def _store_result(self, report):
        # 逐帧记录独立于只保留最新结果的界面刷新。
        if self.field_test_enabled:
            with self._field_test_lock:
                try:
                    self._record_test_result(report)
                except (OSError, ValueError, TypeError) as exc:
                    self._field_test_counts["result_errors"] += 1
                    self.write_log("测试结果保存失败：" + str(exc))
        with self._display_lock:
            self._latest_result = report

    def _flush_previews(self):
        with self._display_lock:
            frames, self._preview_frames = self._preview_frames, {}
        worker = self.workers.get("reid")
        if worker is not None and worker.isRunning():
            image, sequence, _ = worker.latest_preview(self._preview_sequences.get("reid", -1))
            if image is not None:
                frames["reid"] = image
                self._preview_sequences["reid"] = sequence
            elif is_stream_source(self.settings["reid"]["source"]):
                frames.pop("reid", None)
        for stage, image in frames.items():
            self.frame.emit(stage, image)

    def set_debug(self, enabled):
        self.debug_enabled = bool(enabled)
        if self.running and enabled and "performance" not in self._handles:
            self._handles["performance"] = (
                self.run_dir / "performance.jsonl").open("a", encoding="utf-8")

    def _flush_result(self):
        with self._display_lock:
            report, self._latest_result = self._latest_result, None
        if report is None:
            return
        metrics = report["metrics"]
        metrics["ui_delivery_ms"] = max(0, self.inbox.now() - report["completed_at_ms"])
        metrics["mqtt_pending"] = len(self.pending_sends)
        with self.inbox.lock:
            metrics["message_pending"] = len(self.inbox.items)
        metrics["position_age_ms"] = dict(self._position_ages)
        self.recognition.emit(report)
        now = time.monotonic()
        if self.debug_enabled and now >= self._next_debug:
            self._next_debug = now + 1
            performance_report = {
                key: value for key, value in report.items()
                if key not in ("_frame", "_source_frame")}
            self._handles["performance"].write(
                json.dumps(performance_report, ensure_ascii=False) + "\n")
            self._handles["performance"].flush()
            self.write_log("[性能] 设备={} 处理={:.0f}ms 检测={:.0f}ms 特征={:.0f}ms "
                           "界面等待={}ms 跳帧={} 入库等待={}".format(
                               metrics["device"], metrics["processing_ms"],
                               metrics["detection_ms"], metrics["reid_ms"],
                               metrics["ui_delivery_ms"], metrics["skipped"],
                               metrics["gallery_pending"]))

    def write_log(self, text):
        text = re.sub(r"(rtsps?://)[^/@\s]+:[^/@\s]+@", r"\1***@", str(text))
        line = datetime.now().strftime("%H:%M:%S") + " " + text
        self.logger.info(line)
        self.log.emit(line)

    def _write_field_test_meta(self):
        if not self.field_test_enabled or self._field_test_meta is None:
            return
        (self.run_dir / "field_test.json").write_text(
            json.dumps(self._field_test_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    @staticmethod
    def _safe_image_part(value):
        value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value if value is not None else "unknown"))
        return value[:80] or "unknown"

    def _next_field_test_stem(self, prefix):
        self._field_test_image_index += 1
        return "{}_{:06d}".format(prefix, self._field_test_image_index)

    def _save_field_test_image(self, folder, stem, image, extension=".png"):
        if not self.field_test_enabled or image is None:
            return None
        if not hasattr(image, "size") or image.size == 0:
            return None
        import cv2

        path = self.run_dir / "test_images" / folder / (stem + extension)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(
                extension, image,
                [cv2.IMWRITE_JPEG_QUALITY, 95] if extension == ".jpg"
                else [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise ValueError("图像编码失败")
            path.write_bytes(encoded.tobytes())
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            self._field_test_counts["image_errors"] += 1
            self.write_log("测试图像保存失败：{}；{}".format(path, exc))
            return None
        self._field_test_counts["images_saved"] += 1
        return path.relative_to(self.run_dir).as_posix()

    @staticmethod
    def _crop_field_test_image(image, box):
        if image is None or not hasattr(image, "shape") or box is None:
            return None
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in box]
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        x2, y2 = max(x1, min(width, x2)), max(y1, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()

    def _save_field_test_person_images(self, payload):
        if not self.field_test_enabled:
            return {}
        from pose_project.reid.messaging import decode_base64_image

        images = payload.get("images") or {}
        if not isinstance(images, dict):
            raise ValueError("images 必须是 JSON 对象")
        local_id = payload.get("local_id") or payload.get("passenger_id")
        stem = self._next_field_test_stem(
            "gallery_" + self._safe_image_part(local_id))
        saved = {}
        for name in ("upper", "face"):
            image = decode_base64_image(images.get(name))
            if images.get(name) and image is None:
                self._field_test_counts["image_errors"] += 1
                self.write_log("测试图像解码失败：{} / {}".format(local_id, name))
            path = self._save_field_test_image("gallery", stem + "_" + name, image)
            if path:
                saved[name] = path
        passenger = payload.get("passenger_id") or payload.get("person_id")
        if isinstance(passenger, str):
            self._field_test_gallery[passenger] = saved
        return saved

    def _record_test_result(self, report):
        import cv2

        data = {
            key: deepcopy(value)
            for key, value in report.items()
            if key not in ("_frame", "_source_frame")
        }
        source_frame = report.get("_source_frame")
        if source_frame is None:
            source_frame = report.get("_frame")
        captured_at = report.get("captured_at_ms", self.inbox.now())
        stem = self._next_field_test_stem(
            "reid_{:013d}".format(int(captured_at)))
        images = {}
        source_path = self._save_field_test_image(
            "reid_frames", stem + "_source", source_frame, ".jpg")
        if source_path:
            images["source_frame"] = source_path
        annotated = source_frame.copy() if source_frame is not None else None
        crops = []
        for region_id, region in data.get("regions", {}).items():
            for index, candidate in enumerate(region.get("candidates") or []):
                candidate["reference_images"] = dict(
                    self._field_test_gallery.get(candidate.get("label"), {}))
                for ranked in candidate.get("ranked_candidates", []):
                    ranked["reference_images"] = dict(
                        self._field_test_gallery.get(ranked["label"], {}))
                crop = self._crop_field_test_image(
                    source_frame, candidate.get("bbox_xyxy"))
                crop_path = self._save_field_test_image(
                    "reid_crops",
                    "{}_{}_{}_{}".format(
                        stem, self._safe_image_part(region_id),
                        self._safe_image_part(candidate.get("tid")),
                        index),
                    crop,
                )
                if crop_path:
                    candidate["image_path"] = crop_path
                    crops.append(crop_path)
                box = candidate.get("bbox_xyxy")
                if annotated is not None and box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    color = (60, 185, 70) if candidate.get("status") == "matched" else (0, 180, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    text = "{} T{} {} {}".format(
                        region_id, candidate.get("tid"), candidate.get("score"),
                        candidate.get("status"))
                    cv2.putText(annotated, text, (max(0, x1), max(14, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        frame_path = self._save_field_test_image(
            "reid_frames", stem + "_annotated", annotated, ".jpg")
        if frame_path:
            images["annotated_frame"] = frame_path
        if crops:
            images["candidate_crops"] = crops
        data["images"] = images
        self._handles["test_results"].write(json.dumps({
            "recorded_at_ms": self.inbox.now(),
            "data": data,
        }, ensure_ascii=False) + "\n")
        self._handles["test_results"].flush()
        self._field_test_counts["test_results"] += 1
        self._record_field_event("recognition", data)
        return data

    def _record_field_event(self, event_type, data=None):
        with self._field_test_lock:
            handle = self._handles.get("field_test")
            if handle is None:
                return
            record = {
                "event_type": event_type,
                "recorded_at_ms": self.inbox.now(),
                "data": data or {},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            self._field_test_counts[event_type] += 1

    def record_field_observation(self, category, result, expected, actual, note,
                                 context=None):
        if not self.running or not self.field_test_enabled:
            raise ValueError("请先启用现场测试记录并启动联调")
        values = {
            "category": str(category).strip(),
            "result": str(result).strip(),
            "expected": str(expected).strip(),
            "actual": str(actual).strip(),
            "note": str(note).strip(),
            "context": context or {},
        }
        if not values["category"] or not values["result"]:
            raise ValueError("现场记录类别和结论不能为空")
        self._record_field_event("manual_observation", values)
        self.write_log("[现场记录] {} / {}".format(
            values["category"], values["result"]))
        return values

    def start(self, settings, start_workers=True):
        # 每次启动新建会话目录和算法状态，避免沿用上次的旅客与轨迹。
        if self.running or self.stopping:
            raise ValueError("上一轮调试仍在运行或停止中")
        validate(settings, sources=start_workers)
        self.settings = deepcopy(settings)
        self.settings["binding"] = active_binding(settings)
        self.run_dir = new_run_directory(settings)
        save_settings(settings, self.run_dir / "settings.json")
        handler = logging.FileHandler(self.run_dir / "runtime.log", encoding="utf-8")
        self.logger.addHandler(handler)
        self._file_handler = handler
        self._handles = {
            key: (self.run_dir / (key + ".jsonl")).open("a", encoding="utf-8")
            for key in ("messages", "decisions")
        }
        self.binder = PassengerTrayBinder(self.settings["binding"])
        self.inbox = LiveInbox()
        field_test = self.settings.get("debug", {}).get("field_test", {})
        self.field_test_enabled = bool(field_test.get("enabled", False))
        self._field_test_counts.clear()
        self._field_test_image_index = 0
        self._field_test_gallery.clear()
        self._field_test_meta = None
        if self.field_test_enabled:
            self._handles["field_test"] = (
                self.run_dir / "field_test_events.jsonl").open("a", encoding="utf-8")
            self._handles["test_results"] = (
                self.run_dir / "test_results.jsonl").open("a", encoding="utf-8")
            started_at_ms = self.inbox.now()
            self._field_test_meta = {
                "schema_version": 2,
                "status": "running",
                "session_id": self.run_dir.name,
                "started_at_ms": started_at_ms,
                "started_at": _timestamp_text(started_at_ms),
                "test": {
                    key: str(field_test.get(key, "")).strip()
                    for key in ("test_id", "operator", "site", "scenario", "notes")
                },
                "runtime": {
                    "packaged": bool(getattr(sys, "frozen", False)),
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "configuration": {
                    "transport": self.settings["transport"],
                    "gate_source": _redact_source(self.settings["gate"]["source"]),
                    "reid_source": _redact_source(self.settings["reid"]["source"]),
                    "enabled_regions": list(self.settings["reid"]["enabled_regions"]),
                    "threshold": self.settings["reid"]["threshold"],
                    "sample_interval_ms": self.settings["reid"]["sample_interval_ms"],
                    "image_size": list(self.settings["reid"]["image_size"]),
                },
                "files": [
                    "field_test.json", "field_test_events.jsonl",
                    "test_results.jsonl", "test_images/",
                    "settings.json", "runtime.log", "messages.jsonl",
                    "decisions.jsonl", "summary.json",
                ],
            }
            self._write_field_test_meta()
            self._record_field_event("session_started", {
                "run_dir": str(self.run_dir),
                "configuration": self._field_test_meta["configuration"],
            })
        self.seen_events = {}
        self.records.clear()
        self.counts.clear()
        self.pending_sends.clear()
        self.output_index = 0
        self.workers.clear()
        self._preview_frames.clear()
        self._preview_sequences.clear()
        self._latest_result = None
        self._position_ages.clear()
        self._next_preview = self._next_debug = 0.0
        self.running = True
        self.stopping = False
        self.set_debug(settings.get("debug", {}).get("enabled", False))
        self.running_changed.emit(True)
        try:
            if settings["transport"] == "mqtt":
                import os
                mqtt = dict(settings["mqtt"])
                mqtt["password"] = mqtt.get("password") or os.environ.get("MQTT_PASSWORD", "")
                self.bridge = MqttLink(
                    mqtt, [mqtt["person_topic"], *settings["binding"]["topics"].values()],
                    self.inbox, self)
                self.bridge.state_changed.connect(self._mqtt_state)
                self.bridge.error.connect(self.write_log)
                self.bridge.delivery.connect(self._delivery)
                self.bridge.start()
                self.state.emit("transport", "连接中")
            else:
                self.state.emit("transport", "本机直连")
            self.state.emit("binding", "接收中")
            self.timer.start()
            self.write_log("会话开始，输出目录：" + str(self.run_dir))
            if start_workers:
                gate = GateWorker(deepcopy(settings["gate"]), self.run_dir / "gate",
                                  clock=self.inbox.now)
                reid = RecognitionWorker(deepcopy(settings["reid"]),
                                         deepcopy(self.settings["binding"]["regions"]),
                                         clock=self.inbox.now)
                self.workers = {"gate": gate, "reid": reid}
                # 界面忙碌时仍限制待显示画面数量，避免预览持续显示旧画面。
                gate.frame.connect(lambda image: self._store_frame("gate", image),
                                   type=Qt.DirectConnection)
                gate.person.connect(self.receive_collected)
                reid.frame.connect(lambda image: self._store_frame("reid", image),
                                   type=Qt.DirectConnection)
                reid.result.connect(self._store_result, type=Qt.DirectConnection)
                reid.snapshot.connect(self.receive_snapshot)
                reid.person_added.connect(self.person)
                reid.ready.connect(
                    lambda worker=reid: self.state.emit("reid", worker.runtime_status))
                for name, worker in self.workers.items():
                    worker.log.connect(lambda text, stage=name: self.write_log(
                        "[{}] {}".format(stage, text)))
                    worker.failed.connect(lambda text, stage=name: self._worker_failed(stage, text))
                    worker.finished.connect(lambda stage=name: self._worker_finished(stage))
                    self.state.emit(name, "采集中" if name == "gate" else "模型加载中")
                    worker.start()
        except Exception:
            self.stop()
            raise

    def _mqtt_state(self, connected, text):
        if self.bridge is not None:
            self.bridge.connected = connected
        self.state.emit("transport", text)
        self.write_log("[MQTT] " + text)
        self._record_field_event("mqtt_state", {
            "connected": bool(connected), "text": text})

    def _delivery(self, index, status):
        if status in ("已确认", "未确认"):
            self.counts["mqtt_confirmations"] += int(status == "已确认")

    def _send(self, topic, payload):
        if not self.running or self.stopping:
            raise ValueError("请先启动调试")
        if self.bridge is None:
            self.inbox.put(topic, payload)
        else:
            if len(self.pending_sends) >= 1024:
                raise ValueError("MQTT 待发送消息超过 1024 条，请检查连接")
            self.pending_sends.append((topic, deepcopy(payload)))
        return True

    def receive_collected(self, payload):
        if not self.running or self.stopping:
            return
        try:
            self._send(self.settings["mqtt"]["person_topic"], payload)
        except (ValueError, TypeError) as exc:
            self.write_log(str(exc))

    def receive_snapshot(self, payload):
        if not self.running or self.stopping:
            return
        try:
            self._send(self.settings["binding"]["topics"]["detections"], payload)
        except (ValueError, TypeError) as exc:
            self.write_log(str(exc))

    def inject_rfid(self, region, rfid):
        if not isinstance(rfid, str) or not rfid.strip():
            raise ValueError("行李框 RFID 不能为空")
        if self.settings is None or region not in self.settings["binding"]["regions"]:
            raise ValueError("请选择已配置的区域")
        payload = {"roi_id": region, "rfid": rfid.strip(), "event_id": uuid.uuid4().hex}
        self._send(self.settings["binding"]["topics"]["rfid"], payload)
        self.write_log("[测试注入] RFID {} / {}".format(region, rfid.strip()))

    def import_image(self, path):
        import cv2
        import numpy as np
        if not self.running:
            raise ValueError("请先启动调试")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取上半身图片")
        passenger = uuid.uuid4().hex + "-" + datetime.now().strftime("%Y-%m-%d")
        payload = {
            "person_id": passenger, "passenger_id": passenger, "local_id": Path(path).stem,
            "capture_time": datetime.now().isoformat(), "source_video": "manual-image",
            "images": {"upper": encode_image(image), "face": None},
        }
        self.receive_collected(payload)
        return passenger

    def _record_message(self, topic, payload, received):
        record = {"topic": topic, "received_at_ms": received,
                  "payload": message_preview(payload)}
        self._handles["messages"].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        self.counts["messages"] += 1
        self.message.emit(record)

    def _accept(self, topic, payload, received, retained=False):
        if not isinstance(payload, dict):
            raise ValueError("消息必须是 JSON 对象")
        self._record_message(topic, payload, received)
        if topic == self.settings["mqtt"]["person_topic"]:
            worker = self.workers.get("reid")
            if worker is not None and worker.isRunning():
                worker.add_person(payload)
            self.counts["persons"] += 1
            with self._field_test_lock:
                self._record_field_event("person_collected", {
                    key: payload.get(key) for key in (
                        "passenger_id", "person_id", "local_id",
                        "capture_time", "source_video")
                } | {"images": self._save_field_test_person_images(payload)})
            return
        if topic == self.settings["binding"]["topics"]["rfid"]:
            if retained:
                self.write_log("忽略保留的 RFID 消息")
                return
            event_id = payload.get("event_id")
            if event_id is not None:
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError("event_id 必须是非空字符串")
                key = (payload["roi_id"], event_id)
                if key in self.seen_events:
                    if self.seen_events[key] != payload["rfid"]:
                        raise ValueError("同一事件编号对应不同 RFID")
                    return
        normalized = normalize_message(
            topic, payload, received, self.settings["binding"], DEFAULT_FIELDS)
        self._record_decisions(self.binder.ingest(normalized))
        if topic == self.settings["binding"]["topics"]["rfid"] and payload.get("event_id"):
            self.seen_events[(payload["roi_id"], payload["event_id"])] = payload["rfid"]
        if topic == self.settings["binding"]["topics"]["detections"]:
            age = received - payload["captured_at_ms"]
            self._position_ages[payload["roi_id"]] = age
            self.state.emit("binding", "接收中 / 位置延迟 {} ms".format(age))

    def _record_decisions(self, decisions):
        for result in decisions:
            self._handles["decisions"].write(json.dumps(result, ensure_ascii=False) + "\n")
            self._handles["decisions"].flush()
            self.counts[result["status"]] += 1
            self._record_field_event("binding_decision", result)
            self.decision.emit(result)
            self.write_log("[绑定] {} {} {}".format(
                result["rfid"], result["status"], result["passenger_id"] or result["reason"]))
            if result["binding"] is not None and self.settings["mqtt"]["output_enabled"]:
                if self.bridge is None:
                    self.write_log("当前为本机模式，绑定结果未发送到外部 MQTT")
                elif self.stopping:
                    self.write_log("正在停止，绑定结果仅保存在本次会话")
                    self.counts["unsent"] += 1
                else:
                    try:
                        self._send(self.settings["mqtt"]["output_topic"], result["binding"])
                    except ValueError as exc:
                        self.counts["unsent"] += 1
                        self.write_log("绑定输出未发送：" + str(exc))

    def _flush_sends(self):
        if self.bridge is None or not self.bridge.connected:
            return
        for _ in range(min(50, len(self.pending_sends))):
            topic, payload = self.pending_sends[0]
            self.output_index += 1
            if not self.bridge.publish(self.output_index, payload, topic, 1):
                break
            self.pending_sends.popleft()
            if topic == self.settings["mqtt"]["output_topic"]:
                self._record_message(topic, payload, self.inbox.now())

    def tick(self):
        if not self.running:
            return
        self._flush_sends()
        items, watermark = self.inbox.drain()
        for topic, raw, received, retained in items:
            try:
                payload = json.loads(raw.decode("utf-8")) if isinstance(raw, bytes) else raw
                self._accept(topic, payload, received, retained)
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                self.counts["invalid"] += 1
                self.write_log("消息无效：" + str(exc))
        if watermark is not None and watermark >= self.binder.last_received_at_ms:
            self._record_decisions(self.binder.advance(watermark))
        self._flush_result()
        now = time.monotonic()
        if now >= self._next_preview:
            self._next_preview = now + 1 / self.settings["reid"].get("preview_fps", 15)
            self._flush_previews()
        if self.stopping and not any(w.isRunning() for w in self.workers.values()):
            self._finish_stop()

    def _worker_failed(self, name, text):
        self.state.emit(name, "错误")
        self.write_log("[{}] {}".format(name, text))
        self.counts[name + "_errors"] += 1
        self._record_field_event("worker_error", {"worker": name, "text": text})

    def _worker_finished(self, name):
        if not self.counts[name + "_errors"]:
            self.state.emit(name, "已停止" if self.stopping else "已结束")

    def stop(self):
        if not self.running:
            return
        self.stopping = True
        self.write_log("正在停止后台任务")
        for worker in self.workers.values():
            worker.stop()
        if not any(worker.isRunning() for worker in self.workers.values()):
            self._finish_stop()

    def _finish_stop(self):
        self.timer.stop()
        if self.bridge is not None:
            if self.pending_sends:
                self.write_log("停止时尚有 {} 条 MQTT 消息未发送".format(len(self.pending_sends)))
                self.counts["unsent"] += len(self.pending_sends)
            self.pending_sends.clear()
            if self.bridge.waiting:
                self.write_log("停止时尚有 {} 条 MQTT 消息未确认".format(len(self.bridge.waiting)))
            self.bridge.close()
            self.bridge.deleteLater()
            self.bridge = None
        self._record_field_event("session_stopped", {"counts": dict(self.counts)})
        self.write_log("会话已结束：" + json.dumps(dict(self.counts), ensure_ascii=False))
        summary = dict(self.counts)
        if self.field_test_enabled:
            summary["field_test"] = {
                "manual_observations": self._field_test_counts["manual_observation"],
                "recognition_records": self._field_test_counts["recognition"],
                "test_results": self._field_test_counts["test_results"],
                "images_saved": self._field_test_counts["images_saved"],
                "image_errors": self._field_test_counts["image_errors"],
                "result_errors": self._field_test_counts["result_errors"],
                "event_counts": dict(self._field_test_counts),
            }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.field_test_enabled:
            ended_at_ms = self.inbox.now()
            self._field_test_meta.update({
                "status": "finished",
                "ended_at_ms": ended_at_ms,
                "ended_at": _timestamp_text(ended_at_ms),
                "duration_ms": ended_at_ms - self._field_test_meta["started_at_ms"],
                "counts": dict(self.counts),
                "event_counts": dict(self._field_test_counts),
            })
            self._write_field_test_meta()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self.logger.removeHandler(self._file_handler)
        self._file_handler.close()
        for worker in self.workers.values():
            worker.deleteLater()
        self.workers.clear()
        self.running = self.stopping = False
        self.field_test_enabled = False
        self.state.emit("binding", "已停止")
        self.running_changed.emit(False)
        self.stopped.emit()
