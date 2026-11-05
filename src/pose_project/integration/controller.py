"""One event path for local and MQTT integration sessions."""

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import time
import uuid

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from pose_project.binding.binding_runtime import (
    DEFAULT_FIELDS, LiveInbox, MqttLink, normalize_message,
)
from pose_project.binding.passenger_tray_binding import PassengerTrayBinder
from pose_project.integration.configuration import new_run_directory, validate, save_settings
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


class IntegrationController(QObject):
    frame = pyqtSignal(str, object)
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

    def write_log(self, text):
        text = re.sub(r"(rtsps?://)[^/@\s]+:[^/@\s]+@", r"\1***@", str(text))
        line = datetime.now().strftime("%H:%M:%S") + " " + text
        self.logger.info(line)
        self.log.emit(line)

    def start(self, settings, start_workers=True):
        if self.running or self.stopping:
            raise ValueError("上一轮调试仍在运行或停止中")
        validate(settings, sources=start_workers)
        self.settings = deepcopy(settings)
        self.run_dir = new_run_directory(settings)
        save_settings(settings, self.run_dir / "settings.json")
        handler = logging.FileHandler(self.run_dir / "runtime.log", encoding="utf-8")
        self.logger.addHandler(handler)
        self._file_handler = handler
        self._handles = {
            key: (self.run_dir / (key + ".jsonl")).open("a", encoding="utf-8")
            for key in ("messages", "decisions")
        }
        self.binder = PassengerTrayBinder(settings["binding"])
        self.inbox = LiveInbox()
        self.seen_events = {}
        self.records.clear()
        self.counts.clear()
        self.pending_sends.clear()
        self.output_index = 0
        self.workers.clear()
        self.running = True
        self.stopping = False
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
                                         deepcopy(settings["binding"]["regions"]),
                                         clock=self.inbox.now)
                self.workers = {"gate": gate, "reid": reid}
                gate.frame.connect(lambda image: self.frame.emit("gate", image))
                gate.person.connect(self.receive_collected)
                reid.frame.connect(lambda image: self.frame.emit("reid", image))
                reid.snapshot.connect(self.receive_snapshot)
                reid.person_added.connect(self.person)
                reid.ready.connect(lambda: self.state.emit("reid", "识别中"))
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
            self.state.emit("binding", "接收中 / 位置延迟 {} ms".format(age))

    def _record_decisions(self, decisions):
        for result in decisions:
            self._handles["decisions"].write(json.dumps(result, ensure_ascii=False) + "\n")
            self._handles["decisions"].flush()
            self.counts[result["status"]] += 1
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
        if self.stopping and not any(w.isRunning() for w in self.workers.values()):
            self._finish_stop()

    def _worker_failed(self, name, text):
        self.state.emit(name, "错误")
        self.write_log("[{}] {}".format(name, text))
        self.counts[name + "_errors"] += 1

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
        self.write_log("会话已结束：" + json.dumps(dict(self.counts), ensure_ascii=False))
        (self.run_dir / "summary.json").write_text(
            json.dumps(dict(self.counts), ensure_ascii=False, indent=2), encoding="utf-8")
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self.logger.removeHandler(self._file_handler)
        self._file_handler.close()
        for worker in self.workers.values():
            worker.deleteLater()
        self.workers.clear()
        self.running = self.stopping = False
        self.state.emit("binding", "已停止")
        self.running_changed.emit(False)
        self.stopped.emit()
