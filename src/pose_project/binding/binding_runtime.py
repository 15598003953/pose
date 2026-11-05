"""Qt session control and data adapters for the passenger/tray binder."""

from collections import OrderedDict, deque
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from pose_project.binding.passenger_tray_binding import PassengerTrayBinder, require_text
from pose_project.binding.replay_binding import read_jsonl


DEFAULT_FIELDS = {
    "roi_id": "roi_id",
    "captured_at_ms": "captured_at_ms",
    "detections": "detections",
    "passenger_id": "passenger_id",
    "bbox_xyxy": "bbox_xyxy",
    "rfid": "rfid",
    "event_id": "event_id",
}


def field_value(data, path):
    for part in path.split("."):
        data = data[part]
    return data


def normalize_message(topic, data, received, config, fields):
    if not isinstance(data, dict):
        raise ValueError("消息必须是 JSON 对象")
    payload = data.get("payload", data)
    roi_id = field_value(payload, fields["roi_id"])
    result = {"roi_id": roi_id}
    if topic == config["topics"]["detections"]:
        result["captured_at_ms"] = field_value(payload, fields["captured_at_ms"])
        people = field_value(payload, fields["detections"])
        if not isinstance(people, list):
            raise ValueError("人员列表必须是数组")
        result["detections"] = [{
            "passenger_id": field_value(person, fields["passenger_id"]),
            "bbox_xyxy": field_value(person, fields["bbox_xyxy"]),
        } for person in people]
    elif topic == config["topics"]["rfid"]:
        result["rfid"] = field_value(payload, fields["rfid"])
        if fields.get("event_id"):
            try:
                result["event_id"] = field_value(payload, fields["event_id"])
            except KeyError:
                pass
    else:
        raise ValueError("未知消息主题")
    return {"topic": topic, "received_at_ms": received, "payload": result}


def validate_broker(settings):
    require_text(settings["host"], "MQTT 地址")
    if type(settings["port"]) is not int or not 1 <= settings["port"] <= 65535:
        raise ValueError("MQTT 端口必须在 1 到 65535 之间")


def validate_topic(topic):
    require_text(topic, "MQTT 主题")
    if any(character in topic for character in ("+", "#", "\x00")):
        raise ValueError("消息主题不能包含通配符或空字符")


class LiveInbox:
    """Stamp reception and take a timer watermark under the same lock."""

    def __init__(self, limit=10000):
        self.items = deque()
        self.lock = threading.Lock()
        self.limit = limit
        self.epoch = time.time_ns() // 1000000
        self.started = time.monotonic_ns()

    def now(self):
        return self.epoch + (time.monotonic_ns() - self.started) // 1000000

    def put(self, topic, data, retained=False):
        with self.lock:
            if len(self.items) >= self.limit:
                raise ValueError("消息积压超过上限，请停止后检查发送频率")
            received = self.now()
            self.items.append((topic, data, received, retained))
            return received

    def drain(self, limit=500):
        with self.lock:
            items = [self.items.popleft() for _ in range(min(limit, len(self.items)))]
            # A later callback can share this millisecond, so leave it uncommitted.
            watermark = self.now() - 1 if not self.items else None
            return items, watermark


class MqttLink(QObject):
    state_changed = pyqtSignal(bool, str)
    delivery = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, settings, topics=None, inbox=None, parent=None):
        super().__init__(parent)
        validate_broker(settings)
        self.settings = settings
        self.topics = topics or []
        self.inbox = inbox
        self.connected = False
        self.client = None
        self.waiting = {}
        self.received_packets = OrderedDict()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.check_delivery)

    def start(self):
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="binding-{}".format(time.monotonic_ns()),
        )
        self.client.on_connect = self._connected
        self.client.on_disconnect = self._disconnected
        self.client.on_message = self._message
        self.client.on_subscribe = self._subscribed
        self.client.on_connect_fail = lambda client, userdata: self.state_changed.emit(
            False, "连接失败，正在重试"
        )
        if self.settings.get("username"):
            self.client.username_pw_set(
                self.settings["username"], self.settings.get("password") or None
            )
        if self.settings.get("tls"):
            self.client.tls_set()
        self.client.connect_async(self.settings["host"], self.settings["port"], keepalive=30)
        self.client.loop_start()
        self.timer.start()

    def _connected(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self.state_changed.emit(False, "连接被拒绝：{}".format(reason_code))
            return
        self.received_packets.clear()
        if self.topics:
            client.subscribe([(topic, 1) for topic in self.topics])
        else:
            self.state_changed.emit(True, "已连接")

    def _subscribed(self, client, userdata, mid, reasons, properties):
        if any(reason.is_failure for reason in reasons):
            self.state_changed.emit(False, "主题订阅被拒绝")
        else:
            self.state_changed.emit(True, "已连接并订阅")

    def _disconnected(self, client, userdata, flags, reason_code, properties):
        self.state_changed.emit(False, "连接已断开")

    def _message(self, client, userdata, message):
        try:
            raw = bytes(message.payload)
            if message.qos:
                key = (message.topic, message.mid)
                if message.dup and self.received_packets.get(key) == raw:
                    return
                self.received_packets[key] = raw
                self.received_packets.move_to_end(key)
                if len(self.received_packets) > 4096:
                    self.received_packets.popitem(last=False)
            # Stamp before decoding or scheduling any GUI work.
            self.inbox.put(message.topic, raw, bool(getattr(message, "retain", False)))
        except (ValueError, TypeError) as exc:
            self.error.emit(str(exc))

    def publish(self, index, binding, topic, qos):
        if not self.connected or self.client is None:
            return False
        info = self.client.publish(
            topic, json.dumps(binding, ensure_ascii=False), qos=qos, retain=False
        )
        queued_offline = qos > 0 and info.rc == mqtt.MQTT_ERR_NO_CONN
        if info.rc != mqtt.MQTT_ERR_SUCCESS and not queued_offline:
            self.error.emit("结果发送失败：{}".format(mqtt.error_string(info.rc)))
            return False
        self.waiting[index] = (info, qos)
        self.delivery.emit(index, "等待重连" if queued_offline else "发送中")
        return True

    def check_delivery(self):
        for index, (info, qos) in list(self.waiting.items()):
            if info.is_published():
                del self.waiting[index]
                self.delivery.emit(index, "已确认" if qos else "已发送")

    def close(self):
        self.timer.stop()
        if self.client is not None:
            client, self.client = self.client, None
            client.disconnect()
            client.loop_stop()
        self.connected = False


class HttpInput:
    def __init__(self, host, port, topics, inbox):
        self.topics, self.inbox = topics, inbox
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    if self.path not in ("/detections", "/rfid"):
                        self.send_error(404)
                        return
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 1024 * 1024:
                        self.send_error(413)
                        return
                    raw = self.rfile.read(size)
                    data = json.loads(raw.decode("utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("消息必须是 JSON 对象")
                    topic = owner.topics[self.path[1:]]
                    received = owner.inbox.put(topic, data)
                    reply = json.dumps({"accepted": True, "received_at_ms": received}).encode()
                    self.send_response(202)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(reply)))
                    self.end_headers()
                    self.wfile.write(reply)
                except (ValueError, UnicodeError, KeyError) as exc:
                    self.send_error(400, str(exc).encode("ascii", "replace").decode())

            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )

    @property
    def address(self):
        return self.server.server_address

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


class BindingSession(QObject):
    snapshot = pyqtSignal(dict)
    decision = pyqtSignal(int, dict)
    delivery = pyqtSignal(int, str)
    state_changed = pyqtSignal(str)
    connection_changed = pyqtSignal(str, bool, str)
    progress = pyqtSignal(int, int, int)
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.tick)
        self.input_link = self.output_link = self.http = None
        self.iterator = self.next_message = None
        self.running = self.paused = False
        self.decisions = []
        self.outbox = deque()
        self.message_count = self.invalid_count = 0
        self.seen_events = {}

    def start(self, config, options):
        self.stop()
        self.binder = PassengerTrayBinder(config)
        self.config, self.options = deepcopy(config), deepcopy(options)
        for topic in config["topics"].values():
            validate_topic(topic)
        if options["source"] != "file":
            for key in DEFAULT_FIELDS:
                if key != "event_id":
                    require_text(options["fields"][key], "消息字段 {}".format(key))
        if options.get("output_enabled"):
            validate_topic(options["output_topic"])
            if options["output_topic"] in config["topics"].values():
                raise ValueError("结果主题必须与检测和 RFID 主题不同")
        self.decisions, self.outbox, self.seen_events = [], deque(), {}
        self.message_count = self.invalid_count = 0
        self.latest_captures = {}
        self.inbox = LiveInbox()
        self.running, self.paused = True, False
        try:
            if options.get("output_enabled"):
                self.output_link = MqttLink(options["output_mqtt"], parent=self)
                self.output_link.state_changed.connect(self._output_state)
                self.output_link.delivery.connect(self.delivery)
                self.output_link.error.connect(self.log)
                self.output_link.start()
                self.connection_changed.emit("output", False, "连接中")
            source = options["source"]
            if source == "file":
                self.iterator = iter(read_jsonl(options["file_path"]))
                self.next_message = next(self.iterator, None)
                if self.next_message is None:
                    raise ValueError("回放文件为空")
                self.replay_start = self.next_message["received_at_ms"]
                self.virtual_now = self.replay_start
                self.anchor = time.monotonic()
                self.anchor_virtual = self.virtual_now
                self.connection_changed.emit("input", True, "文件已打开")
            elif source == "mqtt":
                self.input_link = MqttLink(
                    options["input_mqtt"], list(config["topics"].values()), self.inbox, self
                )
                self.input_link.state_changed.connect(self._input_state)
                self.input_link.error.connect(self._fatal)
                self.input_link.start()
                self.connection_changed.emit("input", False, "连接中")
            elif source == "http":
                self.http = HttpInput(
                    options["http_host"], options["http_port"], config["topics"], self.inbox
                )
                self.http.start()
                self.connection_changed.emit(
                    "input", True, "HTTP {}:{}".format(*self.http.address)
                )
            else:
                raise ValueError("未知数据接口")
            self.timer.start()
            self.state_changed.emit("running")
        except Exception:
            self.stop()
            raise

    def _input_state(self, connected, text):
        if self.sender() is self.input_link and self.input_link is not None:
            self.input_link.connected = connected
            self.connection_changed.emit("input", connected, text)

    def _output_state(self, connected, text):
        if self.sender() is self.output_link and self.output_link is not None:
            self.output_link.connected = connected
            self.connection_changed.emit("output", connected, text)

    def set_paused(self, paused):
        if self.options["source"] != "file" or not self.running:
            return
        self.paused = paused
        if not paused:
            self.anchor_virtual, self.anchor = self.virtual_now, time.monotonic()
        self.state_changed.emit("paused" if paused else "running")

    def _accept(self, message):
        if message["topic"] == self.config["topics"]["rfid"]:
            payload = message["payload"]
            event_id = payload.get("event_id")
            if event_id is not None:
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError("RFID event_id 必须是非空字符串")
                key = (payload["roi_id"], event_id)
                fingerprint = payload["rfid"]
                if key in self.seen_events:
                    if self.seen_events[key] != fingerprint:
                        raise ValueError("同一 RFID 事件编号对应不同 RFID")
                    self.log.emit("忽略重复 RFID 事件：{}".format(event_id))
                    return
        decisions = self.binder.ingest(message)
        if message["topic"] == self.config["topics"]["rfid"] and event_id is not None:
            self.seen_events[key] = fingerprint
        self.message_count += 1
        payload = message["payload"]
        if message["topic"] == self.config["topics"]["detections"]:
            roi, captured = payload["roi_id"], payload["captured_at_ms"]
            if captured >= self.latest_captures.get(roi, -1):
                self.latest_captures[roi] = captured
                self.pending_snapshots[roi] = payload
        self._record(decisions)

    def _record(self, decisions):
        for item in decisions:
            index = len(self.decisions)
            self.decisions.append(item)
            self.decision.emit(index, item)
            if item["binding"] is not None and self.output_link is not None:
                self.outbox.append(index)
                self.delivery.emit(index, "等待连接")

    def _flush_output(self):
        if self.output_link is None or not self.output_link.connected:
            return
        for _ in range(min(50, len(self.outbox))):
            index = self.outbox[0]
            if not self.output_link.publish(
                index, self.decisions[index]["binding"],
                self.options["output_topic"], self.options["output_qos"]
            ):
                break
            self.outbox.popleft()

    def tick(self):
        if not self.running:
            return
        self.pending_snapshots = {}
        try:
            if self.options["source"] == "file":
                if not self.paused:
                    self._tick_file()
            else:
                items, watermark = self.inbox.drain()
                for topic, raw, received, retained in items:
                    try:
                        if retained and topic == self.config["topics"]["rfid"]:
                            self.log.emit("忽略 MQTT 保留的 RFID 消息")
                            continue
                        data = json.loads(raw.decode("utf-8")) if isinstance(raw, bytes) else raw
                        message = normalize_message(
                            topic, data, received, self.config, self.options["fields"]
                        )
                        self._accept(message)
                    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                        self.invalid_count += 1
                        self.log.emit("输入消息无效：{}".format(exc))
                if watermark is not None and watermark >= self.binder.last_received_at_ms:
                    self._record(self.binder.advance(watermark))
            for payload in self.pending_snapshots.values():
                self.snapshot.emit(payload)
            self._flush_output()
            self.progress.emit(self.message_count, len(self.decisions), self.invalid_count)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._fatal(str(exc))

    def _tick_file(self):
        self.virtual_now = int(
            self.anchor_virtual + (time.monotonic() - self.anchor) * 1000
            * self.options["speed"]
        )
        count = 0
        while (
            self.next_message is not None
            and self.next_message["received_at_ms"] <= self.virtual_now and count < 500
        ):
            self._accept(self.next_message)
            self.next_message = next(self.iterator, None)
            count += 1
        if self.next_message is None:
            self._record(self.binder.finish())
            self.paused = True
            self.state_changed.emit("finished")
        elif self.next_message["received_at_ms"] > self.virtual_now:
            self._record(self.binder.advance(self.virtual_now))

    def _fatal(self, message):
        self.stop()
        self.failed.emit(message)

    def stop(self):
        self.timer.stop()
        self.running = False
        if self.output_link is not None:
            for index in self.outbox:
                self.delivery.emit(index, "未发送")
            for index in self.output_link.waiting:
                self.delivery.emit(index, "未确认")
        for name in ("input_link", "output_link", "http"):
            connection = getattr(self, name)
            setattr(self, name, None)
            if connection is not None:
                connection.close()
                if isinstance(connection, QObject):
                    connection.deleteLater()
        if self.iterator is not None:
            self.iterator.close()
        self.iterator = self.next_message = None
        self.state_changed.emit("stopped")

    def export(self, output):
        from pose_project.binding.replay_binding import write_results
        return write_results(output, self.decisions, {
            "message_count": self.message_count,
            "decision_count": len(self.decisions),
            "invalid_messages": self.invalid_count,
            "config": self.config,
            "source": self.options["source"],
            "exported_at": datetime.now().astimezone().isoformat(),
        })
