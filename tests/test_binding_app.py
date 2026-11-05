"""Exercise Qt controls and real loopback HTTP/MQTT transport."""

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import socketserver
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from pose_project.apps.binding_app import BindingWindow
from pose_project.apps.binding_simulation_dialog import SimulationDialog
from pose_project.binding.binding_runtime import (
    BindingSession, DEFAULT_FIELDS, LiveInbox, MqttLink, normalize_message,
)
from pose_project.binding.replay_binding import DEFAULT_CONFIG, DEFAULT_INPUT, evaluate, read_jsonl
from pose_project.binding.generate_binding_simulation import DEFAULT_CONFIG as SIMULATION_CONFIG


def packet(header, body):
    remaining = len(body)
    encoded = bytearray([header])
    while True:
        digit = remaining % 128
        remaining //= 128
        encoded.append(digit | (128 if remaining else 0))
        if not remaining:
            return bytes(encoded) + body


def mqtt_string(value):
    raw = value.encode("utf-8")
    return len(raw).to_bytes(2, "big") + raw


def read_exact(connection, size):
    result = bytearray()
    while len(result) < size:
        part = connection.recv(size - len(result))
        if not part:
            raise EOFError
        result.extend(part)
    return bytes(result)


class LoopbackBroker(socketserver.ThreadingTCPServer):
    """Minimal broker fixture; production networking uses paho-mqtt."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.clients = []
        self.received = []
        self.lock = threading.Lock()
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def setup(self):
                self.topics = set()
                self.write_lock = threading.Lock()
                self.request.settimeout(3)
                with owner.lock:
                    owner.clients.append(self)

            def send(self, data):
                with self.write_lock:
                    self.request.sendall(data)

            def handle(self):
                try:
                    while True:
                        header = read_exact(self.request, 1)[0]
                        size, multiplier = 0, 1
                        while True:
                            digit = read_exact(self.request, 1)[0]
                            size += (digit & 127) * multiplier
                            if not digit & 128:
                                break
                            multiplier *= 128
                        body = read_exact(self.request, size)
                        kind = header >> 4
                        if kind == 1:
                            self.send(b"\x20\x02\x00\x00")
                        elif kind == 8:
                            position = 2
                            granted = bytearray()
                            while position < len(body):
                                length = int.from_bytes(body[position:position + 2], "big")
                                position += 2
                                self.topics.add(body[position:position + length].decode())
                                position += length
                                granted.append(body[position])
                                position += 1
                            self.send(packet(0x90, body[:2] + granted))
                        elif kind == 3:
                            length = int.from_bytes(body[:2], "big")
                            topic = body[2:2 + length].decode()
                            position = 2 + length
                            qos = (header >> 1) & 3
                            mid = body[position:position + 2] if qos else b""
                            position += 2 if qos else 0
                            with owner.lock:
                                owner.received.append((topic, body[position:], header))
                            if qos == 1:
                                self.send(packet(0x40, mid))
                            elif qos == 2:
                                self.send(packet(0x50, mid))
                        elif kind == 6:
                            self.send(packet(0x70, body[:2]))
                        elif kind == 12:
                            self.send(b"\xd0\x00")
                        elif kind == 14:
                            return
                except (EOFError, OSError):
                    pass

            def finish(self):
                with owner.lock:
                    owner.clients.remove(self)

        super().__init__(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    def send_message(self, topic, data, retained=False):
        raw = json.dumps(data).encode("utf-8")
        with self.lock:
            subscribers = [client for client in self.clients if topic in client.topics]
        for client in subscribers:
            client.send(packet(0x30 | int(retained), mqtt_string(topic) + raw))

    def close(self):
        self.shutdown()
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            try:
                client.request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server_close()
        self.thread.join(timeout=1)


class BindingAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if not QFontDatabase().families():
            QFontDatabase.addApplicationFont(
                str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc")
            )
        cls.app.setStyle("Fusion")
        cls.app.setFont(QFont("Microsoft YaHei", 9))
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def setUp(self):
        self.sessions = []
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        for session in self.sessions:
            session.stop()
            session.deleteLater()
        self.app.processEvents()

    def wait_for(self, predicate, timeout=5000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate(), "timed out waiting for Qt/transport")

    def options(self, source="file"):
        broker = {
            "host": "127.0.0.1", "port": 1883, "username": "", "password": "", "tls": False
        }
        return {
            "source": source, "file_path": str(DEFAULT_INPUT), "speed": 2000,
            "input_mqtt": dict(broker), "output_mqtt": dict(broker),
            "output_enabled": False, "output_topic": "test/bindings", "output_qos": 1,
            "http_host": "127.0.0.1", "http_port": 0, "fields": dict(DEFAULT_FIELDS),
        }

    def new_session(self):
        session = BindingSession()
        self.sessions.append(session)
        return session

    def test_live_field_mapping_and_receiver_owned_timestamp(self):
        fields = dict(DEFAULT_FIELDS, roi_id="station.zone", captured_at_ms="frame.time",
                      detections="frame.people", passenger_id="person.id", bbox_xyxy="box")
        raw = {"received_at_ms": 1, "payload": {
            "station": {"zone": "A"},
            "frame": {"time": 900, "people": [
                {"person": {"id": "passenger"}, "box": [320, 255, 480, 495]}
            ]},
        }}
        normalized = normalize_message(
            self.config["topics"]["detections"], raw, 1000, self.config, fields
        )
        self.assertEqual(1000, normalized["received_at_ms"])
        self.assertEqual(900, normalized["payload"]["captured_at_ms"])
        self.assertEqual("passenger", normalized["payload"]["detections"][0]["passenger_id"])
        self.assertEqual("A", normalized["payload"]["roi_id"])

    def test_inbox_stamps_in_order_and_does_not_commit_future_arrivals(self):
        inbox = LiveInbox(limit=100)
        threads = [
            threading.Thread(target=inbox.put, args=("topic", {"index": index}))
            for index in range(30)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        rows, watermark = inbox.drain()
        times = [row[2] for row in rows]
        self.assertEqual(sorted(times), times)
        self.assertEqual(30, len(rows))
        self.assertGreater(inbox.put("topic", {}), watermark)
        tiny = LiveInbox(limit=1)
        tiny.put("topic", {})
        with self.assertRaises(ValueError):
            tiny.put("topic", {})

    def test_mqtt_retry_packets_are_deduplicated_without_dropping_new_events(self):
        inbox = LiveInbox()
        link = MqttLink(self.options()["input_mqtt"], inbox=inbox)
        message = SimpleNamespace(topic="test/rfid", mid=9, qos=1, dup=False,
                                  payload=b'{"rfid": "tray-1"}')
        link._message(None, None, message)
        message.dup = True
        link._message(None, None, message)
        message.dup, message.payload = False, b'{"rfid": "tray-2"}'
        link._message(None, None, message)
        message.dup = True
        link._message(None, None, message)
        rows, _ = inbox.drain()
        self.assertEqual(2, len(rows))
        self.assertEqual([b'{"rfid": "tray-1"}', b'{"rfid": "tray-2"}'],
                         [row[1] for row in rows])

    def test_qos_one_queued_during_disconnect_is_not_resubmitted(self):
        link = MqttLink(self.options()["output_mqtt"])
        info = SimpleNamespace(rc=4)
        link.client = SimpleNamespace(publish=lambda *args, **kwargs: info)
        link.connected = True
        self.assertTrue(link.publish(
            0, {"roi_id": "A", "passenger_id": "p", "rfid": "tray"}, "test/bindings", 1
        ))
        self.assertEqual((info, 1), link.waiting[0])
        link.client = None

    def test_file_session_matches_independent_labels_and_exports(self):
        session = self.new_session()
        errors = []
        session.failed.connect(errors.append)
        session.start(self.config, self.options())
        self.wait_for(lambda: session.paused)
        self.assertEqual([], errors)
        self.assertEqual(3609, session.message_count)
        expected = list(read_jsonl(DEFAULT_INPUT.with_name("expected_bindings.jsonl")))
        self.assertTrue(evaluate(session.decisions, expected)["all_events_match"])
        with tempfile.TemporaryDirectory() as tmp:
            session.export(tmp)
            self.assertEqual(9, len(list(read_jsonl(Path(tmp) / "bindings.jsonl"))))
            saved = json.loads((Path(tmp) / "run.json").read_text(encoding="utf-8"))
            self.assertNotIn("password", json.dumps(saved))

    def test_pause_holds_virtual_time_and_stop_discards_pending_events(self):
        session = self.new_session()
        options = self.options()
        options["speed"] = 1
        session.start(self.config, options)
        session.set_paused(True)
        count = session.message_count
        QTest.qWait(80)
        self.assertEqual(count, session.message_count)
        session.set_paused(False)
        self.wait_for(lambda: session.message_count > count)
        session.stop()
        self.assertFalse(session.timer.isActive())
        self.assertFalse(session.running)
        self.assertEqual([], session.decisions)

    def test_http_input_receiver_timestamp_invalid_json_and_event_deduplication(self):
        session = self.new_session()
        config = deepcopy(self.config)
        config["binding_parameters"]["binding_wait_ms"] = 50
        session.start(config, self.options("http"))
        host, port = session.http.address
        base = session.inbox.now() - 6000

        def post(path, payload):
            request = Request("http://{}:{}{}".format(host, port, path),
                              json.dumps(payload).encode(), {"Content-Type": "application/json"})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(202, response.status)
                return json.load(response)

        for offset in range(0, 6001, 200):
            post("/detections", {
                "roi_id": "A", "captured_at_ms": base + offset,
                "detections": [{"passenger_id": "person", "bbox_xyxy": [320, 255, 480, 495]}],
            })
        reply = post("/rfid", {
            "roi_id": "A", "rfid": "tray-http", "event_id": "event-1", "received_at_ms": 1,
        })
        post("/rfid", {"roi_id": "A", "rfid": "tray-http", "event_id": "event-1"})
        self.wait_for(lambda: len(session.decisions) == 1)
        self.assertEqual("bound", session.decisions[0]["status"])
        self.assertEqual(reply["received_at_ms"], session.decisions[0]["rfid_received_at_ms"])
        self.assertEqual("tray-http", session.decisions[0]["binding"]["rfid"])
        request = Request("http://{}:{}/rfid".format(host, port), b"[]")
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=2)
        self.assertEqual(400, raised.exception.code)
        post("/detections", {"roi_id": "unknown"})
        self.wait_for(lambda: session.invalid_count == 1)
        session.stop()
        self.assertIsNone(session.http)

    def test_real_mqtt_subscription_output_and_qos_acknowledgement(self):
        broker = LoopbackBroker()
        session = self.new_session()
        try:
            options = self.options("mqtt")
            options["input_mqtt"]["port"] = broker.server_address[1]
            options["output_mqtt"]["port"] = broker.server_address[1]
            options["output_enabled"] = True
            config = deepcopy(self.config)
            config["binding_parameters"]["binding_wait_ms"] = 50
            deliveries = []
            session.delivery.connect(lambda index, status: deliveries.append((index, status)))
            session.start(config, options)
            self.wait_for(lambda: session.input_link.connected and session.output_link.connected)
            base = session.inbox.now() - 6000
            for offset in range(0, 6001, 200):
                broker.send_message(config["topics"]["detections"], {
                    "roi_id": "A", "captured_at_ms": base + offset,
                    "detections": [{"passenger_id": "person",
                                    "bbox_xyxy": [320, 255, 480, 495]}],
                })
            broker.send_message(config["topics"]["rfid"], {
                "roi_id": "A", "rfid": "tray-mqtt", "event_id": "event-mqtt-1"
            })
            self.wait_for(lambda: (0, "已确认") in deliveries)
            with broker.lock:
                received = list(broker.received)
            self.assertEqual(1, len(received))
            topic, body, header = received[0]
            self.assertEqual("test/bindings", topic)
            self.assertEqual({
                "roi_id": "A", "passenger_id": "person", "rfid": "tray-mqtt"
            }, json.loads(body))
            self.assertEqual(1, (header >> 1) & 3)
            self.assertEqual(0, header & 1)
            broker.send_message(config["topics"]["rfid"], {
                "roi_id": "A", "rfid": "historical-retained-tray"
            }, retained=True)
            QTest.qWait(100)
            self.assertEqual(1, len(session.decisions))
            session.stop()
        finally:
            session.stop()
            broker.close()

    def test_file_replay_publishes_exact_binding_payloads_at_qos_zero_and_two(self):
        broker = LoopbackBroker()
        try:
            for qos in (0, 2):
                with self.subTest(qos=qos):
                    session = self.new_session()
                    options = self.options()
                    options["output_enabled"] = True
                    options["output_mqtt"]["port"] = broker.server_address[1]
                    options["output_qos"] = qos
                    deliveries = []
                    session.delivery.connect(
                        lambda index, status: deliveries.append((index, status))
                    )
                    with broker.lock:
                        previous_count = len(broker.received)
                    session.start(self.config, options)
                    status = "已确认" if qos else "已发送"
                    self.wait_for(
                        lambda: sum(value == status for _, value in deliveries) == 9
                    )
                    with broker.lock:
                        rows = list(broker.received[previous_count:])
                    self.assertEqual(9, len(rows))
                    for topic, body, header in rows:
                        payload = json.loads(body)
                        self.assertEqual({"roi_id", "passenger_id", "rfid"}, set(payload))
                        self.assertRegex(
                            payload["passenger_id"], r"^[0-9a-f]{32}-2026-10-20$"
                        )
                        self.assertEqual(qos, (header >> 1) & 3)
                        self.assertEqual(0, header & 1)
                    session.stop()
        finally:
            broker.close()

    def test_bad_start_rolls_back_connections_and_output_topic_cannot_loop(self):
        session = self.new_session()
        options = self.options()
        options["output_enabled"] = True
        options["output_topic"] = self.config["topics"]["rfid"]
        with self.assertRaisesRegex(ValueError, "主题"):
            session.start(self.config, options)
        self.assertFalse(session.running)
        self.assertIsNone(session.output_link)
        options = self.options("http")
        options["http_host"] = "invalid host"
        with self.assertRaises(OSError):
            session.start(self.config, options)
        self.assertFalse(session.running)
        self.assertFalse(session.timer.isActive())

    def test_generate_from_controls_selects_replay_and_matches_independent_labels(self):
        window = BindingWindow()
        self.windows.append(window)
        window.show()
        window.parameters["binding_wait_ms"].setValue(750)
        window.output_topic.setText("custom/bindings")
        window.output_broker.host.setText("192.0.2.10")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "pose_project.apps.binding_simulation_dialog.QMessageBox.critical"
        ) as errors:
            config = json.loads(SIMULATION_CONFIG.read_text(encoding="utf-8"))
            config["regions"]["C"] = config["regions"].pop("A")
            for item in config["tracks"] + config["push_events"]:
                if item["roi_id"] == "A":
                    item["roi_id"] = "C"
            config["topics"] = {
                "detections": "custom/detections", "rfid": "custom/rfid"
            }
            template = Path(tmp) / "自定义场景.json"
            template.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            original = template.read_bytes()
            QTest.mouseClick(window.simulate_button, Qt.LeftButton)
            dialog = window.simulation_dialog
            self.assertTrue(dialog.isVisible())
            with patch("pose_project.apps.binding_simulation_dialog.QFileDialog.getOpenFileName",
                       return_value=(str(template), "")):
                dialog.browse_template()
            output = Path(tmp) / "模拟数据"
            output.mkdir()
            with patch("pose_project.apps.binding_simulation_dialog.QFileDialog.getExistingDirectory",
                       return_value=str(output)):
                dialog.browse_output()
            dialog.seed.setText("4567")
            dialog.start_time.setText("2026-10-21T09:00:00+08:00")
            dialog.interval.setValue(100)
            dialog.jitter.setValue(4)
            if os.environ.get("BINDING_UI_ARTIFACT_DIR"):
                folder = Path(os.environ["BINDING_UI_ARTIFACT_DIR"])
                folder.mkdir(parents=True, exist_ok=True)
                QTest.qWait(50)
                self.assertTrue(dialog.grab().save(str(folder / "simulation_dialog.png")))
            QTest.mouseClick(dialog.generate_button, Qt.LeftButton)
            self.wait_for(lambda: dialog.result() == dialog.Accepted)
            errors.assert_not_called()
            self.assertEqual(original, template.read_bytes())
            self.assertEqual(str(output / "replay.jsonl"), window.file_path.text())
            self.assertEqual("file", window.source.currentData())
            self.assertFalse(window.output_enabled.isChecked())
            self.assertEqual("custom/bindings", window.output_topic.text())
            self.assertEqual("192.0.2.10", window.output_broker.host.text())
            self.assertEqual(750, window.parameters["binding_wait_ms"].value())
            self.assertEqual({"B", "C"}, set(window.current_config()["regions"]))
            self.assertEqual(config["topics"], window.current_config()["topics"])
            saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(4567, saved["config"]["seed"])
            self.assertEqual(100, saved["config"]["update_interval_ms"])
            self.assertEqual(4, saved["config"]["bbox_jitter_px"])
            messages = list(read_jsonl(output / "replay.jsonl"))
            self.assertEqual(7209, len(messages))
            self.assertEqual(7209, saved["counts"]["replay.jsonl"])
            self.assertEqual(
                int(datetime.fromisoformat("2026-10-21T09:00:00+08:00").timestamp() * 1000),
                min(row["payload"]["captured_at_ms"] for row in messages
                    if row["topic"] == config["topics"]["detections"])
            )
            expected = list(read_jsonl(output / "expected_bindings.jsonl"))
            self.assertTrue(all(
                row["passenger_id"].endswith("-2026-10-21") for row in expected
            ))
            window.speed.setValue(2000)
            QTest.mouseClick(window.start_button, Qt.LeftButton)
            self.assertFalse(window.simulate_button.isEnabled())
            self.wait_for(lambda: window.session.paused)
            self.assertTrue(evaluate(window.session.decisions, expected)["all_events_match"])
            self.assertEqual(9, window.results_table.rowCount())
            self.assertIn("模拟数据已生成", window.logs.toPlainText())
            if os.environ.get("BINDING_UI_ARTIFACT_DIR"):
                self.assertTrue(window.grab().save(str(folder / "generated_replay.png")))
            window.stop()

    def test_generation_validation_and_nonempty_output_preserve_existing_files(self):
        dialog = SimulationDialog()
        self.windows.append(dialog)
        dialog.show()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "pose_project.apps.binding_simulation_dialog.QMessageBox.critical"
        ) as errors:
            output = Path(tmp) / "new"
            dialog.output_path.setText(str(output))
            for mutate in (
                lambda: dialog.seed.setText("-1"),
                lambda: dialog.seed.setText("invalid"),
                lambda: dialog.start_time.setText("2026-10-21T09:00:00"),
                lambda: dialog.interval.setValue(201),
                lambda: dialog.delays["detection_delay_ms"][0].setValue(500),
                lambda: dialog.output_path.clear(),
            ):
                dialog.load_template(SIMULATION_CONFIG)
                dialog.output_path.setText(str(output))
                mutate()
                before = errors.call_count
                QTest.mouseClick(dialog.generate_button, Qt.LeftButton)
                self.assertEqual(before + 1, errors.call_count)
                self.assertIsNone(dialog.worker)
                self.assertFalse(output.exists())
            dialog.load_template(SIMULATION_CONFIG)
            output.mkdir()
            old_file = output / "replay.jsonl"
            old_file.write_bytes(b"original dataset\n")
            dialog.output_path.setText(str(output))
            QTest.mouseClick(dialog.generate_button, Qt.LeftButton)
            self.assertIsNone(dialog.worker)
            self.assertEqual(b"original dataset\n", old_file.read_bytes())
            self.assertEqual([old_file], list(output.iterdir()))
            self.assertIn("新目录或空目录", dialog.status.text())

    def test_background_generation_failure_unlocks_controls_and_can_retry(self):
        window = BindingWindow()
        self.windows.append(window)
        window.show()
        window.open_simulation()
        dialog = window.simulation_dialog
        old_path = window.file_path.text()
        entered, release = threading.Event(), threading.Event()
        gui_events = []
        dialog.seed.textChanged.connect(gui_events.append)

        def fail_write(*args):
            entered.set()
            release.wait(timeout=5)
            raise OSError("测试磁盘写入失败")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "pose_project.apps.binding_simulation_dialog.QMessageBox.critical"
        ) as errors:
            dialog.output_path.setText(str(Path(tmp) / "retry"))
            try:
                with patch("pose_project.apps.binding_simulation_dialog.write_dataset", side_effect=fail_write):
                    QTest.mouseClick(dialog.generate_button, Qt.LeftButton)
                    self.wait_for(entered.is_set)
                    self.assertTrue(dialog.worker.isRunning())
                    self.assertFalse(dialog.form_panel.isEnabled())
                    self.assertFalse(dialog.cancel_button.isEnabled())
                    dialog.reject()
                    self.assertTrue(dialog.isVisible())
                    self.assertFalse(window.close())
                    dialog.seed.setText("1234")
                    QTest.qWait(20)
                    self.assertEqual(["1234"], gui_events)
                    release.set()
                    self.wait_for(lambda: errors.call_count == 1)
                self.assertEqual(old_path, window.file_path.text())
                self.assertTrue(dialog.isVisible())
                self.assertTrue(dialog.form_panel.isEnabled())
                self.assertTrue(dialog.generate_button.isEnabled())
                self.assertTrue(dialog.cancel_button.isEnabled())
                self.assertIn("测试磁盘写入失败", dialog.status.text())
                self.assertFalse(dialog.progress.isVisible())
                QTest.mouseClick(dialog.generate_button, Qt.LeftButton)
                self.wait_for(lambda: dialog.result() == dialog.Accepted)
                self.assertEqual(1, errors.call_count)
                saved = json.loads(
                    (Path(tmp) / "retry" / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(1234, saved["config"]["seed"])
            finally:
                release.set()
                self.wait_for(lambda: dialog.worker is None or not dialog.worker.isRunning())

    def test_invalid_template_keeps_previous_valid_scenario(self):
        dialog = SimulationDialog()
        self.windows.append(dialog)
        original = deepcopy(dialog.config)
        old_path = dialog.template_path.text()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "pose_project.apps.binding_simulation_dialog.QMessageBox.critical"
        ) as errors:
            template = Path(tmp) / "invalid.json"
            template.write_text('{"seed":', encoding="utf-8")
            dialog.load_template(template)
            dialog.load_template(Path(tmp) / "missing.json")
            self.assertEqual(2, errors.call_count)
            self.assertEqual(original, dialog.config)
            self.assertEqual(old_path, dialog.template_path.text())
            self.assertTrue(dialog.generate_button.isEnabled())

    def test_native_window_replay_details_filtering_and_resize(self):
        window = BindingWindow()
        self.windows.append(window)
        errors = []
        with patch("pose_project.apps.binding_app.QMessageBox.critical",
                   side_effect=lambda parent, title, text: errors.append(text)):
            window.show()
            self.assertTrue(QFontDatabase().families())
            window.speed.setValue(2000)
            QTest.mouseClick(window.start_button, Qt.LeftButton)
            self.wait_for(lambda: window.session.paused)
            self.assertEqual([], errors)
            self.assertEqual(9, window.results_table.rowCount())
            self.assertEqual(9, window.counts["bound"])
            self.assertFalse(window.region_view.scene().itemsBoundingRect().isEmpty())
            self.assertEqual(9, len(window.results))
            binding = json.loads(window.payload_view.toPlainText())
            self.assertEqual({"roi_id", "passenger_id", "rfid"}, set(binding))
            window.result_filter.setCurrentIndex(2)
            self.assertTrue(all(window.results_table.isRowHidden(row) for row in range(9)))
            window.result_filter.setCurrentIndex(0)
            self.assertTrue(all(not window.results_table.isRowHidden(row) for row in range(9)))
            for width, height in ((1320, 900), (1000, 720)):
                window.resize(width, height)
                QTest.qWait(40)
                self.assertTrue(window.start_button.isVisible())
                self.assertTrue(window.results_table.isVisible())
                self.assertGreater(window.region_view.width(), 300)
                image = window.region_view.grab().toImage()
                colors = {
                    image.pixelColor(x, y).name()
                    for x in range(0, image.width(), 10)
                    for y in range(0, image.height(), 10)
                }
                self.assertGreater(len(colors), 8)
                if os.environ.get("BINDING_UI_ARTIFACT_DIR"):
                    folder = Path(os.environ["BINDING_UI_ARTIFACT_DIR"])
                    folder.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(window.grab().save(
                        str(folder / "binding_{}x{}.png".format(width, height))
                    ))
            window.stop()


if __name__ == "__main__":
    unittest.main()
