import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from mqtt_publisher import (
    MqttConfig,
    build_action_payload,
    load_mqtt_config,
    save_mqtt_config,
)


class FakeClient:
    def __init__(self):
        self.connected = None
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False
        self.published = []

    def connect(self, host, port, keepalive=30):
        self.connected = (host, port, keepalive)

    def loop_start(self):
        self.loop_started = True

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        return SimpleNamespace(rc=0)

    def loop_stop(self):
        self.loop_stopped = True

    def disconnect(self):
        self.disconnected = True


class MqttPublisherTest(unittest.TestCase):
    def test_load_missing_config_returns_disabled_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_mqtt_config(Path(tmp) / "mqtt_config.json")

        self.assertFalse(config.enabled)
        self.assertEqual("127.0.0.1", config.host)
        self.assertEqual(1883, config.port)
        self.assertEqual("pose/put_luggage", config.topic)

    def test_save_and_load_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mqtt_config.json"
            expected = MqttConfig(
                enabled=True,
                host="10.0.0.8",
                port=1884,
                topic="airport/security/action",
                client_id="pose-client-1",
                username="user",
                password="secret",
            )

            save_mqtt_config(expected, path)
            actual = load_mqtt_config(path)

        self.assertEqual(expected, actual)

    def test_build_action_payload_contains_time_action_and_roi(self):
        event = {
            "event": "put_luggage",
            "confidence": 0.82,
            "frame_idx": 123,
            "timestamp_ms": 4560,
        }

        payload = build_action_payload(event, roi_id=2, video_path="rtsp://camera/main")

        self.assertEqual("旅客正在放置行李", payload["action"])
        self.assertEqual("put_luggage", payload["action_code"])
        self.assertEqual(2, payload["roi_id"])
        self.assertEqual(0.82, payload["confidence"])
        self.assertEqual(123, payload["frame_idx"])
        self.assertEqual(4560, payload["timestamp_ms"])
        self.assertEqual("rtsp://camera/main", payload["source"])
        self.assertRegex(payload["time"], r"^\d{4}-\d{2}-\d{2}T")

    def test_invalid_json_config_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mqtt_config.json"
            path.write_text("{bad", encoding="utf-8")

            config = load_mqtt_config(path)

        self.assertFalse(config.enabled)

    def test_publish_action_strips_wildcards_from_publish_topic(self):
        client = FakeClient()
        config = MqttConfig(
            enabled=True,
            host="127.0.0.1",
            port=1883,
            topic="testtopic/#",
            client_id="pose-client",
        )
        publisher = __import__("mqtt_publisher").MqttPublisher(
            config,
            client_factory=lambda client_id: client,
        )

        ok = publisher.publish_action(
            {"event": "idle", "confidence": 0.0},
            roi_id="FULL",
            video_path="startup",
        )

        self.assertTrue(ok)
        self.assertEqual("testtopic", client.published[0][0])
        self.assertTrue(client.loop_started)
        self.assertEqual(("127.0.0.1", 1883, 30), client.connected)

    def test_save_mqtt_config_strips_wildcards_from_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mqtt_config.json"
            expected = MqttConfig(
                enabled=True,
                host="192.168.1.85",
                port=1883,
                topic="testtopic/#",
                client_id="pose-client",
            )

            save_mqtt_config(expected, path)
            actual = load_mqtt_config(path)
            raw = path.read_text(encoding="utf-8")

        self.assertEqual("testtopic", actual.topic)
        self.assertNotIn("#", raw)


if __name__ == "__main__":
    unittest.main()
