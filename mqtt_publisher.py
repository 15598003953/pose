import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_TOPIC = "pose/put_luggage"
PUT_LUGGAGE_ACTION_TEXT = "旅客正在放置行李"
IDLE_ACTION_TEXT = "idle/heartbeat"


@dataclass(eq=True)
class MqttConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    topic: str = DEFAULT_TOPIC
    client_id: str = "pose-action-client"
    username: str = ""
    password: str = ""


def load_mqtt_config(path: Path) -> MqttConfig:
    path = Path(path)
    if not path.is_file():
        return MqttConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return MqttConfig()
    return MqttConfig(
        enabled=bool(payload.get("enabled", False)),
        host=str(payload.get("host") or "127.0.0.1"),
        port=_safe_port(payload.get("port", 1883)),
        topic=_sanitize_publish_topic(payload.get("topic") or DEFAULT_TOPIC),
        client_id=str(payload.get("client_id") or "pose-action-client"),
        username=str(payload.get("username") or ""),
        password=str(payload.get("password") or ""),
    )


def save_mqtt_config(config: MqttConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = MqttConfig(
        enabled=bool(config.enabled),
        host=str(config.host or "127.0.0.1"),
        port=_safe_port(config.port),
        topic=_sanitize_publish_topic(config.topic),
        client_id=str(config.client_id or "pose-action-client"),
        username=str(config.username or ""),
        password=str(config.password or ""),
    )
    path.write_text(
        json.dumps(asdict(normalized), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_action_payload(
        event: Dict[str, Any],
        roi_id: Any = None,
        video_path: str = "") -> Dict[str, Any]:
    action_code = str(event.get("event") or event.get("state") or "")
    action = {
        "put_luggage": PUT_LUGGAGE_ACTION_TEXT,
        "idle": IDLE_ACTION_TEXT,
        "heartbeat": IDLE_ACTION_TEXT,
    }.get(action_code, action_code)
    return {
        "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "action": action,
        "action_code": action_code,
        "roi_id": roi_id,
        "confidence": event.get("confidence"),
        "frame_idx": event.get("frame_idx"),
        "timestamp_ms": event.get("timestamp_ms"),
        "source": video_path,
    }


class MqttPublisher:
    def __init__(self, config: MqttConfig, client_factory=None):
        self.config = config
        self.client_factory = client_factory
        self._client = None
        self._available_error = ""

    @property
    def available_error(self) -> str:
        return self._available_error

    def publish_action(self, event: Dict[str, Any], roi_id=None, video_path: str = "") -> bool:
        if not self.config.enabled:
            print(
                "[MQTT] skipped disabled topic={} host={}".format(
                    self.config.topic, self.config.host
                ),
                flush=True,
            )
            return False
        client = self._get_client()
        if client is None:
            print(
                "[MQTT] connect failed topic={} host={} error={}".format(
                    self.config.topic, self.config.host, self._available_error
                ),
                flush=True,
            )
            return False
        payload = build_action_payload(event, roi_id=roi_id, video_path=video_path)
        payload_json = json.dumps(payload, ensure_ascii=False)
        try:
            publish_topic = _sanitize_publish_topic(self.config.topic)
            client.publish(
                publish_topic,
                payload_json,
                qos=0,
                retain=False,
            )
            print(
                "[MQTT] topic={} host={} payload={}".format(
                    publish_topic, self.config.host, payload_json
                ),
                flush=True,
            )
            return True
        except Exception as exc:
            self._available_error = str(exc)
            print(
                "[MQTT] publish failed topic={} host={} error={} payload={}".format(
                    self.config.topic, self.config.host, exc, payload_json
                ),
                flush=True,
            )
            return False

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            factory = self.client_factory or _default_client_factory
            client = factory(self.config.client_id)
            if self.config.username:
                client.username_pw_set(self.config.username, self.config.password or None)
            client.connect(self.config.host, self.config.port, keepalive=30)
            client.loop_start()
            self._client = client
            return self._client
        except Exception as exc:
            self._available_error = str(exc)
            return None

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        finally:
            self._client = None


def _default_client_factory(client_id: str):
    try:
        import paho.mqtt.client as mqtt
    except Exception as exc:
        raise RuntimeError("paho-mqtt is not installed") from exc
    return mqtt.Client(client_id=client_id)


def _safe_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return 1883
    return port if 0 < port < 65536 else 1883


def _sanitize_publish_topic(topic: str) -> str:
    clean = str(topic or "").strip()
    while clean.endswith("/") or clean.endswith("#") or clean.endswith("+"):
        clean = clean[:-1].rstrip()
    return clean or DEFAULT_TOPIC
