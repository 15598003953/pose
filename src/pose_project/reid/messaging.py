"""重识别独立界面的 MQTT 通信与旅客图片解码。"""

import base64
import json
import re

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, pyqtSignal


class MqttReceiver(QObject):
    """通过 MQTT 接收人物消息（后台网络线程 -> Qt 信号转发到主线程）。

    连接 broker 并订阅指定 topic；每条 JSON 消息解析出 images.upper
    （base64 上半身图）后 emit 给主线程建立人物库条目。
    """

    person_received = pyqtSignal(dict)   # 解析成功的 JSON dict
    log = pyqtSignal(str)
    connected_changed = pyqtSignal(bool)  # 连接状态变化

    def __init__(self, host, port, topic):
        super().__init__()
        self.host, self.port, self.topic = host, port, topic
        # paho-mqtt 2.x 要求显式 CallbackAPIVersion；1.x 无此参数
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self._running = False

    def start(self):
        self.client.connect_async(self.host, self.port, keepalive=60)
        self.client.loop_start()
        self._running = True

    def stop(self):
        if self._running:
            self._running = False
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client.loop_stop()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self.topic)
            self.log.emit(f"MQTT 已连接 {self.host}:{self.port}，订阅 topic：{self.topic}")
            self.connected_changed.emit(True)
        else:
            self.log.emit(f"MQTT 连接失败（rc={rc}）：{self.host}:{self.port}")
            self.connected_changed.emit(False)

    def _on_disconnect(self, client, userdata, rc):
        self.log.emit("MQTT 已断开。")
        self.connected_changed.emit(False)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
            if not isinstance(payload, dict):
                raise ValueError("消息不是 JSON 对象")
        except Exception as e:
            self.log.emit(f"MQTT 消息解析失败：{e}")
            return
        self.person_received.emit(payload)


class MqttPublisher(QObject):
    """ROI 检测结果发布器：连接 broker 并向 topic「reidtest」发送 JSON。

    与 Step1 的 MqttReceiver 使用各自的客户端（broker 允许多连接），
    broker 地址取 Step1 输入框的值。连接建立前 publish 的消息会缓存，
    连上后自动补发，避免检索刚开始时的消息丢失。
    """

    log = pyqtSignal(str)

    def __init__(self, host, port, topic):
        super().__init__()
        self.host, self.port, self.topic = host, port, topic
        # paho-mqtt 2.x 要求显式 CallbackAPIVersion；1.x 无此参数
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._pending = []      # 连接建立前的缓存消息（最多保留最近 100 条）
        self._running = False

    def start(self):
        self._running = True
        self.client.connect_async(self.host, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        if self._running:
            self._running = False
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client.loop_stop()

    def publish(self, payload):
        """发布一条 JSON 字符串；未连上时先缓存，连上后自动补发。"""
        if self._running and self.client.is_connected():
            try:
                self.client.publish(self.topic, payload, qos=1)
            except Exception as e:
                self.log.emit(f"MQTT 发布失败：{e}")
        else:
            self._pending.append(payload)
            if len(self._pending) > 100:
                self._pending = self._pending[-100:]

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            n = len(self._pending)
            for p in self._pending:
                try:
                    client.publish(self.topic, p, qos=1)
                except Exception:
                    pass
            self._pending.clear()
            msg = f"ROI 上报已连接 MQTT {self.host}:{self.port}，topic：{self.topic}"
            self.log.emit(msg + (f"（补发缓存 {n} 条）" if n else ""))
        else:
            self.log.emit(f"ROI 上报 MQTT 连接失败（rc={rc}）：{self.host}:{self.port}")

    def _on_disconnect(self, client, userdata, rc):
        # 断线由 paho 自动重连，期间消息进入缓存，无需提示
        pass


def decode_base64_image(b64):
    """base64 字符串 -> BGR 图像（兼容 data:image/...;base64, 前缀）。

    解码或解码后不是有效图像时返回 None。
    """
    if not b64 or not isinstance(b64, str):
        return None
    data = re.sub(r"^data:image/\w+;base64,", "", b64.strip())
    try:
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None
