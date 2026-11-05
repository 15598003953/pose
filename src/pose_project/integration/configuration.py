"""Independent, validated settings for the integration workbench."""

from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit
import uuid

from pose_project.binding.binding_runtime import validate_broker, validate_topic
from pose_project.binding.passenger_tray_binding import validate_config
from pose_project.common.project_paths import APP_DIR, CONFIG_DIR, RESULT_DIR, ROOT


DEFAULT_CONFIG = CONFIG_DIR / "integration_config.json"


def is_stream_source(source):
    return "://" in source


def validate_video_source(source, name="视频"):
    if not source:
        raise ValueError("请选择{}文件或填写 RTSP 地址".format(name))
    if is_stream_source(source):
        try:
            url = urlsplit(source)
            if (url.scheme.lower() not in
                    ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "udp", "tcp")
                    or not url.hostname or any(char.isspace() for char in source)
                    or (url.port is not None and not 1 <= url.port <= 65535)):
                raise ValueError()
        except ValueError:
            raise ValueError("{}流地址无效，请检查协议、主机和端口".format(name)) from None
    elif not Path(source).is_file():
        raise ValueError("{}文件不存在：{}".format(name, source))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def defaults():
    gate = read_json(ROOT / "check_gate" / "extractor_config.json")
    binding = read_json(CONFIG_DIR / "binding_config.json")
    binding["topics"] = {"detections": "reidtest", "rfid": "pose/rfid"}
    return {
        "transport": "local",
        "gate": dict(gate, source=""),
        "reid": {
            "source": "", "rotate180": False, "threshold": 0.65,
            "image_size": [1920, 1080], "enabled_regions": ["A"],
            "detector_weights": str(ROOT / "reid1026001" / "yolov8s.pt"),
            "reid_weights": str(ROOT / "reid1026001" / "weights" / "msmt_sbs_R50.pth"),
        },
        "binding": binding,
        "mqtt": {
            "host": "127.0.0.1", "port": 1883, "username": "", "tls": False,
            "person_topic": "check_gate/person/collected",
            "output_topic": "pose/bindings", "output_enabled": False,
        },
        "output_dir": str(RESULT_DIR / "integration"),
    }


def validate(settings, sources=False):
    if settings["transport"] not in ("local", "mqtt"):
        raise ValueError("消息连接方式必须为 local 或 mqtt")
    validate_config(settings["binding"])
    mqtt = settings["mqtt"]
    validate_broker(mqtt)
    topics = [mqtt["person_topic"], mqtt["output_topic"],
              *settings["binding"]["topics"].values()]
    for topic in topics:
        validate_topic(topic)
    if len(set(topics)) != len(topics):
        raise ValueError("旅客、位置、RFID 和绑定结果必须使用不同主题")
    reid = settings["reid"]
    width, height = reid["image_size"]
    if any(type(v) is not int or v < 16 for v in (width, height)):
        raise ValueError("供框台画面尺寸必须是大于 15 的整数")
    if not 0 <= reid["threshold"] <= 1:
        raise ValueError("重识别阈值必须位于 0 到 1")
    if not reid["enabled_regions"] or set(reid["enabled_regions"]) - {"A", "B"}:
        raise ValueError("至少启用 A、B 中的一个区域")
    for key in reid["enabled_regions"]:
        if key not in settings["binding"]["regions"]:
            raise ValueError("缺少区域配置：" + key)
    for region in settings["binding"]["regions"].values():
        x1, y1, x2, y2 = region["roi_xyxy"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("供框台区域超出画面尺寸，请重新标定")
    roi = settings["gate"]["roi"]
    if not (0 <= roi["x"] < roi["x"] + roi["w"] <= 1
            and 0 <= roi["y"] < roi["y"] + roi["h"] <= 1):
        raise ValueError("闸机抓拍区域必须位于画面内")
    if not str(settings["output_dir"]).strip():
        raise ValueError("输出目录不能为空")
    if sources:
        for stage, name in (("gate", "闸机视频"), ("reid", "供框台视频")):
            validate_video_source(settings[stage]["source"].strip(), name)


def load_settings(path=DEFAULT_CONFIG):
    path = Path(path)
    settings = read_json(path) if path.exists() else defaults()
    for values, key in _file_paths(settings):
        value = values[key]
        if value and not is_stream_source(value) and not Path(value).is_absolute():
            values[key] = str((path.resolve().parent / value).resolve())
    validate(settings)
    return settings


def _file_paths(settings):
    yield settings, "output_dir"
    for stage in ("gate", "reid"):
        yield settings[stage], "source"
    for key in ("detector_weights", "reid_weights"):
        yield settings["reid"], key


def save_settings(settings, path=DEFAULT_CONFIG):
    validate(settings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".bak")
        shutil.copy2(path, backup)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    # Passwords are supplied per launch; never put them in the saved config.
    saved = deepcopy(settings)
    saved["mqtt"].pop("password", None)
    if getattr(sys, "frozen", False):
        for values, key in _file_paths(saved):
            value = values[key]
            if value and not is_stream_source(value):
                target = Path(value)
                if target.is_absolute() and target.resolve().is_relative_to(APP_DIR.resolve()):
                    values[key] = os.path.relpath(target, path.resolve().parent)
    try:
        temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        if read_json(temporary) != saved:
            raise OSError("配置回读不一致")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def new_run_directory(settings):
    parent = Path(settings["output_dir"]).resolve()
    path = parent / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_report(settings):
    modules = ("PyQt5", "cv2", "numpy", "paho.mqtt.client", "torch",
               "torchvision", "ultralytics", "lap")
    available = {}
    for name in modules:
        try:
            available[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available[name] = False
    files = {
        "gate_detector": ROOT / "check_gate/models/yolo11n-pose.onnx",
        "gate_face": ROOT / "check_gate/models/face_detection_yunet_2023mar.onnx",
        "person_detector": Path(settings["reid"]["detector_weights"]),
        "reidentification": Path(settings["reid"]["reid_weights"]),
    }
    weights = {name: {"path": str(path), "exists": path.is_file()}
               for name, path in files.items()}
    return {"ok": all(available.values()) and all(v["exists"] for v in weights.values()),
            "packaged": bool(getattr(sys, "frozen", False)),
            "python": sys.executable, "modules": available, "weights": weights}
